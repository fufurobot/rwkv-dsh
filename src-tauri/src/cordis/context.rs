use serde_json::Value;
use std::any::Any;
use std::collections::{BTreeMap, HashMap};
use std::sync::Arc;

use super::events::{is_persistent, EventHandler, EventRecord, Listener};
use super::fiber::{Fiber, FiberState};
use super::{Error, Plugin};

type Service = Arc<dyn Any + Send + Sync>;

struct ServiceSlot {
    owner: &'static str,
    value: Service,
}

pub type ToolExecutor = Arc<dyn Fn(&Value) -> Result<Value, Error> + Send + Sync>;

#[derive(Clone)]
pub struct ToolRegistration {
    pub name: String,
    pub description: String,
    pub arguments: Value,
    pub execute: ToolExecutor,
}

struct ToolSlot {
    owner: &'static str,
    registration: ToolRegistration,
}

#[derive(Default)]
pub struct Context {
    fibers: BTreeMap<&'static str, Fiber>,
    services: HashMap<&'static str, ServiceSlot>,
    listeners: HashMap<String, Vec<Listener>>,
    tools: BTreeMap<String, ToolSlot>,
    events: Vec<EventRecord>,
    current_fiber: Option<&'static str>,
    next_listener_id: u64,
}

impl Context {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn load<P>(&mut self, plugin: P) -> Result<(), Error>
    where
        P: Plugin + 'static,
    {
        self.load_arc(Arc::new(plugin))
    }

    pub fn load_arc(&mut self, plugin: Arc<dyn Plugin>) -> Result<(), Error> {
        let name = plugin.name();
        if self.fibers.contains_key(name) {
            return Err(Error::PluginExists(name));
        }
        self.fibers.insert(name, Fiber::new(plugin));
        self.reconcile()
    }

    pub fn unload(&mut self, name: &str) -> Result<(), Error> {
        let owned_name = self
            .fibers
            .keys()
            .copied()
            .find(|candidate| *candidate == name)
            .ok_or_else(|| Error::PluginNotFound(name.to_string()))?;
        self.deactivate_tree(owned_name);
        self.fibers.remove(owned_name);
        Ok(())
    }

    pub fn fiber_state(&self, name: &str) -> Option<FiberState> {
        self.fibers.get(name).map(|fiber| fiber.state)
    }

    pub fn fiber(&self, name: &str) -> Option<&Fiber> {
        self.fibers.get(name)
    }

    pub fn fibers(&self) -> impl Iterator<Item = &Fiber> {
        self.fibers.values()
    }

    pub fn active_plugins(&self) -> Vec<&'static str> {
        self.fibers
            .iter()
            .filter_map(|(name, fiber)| (fiber.state == FiberState::Active).then_some(*name))
            .collect()
    }

    pub fn provide<T>(&mut self, name: &'static str, value: Arc<T>) -> Result<(), Error>
    where
        T: Any + Send + Sync,
    {
        let owner = self.owner()?;
        if self.services.contains_key(name) {
            return Err(Error::ServiceExists(name));
        }
        self.services.insert(name, ServiceSlot { owner, value });
        self.effect(move |ctx| {
            let remove = ctx
                .services
                .get(name)
                .is_some_and(|slot| slot.owner == owner);
            if remove {
                ctx.services.remove(name);
            }
        })
    }

    pub fn service<T>(&self, name: &str) -> Option<Arc<T>>
    where
        T: Any + Send + Sync,
    {
        self.services
            .get(name)
            .map(|slot| Arc::clone(&slot.value))
            .and_then(|value| value.downcast::<T>().ok())
    }

    pub fn has_service(&self, name: &str) -> bool {
        self.services.contains_key(name)
    }

    pub fn effect<F>(&mut self, disposer: F) -> Result<(), Error>
    where
        F: FnOnce(&mut Context) + Send + 'static,
    {
        let owner = self.owner()?;
        let fiber = self.fibers.get_mut(owner).ok_or(Error::NoActiveFiber)?;
        fiber.effects.push(Box::new(disposer));
        Ok(())
    }

    pub fn on(&mut self, name: impl Into<String>, handler: EventHandler) -> Result<u64, Error> {
        let owner = self.owner()?;
        self.next_listener_id = self.next_listener_id.wrapping_add(1).max(1);
        let id = self.next_listener_id;
        let name = name.into();
        self.listeners
            .entry(name.clone())
            .or_default()
            .push(Listener { id, owner, handler });
        self.effect(move |ctx| {
            if let Some(listeners) = ctx.listeners.get_mut(&name) {
                listeners.retain(|listener| listener.id != id || listener.owner != owner);
                if listeners.is_empty() {
                    ctx.listeners.remove(&name);
                }
            }
        })?;
        Ok(id)
    }

    pub fn emit(&mut self, name: &str, payload: Value) -> Result<Value, Error> {
        let payload = self.dispatch(name, payload)?;
        if is_persistent(name) {
            self.events.push(EventRecord {
                name: name.to_string(),
                payload: payload.clone(),
            });
        }
        Ok(payload)
    }

    pub fn waterfall(&self, name: &str, payload: Value) -> Result<Value, Error> {
        self.dispatch(name, payload)
    }

    pub fn event_log(&self) -> &[EventRecord] {
        &self.events
    }

    pub fn event_handlers(&self, name: &str) -> Vec<EventHandler> {
        self.listeners
            .get(name)
            .into_iter()
            .flatten()
            .map(|listener| Arc::clone(&listener.handler))
            .collect()
    }

    pub fn record_event(&mut self, name: &str, payload: Value) {
        if is_persistent(name) {
            self.events.push(EventRecord {
                name: name.to_string(),
                payload,
            });
        }
    }

    pub fn take_event_log(&mut self) -> Vec<EventRecord> {
        std::mem::take(&mut self.events)
    }

    pub fn register_tool(&mut self, tool: ToolRegistration) -> Result<(), Error> {
        let owner = self.owner()?;
        if self.tools.contains_key(&tool.name) {
            return Err(Error::ToolExists(tool.name));
        }
        let name = tool.name.clone();
        self.tools.insert(
            name.clone(),
            ToolSlot {
                owner,
                registration: tool,
            },
        );
        self.effect(move |ctx| {
            let remove = ctx.tools.get(&name).is_some_and(|slot| slot.owner == owner);
            if remove {
                ctx.tools.remove(&name);
            }
        })
    }

    pub fn tool(&self, name: &str) -> Option<ToolRegistration> {
        self.tools.get(name).map(|slot| slot.registration.clone())
    }

    pub fn tool_names(&self) -> Vec<String> {
        self.tools.keys().cloned().collect()
    }

    fn owner(&self) -> Result<&'static str, Error> {
        self.current_fiber.ok_or(Error::NoActiveFiber)
    }

    fn dispatch(&self, name: &str, mut payload: Value) -> Result<Value, Error> {
        let handlers = self.event_handlers(name);
        for handler in handlers {
            handler(&mut payload)?;
        }
        Ok(payload)
    }

    fn reconcile(&mut self) -> Result<(), Error> {
        loop {
            let next = self.fibers.iter().find_map(|(name, fiber)| {
                (fiber.state == FiberState::Pending
                    && fiber
                        .plugin
                        .inject()
                        .iter()
                        .all(|dependency| self.services.contains_key(dependency)))
                .then_some(*name)
            });
            let Some(name) = next else {
                return Ok(());
            };
            let Some(plugin) = self.fibers.get(name).map(|fiber| Arc::clone(&fiber.plugin)) else {
                return Err(Error::PluginNotFound(name.to_string()));
            };
            if let Some(fiber) = self.fibers.get_mut(name) {
                fiber.state = FiberState::Applying;
            }
            self.current_fiber = Some(name);
            let result = plugin.apply(self);
            self.current_fiber = None;
            match result {
                Ok(()) => {
                    if let Some(fiber) = self.fibers.get_mut(name) {
                        fiber.state = FiberState::Active;
                    }
                }
                Err(error) => {
                    self.dispose(name);
                    if let Some(fiber) = self.fibers.get_mut(name) {
                        fiber.state = FiberState::Pending;
                    }
                    return Err(Error::apply(name, error.to_string()));
                }
            }
        }
    }

    fn deactivate_tree(&mut self, name: &'static str) {
        let provided: Vec<&'static str> = self
            .services
            .iter()
            .filter_map(|(service, slot)| (slot.owner == name).then_some(*service))
            .collect();
        let dependents: Vec<&'static str> = self
            .fibers
            .iter()
            .filter_map(|(candidate, fiber)| {
                (*candidate != name
                    && fiber.state == FiberState::Active
                    && fiber
                        .plugin
                        .inject()
                        .iter()
                        .any(|dependency| provided.contains(dependency)))
                .then_some(*candidate)
            })
            .collect();
        for dependent in dependents {
            self.deactivate_tree(dependent);
        }
        self.dispose(name);
        if let Some(fiber) = self.fibers.get_mut(name) {
            fiber.state = FiberState::Pending;
        }
    }

    fn dispose(&mut self, name: &'static str) {
        loop {
            let effect = self
                .fibers
                .get_mut(name)
                .and_then(|fiber| fiber.effects.pop());
            let Some(effect) = effect else {
                break;
            };
            effect(self);
        }
    }
}
