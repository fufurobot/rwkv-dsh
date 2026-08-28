use std::sync::Arc;

use super::Plugin;

pub(crate) type Disposer = Box<dyn FnOnce(&mut super::Context) + Send>;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FiberState {
    Pending,
    Applying,
    Active,
}

pub struct Fiber {
    pub(crate) plugin: Arc<dyn Plugin>,
    pub(crate) state: FiberState,
    pub(crate) effects: Vec<Disposer>,
}

impl Fiber {
    pub fn new(plugin: Arc<dyn Plugin>) -> Self {
        Self {
            plugin,
            state: FiberState::Pending,
            effects: Vec::new(),
        }
    }

    pub fn name(&self) -> &'static str {
        self.plugin.name()
    }

    pub fn inject(&self) -> &'static [&'static str] {
        self.plugin.inject()
    }

    pub fn state(&self) -> FiberState {
        self.state
    }

    pub fn effect_count(&self) -> usize {
        self.effects.len()
    }
}
