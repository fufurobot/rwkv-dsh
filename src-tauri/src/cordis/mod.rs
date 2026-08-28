mod context;
mod error;
mod events;
mod fiber;
mod plugin;

pub use context::{Context, ToolExecutor, ToolRegistration};
pub use error::Error;
pub use events::{is_persistent, EventHandler, EventRecord};
pub use fiber::{Fiber, FiberState};
pub use plugin::Plugin;

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use std::sync::{Arc, Mutex};

    struct Consumer {
        disposed: Arc<Mutex<Vec<&'static str>>>,
    }

    impl Plugin for Consumer {
        fn name(&self) -> &'static str {
            "consumer"
        }

        fn inject(&self) -> &'static [&'static str] {
            &["tools"]
        }

        fn apply(&self, ctx: &mut Context) -> Result<(), Error> {
            let disposed = Arc::clone(&self.disposed);
            ctx.effect(move |_| disposed.lock().expect("dispose lock").push("first"))?;
            let disposed = Arc::clone(&self.disposed);
            ctx.effect(move |_| disposed.lock().expect("dispose lock").push("second"))?;
            ctx.on(
                "intent/classify",
                Arc::new(|payload| {
                    payload["seen"] = json!(true);
                    Ok(())
                }),
            )?;
            ctx.register_tool(ToolRegistration {
                name: "read_file".into(),
                description: "read".into(),
                arguments: json!({"path": {"type": "string"}}),
                execute: Arc::new(|_| Ok(json!({"ok": true}))),
            })?;
            Ok(())
        }
    }

    struct ToolsProvider;

    impl Plugin for ToolsProvider {
        fn name(&self) -> &'static str {
            "tools-provider"
        }

        fn apply(&self, ctx: &mut Context) -> Result<(), Error> {
            ctx.provide("tools", Arc::new(String::from("registry")))
        }
    }

    struct FailingConsumer {
        disposed: Arc<Mutex<Vec<&'static str>>>,
    }

    impl Plugin for FailingConsumer {
        fn name(&self) -> &'static str {
            "failing-consumer"
        }

        fn inject(&self) -> &'static [&'static str] {
            &["tools"]
        }

        fn apply(&self, ctx: &mut Context) -> Result<(), Error> {
            ctx.provide("partial", Arc::new(String::from("temporary")))?;
            ctx.on("agent/pre-step", Arc::new(|_| Ok(())))?;
            ctx.register_tool(ToolRegistration {
                name: "partial_tool".into(),
                description: "temporary".into(),
                arguments: json!({}),
                execute: Arc::new(|_| Ok(json!(null))),
            })?;
            let disposed = Arc::clone(&self.disposed);
            ctx.effect(move |_| disposed.lock().expect("dispose lock").push("custom"))?;
            Err(Error::Event("apply failed".into()))
        }
    }

    struct OrderedListeners;

    impl Plugin for OrderedListeners {
        fn name(&self) -> &'static str {
            "ordered-listeners"
        }

        fn apply(&self, ctx: &mut Context) -> Result<(), Error> {
            ctx.on(
                "agent/pre-step",
                Arc::new(|payload| {
                    payload
                        .as_array_mut()
                        .ok_or_else(|| Error::Event("expected array".into()))?
                        .push(json!("first"));
                    Ok(())
                }),
            )?;
            ctx.on(
                "agent/pre-step",
                Arc::new(|payload| {
                    payload
                        .as_array_mut()
                        .ok_or_else(|| Error::Event("expected array".into()))?
                        .push(json!("second"));
                    Ok(())
                }),
            )?;
            Ok(())
        }
    }

    #[test]
    fn pending_fiber_activates_when_dependency_arrives() {
        let disposed = Arc::new(Mutex::new(Vec::new()));
        let mut ctx = Context::new();
        ctx.load(Consumer {
            disposed: Arc::clone(&disposed),
        })
        .unwrap();
        assert_eq!(ctx.fiber_state("consumer"), Some(FiberState::Pending));
        assert!(ctx.tool_names().is_empty());

        ctx.load(ToolsProvider).unwrap();
        assert_eq!(ctx.fiber_state("consumer"), Some(FiberState::Active));
        let fiber = ctx.fiber("consumer").expect("consumer fiber");
        assert_eq!(fiber.name(), "consumer");
        assert_eq!(fiber.inject(), &["tools"]);
        assert_eq!(fiber.state(), FiberState::Active);
        assert_eq!(fiber.effect_count(), 4);
        assert_eq!(
            ctx.service::<String>("tools").as_deref(),
            Some(&String::from("registry"))
        );
        assert_eq!(ctx.tool_names(), vec!["read_file"]);
        let payload = ctx.waterfall("intent/classify", json!({})).unwrap();
        assert_eq!(payload["seen"], true);
    }

    #[test]
    fn unloading_provider_cascades_and_disposes_lifo() {
        let disposed = Arc::new(Mutex::new(Vec::new()));
        let mut ctx = Context::new();
        ctx.load(ToolsProvider).unwrap();
        ctx.load(Consumer {
            disposed: Arc::clone(&disposed),
        })
        .unwrap();

        ctx.unload("tools-provider").unwrap();

        assert!(!ctx.has_service("tools"));
        assert_eq!(ctx.fiber_state("consumer"), Some(FiberState::Pending));
        assert!(ctx.tool_names().is_empty());
        assert_eq!(
            ctx.waterfall("intent/classify", json!({})).unwrap(),
            json!({})
        );
        assert_eq!(
            *disposed.lock().expect("dispose lock"),
            vec!["second", "first"]
        );
    }

    #[test]
    fn persistent_events_are_logged_but_waterfalls_are_not() {
        let mut ctx = Context::new();
        ctx.emit("intent/resolved", json!({"intent": "chat"}))
            .unwrap();
        ctx.emit("intent/classify", json!({"text": "hello"}))
            .unwrap();
        assert_eq!(ctx.event_log().len(), 1);
        assert_eq!(ctx.event_log()[0].name, "intent/resolved");
    }

    #[test]
    fn failed_apply_rolls_back_every_partial_effect() {
        let disposed = Arc::new(Mutex::new(Vec::new()));
        let mut ctx = Context::new();
        ctx.load(ToolsProvider).unwrap();

        let error = ctx
            .load(FailingConsumer {
                disposed: Arc::clone(&disposed),
            })
            .expect_err("apply must fail");

        assert!(matches!(
            error,
            Error::Apply {
                plugin: "failing-consumer",
                ..
            }
        ));
        assert_eq!(
            ctx.fiber_state("failing-consumer"),
            Some(FiberState::Pending)
        );
        assert!(!ctx.has_service("partial"));
        assert!(ctx.tool("partial_tool").is_none());
        assert_eq!(
            ctx.waterfall("agent/pre-step", json!([])).unwrap(),
            json!([])
        );
        assert_eq!(*disposed.lock().expect("dispose lock"), vec!["custom"]);
    }

    #[test]
    fn replacement_provider_reactivates_consumer_without_duplicate_effects() {
        let disposed = Arc::new(Mutex::new(Vec::new()));
        let mut ctx = Context::new();
        ctx.load(Consumer {
            disposed: Arc::clone(&disposed),
        })
        .unwrap();
        ctx.load(ToolsProvider).unwrap();
        ctx.unload("tools-provider").unwrap();
        assert_eq!(ctx.fiber_state("consumer"), Some(FiberState::Pending));

        ctx.load(ToolsProvider).unwrap();

        assert_eq!(ctx.fiber_state("consumer"), Some(FiberState::Active));
        assert_eq!(ctx.tool_names(), vec!["read_file"]);
        assert_eq!(
            ctx.waterfall("intent/classify", json!({})).unwrap(),
            json!({"seen": true})
        );
        assert_eq!(
            *disposed.lock().expect("dispose lock"),
            vec!["second", "first"]
        );
    }

    #[test]
    fn waterfall_listeners_run_in_registration_order() {
        let mut ctx = Context::new();
        ctx.load(OrderedListeners).unwrap();

        let payload = ctx.waterfall("agent/pre-step", json!([])).unwrap();

        assert_eq!(payload, json!(["first", "second"]));
    }
}
