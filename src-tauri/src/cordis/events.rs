use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::sync::Arc;

use super::Error;

pub type EventHandler = Arc<dyn Fn(&mut Value) -> Result<(), Error> + Send + Sync>;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EventRecord {
    pub name: String,
    pub payload: Value,
}

pub(crate) struct Listener {
    pub id: u64,
    pub owner: &'static str,
    pub handler: EventHandler,
}

pub fn is_persistent(name: &str) -> bool {
    name.starts_with("turn/")
        || name.starts_with("step/")
        || name == "user/message"
        || name == "assistant/chunk"
        || name.starts_with("tool/")
        || name == "intent/resolved"
}
