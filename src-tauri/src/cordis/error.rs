use std::fmt::{Display, Formatter};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Error {
    PluginExists(&'static str),
    PluginNotFound(String),
    NoActiveFiber,
    ServiceExists(&'static str),
    ToolExists(String),
    Apply {
        plugin: &'static str,
        message: String,
    },
    Event(String),
}

impl Error {
    pub fn apply(plugin: &'static str, message: impl Into<String>) -> Self {
        Self::Apply {
            plugin,
            message: message.into(),
        }
    }
}

impl Display for Error {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::PluginExists(name) => write!(formatter, "plugin {name:?} is already loaded"),
            Self::PluginNotFound(name) => write!(formatter, "plugin {name:?} is not loaded"),
            Self::NoActiveFiber => formatter.write_str("effect requires an applying plugin fiber"),
            Self::ServiceExists(name) => {
                write!(formatter, "service {name:?} already has a provider")
            }
            Self::ToolExists(name) => write!(formatter, "tool {name:?} is already registered"),
            Self::Apply { plugin, message } => {
                write!(formatter, "plugin {plugin:?} failed to apply: {message}")
            }
            Self::Event(message) => formatter.write_str(message),
        }
    }
}

impl std::error::Error for Error {}
