use std::sync::{Arc, Mutex};

use crate::agent::{Message, SessionStore};
use crate::cordis::{Context, Error, Plugin};

pub struct SessionsService {
    pub store: Arc<Mutex<SessionStore>>,
    pub history: Arc<Mutex<Vec<Message>>>,
}

pub struct SessionsPlugin {
    service: Arc<SessionsService>,
}

impl SessionsPlugin {
    pub fn new(store: Arc<Mutex<SessionStore>>, history: Arc<Mutex<Vec<Message>>>) -> Self {
        Self {
            service: Arc::new(SessionsService { store, history }),
        }
    }
}

impl Plugin for SessionsPlugin {
    fn name(&self) -> &'static str {
        "sessions/store"
    }

    fn inject(&self) -> &'static [&'static str] {
        &["settings"]
    }

    fn apply(&self, ctx: &mut Context) -> Result<(), Error> {
        ctx.provide("sessions", Arc::clone(&self.service))
    }
}
