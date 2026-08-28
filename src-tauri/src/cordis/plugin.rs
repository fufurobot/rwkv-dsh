use super::{Context, Error};

pub trait Plugin: Send + Sync {
    fn name(&self) -> &'static str;

    fn inject(&self) -> &'static [&'static str] {
        &[]
    }

    fn apply(&self, ctx: &mut Context) -> Result<(), Error>;
}
