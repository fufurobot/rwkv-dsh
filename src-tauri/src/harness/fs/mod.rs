use std::sync::Arc;

use crate::cordis::{Context, Error, Plugin};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FsKind {
    Workspace,
    Emulated,
}

#[derive(Debug)]
pub struct FsService {
    kind: FsKind,
}

impl FsService {
    pub fn kind(&self) -> FsKind {
        self.kind
    }
}

pub struct FsPlugin {
    kind: FsKind,
}

impl FsPlugin {
    pub fn workspace() -> Self {
        Self {
            kind: FsKind::Workspace,
        }
    }

    pub fn emulated() -> Self {
        Self {
            kind: FsKind::Emulated,
        }
    }
}

impl Plugin for FsPlugin {
    fn name(&self) -> &'static str {
        match self.kind {
            FsKind::Workspace => "fs/workspace",
            FsKind::Emulated => "fs/emulated",
        }
    }

    fn inject(&self) -> &'static [&'static str] {
        &["settings"]
    }

    fn apply(&self, ctx: &mut Context) -> Result<(), Error> {
        ctx.provide("fs", Arc::new(FsService { kind: self.kind }))
    }
}
