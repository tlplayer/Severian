use std::fmt;

use crate::{ModuleId, Symbol, SymbolId};

#[derive(Clone, Debug, PartialEq)]
pub struct ModuleInterface {
    pub id: ModuleId,
    pub path: ModulePath,
    pub symbols: Vec<Symbol>,
    pub exports: Vec<SymbolId>,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct ModulePath(pub Vec<String>);

impl ModulePath {
    pub fn root() -> Self {
        Self(Vec::new())
    }

    pub fn new<I, S>(parts: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        Self(parts.into_iter().map(Into::into).collect())
    }

    pub fn child(&self, part: impl Into<String>) -> Self {
        let mut parts = self.0.clone();
        parts.push(part.into());
        Self(parts)
    }
}

impl fmt::Display for ModulePath {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.0.is_empty() {
            return f.write_str("<root>");
        }
        f.write_str(&self.0.join("."))
    }
}
