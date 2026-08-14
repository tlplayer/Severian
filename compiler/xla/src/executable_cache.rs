use crate::{LoadedExecutable, Result, XlaError};
use severian_model_ir::CompileKey;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

/// Process-local cache for specialized PJRT executables.
///
/// Compilation runs while the cache lock is held. This deliberately favors a
/// simple guarantee—one compilation per key—over parallel compilation. A
/// persistent cache can implement the same key contract without changing model
/// loading or generation APIs.
#[derive(Default)]
pub struct ExecutableCache {
    entries: Mutex<HashMap<CompileKey, Arc<LoadedExecutable>>>,
}

impl ExecutableCache {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn get(&self, key: &CompileKey) -> Result<Option<Arc<LoadedExecutable>>> {
        let entries = self
            .entries
            .lock()
            .map_err(|_| XlaError::Compilation("executable cache lock was poisoned".to_owned()))?;
        Ok(entries.get(key).cloned())
    }

    pub fn get_or_compile(
        &self,
        key: CompileKey,
        compile: impl FnOnce() -> Result<LoadedExecutable>,
    ) -> Result<Arc<LoadedExecutable>> {
        let mut entries = self
            .entries
            .lock()
            .map_err(|_| XlaError::Compilation("executable cache lock was poisoned".to_owned()))?;
        if let Some(executable) = entries.get(&key) {
            return Ok(Arc::clone(executable));
        }

        let executable = Arc::new(compile()?);
        entries.insert(key, Arc::clone(&executable));
        Ok(executable)
    }

    pub fn len(&self) -> Result<usize> {
        let entries = self
            .entries
            .lock()
            .map_err(|_| XlaError::Compilation("executable cache lock was poisoned".to_owned()))?;
        Ok(entries.len())
    }

    pub fn is_empty(&self) -> Result<bool> {
        Ok(self.len()? == 0)
    }

    pub fn clear(&self) -> Result<()> {
        let mut entries = self
            .entries
            .lock()
            .map_err(|_| XlaError::Compilation("executable cache lock was poisoned".to_owned()))?;
        entries.clear();
        Ok(())
    }
}
