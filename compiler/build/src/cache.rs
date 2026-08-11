use crate::{fingerprint::Fingerprint, node::BuildNode};
use std::{
    fs, io,
    path::{Path, PathBuf},
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CacheStatus {
    Fresh,
    Dirty,
    MissingOutput,
    Uncacheable,
}

#[derive(Debug, Clone)]
pub struct BuildCache {
    root: PathBuf,
}

impl BuildCache {
    pub fn new(target_directory: impl Into<PathBuf>) -> Self {
        Self {
            root: target_directory.into().join(".sev-cache"),
        }
    }

    pub fn root(&self) -> &Path {
        &self.root
    }

    pub fn status(&self, node: &BuildNode, fingerprint: Fingerprint) -> io::Result<CacheStatus> {
        if !node.stage.cacheable() {
            return Ok(CacheStatus::Uncacheable);
        }

        if node.outputs.iter().any(|output| !output.exists()) {
            return Ok(CacheStatus::MissingOutput);
        }

        let path = self.fingerprint_path(node);
        let existing = match fs::read_to_string(path) {
            Ok(value) => value,
            Err(error) if error.kind() == io::ErrorKind::NotFound => return Ok(CacheStatus::Dirty),
            Err(error) => return Err(error),
        };

        Ok(if existing.trim() == fingerprint.hex() {
            CacheStatus::Fresh
        } else {
            CacheStatus::Dirty
        })
    }

    pub fn commit(&self, node: &BuildNode, fingerprint: Fingerprint) -> io::Result<()> {
        if !node.stage.cacheable() {
            return Ok(());
        }

        fs::create_dir_all(&self.root)?;
        let destination = self.fingerprint_path(node);
        let temporary = destination.with_extension("tmp");
        fs::write(&temporary, fingerprint.hex())?;
        fs::rename(temporary, destination)
    }

    pub fn invalidate(&self, node: &BuildNode) -> io::Result<()> {
        match fs::remove_file(self.fingerprint_path(node)) {
            Ok(()) => Ok(()),
            Err(error) if error.kind() == io::ErrorKind::NotFound => Ok(()),
            Err(error) => Err(error),
        }
    }

    fn fingerprint_path(&self, node: &BuildNode) -> PathBuf {
        let safe = node
            .label()
            .chars()
            .map(|character| {
                if character.is_ascii_alphanumeric() || matches!(character, '-' | '_') {
                    character
                } else {
                    '-'
                }
            })
            .collect::<String>();

        self.root.join(format!("{safe}.fingerprint"))
    }
}
