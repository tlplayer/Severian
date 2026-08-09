use super::key::CacheKey;
use crate::executable::metadata::ExecutableManifest;
use std::{
    fs,
    io,
    path::{Path, PathBuf},
};

#[derive(Debug, Clone)]
pub struct CacheEntry {
    pub key: CacheKey,
    pub directory: PathBuf,
    pub executable_path: PathBuf,
    pub manifest_path: PathBuf,
}

#[derive(Debug, Clone)]
pub struct DiskCache {
    root: PathBuf,
}

impl DiskCache {
    pub fn new(root: impl Into<PathBuf>) -> Self {
        Self { root: root.into() }
    }

    pub fn default_user_cache() -> Self {
        let root = std::env::var_os("XDG_CACHE_HOME")
            .map(PathBuf::from)
            .or_else(|| {
                std::env::var_os("HOME")
                    .map(PathBuf::from)
                    .map(|home| home.join(".cache"))
            })
            .unwrap_or_else(std::env::temp_dir)
            .join("severian")
            .join("xla");

        Self::new(root)
    }

    pub fn root(&self) -> &Path {
        &self.root
    }

    pub fn entry(&self, key: CacheKey) -> CacheEntry {
        let directory = self.root.join(key.shard()).join(key.hex());

        CacheEntry {
            key,
            executable_path: directory.join("executable.pjrt"),
            manifest_path: directory.join("manifest.txt"),
            directory,
        }
    }

    pub fn contains(&self, key: CacheKey) -> bool {
        let entry = self.entry(key);
        entry.executable_path.is_file() && entry.manifest_path.is_file()
    }

    pub fn load(
        &self,
        key: CacheKey,
    ) -> io::Result<Option<(Vec<u8>, ExecutableManifest)>> {
        let entry = self.entry(key);

        if !entry.executable_path.is_file() || !entry.manifest_path.is_file() {
            return Ok(None);
        }

        let executable = fs::read(&entry.executable_path)?;
        let manifest_text = fs::read_to_string(&entry.manifest_path)?;
        let manifest = ExecutableManifest::decode(&manifest_text)
            .map_err(|message| io::Error::new(io::ErrorKind::InvalidData, message))?;

        if manifest.cache_key != key {
            return Ok(None);
        }

        Ok(Some((executable, manifest)))
    }

    pub fn store(
        &self,
        key: CacheKey,
        executable: &[u8],
        manifest: &ExecutableManifest,
    ) -> io::Result<CacheEntry> {
        let entry = self.entry(key);
        fs::create_dir_all(&entry.directory)?;

        atomic_write(&entry.executable_path, executable)?;
        atomic_write(&entry.manifest_path, manifest.encode().as_bytes())?;

        Ok(entry)
    }

    pub fn remove(&self, key: CacheKey) -> io::Result<()> {
        let entry = self.entry(key);
        match fs::remove_dir_all(entry.directory) {
            Ok(()) => Ok(()),
            Err(error) if error.kind() == io::ErrorKind::NotFound => Ok(()),
            Err(error) => Err(error),
        }
    }

    pub fn clear(&self) -> io::Result<()> {
        match fs::remove_dir_all(&self.root) {
            Ok(()) => Ok(()),
            Err(error) if error.kind() == io::ErrorKind::NotFound => Ok(()),
            Err(error) => Err(error),
        }
    }
}

fn atomic_write(path: &Path, bytes: &[u8]) -> io::Result<()> {
    let parent = path.parent().unwrap_or_else(|| Path::new("."));
    fs::create_dir_all(parent)?;

    let temporary = parent.join(format!(
        ".{}.{}.tmp",
        path.file_name().and_then(|name| name.to_str()).unwrap_or("cache"),
        std::process::id(),
    ));

    fs::write(&temporary, bytes)?;

    match fs::rename(&temporary, path) {
        Ok(()) => Ok(()),
        Err(error) => {
            let _ = fs::remove_file(&temporary);
            Err(error)
        }
    }
}
