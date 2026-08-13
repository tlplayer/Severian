use severian_package::BuildGate;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::UNIX_EPOCH;

pub struct BuildGateCache {
    directory: PathBuf,
    fingerprint: String,
}

impl BuildGateCache {
    pub fn discover(root: &Path, input: &Path) -> Result<Self, String> {
        let directory = root.join("target").join("build-gates");
        let fingerprint = build_fingerprint(root, input)?;
        Ok(Self {
            directory,
            fingerprint,
        })
    }

    pub fn is_fresh(&self, gate: BuildGate) -> bool {
        fs::read_to_string(self.stamp(gate))
            .is_ok_and(|contents| contents.trim() == self.fingerprint)
    }

    pub fn invalidate_from(&self, gate: BuildGate, pipeline: &[BuildGate]) -> Result<(), String> {
        let Some(start) = pipeline.iter().position(|candidate| *candidate == gate) else {
            return Ok(());
        };
        for gate in &pipeline[start..] {
            let stamp = self.stamp(*gate);
            match fs::remove_file(&stamp) {
                Ok(()) => {}
                Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
                Err(error) => {
                    return Err(format!("could not invalidate {}: {error}", stamp.display()))
                }
            }
        }
        Ok(())
    }

    pub fn record(&self, gate: BuildGate) -> Result<(), String> {
        fs::create_dir_all(&self.directory).map_err(|error| error.to_string())?;
        let stamp = self.stamp(gate);
        let temporary = stamp.with_extension(format!("{}.tmp", std::process::id()));
        fs::write(&temporary, format!("{}\n", self.fingerprint))
            .map_err(|error| error.to_string())?;
        fs::rename(&temporary, &stamp).map_err(|error| error.to_string())
    }

    fn stamp(&self, gate: BuildGate) -> PathBuf {
        self.directory.join(format!("{}.stamp", gate.name()))
    }
}

fn build_fingerprint(root: &Path, input: &Path) -> Result<String, String> {
    let mut files = Vec::new();
    collect_inputs(root, root, &mut files).map_err(|error| error.to_string())?;
    files.sort();

    let mut hash = Fingerprint::new();
    hash.add(env!("CARGO_PKG_VERSION").as_bytes());
    hash.add(std::env::consts::OS.as_bytes());
    hash.add(std::env::consts::ARCH.as_bytes());
    hash.add(input.to_string_lossy().as_bytes());
    if let Ok(executable) = std::env::current_exe() {
        hash.add(executable.to_string_lossy().as_bytes());
        if let Ok(metadata) = executable.metadata() {
            hash.add(&metadata.len().to_le_bytes());
            if let Ok(modified) = metadata.modified().and_then(|time| {
                time.duration_since(UNIX_EPOCH)
                    .map_err(std::io::Error::other)
            }) {
                hash.add(&modified.as_nanos().to_le_bytes());
            }
        }
    }
    for path in files {
        let relative = path.strip_prefix(root).unwrap_or(&path);
        hash.add(relative.to_string_lossy().as_bytes());
        let contents = fs::read(&path)
            .map_err(|error| format!("could not fingerprint {}: {error}", path.display()))?;
        hash.add(&contents);
    }
    Ok(hash.finish())
}

fn collect_inputs(root: &Path, directory: &Path, output: &mut Vec<PathBuf>) -> std::io::Result<()> {
    for entry in fs::read_dir(directory)? {
        let entry = entry?;
        let path = entry.path();
        if path.is_dir() {
            let name = path
                .file_name()
                .and_then(|name| name.to_str())
                .unwrap_or("");
            if matches!(
                name,
                "target" | ".git" | ".codex" | ".agents" | "node_modules"
            ) || name == ".venv"
                || name.starts_with(".venv-")
            {
                continue;
            }
            collect_inputs(root, &path, output)?;
        } else if path.is_file()
            && path.starts_with(root)
            && matches!(
                path.extension().and_then(|extension| extension.to_str()),
                Some("sev" | "toml" | "lock")
            )
        {
            output.push(path);
        }
    }
    Ok(())
}

struct Fingerprint {
    left: u64,
    right: u64,
}

impl Fingerprint {
    const fn new() -> Self {
        Self {
            left: 0xcbf29ce484222325,
            right: 0x84222325cbf29ce4,
        }
    }

    fn add(&mut self, bytes: &[u8]) {
        for byte in (bytes.len() as u64).to_le_bytes().iter().chain(bytes) {
            self.left ^= u64::from(*byte);
            self.left = self.left.wrapping_mul(0x100000001b3);
            self.right ^= u64::from(*byte);
            self.right = self.right.wrapping_mul(0x9e3779b185ebca87);
        }
    }

    fn finish(self) -> String {
        format!("{:016x}{:016x}", self.left, self.right)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temporary_root(label: &str) -> PathBuf {
        std::env::temp_dir().join(format!(
            "severian-build-cache-{label}-{}",
            std::process::id()
        ))
    }

    #[test]
    fn reuses_only_matching_successful_gates_and_invalidates_downstream() {
        let root = temporary_root("progressive");
        let _ = fs::remove_dir_all(&root);
        fs::create_dir_all(root.join("src")).unwrap();
        fs::write(root.join("package.toml"), "[package]\nname = \"cache\"\n").unwrap();
        fs::write(root.join("src/main.sev"), "def main():\n    print(1)\n").unwrap();
        let pipeline = [BuildGate::Compile, BuildGate::Test, BuildGate::Integration];
        let cache = BuildGateCache::discover(&root, &root).unwrap();
        cache.record(BuildGate::Compile).unwrap();
        cache.record(BuildGate::Test).unwrap();
        cache.record(BuildGate::Integration).unwrap();
        assert!(cache.is_fresh(BuildGate::Test));

        cache.invalidate_from(BuildGate::Test, &pipeline).unwrap();
        assert!(cache.is_fresh(BuildGate::Compile));
        assert!(!cache.is_fresh(BuildGate::Test));
        assert!(!cache.is_fresh(BuildGate::Integration));

        let changed = "def main():\n    print(2)\n";
        fs::write(root.join("src/main.sev"), changed).unwrap();
        let changed_cache = BuildGateCache::discover(&root, &root).unwrap();
        assert!(!changed_cache.is_fresh(BuildGate::Compile));
        let _ = fs::remove_dir_all(root);
    }
}
