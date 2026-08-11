use std::path::{Path, PathBuf};

const XLA_RUNTIME: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/libseverian_xla.a"));

pub fn materialize_xla_runtime(directory: &Path) -> std::io::Result<PathBuf> {
    let hash = XLA_RUNTIME.iter().fold(0xcbf29ce484222325_u64, |hash, byte| {
        (hash ^ u64::from(*byte)).wrapping_mul(0x100000001b3)
    });
    let assets = directory.join(".severian");
    std::fs::create_dir_all(&assets)?;
    let output = assets.join(format!("libseverian_xla-{hash:016x}.a"));
    if !output.is_file() {
        std::fs::write(&output, XLA_RUNTIME)?;
    }
    Ok(output)
}
