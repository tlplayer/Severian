//! PJRT compiled executable ownership, metadata, persistence, and loading.

pub mod fingerprint;
pub mod load;
pub mod metadata;
pub mod serialize;

pub use fingerprint::ExecutableFingerprint;
pub use load::{load_cached_executable, deserialize_and_load};
pub use metadata::{
    CompiledMemoryStats, ExecutableManifest, ExecutableMetadata,
};
pub use serialize::{serialize_executable, SerializedExecutable};
