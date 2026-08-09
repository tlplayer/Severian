//! Persistent XLA executable cache.
//!
//! PJRT executable serialization is explicitly platform/library-version
//! specific, so cache entries are namespaced by a deterministic CacheKey and
//! carry a compatibility manifest beside the raw executable bytes.

pub mod disk;
pub mod key;

pub use disk::{CacheEntry, DiskCache};
pub use key::{CacheKey, CacheKeyBuilder};
