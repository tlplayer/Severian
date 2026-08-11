#![forbid(unsafe_code)]

pub mod cache;
pub mod dependency;
pub mod executor;
pub mod fingerprint;
pub mod graph;
pub mod lockfile;
pub mod manifest;
pub mod node;
pub mod package;
pub mod profile;
pub mod workspace;

pub use cache::{BuildCache, CacheStatus};
pub use dependency::{Dependency, DependencySource};
pub use executor::{BuildContext, BuildExecutor, BuildFailure, BuildOutcome, BuildRunner};
pub use fingerprint::{Fingerprint, FingerprintBuilder};
pub use graph::{BuildGraph, GraphError};
pub use lockfile::{LockedDependency, LockedPackage, SeverianLockfile};
pub use manifest::{Manifest, ManifestError};
pub use node::{BuildNode, BuildNodeId, BuildStage};
pub use package::{Package, PackageTarget, PackageTargetKind};
pub use profile::{BuildProfile, DebugInfo, LtoMode, Sanitizer};
pub use workspace::{Workspace, WorkspaceError};

pub const MANIFEST_FILE: &str = "package.toml";
pub const LEGACY_MANIFEST_FILE: &str = "Severian.toml";
pub const LOCK_FILE: &str = "Severian.lock";
pub const DEFAULT_TARGET_DIRECTORY: &str = "target";
