//! StableHLO representation and portable artifact support.

pub mod export;
pub mod import;
pub mod types;

pub use types::{StableHloFormat, StableHloModule, StableHloVersion};

use crate::Result;
use export::PortableArtifactOptions;

impl StableHloModule {
    pub fn to_portable_artifact(
        &self,
        options: &PortableArtifactOptions,
    ) -> Result<StableHloModule> {
        export::serialize_portable(self, options)
    }

    pub fn to_text(&self) -> Result<StableHloModule> {
        import::deserialize_to_text(self)
    }
}
