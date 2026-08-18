use crate::signature::verify_sha256;
use crate::PackageError;
use std::fs;
use std::io;
use std::path::{Path, PathBuf};

const MAX_ARTIFACT_BYTES: u64 = 8 * 1024 * 1024 * 1024;

/// Downloads one registry-authorized HTTPS artifact. This transport is called
/// only by `sev install` after trust, domain, date, and signature validation;
/// package builds never receive this network capability.
pub fn download_verified(
    source: &str,
    sha256: &str,
    cache: &Path,
) -> Result<PathBuf, PackageError> {
    if let Some(parent) = cache.parent() {
        fs::create_dir_all(parent)?;
    }
    let temporary = cache.with_extension(format!("{}.download", std::process::id()));
    let agent: ureq::Agent = ureq::config::Config::builder()
        .https_only(true)
        .max_redirects(0)
        .build()
        .into();
    let mut response = agent.get(source).call().map_err(|error| {
        PackageError::Manifest(format!("secure download failed for `{source}`: {error}"))
    })?;
    if !response.status().is_success() {
        return Err(PackageError::Manifest(format!(
            "secure download of `{source}` returned HTTP {}",
            response.status()
        )));
    }
    let result = (|| {
        let mut output = fs::File::create(&temporary)?;
        let mut reader = response
            .body_mut()
            .with_config()
            .limit(MAX_ARTIFACT_BYTES)
            .reader();
        io::copy(&mut reader, &mut output).map_err(|error| {
            PackageError::Manifest(format!("could not download `{source}`: {error}"))
        })?;
        output.sync_all()?;
        verify_sha256(&temporary, sha256)?;
        fs::rename(&temporary, cache)?;
        Ok(cache.to_path_buf())
    })();
    if result.is_err() {
        let _ = fs::remove_file(&temporary);
    }
    result
}
