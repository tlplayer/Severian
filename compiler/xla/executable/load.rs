use crate::{
    cache::{CacheKey, DiskCache},
    executable::metadata::ExecutableManifest,
    options::{serialize_compile_options, XlaCompileOptions},
    pjrt::{
        api,
        compile::{RawClient, RawLoadedExecutable},
        error,
    },
    Result,
};
use std::ptr::NonNull;

pub fn deserialize_and_load(
    client: &RawClient,
    bytes: &[u8],
    overridden_options: Option<&XlaCompileOptions>,
) -> Result<RawLoadedExecutable> {
    let api = client.plugin().api();

    let serialized_options = overridden_options
        .map(serialize_compile_options)
        .transpose()
        .map_err(|error| crate::XlaError::Pjrt(error.to_string()))?;

    let (options_pointer, options_size) = serialized_options
        .as_deref()
        .map(|bytes| (bytes.as_ptr().cast(), bytes.len()))
        .unwrap_or((std::ptr::null(), 0));

    let mut load_options = api::PJRT_LoadOptions {
        struct_size: api::struct_size::<api::PJRT_LoadOptions>(),
        computation_origin: std::ptr::null(),
        computation_origin_size: 0,
        multi_slice_config: std::ptr::null_mut(),
    };

    let mut args = api::PJRT_Executable_DeserializeAndLoad_Args {
        struct_size: api::struct_size::<api::PJRT_Executable_DeserializeAndLoad_Args>(),
        extension_start: api::null_extension(),
        client: client.raw(),
        serialized_executable: bytes.as_ptr().cast(),
        serialized_executable_size: bytes.len(),
        loaded_executable: std::ptr::null_mut(),
        overridden_serialized_compile_options: options_pointer,
        overridden_serialized_compile_options_size: options_size,
        load_options: &mut load_options,
    };

    let result = unsafe { (api.PJRT_Executable_DeserializeAndLoad)(&mut args) };
    unsafe { error::check(api, result)? };

    let executable = NonNull::new(args.loaded_executable)
        .ok_or_else(|| error::invalid_raw_pointer("PJRT_LoadedExecutable"))?;

    Ok(unsafe {
        RawLoadedExecutable::from_raw_parts(
            client.plugin().clone(),
            executable,
        )
    })
}

pub fn load_cached_executable(
    client: &RawClient,
    cache: &DiskCache,
    key: CacheKey,
    expected_platform_name: &str,
    expected_platform_version: &str,
    overridden_options: Option<&XlaCompileOptions>,
) -> Result<Option<RawLoadedExecutable>> {
    let Some((bytes, manifest)) = cache
        .load(key)
        .map_err(crate::XlaError::Io)?
    else {
        return Ok(None);
    };

    if !compatible(
        &manifest,
        expected_platform_name,
        expected_platform_version,
        client.plugin().version().major_version,
        client.plugin().version().minor_version,
    ) {
        return Ok(None);
    }

    match deserialize_and_load(client, &bytes, overridden_options) {
        Ok(executable) => Ok(Some(executable)),
        Err(_) => {
            // Serialization is not stable over time. Treat stale/invalid
            // binaries as cache misses and discard them.
            let _ = cache.remove(key);
            Ok(None)
        }
    }
}

fn compatible(
    manifest: &ExecutableManifest,
    platform_name: &str,
    platform_version: &str,
    pjrt_api_major: i32,
    pjrt_api_minor: i32,
) -> bool {
    manifest.format_version == ExecutableManifest::CURRENT_VERSION
        && manifest.platform_name == platform_name
        && manifest.platform_version == platform_version
        && manifest.pjrt_api_major == pjrt_api_major
        // Allow a plugin with a newer compatible minor only if the actual
        // platform/library version matches exactly.
        && pjrt_api_minor >= manifest.pjrt_api_minor
}
