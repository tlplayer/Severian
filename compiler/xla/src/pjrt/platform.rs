//! PJRT client/platform metadata.

use super::{api, compile::RawClient, error};
use crate::Result;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PlatformInfo {
    pub name: String,
    pub version: String,
    pub process_index: i32,
}

impl RawClient {
    pub fn platform_info(&self) -> Result<PlatformInfo> {
        Ok(PlatformInfo {
            name: self.platform_name()?,
            version: self.platform_version()?,
            process_index: self.process_index()?,
        })
    }

    pub fn platform_version(&self) -> Result<String> {
        let api = self.plugin().api();
        let mut args = api::PJRT_Client_PlatformVersion_Args {
            struct_size: api::struct_size::<api::PJRT_Client_PlatformVersion_Args>(),
            extension_start: api::null_extension(),
            client: self.raw(),
            platform_version: std::ptr::null(),
            platform_version_size: 0,
        };
        let result = unsafe { (api.PJRT_Client_PlatformVersion)(&mut args) };
        unsafe { error::check(api, result)? };
        borrowed_string(args.platform_version, args.platform_version_size)
    }

    pub fn process_index(&self) -> Result<i32> {
        let api = self.plugin().api();
        let mut args = api::PJRT_Client_ProcessIndex_Args {
            struct_size: api::struct_size::<api::PJRT_Client_ProcessIndex_Args>(),
            extension_start: api::null_extension(),
            client: self.raw(),
            process_index: 0,
        };
        let result = unsafe { (api.PJRT_Client_ProcessIndex)(&mut args) };
        unsafe { error::check(api, result)? };
        Ok(args.process_index)
    }
}

pub(crate) fn borrowed_string(
    pointer: *const std::ffi::c_char,
    len: usize,
) -> Result<String> {
    if len == 0 { return Ok(String::new()); }
    if pointer.is_null() {
        return Err(error::invalid_raw_pointer("PJRT string"));
    }
    let bytes = unsafe { std::slice::from_raw_parts(pointer.cast::<u8>(), len) };
    Ok(String::from_utf8_lossy(bytes).into_owned())
}
