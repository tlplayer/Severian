use crate::{
    pjrt::{
        api,
        compile::RawExecutable,
        error,
    },
    Result,
};
use std::fmt;

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ExecutableFingerprint(pub Vec<u8>);

impl ExecutableFingerprint {
    pub fn as_bytes(&self) -> &[u8] {
        &self.0
    }

    pub fn display_lossy(&self) -> String {
        match std::str::from_utf8(&self.0) {
            Ok(text) => text.to_owned(),
            Err(_) => self
                .0
                .iter()
                .map(|byte| format!("{byte:02x}"))
                .collect(),
        }
    }
}

impl fmt::Display for ExecutableFingerprint {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.display_lossy())
    }
}

impl RawExecutable {
    pub fn fingerprint(&self) -> Result<ExecutableFingerprint> {
        let api = self.plugin().api();
        let mut args = api::PJRT_Executable_Fingerprint_Args {
            struct_size: api::struct_size::<api::PJRT_Executable_Fingerprint_Args>(),
            extension_start: api::null_extension(),
            executable: self.raw(),
            executable_fingerprint: std::ptr::null(),
            executable_fingerprint_size: 0,
        };

        let result = unsafe { (api.PJRT_Executable_Fingerprint)(&mut args) };
        unsafe { error::check(api, result)? };

        if args.executable_fingerprint_size == 0 {
            return Ok(ExecutableFingerprint(Vec::new()));
        }

        if args.executable_fingerprint.is_null() {
            return Err(error::invalid_raw_pointer("executable fingerprint"));
        }

        let bytes = unsafe {
            std::slice::from_raw_parts(
                args.executable_fingerprint.cast::<u8>(),
                args.executable_fingerprint_size,
            )
        };

        Ok(ExecutableFingerprint(bytes.to_vec()))
    }
}
