use super::api;
use crate::XlaError;
use std::{ffi::CStr, ptr::NonNull};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PjrtStatus {
    pub message: String,
}

impl PjrtStatus {
    pub fn into_xla_error(self) -> XlaError {
        XlaError::Pjrt(self.message)
    }
}

pub unsafe fn status_from_error(
    api: &api::PJRT_Api,
    error: *mut api::PJRT_Error,
) -> Option<PjrtStatus> {
    let error = NonNull::new(error)?;

    let mut message_args = api::PJRT_Error_Message_Args {
        struct_size: api::struct_size::<api::PJRT_Error_Message_Args>(),
        extension_start: api::null_extension(),
        error: error.as_ptr(),
        message: std::ptr::null(),
        message_size: 0,
    };

    (api.PJRT_Error_Message)(&mut message_args);

    let message = if message_args.message.is_null() {
        "unknown PJRT error".to_string()
    } else {
        let bytes = std::slice::from_raw_parts(
            message_args.message.cast::<u8>(),
            message_args.message_size,
        );
        String::from_utf8_lossy(bytes).into_owned()
    };

    let mut destroy_args = api::PJRT_Error_Destroy_Args {
        struct_size: api::struct_size::<api::PJRT_Error_Destroy_Args>(),
        extension_start: api::null_extension(),
        error: error.as_ptr(),
    };
    (api.PJRT_Error_Destroy)(&mut destroy_args);

    Some(PjrtStatus { message })
}

pub unsafe fn check(
    api: &api::PJRT_Api,
    error: *mut api::PJRT_Error,
) -> crate::Result<()> {
    match status_from_error(api, error) {
        None => Ok(()),
        Some(status) => Err(status.into_xla_error()),
    }
}

pub fn invalid_raw_pointer(name: &str) -> XlaError {
    XlaError::Pjrt(format!("PJRT returned a null `{name}` pointer"))
}
