use crate::{
    pjrt::{
        api,
        compile::RawExecutable,
        error,
    },
    Result,
};

#[derive(Debug, Clone)]
pub struct SerializedExecutable {
    pub bytes: Vec<u8>,
}

impl SerializedExecutable {
    pub fn as_bytes(&self) -> &[u8] {
        &self.bytes
    }

    pub fn into_bytes(self) -> Vec<u8> {
        self.bytes
    }
}

pub fn serialize_executable(
    executable: &RawExecutable,
) -> Result<SerializedExecutable> {
    let api = executable.plugin().api();

    let mut args = api::PJRT_Executable_Serialize_Args {
        struct_size: api::struct_size::<api::PJRT_Executable_Serialize_Args>(),
        extension_start: api::null_extension(),
        executable: executable.raw(),
        serialized_bytes: std::ptr::null(),
        serialized_bytes_size: 0,
        serialized_executable: std::ptr::null_mut(),
        serialized_executable_deleter: None,
    };

    let result = unsafe { (api.PJRT_Executable_Serialize)(&mut args) };
    unsafe { error::check(api, result)? };

    if args.serialized_bytes_size > 0 && args.serialized_bytes.is_null() {
        destroy_backing(&mut args);
        return Err(error::invalid_raw_pointer("serialized executable bytes"));
    }

    let bytes = if args.serialized_bytes_size == 0 {
        Vec::new()
    } else {
        unsafe {
            std::slice::from_raw_parts(
                args.serialized_bytes.cast::<u8>(),
                args.serialized_bytes_size,
            )
        }
        .to_vec()
    };

    destroy_backing(&mut args);

    Ok(SerializedExecutable { bytes })
}

fn destroy_backing(args: &mut api::PJRT_Executable_Serialize_Args) {
    if let Some(deleter) = args.serialized_executable_deleter.take() {
        if !args.serialized_executable.is_null() {
            unsafe { deleter(args.serialized_executable) };
            args.serialized_executable = std::ptr::null_mut();
        }
    }
}
