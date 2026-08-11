use super::{
    api,
    error,
    plugin::RawPjrtPlugin,
};
use crate::{
    pipeline::{CompileOptions, OptimizationLevel},
    stablehlo::StableHloModule,
    Result, XlaError,
};
use std::{
    ffi::c_char,
    ptr::NonNull,
};

pub struct RawClient {
    plugin: RawPjrtPlugin,
    client: NonNull<api::PJRT_Client>,
}

unsafe impl Send for RawClient {}
unsafe impl Sync for RawClient {}

pub struct RawLoadedExecutable {
    plugin: RawPjrtPlugin,
    executable: NonNull<api::PJRT_LoadedExecutable>,
}

unsafe impl Send for RawLoadedExecutable {}
unsafe impl Sync for RawLoadedExecutable {}

impl RawClient {
    pub fn create(plugin: RawPjrtPlugin) -> Result<Self> {
        let api = plugin.api();
        let mut args = api::PJRT_Client_Create_Args {
            struct_size: api::struct_size::<api::PJRT_Client_Create_Args>(),
            extension_start: api::null_extension(),
            create_options: std::ptr::null(),
            num_options: 0,
            kv_get_callback: std::ptr::null(),
            kv_get_user_arg: std::ptr::null_mut(),
            kv_put_callback: std::ptr::null(),
            kv_put_user_arg: std::ptr::null_mut(),
            client: std::ptr::null_mut(),
            kv_try_get_callback: std::ptr::null(),
            kv_try_get_user_arg: std::ptr::null_mut(),
        };

        let result = unsafe { (api.PJRT_Client_Create)(&mut args) };
        unsafe { error::check(api, result)? };

        let client =
            NonNull::new(args.client).ok_or_else(|| error::invalid_raw_pointer("PJRT_Client"))?;

        Ok(Self { plugin, client })
    }

    pub fn plugin(&self) -> &RawPjrtPlugin {
        &self.plugin
    }

    pub fn raw(&self) -> *mut api::PJRT_Client {
        self.client.as_ptr()
    }

    pub fn platform_name(&self) -> Result<String> {
        let api = self.plugin.api();
        let mut args = api::PJRT_Client_PlatformName_Args {
            struct_size: api::struct_size::<api::PJRT_Client_PlatformName_Args>(),
            extension_start: api::null_extension(),
            client: self.raw(),
            platform_name: std::ptr::null(),
            platform_name_size: 0,
        };

        let result = unsafe { (api.PJRT_Client_PlatformName)(&mut args) };
        unsafe { error::check(api, result)? };

        if args.platform_name.is_null() {
            return Err(error::invalid_raw_pointer("platform_name"));
        }

        let bytes = unsafe {
            std::slice::from_raw_parts(
                args.platform_name.cast::<u8>(),
                args.platform_name_size,
            )
        };

        Ok(String::from_utf8_lossy(bytes).into_owned())
    }

    pub fn compile(
        &self,
        module: &StableHloModule,
        options: &CompileOptions,
    ) -> Result<RawLoadedExecutable> {
        module.validate_basic()?;

        // PJRT's "mlir" program format accepts MLIR text or bytecode. The
        // StableHLO portable artifact is MLIR bytecode, so all three Severian
        // StableHLO representations stay behind one program format.
        let mut code = module.bytes().to_vec();
        let format = b"mlir";

        let program = api::PJRT_Program {
            struct_size: api::struct_size::<api::PJRT_Program>(),
            extension_start: api::null_extension(),
            code: code.as_mut_ptr().cast::<c_char>(),
            code_size: code.len(),
            format: format.as_ptr().cast::<c_char>(),
            format_size: format.len(),
        };

        let serialized_options = serialized_compile_options(options)?;

        let api = self.plugin.api();
        let mut args = api::PJRT_Client_Compile_Args {
            struct_size: api::struct_size::<api::PJRT_Client_Compile_Args>(),
            extension_start: api::null_extension(),
            client: self.raw(),
            program: &program,
            compile_options: serialized_options.as_ptr().cast::<c_char>(),
            compile_options_size: serialized_options.len(),
            executable: std::ptr::null_mut(),
        };

        let result = unsafe { (api.PJRT_Client_Compile)(&mut args) };
        unsafe { error::check(api, result)? };

        let executable = NonNull::new(args.executable)
            .ok_or_else(|| error::invalid_raw_pointer("PJRT_LoadedExecutable"))?;

        Ok(RawLoadedExecutable {
            plugin: self.plugin.clone(),
            executable,
        })
    }
}

impl Drop for RawClient {
    fn drop(&mut self) {
        let api = self.plugin.api();
        let mut args = api::PJRT_Client_Destroy_Args {
            struct_size: api::struct_size::<api::PJRT_Client_Destroy_Args>(),
            extension_start: api::null_extension(),
            client: self.client.as_ptr(),
        };

        let error = unsafe { (api.PJRT_Client_Destroy)(&mut args) };
        let _ = unsafe { error::check(api, error) };
    }
}

impl RawLoadedExecutable {
    pub fn raw(&self) -> *mut api::PJRT_LoadedExecutable {
        self.executable.as_ptr()
    }

    pub(crate) fn plugin(&self) -> &RawPjrtPlugin { &self.plugin }

}

impl Drop for RawLoadedExecutable {
    fn drop(&mut self) {
        let api = self.plugin.api();
        let mut args = api::PJRT_LoadedExecutable_Destroy_Args {
            struct_size: api::struct_size::<api::PJRT_LoadedExecutable_Destroy_Args>(),
            extension_start: api::null_extension(),
            executable: self.raw(),
        };

        let error = unsafe { (api.PJRT_LoadedExecutable_Destroy)(&mut args) };
        let _ = unsafe { error::check(api, error) };
    }
}

/// `PJRT_Client_Compile` consumes an XLA CompileOptionsProto serialization.
/// An empty protobuf is the canonical default instance. Non-default options
/// remain unsupported until the crate owns generated XLA protobuf bindings.
fn serialized_compile_options(options: &CompileOptions) -> Result<Vec<u8>> {
    if options.optimization == OptimizationLevel::O2
        && options.num_replicas == 1
        && options.num_partitions == 1
        && !options.parameter_is_tupled_arguments
        && !options.use_spmd_partitioning
        && options.device_ordinal.is_none()
        && options.debug_options.is_empty()
    {
        Ok(Vec::new())
    } else {
        Err(XlaError::Compilation(
            "this PJRT plugin bridge currently accepts the canonical default CompileOptionsProto; requested non-default fields cannot be serialized losslessly"
                .into(),
        ))
    }
}
