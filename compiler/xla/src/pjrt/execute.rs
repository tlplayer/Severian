use super::{
    api,
    buffer::Buffer,
    compile::RawLoadedExecutable,
    compile::RawClient,
    device::Device,
    error,
    host_buffer::{await_and_destroy_event, RawBuffer},
};
use crate::Result;
use std::sync::Arc;

struct BorrowedExecutable<'a> {
    api: &'a api::PJRT_Api,
    raw: *mut api::PJRT_Executable,
}

impl Drop for BorrowedExecutable<'_> {
    fn drop(&mut self) {
        let mut args = api::PJRT_Executable_Destroy_Args {
            struct_size: api::struct_size::<api::PJRT_Executable_Destroy_Args>(),
            extension_start: api::null_extension(),
            executable: self.raw,
        };
        let result = unsafe { (self.api.PJRT_Executable_Destroy)(&mut args) };
        let _ = unsafe { error::check(self.api, result) };
    }
}

pub(crate) fn execute(
    loaded: &RawLoadedExecutable,
    client: Arc<RawClient>,
    arguments: &[&Buffer],
    device: &Device,
) -> Result<Vec<Buffer>> {
    let plugin = loaded.plugin();
    let api = plugin.api();
    for (index, argument) in arguments.iter().enumerate() {
        if argument.raw().device()? != device.raw().raw() {
            return Err(crate::XlaError::Pjrt(format!(
                "argument {index} is not resident on the selected execute device"
            )));
        }
    }

    let mut get_args = api::PJRT_LoadedExecutable_GetExecutable_Args {
        struct_size: api::struct_size::<api::PJRT_LoadedExecutable_GetExecutable_Args>(),
        extension_start: api::null_extension(),
        loaded_executable: loaded.raw(),
        executable: std::ptr::null_mut(),
    };
    let result = unsafe { (api.PJRT_LoadedExecutable_GetExecutable)(&mut get_args) };
    unsafe { error::check(api, result)? };
    if get_args.executable.is_null() {
        return Err(error::invalid_raw_pointer("PJRT_Executable"));
    }
    let executable = BorrowedExecutable { api, raw: get_args.executable };

    let mut outputs_args = api::PJRT_Executable_NumOutputs_Args {
        struct_size: api::struct_size::<api::PJRT_Executable_NumOutputs_Args>(),
        extension_start: api::null_extension(),
        executable: executable.raw,
        num_outputs: 0,
    };
    let result = unsafe { (api.PJRT_Executable_NumOutputs)(&mut outputs_args) };
    unsafe { error::check(api, result)? };

    let argument_row: Vec<*mut api::PJRT_Buffer> =
        arguments.iter().map(|buffer| buffer.raw().raw()).collect();
    let argument_lists = [argument_row.as_ptr()];
    let mut output_row = vec![std::ptr::null_mut(); outputs_args.num_outputs];
    let output_lists = [output_row.as_mut_ptr()];
    let mut completion_event = std::ptr::null_mut();

    let mut options = api::PJRT_ExecuteOptions {
        struct_size: api::struct_size::<api::PJRT_ExecuteOptions>(),
        extension_start: api::null_extension(),
        send_callbacks: std::ptr::null_mut(),
        recv_callbacks: std::ptr::null_mut(),
        num_send_ops: 0,
        num_recv_ops: 0,
        launch_id: 0,
        non_donatable_input_indices: std::ptr::null(),
        num_non_donatable_input_indices: 0,
        context: std::ptr::null_mut(),
        call_location: std::ptr::null(),
        num_tasks: 0,
        task_ids: std::ptr::null_mut(),
        incarnation_ids: std::ptr::null_mut(),
        multi_slice_config: std::ptr::null_mut(),
        use_major_to_minor_data_layout_for_callbacks: false,
        hlo_output_callbacks: std::ptr::null_mut(),
        num_hlo_output_callbacks: 0,
    };
    let mut execute_args = api::PJRT_LoadedExecutable_Execute_Args {
        struct_size: api::struct_size::<api::PJRT_LoadedExecutable_Execute_Args>(),
        extension_start: api::null_extension(),
        executable: loaded.raw(),
        options: &mut options,
        argument_lists: argument_lists.as_ptr(),
        num_devices: 1,
        num_args: arguments.len(),
        output_lists: output_lists.as_ptr(),
        device_complete_events: &mut completion_event,
        execute_device: device.raw().raw(),
    };
    let result = unsafe { (api.PJRT_LoadedExecutable_Execute)(&mut execute_args) };
    unsafe { error::check(api, result)? };
    let outputs: Vec<Buffer> = output_row
        .into_iter()
        .map(|raw| {
            RawBuffer::from_raw(plugin.clone(), raw)
                .map(|raw| Buffer::from_raw(raw, Arc::clone(&client)))
        })
        .collect::<Result<_>>()?;
    await_and_destroy_event(api, completion_event)?;
    Ok(outputs)
}
