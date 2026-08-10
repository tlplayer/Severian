use super::{
    api,
    compile::{RawClient, RawLoadedExecutable},
    error,
    host_buffer::{await_and_destroy_event, RawBuffer},
};
use crate::Result;

pub struct RawExecutionResult {
    pub outputs: Vec<RawBuffer>,
}

impl RawClient {
    /// Executes one replica/partition on one addressable device.
    ///
    /// Multi-device execution can be layered on top later by constructing the
    /// full `[num_devices][num_args]` argument and output matrices required by
    /// PJRT.
    pub fn execute_single_device(
        &self,
        executable: &RawLoadedExecutable,
        inputs: &[RawBuffer],
        device: *mut api::PJRT_Device,
        launch_id: i32,
    ) -> Result<RawExecutionResult> {
        if device.is_null() {
            return Err(error::invalid_raw_pointer("PJRT_Device"));
        }

        let api = executable.plugin().api();
        let num_outputs = executable.num_outputs()?;

        let mut raw_inputs = inputs
            .iter()
            .map(RawBuffer::raw)
            .collect::<Vec<_>>();

        let argument_row = raw_inputs.as_mut_ptr();
        let argument_rows = [argument_row as *const *mut api::PJRT_Buffer];

        let mut raw_outputs = vec![std::ptr::null_mut::<api::PJRT_Buffer>(); num_outputs];
        let output_row = raw_outputs.as_mut_ptr();
        let output_rows = [output_row];

        let mut complete_event = std::ptr::null_mut();

        let mut options = api::PJRT_ExecuteOptions {
            struct_size: api::struct_size::<api::PJRT_ExecuteOptions>(),
            extension_start: api::null_extension(),
            send_callbacks: std::ptr::null_mut(),
            recv_callbacks: std::ptr::null_mut(),
            num_send_ops: 0,
            num_recv_ops: 0,
            launch_id,
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

        let mut args = api::PJRT_LoadedExecutable_Execute_Args {
            struct_size: api::struct_size::<api::PJRT_LoadedExecutable_Execute_Args>(),
            extension_start: api::null_extension(),
            executable: executable.raw(),
            options: &mut options,
            argument_lists: argument_rows.as_ptr(),
            num_devices: 1,
            num_args: inputs.len(),
            output_lists: output_rows.as_ptr(),
            device_complete_events: &mut complete_event,
            execute_device: device,
        };

        let result = unsafe { (api.PJRT_LoadedExecutable_Execute)(&mut args) };
        unsafe { error::check(api, result)? };

        if !complete_event.is_null() {
            await_and_destroy_event(api, complete_event)?;
        }

        let outputs = raw_outputs
            .into_iter()
            .map(|buffer| raw_buffer_from_output(executable, buffer))
            .collect::<Result<Vec<_>>>()?;

        Ok(RawExecutionResult { outputs })
    }
}

fn raw_buffer_from_output(
    executable: &RawLoadedExecutable,
    buffer: *mut api::PJRT_Buffer,
) -> Result<RawBuffer> {
    use crate::pjrt::buffer::Shape;
    use super::host_buffer::raw_to_element_type;
    use std::ptr::NonNull;

    let buffer_ptr =
        NonNull::new(buffer).ok_or_else(|| error::invalid_raw_pointer("PJRT_Buffer output"))?;
    let api = executable.plugin().api();

    let mut type_args = api::PJRT_Buffer_ElementType_Args {
        struct_size: api::struct_size::<api::PJRT_Buffer_ElementType_Args>(),
        extension_start: api::null_extension(),
        buffer,
        type_: api::PJRT_BUFFER_TYPE_INVALID,
    };

    let result = unsafe { (api.PJRT_Buffer_ElementType)(&mut type_args) };
    unsafe { error::check(api, result)? };

    let mut dims_args = api::PJRT_Buffer_Dimensions_Args {
        struct_size: api::struct_size::<api::PJRT_Buffer_Dimensions_Args>(),
        extension_start: api::null_extension(),
        buffer,
        dims: std::ptr::null(),
        num_dims: 0,
    };

    let result = unsafe { (api.PJRT_Buffer_Dimensions)(&mut dims_args) };
    unsafe { error::check(api, result)? };

    let dimensions = if dims_args.dims.is_null() || dims_args.num_dims == 0 {
        Vec::new()
    } else {
        unsafe {
            std::slice::from_raw_parts(dims_args.dims, dims_args.num_dims)
        }
        .to_vec()
    };

    let shape = Shape::new(raw_to_element_type(type_args.type_)?, dimensions);

    Ok(unsafe {
        RawBuffer::from_raw_parts(
            executable.plugin().clone(),
            buffer_ptr,
            shape,
        )
    })
}
