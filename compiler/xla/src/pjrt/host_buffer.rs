use super::{
    api,
    compile::RawClient,
    error,
};
use crate::{
    pjrt::buffer::{ElementType, HostBuffer, Shape},
    Result,
};
use std::{
    ffi::c_void,
    ptr::NonNull,
};

pub struct RawBuffer {
    plugin: super::plugin::RawPjrtPlugin,
    buffer: NonNull<api::PJRT_Buffer>,
    shape: Shape,
}

unsafe impl Send for RawBuffer {}
unsafe impl Sync for RawBuffer {}

impl RawClient {
    pub fn upload_host_buffer(
        &self,
        host: &HostBuffer,
        device: *mut api::PJRT_Device,
    ) -> Result<RawBuffer> {
        if device.is_null() {
            return Err(error::invalid_raw_pointer("PJRT_Device"));
        }

        let api = self.plugin().api();

        let mut args = api::PJRT_Client_BufferFromHostBuffer_Args {
            struct_size: api::struct_size::<api::PJRT_Client_BufferFromHostBuffer_Args>(),
            extension_start: api::null_extension(),
            client: self.raw(),
            data: host.bytes.as_ptr().cast::<c_void>(),
            type_: element_type_to_raw(host.shape.element_type),
            dims: host.shape.dimensions.as_ptr(),
            num_dims: host.shape.dimensions.len(),
            byte_strides: std::ptr::null(),
            num_byte_strides: 0,
            host_buffer_semantics: api::PJRT_HOST_BUFFER_IMMUTABLE_UNTIL_TRANSFER_COMPLETES,
            device,
            memory: std::ptr::null_mut(),
            device_layout: std::ptr::null_mut(),
            done_with_host_buffer: std::ptr::null_mut(),
            buffer: std::ptr::null_mut(),
        };

        let result = unsafe { (api.PJRT_Client_BufferFromHostBuffer)(&mut args) };
        unsafe { error::check(api, result)? };

        // We promised to keep the host bytes alive until the transfer finishes.
        // Await the returned event before returning from this borrowed upload.
        if !args.done_with_host_buffer.is_null() {
            await_and_destroy_event(api, args.done_with_host_buffer)?;
        }

        let buffer =
            NonNull::new(args.buffer).ok_or_else(|| error::invalid_raw_pointer("PJRT_Buffer"))?;

        Ok(RawBuffer {
            plugin: self.plugin().clone(),
            buffer,
            shape: host.shape.clone(),
        })
    }
}

impl RawBuffer {
    pub fn raw(&self) -> *mut api::PJRT_Buffer {
        self.buffer.as_ptr()
    }

    pub fn shape(&self) -> &Shape {
        &self.shape
    }

}

impl Drop for RawBuffer {
    fn drop(&mut self) {
        let api = self.plugin.api();
        let mut args = api::PJRT_Buffer_Destroy_Args {
            struct_size: api::struct_size::<api::PJRT_Buffer_Destroy_Args>(),
            extension_start: api::null_extension(),
            buffer: self.raw(),
        };

        let error = unsafe { (api.PJRT_Buffer_Destroy)(&mut args) };
        let _ = unsafe { error::check(api, error) };
    }
}

pub fn element_type_to_raw(element: ElementType) -> api::PJRT_Buffer_Type {
    match element {
        ElementType::Pred => api::PJRT_BUFFER_TYPE_PRED,
        ElementType::S8 => api::PJRT_BUFFER_TYPE_S8,
        ElementType::S16 => api::PJRT_BUFFER_TYPE_S16,
        ElementType::S32 => api::PJRT_BUFFER_TYPE_S32,
        ElementType::S64 => api::PJRT_BUFFER_TYPE_S64,
        ElementType::U8 => api::PJRT_BUFFER_TYPE_U8,
        ElementType::U16 => api::PJRT_BUFFER_TYPE_U16,
        ElementType::U32 => api::PJRT_BUFFER_TYPE_U32,
        ElementType::U64 => api::PJRT_BUFFER_TYPE_U64,
        ElementType::F16 => api::PJRT_BUFFER_TYPE_F16,
        ElementType::BF16 => api::PJRT_BUFFER_TYPE_BF16,
        ElementType::F32 => api::PJRT_BUFFER_TYPE_F32,
        ElementType::F64 => api::PJRT_BUFFER_TYPE_F64,
    }
}

pub(crate) fn await_and_destroy_event(
    api: &api::PJRT_Api,
    event: *mut api::PJRT_Event,
) -> Result<()> {
    if event.is_null() {
        return Ok(());
    }

    let mut await_args = api::PJRT_Event_Await_Args {
        struct_size: api::struct_size::<api::PJRT_Event_Await_Args>(),
        extension_start: api::null_extension(),
        event,
    };

    let await_error = unsafe { (api.PJRT_Event_Await)(&mut await_args) };
    let await_result = unsafe { error::check(api, await_error) };

    let mut destroy_args = api::PJRT_Event_Destroy_Args {
        struct_size: api::struct_size::<api::PJRT_Event_Destroy_Args>(),
        extension_start: api::null_extension(),
        event,
    };

    let destroy_error = unsafe { (api.PJRT_Event_Destroy)(&mut destroy_args) };
    let destroy_result = unsafe { error::check(api, destroy_error) };

    await_result.and(destroy_result)
}
