use super::{api, compile::RawClient, error};
use crate::{
    pjrt::buffer::{ElementType, HostBuffer, Shape},
    Result,
};
use std::{ffi::c_void, ptr::NonNull};

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

        let buffer =
            NonNull::new(args.buffer).ok_or_else(|| error::invalid_raw_pointer("PJRT_Buffer"))?;
        let buffer = RawBuffer {
            plugin: self.plugin().clone(),
            buffer,
            shape: host.shape.clone(),
        };

        // We promised to keep the host bytes alive until the transfer finishes.
        // Await the returned event before returning from this borrowed upload.
        if !args.done_with_host_buffer.is_null() {
            await_and_destroy_event(api, args.done_with_host_buffer)?;
        }
        Ok(buffer)
    }
}

impl RawBuffer {
    pub(crate) fn from_raw(
        plugin: super::plugin::RawPjrtPlugin,
        buffer: *mut api::PJRT_Buffer,
    ) -> Result<Self> {
        let buffer =
            NonNull::new(buffer).ok_or_else(|| error::invalid_raw_pointer("PJRT_Buffer"))?;
        let api = plugin.api();

        let mut type_args = api::PJRT_Buffer_ElementType_Args {
            struct_size: api::struct_size::<api::PJRT_Buffer_ElementType_Args>(),
            extension_start: api::null_extension(),
            buffer: buffer.as_ptr(),
            type_: api::PJRT_BUFFER_TYPE_INVALID,
        };
        let result = unsafe { (api.PJRT_Buffer_ElementType)(&mut type_args) };
        unsafe { error::check(api, result)? };

        let mut dimensions_args = api::PJRT_Buffer_Dimensions_Args {
            struct_size: api::struct_size::<api::PJRT_Buffer_Dimensions_Args>(),
            extension_start: api::null_extension(),
            buffer: buffer.as_ptr(),
            dims: std::ptr::null(),
            num_dims: 0,
        };
        let result = unsafe { (api.PJRT_Buffer_Dimensions)(&mut dimensions_args) };
        unsafe { error::check(api, result)? };
        if dimensions_args.num_dims != 0 && dimensions_args.dims.is_null() {
            return Err(error::invalid_raw_pointer("PJRT buffer dimensions"));
        }
        let dimensions = if dimensions_args.num_dims == 0 {
            Vec::new()
        } else {
            unsafe { std::slice::from_raw_parts(dimensions_args.dims, dimensions_args.num_dims) }
                .to_vec()
        };

        Ok(Self {
            plugin,
            buffer,
            shape: Shape::new(element_type_from_raw(type_args.type_)?, dimensions),
        })
    }

    pub fn raw(&self) -> *mut api::PJRT_Buffer {
        self.buffer.as_ptr()
    }

    pub fn shape(&self) -> &Shape {
        &self.shape
    }

    pub fn is_on_cpu(&self) -> Result<bool> {
        let api = self.plugin.api();
        let mut args = api::PJRT_Buffer_IsOnCpu_Args {
            struct_size: api::struct_size::<api::PJRT_Buffer_IsOnCpu_Args>(),
            extension_start: api::null_extension(),
            buffer: self.raw(),
            is_on_cpu: true,
        };
        let result = unsafe { (api.PJRT_Buffer_IsOnCpu)(&mut args) };
        unsafe { error::check(api, result)? };
        Ok(args.is_on_cpu)
    }

    pub fn device(&self) -> Result<*mut api::PJRT_Device> {
        let api = self.plugin.api();
        let mut args = api::PJRT_Buffer_Device_Args {
            struct_size: api::struct_size::<api::PJRT_Buffer_Device_Args>(),
            extension_start: api::null_extension(),
            buffer: self.raw(),
            device: std::ptr::null_mut(),
        };
        let result = unsafe { (api.PJRT_Buffer_Device)(&mut args) };
        unsafe { error::check(api, result)? };
        if args.device.is_null() {
            Err(error::invalid_raw_pointer("PJRT buffer device"))
        } else {
            Ok(args.device)
        }
    }

    pub fn to_host(&self) -> Result<Vec<u8>> {
        let mut bytes = vec![0; self.shape.byte_len()?];
        let api = self.plugin.api();
        let mut args = api::PJRT_Buffer_ToHostBuffer_Args {
            struct_size: api::struct_size::<api::PJRT_Buffer_ToHostBuffer_Args>(),
            extension_start: api::null_extension(),
            src: self.raw(),
            host_layout: std::ptr::null_mut(),
            dst: bytes.as_mut_ptr().cast(),
            dst_size: bytes.len(),
            event: std::ptr::null_mut(),
        };
        let result = unsafe { (api.PJRT_Buffer_ToHostBuffer)(&mut args) };
        unsafe { error::check(api, result)? };
        await_and_destroy_event(api, args.event)?;
        Ok(bytes)
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
        ElementType::F8E4M3FN => api::PJRT_BUFFER_TYPE_F8E4M3FN,
        ElementType::F8E5M2 => api::PJRT_BUFFER_TYPE_F8E5M2,
        ElementType::F16 => api::PJRT_BUFFER_TYPE_F16,
        ElementType::BF16 => api::PJRT_BUFFER_TYPE_BF16,
        ElementType::F32 => api::PJRT_BUFFER_TYPE_F32,
        ElementType::F64 => api::PJRT_BUFFER_TYPE_F64,
        ElementType::C64 => api::PJRT_BUFFER_TYPE_C64,
        ElementType::C128 => api::PJRT_BUFFER_TYPE_C128,
    }
}

fn element_type_from_raw(element: api::PJRT_Buffer_Type) -> Result<ElementType> {
    match element {
        api::PJRT_BUFFER_TYPE_PRED => Ok(ElementType::Pred),
        api::PJRT_BUFFER_TYPE_S8 => Ok(ElementType::S8),
        api::PJRT_BUFFER_TYPE_S16 => Ok(ElementType::S16),
        api::PJRT_BUFFER_TYPE_S32 => Ok(ElementType::S32),
        api::PJRT_BUFFER_TYPE_S64 => Ok(ElementType::S64),
        api::PJRT_BUFFER_TYPE_U8 => Ok(ElementType::U8),
        api::PJRT_BUFFER_TYPE_U16 => Ok(ElementType::U16),
        api::PJRT_BUFFER_TYPE_U32 => Ok(ElementType::U32),
        api::PJRT_BUFFER_TYPE_U64 => Ok(ElementType::U64),
        api::PJRT_BUFFER_TYPE_F8E4M3FN => Ok(ElementType::F8E4M3FN),
        api::PJRT_BUFFER_TYPE_F8E5M2 => Ok(ElementType::F8E5M2),
        api::PJRT_BUFFER_TYPE_F16 => Ok(ElementType::F16),
        api::PJRT_BUFFER_TYPE_F32 => Ok(ElementType::F32),
        api::PJRT_BUFFER_TYPE_F64 => Ok(ElementType::F64),
        api::PJRT_BUFFER_TYPE_BF16 => Ok(ElementType::BF16),
        api::PJRT_BUFFER_TYPE_C64 => Ok(ElementType::C64),
        api::PJRT_BUFFER_TYPE_C128 => Ok(ElementType::C128),
        other => Err(crate::XlaError::Pjrt(format!(
            "PJRT returned unknown buffer element type {other}"
        ))),
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
