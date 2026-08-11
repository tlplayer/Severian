//! PJRT device discovery and owned device metadata.

use super::{api, compile::RawClient, error};
use crate::Result;
use std::ptr::NonNull;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RawDeviceInfo {
    pub id: i32,
    pub process_index: i32,
    pub local_hardware_id: Option<i32>,
    pub kind: String,
    pub addressable: bool,
}

#[derive(Debug, Clone, Copy)]
pub struct RawDevice {
    raw: NonNull<api::PJRT_Device>,
}

unsafe impl Send for RawDevice {}
unsafe impl Sync for RawDevice {}

impl RawDevice {
    pub(crate) fn from_raw(raw: *mut api::PJRT_Device) -> Result<Self> {
        Ok(Self {
            raw: NonNull::new(raw).ok_or_else(|| error::invalid_raw_pointer("PJRT_Device"))?,
        })
    }

    pub fn raw(self) -> *mut api::PJRT_Device {
        self.raw.as_ptr()
    }

    pub fn description(self, client: &RawClient) -> Result<*mut api::PJRT_DeviceDescription> {
        let api = client.plugin().api();
        let mut args = api::PJRT_Device_GetDescription_Args {
            struct_size: api::struct_size::<api::PJRT_Device_GetDescription_Args>(),
            extension_start: api::null_extension(),
            device: self.raw(),
            device_description: std::ptr::null_mut(),
        };
        let result = unsafe { (api.PJRT_Device_GetDescription)(&mut args) };
        unsafe { error::check(api, result)? };
        if args.device_description.is_null() {
            Err(error::invalid_raw_pointer("PJRT_DeviceDescription"))
        } else {
            Ok(args.device_description)
        }
    }

    pub fn info(self, client: &RawClient) -> Result<RawDeviceInfo> {
        let api = client.plugin().api();
        let description = self.description(client)?;
        let id = description_id(api, description)?;
        let process_index = description_process_index(api, description)?;
        let kind = description_kind(api, description)?;
        let addressable = is_addressable(api, self.raw())?;

        let local_hardware_id = if addressable {
            let id = local_hardware_id(api, self.raw())?;
            (id >= 0).then_some(id)
        } else {
            None
        };

        Ok(RawDeviceInfo {
            id,
            process_index,
            local_hardware_id,
            kind,
            addressable,
        })
    }
}

impl RawClient {
    pub fn devices(&self) -> Result<Vec<RawDevice>> {
        let api = self.plugin().api();
        let mut args = api::PJRT_Client_Devices_Args {
            struct_size: api::struct_size::<api::PJRT_Client_Devices_Args>(),
            extension_start: api::null_extension(),
            client: self.raw(),
            devices: std::ptr::null(),
            num_devices: 0,
        };
        let result = unsafe { (api.PJRT_Client_Devices)(&mut args) };
        unsafe { error::check(api, result)? };
        pointer_array(args.devices, args.num_devices)?
            .into_iter()
            .map(RawDevice::from_raw)
            .collect()
    }

    pub fn addressable_devices(&self) -> Result<Vec<RawDevice>> {
        let api = self.plugin().api();
        let mut args = api::PJRT_Client_AddressableDevices_Args {
            struct_size: api::struct_size::<api::PJRT_Client_AddressableDevices_Args>(),
            extension_start: api::null_extension(),
            client: self.raw(),
            addressable_devices: std::ptr::null(),
            num_addressable_devices: 0,
        };
        let result = unsafe { (api.PJRT_Client_AddressableDevices)(&mut args) };
        unsafe { error::check(api, result)? };
        pointer_array(args.addressable_devices, args.num_addressable_devices)?
            .into_iter()
            .map(RawDevice::from_raw)
            .collect()
    }

    pub fn default_device(&self) -> Result<RawDevice> {
        self.addressable_devices()?
            .into_iter()
            .next()
            .ok_or_else(|| crate::XlaError::Pjrt("PJRT client has no addressable devices".into()))
    }
}

fn description_id(
    api: &api::PJRT_Api,
    description: *mut api::PJRT_DeviceDescription,
) -> Result<i32> {
    let mut args = api::PJRT_DeviceDescription_Id_Args {
        struct_size: api::struct_size::<api::PJRT_DeviceDescription_Id_Args>(),
        extension_start: api::null_extension(),
        device_description: description,
        id: -1,
    };
    let result = unsafe { (api.PJRT_DeviceDescription_Id)(&mut args) };
    unsafe { error::check(api, result)? };
    Ok(args.id)
}

fn description_process_index(
    api: &api::PJRT_Api,
    description: *mut api::PJRT_DeviceDescription,
) -> Result<i32> {
    let mut args = api::PJRT_DeviceDescription_ProcessIndex_Args {
        struct_size: api::struct_size::<api::PJRT_DeviceDescription_ProcessIndex_Args>(),
        extension_start: api::null_extension(),
        device_description: description,
        process_index: -1,
    };
    let result = unsafe { (api.PJRT_DeviceDescription_ProcessIndex)(&mut args) };
    unsafe { error::check(api, result)? };
    Ok(args.process_index)
}

fn description_kind(
    api: &api::PJRT_Api,
    description: *mut api::PJRT_DeviceDescription,
) -> Result<String> {
    let mut args = api::PJRT_DeviceDescription_Kind_Args {
        struct_size: api::struct_size::<api::PJRT_DeviceDescription_Kind_Args>(),
        extension_start: api::null_extension(),
        device_description: description,
        device_kind: std::ptr::null(),
        device_kind_size: 0,
    };
    let result = unsafe { (api.PJRT_DeviceDescription_Kind)(&mut args) };
    unsafe { error::check(api, result)? };
    if args.device_kind.is_null() {
        return Ok(String::new());
    }
    let bytes =
        unsafe { std::slice::from_raw_parts(args.device_kind.cast::<u8>(), args.device_kind_size) };
    Ok(String::from_utf8_lossy(bytes).into_owned())
}

fn is_addressable(api: &api::PJRT_Api, device: *mut api::PJRT_Device) -> Result<bool> {
    let mut args = api::PJRT_Device_IsAddressable_Args {
        struct_size: api::struct_size::<api::PJRT_Device_IsAddressable_Args>(),
        extension_start: api::null_extension(),
        device,
        is_addressable: false,
    };
    let result = unsafe { (api.PJRT_Device_IsAddressable)(&mut args) };
    unsafe { error::check(api, result)? };
    Ok(args.is_addressable)
}

fn local_hardware_id(api: &api::PJRT_Api, device: *mut api::PJRT_Device) -> Result<i32> {
    let mut args = api::PJRT_Device_LocalHardwareId_Args {
        struct_size: api::struct_size::<api::PJRT_Device_LocalHardwareId_Args>(),
        extension_start: api::null_extension(),
        device,
        local_hardware_id: -1,
    };
    let result = unsafe { (api.PJRT_Device_LocalHardwareId)(&mut args) };
    unsafe { error::check(api, result)? };
    Ok(args.local_hardware_id)
}

fn pointer_array<T>(pointer: *const *mut T, len: usize) -> Result<Vec<*mut T>> {
    if len == 0 {
        return Ok(Vec::new());
    }
    if pointer.is_null() {
        return Err(error::invalid_raw_pointer("PJRT pointer array"));
    }
    Ok(unsafe { std::slice::from_raw_parts(pointer, len) }.to_vec())
}
