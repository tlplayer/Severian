//! Runtime topology discovery.
//!
//! The topology returned by PJRT_Client_TopologyDescription is owned by the
//! client. Severian snapshots the useful information into owned Rust values.

use super::{
    api,
    compile::RawClient,
    error,
    platform::borrowed_string,
};
use crate::Result;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TopologyDevice {
    pub id: i32,
    pub process_index: i32,
    pub kind: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TopologyInfo {
    pub platform_name: String,
    pub platform_version: String,
    pub devices: Vec<TopologyDevice>,
    pub serialized: Option<Vec<u8>>,
}

impl RawClient {
    pub fn topology(&self) -> Result<TopologyInfo> {
        let api = self.plugin().api();

        let mut args = api::PJRT_Client_TopologyDescription_Args {
            struct_size: api::struct_size::<api::PJRT_Client_TopologyDescription_Args>(),
            extension_start: api::null_extension(),
            client: self.raw(),
            topology: std::ptr::null_mut(),
        };

        let result = unsafe { (api.PJRT_Client_TopologyDescription)(&mut args) };
        unsafe { error::check(api, result)? };

        if args.topology.is_null() {
            return Err(error::invalid_raw_pointer("PJRT_TopologyDescription"));
        }

        snapshot_topology(api, args.topology)
    }
}

fn snapshot_topology(
    api: &api::PJRT_Api,
    topology: *mut api::PJRT_TopologyDescription,
) -> Result<TopologyInfo> {
    let platform_name = topology_platform_name(api, topology)?;
    let platform_version = topology_platform_version(api, topology)?;
    let descriptions = topology_device_descriptions(api, topology)?;

    let devices = descriptions
        .into_iter()
        .map(|description| topology_device(api, description))
        .collect::<Result<Vec<_>>>()?;

    let serialized = serialize_topology(api, topology).ok();

    Ok(TopologyInfo {
        platform_name,
        platform_version,
        devices,
        serialized,
    })
}

fn topology_platform_name(
    api: &api::PJRT_Api,
    topology: *mut api::PJRT_TopologyDescription,
) -> Result<String> {
    let mut args = api::PJRT_TopologyDescription_PlatformName_Args {
        struct_size: api::struct_size::<api::PJRT_TopologyDescription_PlatformName_Args>(),
        extension_start: api::null_extension(),
        topology,
        platform_name: std::ptr::null(),
        platform_name_size: 0,
    };
    let result = unsafe { (api.PJRT_TopologyDescription_PlatformName)(&mut args) };
    unsafe { error::check(api, result)? };
    borrowed_string(args.platform_name, args.platform_name_size)
}

fn topology_platform_version(
    api: &api::PJRT_Api,
    topology: *mut api::PJRT_TopologyDescription,
) -> Result<String> {
    let mut args = api::PJRT_TopologyDescription_PlatformVersion_Args {
        struct_size: api::struct_size::<api::PJRT_TopologyDescription_PlatformVersion_Args>(),
        extension_start: api::null_extension(),
        topology,
        platform_version: std::ptr::null(),
        platform_version_size: 0,
    };
    let result = unsafe { (api.PJRT_TopologyDescription_PlatformVersion)(&mut args) };
    unsafe { error::check(api, result)? };
    borrowed_string(args.platform_version, args.platform_version_size)
}

fn topology_device_descriptions(
    api: &api::PJRT_Api,
    topology: *mut api::PJRT_TopologyDescription,
) -> Result<Vec<*mut api::PJRT_DeviceDescription>> {
    let mut args = api::PJRT_TopologyDescription_GetDeviceDescriptions_Args {
        struct_size: api::struct_size::<api::PJRT_TopologyDescription_GetDeviceDescriptions_Args>(),
        extension_start: api::null_extension(),
        topology,
        descriptions: std::ptr::null(),
        num_descriptions: 0,
    };
    let result = unsafe { (api.PJRT_TopologyDescription_GetDeviceDescriptions)(&mut args) };
    unsafe { error::check(api, result)? };

    if args.num_descriptions == 0 { return Ok(Vec::new()); }
    if args.descriptions.is_null() {
        return Err(error::invalid_raw_pointer("topology device descriptions"));
    }

    Ok(unsafe {
        std::slice::from_raw_parts(args.descriptions, args.num_descriptions)
    }.to_vec())
}

fn topology_device(
    api: &api::PJRT_Api,
    description: *mut api::PJRT_DeviceDescription,
) -> Result<TopologyDevice> {
    let mut id = api::PJRT_DeviceDescription_Id_Args {
        struct_size: api::struct_size::<api::PJRT_DeviceDescription_Id_Args>(),
        extension_start: api::null_extension(),
        device_description: description,
        id: -1,
    };
    let result = unsafe { (api.PJRT_DeviceDescription_Id)(&mut id) };
    unsafe { error::check(api, result)? };

    let mut process = api::PJRT_DeviceDescription_ProcessIndex_Args {
        struct_size: api::struct_size::<api::PJRT_DeviceDescription_ProcessIndex_Args>(),
        extension_start: api::null_extension(),
        device_description: description,
        process_index: -1,
    };
    let result = unsafe { (api.PJRT_DeviceDescription_ProcessIndex)(&mut process) };
    unsafe { error::check(api, result)? };

    let mut kind = api::PJRT_DeviceDescription_Kind_Args {
        struct_size: api::struct_size::<api::PJRT_DeviceDescription_Kind_Args>(),
        extension_start: api::null_extension(),
        device_description: description,
        device_kind: std::ptr::null(),
        device_kind_size: 0,
    };
    let result = unsafe { (api.PJRT_DeviceDescription_Kind)(&mut kind) };
    unsafe { error::check(api, result)? };

    Ok(TopologyDevice {
        id: id.id,
        process_index: process.process_index,
        kind: borrowed_string(kind.device_kind, kind.device_kind_size)?,
    })
}

fn serialize_topology(
    api: &api::PJRT_Api,
    topology: *mut api::PJRT_TopologyDescription,
) -> Result<Vec<u8>> {
    let mut args = api::PJRT_TopologyDescription_Serialize_Args {
        struct_size: api::struct_size::<api::PJRT_TopologyDescription_Serialize_Args>(),
        extension_start: api::null_extension(),
        topology,
        serialized_bytes: std::ptr::null(),
        serialized_bytes_size: 0,
        serialized_topology: std::ptr::null_mut(),
        serialized_topology_deleter: None,
    };

    let result = unsafe { (api.PJRT_TopologyDescription_Serialize)(&mut args) };
    unsafe { error::check(api, result)? };

    if args.serialized_bytes_size > 0 && args.serialized_bytes.is_null() {
        return Err(error::invalid_raw_pointer("serialized topology"));
    }

    let bytes = if args.serialized_bytes_size == 0 {
        Vec::new()
    } else {
        unsafe {
            std::slice::from_raw_parts(
                args.serialized_bytes.cast::<u8>(),
                args.serialized_bytes_size,
            )
        }.to_vec()
    };

    if let Some(deleter) = args.serialized_topology_deleter {
        if !args.serialized_topology.is_null() {
            unsafe { deleter(args.serialized_topology) };
        }
    }

    Ok(bytes)
}
