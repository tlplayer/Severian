//! PJRT memory spaces and allocator statistics.

use super::{api, compile::RawClient, devices::RawDevice, error};
use crate::Result;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RawMemoryInfo {
    pub id: i32,
    pub kind: String,
    pub kind_id: Option<i32>,
    pub addressable_device_ids: Vec<i32>,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct DeviceMemoryStats {
    pub bytes_in_use: i64,
    pub peak_bytes_in_use: Option<i64>,
    pub allocations: Option<i64>,
    pub largest_allocation: Option<i64>,
    pub bytes_limit: Option<i64>,
    pub bytes_reserved: Option<i64>,
    pub peak_bytes_reserved: Option<i64>,
    pub reservable_limit: Option<i64>,
    pub largest_free_block: Option<i64>,
    pub pool_bytes: Option<i64>,
    pub peak_pool_bytes: Option<i64>,
    pub peak_allocated_bytes: Option<i64>,
}

impl RawClient {
    pub fn addressable_memories(&self) -> Result<Vec<RawMemoryInfo>> {
        let api = self.plugin().api();
        let mut args = api::PJRT_Client_AddressableMemories_Args {
            struct_size: api::struct_size::<api::PJRT_Client_AddressableMemories_Args>(),
            extension_start: api::null_extension(),
            client: self.raw(),
            addressable_memories: std::ptr::null(),
            num_addressable_memories: 0,
        };
        let result = unsafe { (api.PJRT_Client_AddressableMemories)(&mut args) };
        unsafe { error::check(api, result)? };

        if args.num_addressable_memories == 0 { return Ok(Vec::new()); }
        if args.addressable_memories.is_null() {
            return Err(error::invalid_raw_pointer("addressable_memories"));
        }

        unsafe {
            std::slice::from_raw_parts(
                args.addressable_memories,
                args.num_addressable_memories,
            )
        }
        .iter()
        .copied()
        .map(|memory| memory_info(self, memory))
        .collect()
    }

    pub fn memory_stats(&self, device: RawDevice) -> Result<DeviceMemoryStats> {
        let api = self.plugin().api();
        let mut args = api::PJRT_Device_MemoryStats_Args {
            struct_size: api::struct_size::<api::PJRT_Device_MemoryStats_Args>(),
            extension_start: api::null_extension(),
            device: device.raw(),
            bytes_in_use: 0,
            peak_bytes_in_use: 0,
            peak_bytes_in_use_is_set: false,
            num_allocs: 0,
            num_allocs_is_set: false,
            largest_alloc_size: 0,
            largest_alloc_size_is_set: false,
            bytes_limit: 0,
            bytes_limit_is_set: false,
            bytes_reserved: 0,
            bytes_reserved_is_set: false,
            peak_bytes_reserved: 0,
            peak_bytes_reserved_is_set: false,
            bytes_reservable_limit: 0,
            bytes_reservable_limit_is_set: false,
            largest_free_block_bytes: 0,
            largest_free_block_bytes_is_set: false,
            pool_bytes: 0,
            pool_bytes_is_set: false,
            peak_pool_bytes: 0,
            peak_pool_bytes_is_set: false,
            peak_allocated_bytes: 0,
            peak_allocated_bytes_is_set: false,
        };
        let result = unsafe { (api.PJRT_Device_MemoryStats)(&mut args) };
        unsafe { error::check(api, result)? };

        Ok(DeviceMemoryStats {
            bytes_in_use: args.bytes_in_use,
            peak_bytes_in_use: args.peak_bytes_in_use_is_set.then_some(args.peak_bytes_in_use),
            allocations: args.num_allocs_is_set.then_some(args.num_allocs),
            largest_allocation: args.largest_alloc_size_is_set.then_some(args.largest_alloc_size),
            bytes_limit: args.bytes_limit_is_set.then_some(args.bytes_limit),
            bytes_reserved: args.bytes_reserved_is_set.then_some(args.bytes_reserved),
            peak_bytes_reserved: args.peak_bytes_reserved_is_set.then_some(args.peak_bytes_reserved),
            reservable_limit: args.bytes_reservable_limit_is_set.then_some(args.bytes_reservable_limit),
            largest_free_block: args.largest_free_block_bytes_is_set.then_some(args.largest_free_block_bytes),
            pool_bytes: args.pool_bytes_is_set.then_some(args.pool_bytes),
            peak_pool_bytes: args.peak_pool_bytes_is_set.then_some(args.peak_pool_bytes),
            peak_allocated_bytes: args.peak_allocated_bytes_is_set.then_some(args.peak_allocated_bytes),
        })
    }
}

pub(crate) fn memory_info(
    client: &RawClient,
    memory: *mut api::PJRT_Memory,
) -> Result<RawMemoryInfo> {
    if memory.is_null() {
        return Err(error::invalid_raw_pointer("PJRT_Memory"));
    }

    let api = client.plugin().api();
    let id = memory_id(api, memory)?;
    let kind = memory_kind(api, memory)?;
    let kind_id = memory_kind_id(api, memory).ok();
    let addressable_device_ids = memory_addressable_device_ids(client, memory)?;

    Ok(RawMemoryInfo { id, kind, kind_id, addressable_device_ids })
}

fn memory_id(api: &api::PJRT_Api, memory: *mut api::PJRT_Memory) -> Result<i32> {
    let mut args = api::PJRT_Memory_Id_Args {
        struct_size: api::struct_size::<api::PJRT_Memory_Id_Args>(),
        extension_start: api::null_extension(),
        memory,
        id: -1,
    };
    let result = unsafe { (api.PJRT_Memory_Id)(&mut args) };
    unsafe { error::check(api, result)? };
    Ok(args.id)
}

fn memory_kind(api: &api::PJRT_Api, memory: *mut api::PJRT_Memory) -> Result<String> {
    let mut args = api::PJRT_Memory_Kind_Args {
        struct_size: api::struct_size::<api::PJRT_Memory_Kind_Args>(),
        extension_start: api::null_extension(),
        memory,
        kind: std::ptr::null(),
        kind_size: 0,
    };
    let result = unsafe { (api.PJRT_Memory_Kind)(&mut args) };
    unsafe { error::check(api, result)? };
    if args.kind.is_null() { return Ok(String::new()); }
    let bytes = unsafe { std::slice::from_raw_parts(args.kind.cast::<u8>(), args.kind_size) };
    Ok(String::from_utf8_lossy(bytes).into_owned())
}

fn memory_kind_id(api: &api::PJRT_Api, memory: *mut api::PJRT_Memory) -> Result<i32> {
    let mut args = api::PJRT_Memory_Kind_Id_Args {
        struct_size: api::struct_size::<api::PJRT_Memory_Kind_Id_Args>(),
        extension_start: api::null_extension(),
        memory,
        kind_id: -1,
    };
    let result = unsafe { (api.PJRT_Memory_Kind_Id)(&mut args) };
    unsafe { error::check(api, result)? };
    Ok(args.kind_id)
}

fn memory_addressable_device_ids(
    client: &RawClient,
    memory: *mut api::PJRT_Memory,
) -> Result<Vec<i32>> {
    let api = client.plugin().api();
    let mut args = api::PJRT_Memory_AddressableByDevices_Args {
        struct_size: api::struct_size::<api::PJRT_Memory_AddressableByDevices_Args>(),
        extension_start: api::null_extension(),
        memory,
        devices: std::ptr::null(),
        num_devices: 0,
    };
    let result = unsafe { (api.PJRT_Memory_AddressableByDevices)(&mut args) };
    unsafe { error::check(api, result)? };

    if args.num_devices == 0 { return Ok(Vec::new()); }
    if args.devices.is_null() {
        return Err(error::invalid_raw_pointer("memory addressable devices"));
    }

    let devices = unsafe { std::slice::from_raw_parts(args.devices, args.num_devices) };
    let mut ids = Vec::with_capacity(devices.len());

    for &device in devices {
        let raw = RawDevice::from_raw(device)?;
        let description = raw.description(client)?;
        let mut id_args = api::PJRT_DeviceDescription_Id_Args {
            struct_size: api::struct_size::<api::PJRT_DeviceDescription_Id_Args>(),
            extension_start: api::null_extension(),
            device_description: description,
            id: -1,
        };
        let result = unsafe { (api.PJRT_DeviceDescription_Id)(&mut id_args) };
        unsafe { error::check(api, result)? };
        ids.push(id_args.id);
    }

    Ok(ids)
}
