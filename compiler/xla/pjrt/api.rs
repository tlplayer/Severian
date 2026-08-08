//! Minimal raw PJRT C ABI used by Severian.
//!
//! This intentionally models only the stable prefix and argument structures
//! needed for plugin initialization, client creation, compilation, host-buffer
//! transfer, execution, device inspection, events, and destruction.
//!
//! Pinned against OpenXLA PJRT C API 0.114. Keep all raw ABI changes in this
//! file so the rest of Severian can stay on owned Rust abstractions.

#![allow(non_camel_case_types)]
#![allow(non_snake_case)]
#![allow(dead_code)]

use std::ffi::{c_char, c_void};

pub const PJRT_API_MAJOR: i32 = 0;
pub const PJRT_API_MINOR: i32 = 114;

#[repr(C)]
pub struct PJRT_Extension_Base {
    pub struct_size: usize,
    pub type_: i32,
    pub next: *mut PJRT_Extension_Base,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PJRT_Api_Version {
    pub struct_size: usize,
    pub extension_start: *mut PJRT_Extension_Base,
    pub major_version: i32,
    pub minor_version: i32,
}

#[repr(C)]
pub struct PJRT_Error {
    _private: [u8; 0],
}
#[repr(C)]
pub struct PJRT_Event {
    _private: [u8; 0],
}
#[repr(C)]
pub struct PJRT_Client {
    _private: [u8; 0],
}
#[repr(C)]
pub struct PJRT_Device {
    _private: [u8; 0],
}
#[repr(C)]
pub struct PJRT_DeviceDescription {
    _private: [u8; 0],
}
#[repr(C)]
pub struct PJRT_Memory {
    _private: [u8; 0],
}
#[repr(C)]
pub struct PJRT_Buffer {
    _private: [u8; 0],
}
#[repr(C)]
pub struct PJRT_Executable {
    _private: [u8; 0],
}
#[repr(C)]
pub struct PJRT_LoadedExecutable {
    _private: [u8; 0],
}
#[repr(C)]
pub struct PJRT_ExecuteContext {
    _private: [u8; 0],
}
#[repr(C)]
pub struct PJRT_MultiSlice_Config {
    _private: [u8; 0],
}
#[repr(C)]
pub struct PJRT_Buffer_MemoryLayout {
    _private: [u8; 0],
}

pub type PJRT_Error_Code = i32;
pub const PJRT_ERROR_OK: PJRT_Error_Code = 0;

pub type PJRT_Buffer_Type = i32;
pub const PJRT_BUFFER_TYPE_INVALID: PJRT_Buffer_Type = 0;
pub const PJRT_BUFFER_TYPE_PRED: PJRT_Buffer_Type = 1;
pub const PJRT_BUFFER_TYPE_S8: PJRT_Buffer_Type = 2;
pub const PJRT_BUFFER_TYPE_S16: PJRT_Buffer_Type = 3;
pub const PJRT_BUFFER_TYPE_S32: PJRT_Buffer_Type = 4;
pub const PJRT_BUFFER_TYPE_S64: PJRT_Buffer_Type = 5;
pub const PJRT_BUFFER_TYPE_U8: PJRT_Buffer_Type = 6;
pub const PJRT_BUFFER_TYPE_U16: PJRT_Buffer_Type = 7;
pub const PJRT_BUFFER_TYPE_U32: PJRT_Buffer_Type = 8;
pub const PJRT_BUFFER_TYPE_U64: PJRT_Buffer_Type = 9;
pub const PJRT_BUFFER_TYPE_F16: PJRT_Buffer_Type = 10;
pub const PJRT_BUFFER_TYPE_F32: PJRT_Buffer_Type = 11;
pub const PJRT_BUFFER_TYPE_F64: PJRT_Buffer_Type = 12;
pub const PJRT_BUFFER_TYPE_BF16: PJRT_Buffer_Type = 13;

pub type PJRT_HostBufferSemantics = i32;
pub const PJRT_HOST_BUFFER_IMMUTABLE_ONLY_DURING_CALL: PJRT_HostBufferSemantics = 0;
pub const PJRT_HOST_BUFFER_IMMUTABLE_UNTIL_TRANSFER_COMPLETES: PJRT_HostBufferSemantics = 1;
pub const PJRT_HOST_BUFFER_IMMUTABLE_ZERO_COPY: PJRT_HostBufferSemantics = 2;
pub const PJRT_HOST_BUFFER_MUTABLE_ZERO_COPY: PJRT_HostBufferSemantics = 3;

#[repr(C)]
pub struct PJRT_Plugin_Initialize_Args {
    pub struct_size: usize,
    pub extension_start: *mut PJRT_Extension_Base,
}

#[repr(C)]
pub struct PJRT_Error_Message_Args {
    pub struct_size: usize,
    pub extension_start: *mut PJRT_Extension_Base,
    pub error: *const PJRT_Error,
    pub message: *const c_char,
    pub message_size: usize,
}

#[repr(C)]
pub struct PJRT_Error_Destroy_Args {
    pub struct_size: usize,
    pub extension_start: *mut PJRT_Extension_Base,
    pub error: *mut PJRT_Error,
}

#[repr(C)]
pub struct PJRT_Event_Await_Args {
    pub struct_size: usize,
    pub extension_start: *mut PJRT_Extension_Base,
    pub event: *mut PJRT_Event,
}

#[repr(C)]
pub struct PJRT_Event_Destroy_Args {
    pub struct_size: usize,
    pub extension_start: *mut PJRT_Extension_Base,
    pub event: *mut PJRT_Event,
}

#[repr(C)]
pub struct PJRT_Client_Create_Args {
    pub struct_size: usize,
    pub extension_start: *mut PJRT_Extension_Base,
    pub create_options: *const c_void,
    pub num_options: usize,
    pub kv_get_callback: *const c_void,
    pub kv_get_user_arg: *mut c_void,
    pub kv_put_callback: *const c_void,
    pub kv_put_user_arg: *mut c_void,
    pub client: *mut PJRT_Client,
    pub kv_try_get_callback: *const c_void,
    pub kv_try_get_user_arg: *mut c_void,
}

#[repr(C)]
pub struct PJRT_Client_Destroy_Args {
    pub struct_size: usize,
    pub extension_start: *mut PJRT_Extension_Base,
    pub client: *mut PJRT_Client,
}

#[repr(C)]
pub struct PJRT_Client_PlatformName_Args {
    pub struct_size: usize,
    pub extension_start: *mut PJRT_Extension_Base,
    pub client: *mut PJRT_Client,
    pub platform_name: *const c_char,
    pub platform_name_size: usize,
}

#[repr(C)]
pub struct PJRT_Client_Devices_Args {
    pub struct_size: usize,
    pub extension_start: *mut PJRT_Extension_Base,
    pub client: *mut PJRT_Client,
    pub devices: *const *mut PJRT_Device,
    pub num_devices: usize,
}

#[repr(C)]
pub struct PJRT_Client_AddressableDevices_Args {
    pub struct_size: usize,
    pub extension_start: *mut PJRT_Extension_Base,
    pub client: *mut PJRT_Client,
    pub addressable_devices: *const *mut PJRT_Device,
    pub num_addressable_devices: usize,
}

#[repr(C)]
pub struct PJRT_Program {
    pub struct_size: usize,
    pub extension_start: *mut PJRT_Extension_Base,
    pub code: *mut c_char,
    pub code_size: usize,
    pub format: *const c_char,
    pub format_size: usize,
}

#[repr(C)]
pub struct PJRT_Client_Compile_Args {
    pub struct_size: usize,
    pub extension_start: *mut PJRT_Extension_Base,
    pub client: *mut PJRT_Client,
    pub program: *const PJRT_Program,
    pub compile_options: *const c_char,
    pub compile_options_size: usize,
    pub executable: *mut PJRT_LoadedExecutable,
}

#[repr(C)]
pub struct PJRT_Device_GetDescription_Args {
    pub struct_size: usize,
    pub extension_start: *mut PJRT_Extension_Base,
    pub device: *mut PJRT_Device,
    pub device_description: *mut PJRT_DeviceDescription,
}

#[repr(C)]
pub struct PJRT_DeviceDescription_Id_Args {
    pub struct_size: usize,
    pub extension_start: *mut PJRT_Extension_Base,
    pub device_description: *mut PJRT_DeviceDescription,
    pub id: i32,
}

#[repr(C)]
pub struct PJRT_DeviceDescription_ProcessIndex_Args {
    pub struct_size: usize,
    pub extension_start: *mut PJRT_Extension_Base,
    pub device_description: *mut PJRT_DeviceDescription,
    pub process_index: i32,
}

#[repr(C)]
pub struct PJRT_DeviceDescription_Kind_Args {
    pub struct_size: usize,
    pub extension_start: *mut PJRT_Extension_Base,
    pub device_description: *mut PJRT_DeviceDescription,
    pub device_kind: *const c_char,
    pub device_kind_size: usize,
}

#[repr(C)]
pub struct PJRT_Device_IsAddressable_Args {
    pub struct_size: usize,
    pub extension_start: *mut PJRT_Extension_Base,
    pub device: *mut PJRT_Device,
    pub is_addressable: bool,
}

#[repr(C)]
pub struct PJRT_Device_LocalHardwareId_Args {
    pub struct_size: usize,
    pub extension_start: *mut PJRT_Extension_Base,
    pub device: *mut PJRT_Device,
    pub local_hardware_id: i32,
}

#[repr(C)]
pub struct PJRT_Client_BufferFromHostBuffer_Args {
    pub struct_size: usize,
    pub extension_start: *mut PJRT_Extension_Base,
    pub client: *mut PJRT_Client,
    pub data: *const c_void,
    pub type_: PJRT_Buffer_Type,
    pub dims: *const i64,
    pub num_dims: usize,
    pub byte_strides: *const i64,
    pub num_byte_strides: usize,
    pub host_buffer_semantics: PJRT_HostBufferSemantics,
    pub device: *mut PJRT_Device,
    pub memory: *mut PJRT_Memory,
    pub device_layout: *mut PJRT_Buffer_MemoryLayout,
    pub done_with_host_buffer: *mut PJRT_Event,
    pub buffer: *mut PJRT_Buffer,
}

#[repr(C)]
pub struct PJRT_LoadedExecutable_Destroy_Args {
    pub struct_size: usize,
    pub extension_start: *mut PJRT_Extension_Base,
    pub executable: *mut PJRT_LoadedExecutable,
}

#[repr(C)]
pub struct PJRT_LoadedExecutable_GetExecutable_Args {
    pub struct_size: usize,
    pub extension_start: *mut PJRT_Extension_Base,
    pub loaded_executable: *mut PJRT_LoadedExecutable,
    pub executable: *mut PJRT_Executable,
}

#[repr(C)]
pub struct PJRT_LoadedExecutable_AddressableDevices_Args {
    pub struct_size: usize,
    pub extension_start: *mut PJRT_Extension_Base,
    pub executable: *mut PJRT_LoadedExecutable,
    pub addressable_devices: *const *mut PJRT_Device,
    pub num_addressable_devices: usize,
}

#[repr(C)]
pub struct PJRT_Executable_Destroy_Args {
    pub struct_size: usize,
    pub extension_start: *mut PJRT_Extension_Base,
    pub executable: *mut PJRT_Executable,
}

#[repr(C)]
pub struct PJRT_Executable_NumOutputs_Args {
    pub struct_size: usize,
    pub extension_start: *mut PJRT_Extension_Base,
    pub executable: *mut PJRT_Executable,
    pub num_outputs: usize,
}

#[repr(C)]
pub struct PJRT_ExecuteOptions {
    pub struct_size: usize,
    pub extension_start: *mut PJRT_Extension_Base,
    pub send_callbacks: *mut *mut c_void,
    pub recv_callbacks: *mut *mut c_void,
    pub num_send_ops: usize,
    pub num_recv_ops: usize,
    pub launch_id: i32,
    pub non_donatable_input_indices: *const i64,
    pub num_non_donatable_input_indices: usize,
    pub context: *mut PJRT_ExecuteContext,
    pub call_location: *const c_char,
    pub num_tasks: usize,
    pub task_ids: *mut i32,
    pub incarnation_ids: *mut i64,
    pub multi_slice_config: *mut PJRT_MultiSlice_Config,
    pub use_major_to_minor_data_layout_for_callbacks: bool,
    pub hlo_output_callbacks: *mut c_void,
    pub num_hlo_output_callbacks: usize,
}

#[repr(C)]
pub struct PJRT_LoadedExecutable_Execute_Args {
    pub struct_size: usize,
    pub extension_start: *mut PJRT_Extension_Base,
    pub executable: *mut PJRT_LoadedExecutable,
    pub options: *mut PJRT_ExecuteOptions,
    pub argument_lists: *const *const *mut PJRT_Buffer,
    pub num_devices: usize,
    pub num_args: usize,
    pub output_lists: *const *mut *mut PJRT_Buffer,
    pub device_complete_events: *mut *mut PJRT_Event,
    pub execute_device: *mut PJRT_Device,
}

#[repr(C)]
pub struct PJRT_Buffer_Destroy_Args {
    pub struct_size: usize,
    pub extension_start: *mut PJRT_Extension_Base,
    pub buffer: *mut PJRT_Buffer,
}

#[repr(C)]
pub struct PJRT_Buffer_ElementType_Args {
    pub struct_size: usize,
    pub extension_start: *mut PJRT_Extension_Base,
    pub buffer: *mut PJRT_Buffer,
    pub type_: PJRT_Buffer_Type,
}

#[repr(C)]
pub struct PJRT_Buffer_Dimensions_Args {
    pub struct_size: usize,
    pub extension_start: *mut PJRT_Extension_Base,
    pub buffer: *mut PJRT_Buffer,
    pub dims: *const i64,
    pub num_dims: usize,
}

#[repr(C)]
pub struct PJRT_Buffer_Device_Args {
    pub struct_size: usize,
    pub extension_start: *mut PJRT_Extension_Base,
    pub buffer: *mut PJRT_Buffer,
    pub device: *mut PJRT_Device,
}

#[repr(C)]
pub struct PJRT_Buffer_ToHostBuffer_Args {
    pub struct_size: usize,
    pub extension_start: *mut PJRT_Extension_Base,
    pub src: *mut PJRT_Buffer,
    pub host_layout: *mut PJRT_Buffer_MemoryLayout,
    pub dst: *mut c_void,
    pub dst_size: usize,
    pub event: *mut PJRT_Event,
}

pub type ErrorDestroy = unsafe extern "C" fn(*mut PJRT_Error_Destroy_Args);
pub type ErrorMessage = unsafe extern "C" fn(*mut PJRT_Error_Message_Args);
pub type PluginInitialize =
    unsafe extern "C" fn(*mut PJRT_Plugin_Initialize_Args) -> *mut PJRT_Error;
pub type EventDestroy =
    unsafe extern "C" fn(*mut PJRT_Event_Destroy_Args) -> *mut PJRT_Error;
pub type EventAwait =
    unsafe extern "C" fn(*mut PJRT_Event_Await_Args) -> *mut PJRT_Error;
pub type ClientCreate =
    unsafe extern "C" fn(*mut PJRT_Client_Create_Args) -> *mut PJRT_Error;
pub type ClientDestroy =
    unsafe extern "C" fn(*mut PJRT_Client_Destroy_Args) -> *mut PJRT_Error;
pub type ClientPlatformName =
    unsafe extern "C" fn(*mut PJRT_Client_PlatformName_Args) -> *mut PJRT_Error;
pub type ClientDevices =
    unsafe extern "C" fn(*mut PJRT_Client_Devices_Args) -> *mut PJRT_Error;
pub type ClientAddressableDevices =
    unsafe extern "C" fn(*mut PJRT_Client_AddressableDevices_Args) -> *mut PJRT_Error;
pub type ClientCompile =
    unsafe extern "C" fn(*mut PJRT_Client_Compile_Args) -> *mut PJRT_Error;
pub type ClientBufferFromHostBuffer =
    unsafe extern "C" fn(*mut PJRT_Client_BufferFromHostBuffer_Args) -> *mut PJRT_Error;
pub type DeviceGetDescription =
    unsafe extern "C" fn(*mut PJRT_Device_GetDescription_Args) -> *mut PJRT_Error;
pub type DeviceDescriptionId =
    unsafe extern "C" fn(*mut PJRT_DeviceDescription_Id_Args) -> *mut PJRT_Error;
pub type DeviceDescriptionProcessIndex =
    unsafe extern "C" fn(*mut PJRT_DeviceDescription_ProcessIndex_Args) -> *mut PJRT_Error;
pub type DeviceDescriptionKind =
    unsafe extern "C" fn(*mut PJRT_DeviceDescription_Kind_Args) -> *mut PJRT_Error;
pub type DeviceIsAddressable =
    unsafe extern "C" fn(*mut PJRT_Device_IsAddressable_Args) -> *mut PJRT_Error;
pub type DeviceLocalHardwareId =
    unsafe extern "C" fn(*mut PJRT_Device_LocalHardwareId_Args) -> *mut PJRT_Error;
pub type ExecutableDestroy =
    unsafe extern "C" fn(*mut PJRT_Executable_Destroy_Args) -> *mut PJRT_Error;
pub type ExecutableNumOutputs =
    unsafe extern "C" fn(*mut PJRT_Executable_NumOutputs_Args) -> *mut PJRT_Error;
pub type LoadedExecutableDestroy =
    unsafe extern "C" fn(*mut PJRT_LoadedExecutable_Destroy_Args) -> *mut PJRT_Error;
pub type LoadedExecutableGetExecutable =
    unsafe extern "C" fn(*mut PJRT_LoadedExecutable_GetExecutable_Args) -> *mut PJRT_Error;
pub type LoadedExecutableAddressableDevices =
    unsafe extern "C" fn(*mut PJRT_LoadedExecutable_AddressableDevices_Args) -> *mut PJRT_Error;
pub type LoadedExecutableExecute =
    unsafe extern "C" fn(*mut PJRT_LoadedExecutable_Execute_Args) -> *mut PJRT_Error;
pub type BufferDestroy =
    unsafe extern "C" fn(*mut PJRT_Buffer_Destroy_Args) -> *mut PJRT_Error;
pub type BufferElementType =
    unsafe extern "C" fn(*mut PJRT_Buffer_ElementType_Args) -> *mut PJRT_Error;
pub type BufferDimensions =
    unsafe extern "C" fn(*mut PJRT_Buffer_Dimensions_Args) -> *mut PJRT_Error;
pub type BufferDevice =
    unsafe extern "C" fn(*mut PJRT_Buffer_Device_Args) -> *mut PJRT_Error;
pub type BufferToHostBuffer =
    unsafe extern "C" fn(*mut PJRT_Buffer_ToHostBuffer_Args) -> *mut PJRT_Error;

/// Prefix of PJRT_Api through PJRT_Buffer_ToHostBuffer.
///
/// Unused slots remain opaque pointer-sized entries but preserve the exact
/// field ordering required to reach the functions Severian calls.
#[repr(C)]
pub struct PJRT_Api {
    pub struct_size: usize,
    pub extension_start: *mut PJRT_Extension_Base,
    pub pjrt_api_version: PJRT_Api_Version,

    pub PJRT_Error_Destroy: ErrorDestroy,
    pub PJRT_Error_Message: ErrorMessage,
    pub _error_get_code: *const c_void,
    pub PJRT_Plugin_Initialize: PluginInitialize,
    pub _plugin_attributes: *const c_void,

    pub PJRT_Event_Destroy: EventDestroy,
    pub _event_is_ready: *const c_void,
    pub _event_error: *const c_void,
    pub PJRT_Event_Await: EventAwait,
    pub _event_on_ready: *const c_void,

    pub PJRT_Client_Create: ClientCreate,
    pub PJRT_Client_Destroy: ClientDestroy,
    pub PJRT_Client_PlatformName: ClientPlatformName,
    pub _client_process_index: *const c_void,
    pub _client_platform_version: *const c_void,
    pub PJRT_Client_Devices: ClientDevices,
    pub PJRT_Client_AddressableDevices: ClientAddressableDevices,
    pub _client_lookup_device: *const c_void,
    pub _client_lookup_addressable_device: *const c_void,
    pub _client_addressable_memories: *const c_void,
    pub PJRT_Client_Compile: ClientCompile,
    pub _client_default_device_assignment: *const c_void,
    pub PJRT_Client_BufferFromHostBuffer: ClientBufferFromHostBuffer,

    pub PJRT_DeviceDescription_Id: DeviceDescriptionId,
    pub PJRT_DeviceDescription_ProcessIndex: DeviceDescriptionProcessIndex,
    pub _device_description_attributes: *const c_void,
    pub PJRT_DeviceDescription_Kind: DeviceDescriptionKind,
    pub _device_description_debug_string: *const c_void,
    pub _device_description_to_string: *const c_void,
    pub PJRT_Device_GetDescription: DeviceGetDescription,
    pub PJRT_Device_IsAddressable: DeviceIsAddressable,
    pub PJRT_Device_LocalHardwareId: DeviceLocalHardwareId,
    pub _device_addressable_memories: *const c_void,
    pub _device_default_memory: *const c_void,
    pub _device_memory_stats: *const c_void,

    pub _memory_id: *const c_void,
    pub _memory_kind: *const c_void,
    pub _memory_debug_string: *const c_void,
    pub _memory_to_string: *const c_void,
    pub _memory_addressable_by_devices: *const c_void,

    pub PJRT_Executable_Destroy: ExecutableDestroy,
    pub _executable_name: *const c_void,
    pub _executable_num_replicas: *const c_void,
    pub _executable_num_partitions: *const c_void,
    pub PJRT_Executable_NumOutputs: ExecutableNumOutputs,
    pub _executable_size_generated_code: *const c_void,
    pub _executable_cost_analysis: *const c_void,
    pub _executable_output_memory_kinds: *const c_void,
    pub _executable_optimized_program: *const c_void,
    pub _executable_serialize: *const c_void,

    pub PJRT_LoadedExecutable_Destroy: LoadedExecutableDestroy,
    pub PJRT_LoadedExecutable_GetExecutable: LoadedExecutableGetExecutable,
    pub PJRT_LoadedExecutable_AddressableDevices: LoadedExecutableAddressableDevices,
    pub _loaded_executable_delete: *const c_void,
    pub _loaded_executable_is_deleted: *const c_void,
    pub PJRT_LoadedExecutable_Execute: LoadedExecutableExecute,
    pub _executable_deserialize_and_load: *const c_void,
    pub _loaded_executable_fingerprint: *const c_void,

    pub PJRT_Buffer_Destroy: BufferDestroy,
    pub PJRT_Buffer_ElementType: BufferElementType,
    pub PJRT_Buffer_Dimensions: BufferDimensions,
    pub _buffer_unpadded_dimensions: *const c_void,
    pub _buffer_dynamic_dimension_indices: *const c_void,
    pub _buffer_get_memory_layout: *const c_void,
    pub _buffer_on_device_size: *const c_void,
    pub PJRT_Buffer_Device: BufferDevice,
    pub _buffer_memory: *const c_void,
    pub _buffer_delete: *const c_void,
    pub _buffer_is_deleted: *const c_void,
    pub _buffer_copy_to_device: *const c_void,
    pub PJRT_Buffer_ToHostBuffer: BufferToHostBuffer,
}

pub type GetPjrtApi = unsafe extern "C" fn() -> *const PJRT_Api;

pub const fn struct_size<T>() -> usize {
    std::mem::size_of::<T>()
}

pub fn null_extension() -> *mut PJRT_Extension_Base {
    std::ptr::null_mut()
}
