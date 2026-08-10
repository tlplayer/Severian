//! Owned PJRT asynchronous event wrapper.

use super::{api, error, plugin::RawPjrtPlugin};
use crate::Result;
use std::ptr::NonNull;

pub struct RawEvent {
    plugin: RawPjrtPlugin,
    event: NonNull<api::PJRT_Event>,
}

unsafe impl Send for RawEvent {}
unsafe impl Sync for RawEvent {}

impl RawEvent {
    pub unsafe fn from_raw(
        plugin: RawPjrtPlugin,
        event: *mut api::PJRT_Event,
    ) -> Result<Self> {
        Ok(Self {
            plugin,
            event: NonNull::new(event)
                .ok_or_else(|| error::invalid_raw_pointer("PJRT_Event"))?,
        })
    }

    pub fn raw(&self) -> *mut api::PJRT_Event {
        self.event.as_ptr()
    }

    pub fn is_ready(&self) -> Result<bool> {
        let api = self.plugin.api();
        let mut args = api::PJRT_Event_IsReady_Args {
            struct_size: api::struct_size::<api::PJRT_Event_IsReady_Args>(),
            extension_start: api::null_extension(),
            event: self.raw(),
            is_ready: false,
        };
        let result = unsafe { (api.PJRT_Event_IsReady)(&mut args) };
        unsafe { error::check(api, result)? };
        Ok(args.is_ready)
    }

    pub fn await_ready(&self) -> Result<()> {
        let api = self.plugin.api();
        let mut args = api::PJRT_Event_Await_Args {
            struct_size: api::struct_size::<api::PJRT_Event_Await_Args>(),
            extension_start: api::null_extension(),
            event: self.raw(),
        };
        let result = unsafe { (api.PJRT_Event_Await)(&mut args) };
        unsafe { error::check(api, result) }
    }
}

impl Drop for RawEvent {
    fn drop(&mut self) {
        let api = self.plugin.api();
        let mut args = api::PJRT_Event_Destroy_Args {
            struct_size: api::struct_size::<api::PJRT_Event_Destroy_Args>(),
            extension_start: api::null_extension(),
            event: self.raw(),
        };
        let result = unsafe { (api.PJRT_Event_Destroy)(&mut args) };
        let _ = unsafe { error::check(api, result) };
    }
}
