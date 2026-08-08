//! Native ABI exported by the Severian runtime.
//!
//! The compiler emits `llvm.call @__sev_*` operations. This module is the
//! concrete native side of that boundary.
//!
//! Handles are opaque pointer-shaped integer tokens. The ABI never
//! dereferences foreign pointers here; runtime-owned objects live in registries
//! keyed by those tokens. This makes the first implementation simple enough to
//! stabilize before replacing hot paths with direct pointer-owned objects.
//!
//! NOTE: the current generated runtime crate used `#![forbid(unsafe_code)]`.
//! `#[no_mangle]` is itself considered an unsafe attribute by newer Rust
//! toolchains. Codex should relax that crate-level lint when wiring this module,
//! even though this implementation performs no raw-pointer dereferences.

pub mod boxing;
pub mod channels;
pub mod collections;
pub mod sync;
pub mod tasks;

use std::{
    ffi::c_void,
    sync::atomic::{AtomicUsize, Ordering},
};

static NEXT_HANDLE: AtomicUsize = AtomicUsize::new(1);

pub(crate) fn next_handle_id() -> usize {
    NEXT_HANDLE.fetch_add(1, Ordering::Relaxed)
}

pub(crate) fn handle_from_id(id: usize) -> *mut c_void {
    id as *mut c_void
}

pub(crate) fn id_from_handle(handle: *mut c_void) -> usize {
    handle as usize
}

pub(crate) fn null_handle() -> *mut c_void {
    std::ptr::null_mut()
}

pub(crate) fn is_null_handle(handle: *mut c_void) -> bool {
    handle.is_null()
}
