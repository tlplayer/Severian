use std::ffi::c_void;

pub const STRING_ABI_VERSION: u32 = 1;

/// Owned UTF-8 storage contract used at external and compiled-library
/// boundaries. Length and capacity are byte counts, never character counts.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StringAbiV1 {
    pub data: *mut u8,
    pub length: u64,
    pub capacity: u64,
}

/// Non-owning UTF-8 byte view. The owner must outlive every use of this value.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StringViewAbiV1 {
    pub data: *const u8,
    pub length: u64,
}

#[repr(i32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StringStatusV1 {
    Ok = 0,
    AllocationFailed = 1,
    InvalidUtf8 = 2,
    CapacityOverflow = 3,
    InvalidArgument = 4,
}

impl Default for StringAbiV1 {
    fn default() -> Self {
        Self {
            data: std::ptr::null_mut(),
            length: 0,
            capacity: 0,
        }
    }
}

impl Default for StringViewAbiV1 {
    fn default() -> Self {
        Self {
            data: std::ptr::null(),
            length: 0,
        }
    }
}

impl StringAbiV1 {
    pub const fn is_structurally_valid(self) -> bool {
        self.length <= self.capacity
            && ((self.capacity == 0 && (self.data as *mut c_void).is_null())
                || (self.capacity != 0 && !(self.data as *mut c_void).is_null()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_owned_string_has_the_canonical_null_layout() {
        let value = StringAbiV1::default();
        assert!(value.is_structurally_valid());
        assert!(value.data.is_null());
        assert_eq!(std::mem::size_of::<StringAbiV1>(), 24);
        assert_eq!(std::mem::size_of::<StringViewAbiV1>(), 16);
    }

    #[test]
    fn length_cannot_exceed_capacity() {
        let value = StringAbiV1 {
            data: std::ptr::dangling_mut(),
            length: 5,
            capacity: 4,
        };
        assert!(!value.is_structurally_valid());
    }

    #[test]
    fn mlir_owned_operations_are_not_implemented_in_c() {
        let legacy_c = include_str!("../native/string.c");
        for symbol in [
            "__sev_string_concat",
            "__sev_string_compare",
            "__sev_string_release",
        ] {
            assert!(
                !legacy_c.contains(symbol),
                "{symbol} must remain MLIR-owned"
            );
        }
    }
}
