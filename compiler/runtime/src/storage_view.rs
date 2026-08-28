use core::ffi::c_void;

/// Version of the native storage-view descriptor shared by model loaders and
/// host launchers. Compute kernels never receive this pointer as an MLIR
/// builtin tensor value.
pub const STORAGE_VIEW_ABI_VERSION: u32 = 1;
pub const STORAGE_VIEW_ABI_MAGIC: u64 = 0x5356_5354_4f52_4147;

#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StorageElementKind {
    SignedInteger = 1,
    UnsignedInteger = 2,
    Float = 3,
}

#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StorageFloatFormat {
    None = 0,
    Ieee = 1,
    BrainFloat = 2,
    Float8E4M3Fn = 3,
    Float8E5M2 = 4,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StorageElementRepresentationAbi {
    pub abi_version: u32,
    pub byte_size: u32,
    pub kind: StorageElementKind,
    pub bits: u32,
    pub float_format: StorageFloatFormat,
    pub reserved: u32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct StorageViewAbi {
    pub magic: u64,
    pub abi_version: u32,
    pub byte_size: u32,
    pub flags: u64,
    pub data: *const u8,
    pub byte_length: u64,
    pub rank: u64,
    pub dimensions: *const i64,
    pub strides: *const i64,
    pub offset: i64,
    pub element: StorageElementRepresentationAbi,
    pub owner: *mut c_void,
}

pub const STORAGE_VIEW_READ_ONLY: u64 = 1 << 0;
pub const STORAGE_VIEW_CONTIGUOUS: u64 = 1 << 1;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StorageOwnership {
    Borrowed,
    Runtime,
}

/// Safe, owned copy of the metadata carried by `StorageViewAbi`. Native
/// pointers remain data addresses; rank, dimensions, and strides are copied
/// before compiler specialization begins.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StorageView {
    pub data: u64,
    pub byte_length: u64,
    pub element: StorageElementRepresentationAbi,
    pub dimensions: Vec<u64>,
    pub strides: Vec<i64>,
    pub offset: i64,
    pub ownership: StorageOwnership,
}

impl StorageView {
    pub fn new(
        data: u64,
        byte_length: u64,
        element: StorageElementRepresentationAbi,
        dimensions: Vec<u64>,
        strides: Vec<i64>,
        offset: i64,
        ownership: StorageOwnership,
    ) -> Result<Self, StorageViewError> {
        if element.abi_version != STORAGE_VIEW_ABI_VERSION
            || element.byte_size as usize != core::mem::size_of::<StorageElementRepresentationAbi>()
        {
            return Err(StorageViewError::ElementAbiMismatch);
        }
        if dimensions.len() != strides.len() {
            return Err(StorageViewError::RankStrideMismatch {
                rank: dimensions.len(),
                strides: strides.len(),
            });
        }
        if element.bits == 0 || element.bits % 8 != 0 {
            return Err(StorageViewError::InvalidElementWidth(element.bits));
        }
        Ok(Self {
            data,
            byte_length,
            element,
            dimensions,
            strides,
            offset,
            ownership,
        })
    }

    pub fn rank(&self) -> usize {
        self.dimensions.len()
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StorageViewError {
    ElementAbiMismatch,
    RankStrideMismatch { rank: usize, strides: usize },
    InvalidElementWidth(u32),
}

impl core::fmt::Display for StorageViewError {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(formatter, "{self:?}")
    }
}

impl std::error::Error for StorageViewError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn storage_view_abi_is_versioned_and_pointer_backed() {
        assert_eq!(STORAGE_VIEW_ABI_VERSION, 1);
        assert_eq!(core::mem::size_of::<StorageElementRepresentationAbi>(), 24);
        assert_eq!(core::mem::size_of::<StorageViewAbi>(), 104);
        assert_eq!(core::mem::align_of::<StorageViewAbi>(), 8);
    }

    #[test]
    fn owned_storage_view_copies_rank_shape_stride_and_ownership_metadata() {
        let view = StorageView::new(
            0x2000,
            48,
            StorageElementRepresentationAbi {
                abi_version: STORAGE_VIEW_ABI_VERSION,
                byte_size: core::mem::size_of::<StorageElementRepresentationAbi>() as u32,
                kind: StorageElementKind::Float,
                bits: 16,
                float_format: StorageFloatFormat::BrainFloat,
                reserved: 0,
            },
            vec![2, 3, 4],
            vec![12, 4, 1],
            7,
            StorageOwnership::Runtime,
        )
        .unwrap();
        assert_eq!(view.rank(), 3);
        assert_eq!(view.dimensions, [2, 3, 4]);
        assert_eq!(view.strides, [12, 4, 1]);
        assert_eq!(view.offset, 7);
        assert_eq!(view.ownership, StorageOwnership::Runtime);
    }
}
