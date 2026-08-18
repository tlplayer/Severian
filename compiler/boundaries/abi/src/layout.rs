use std::collections::BTreeMap;
use std::error::Error;
use std::fmt;

use crate::{AbiType, AddressSpaceId, FloatType, IntType, IntWidth, RecordRepr, ResourceRepr};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Endianness {
    Little,
    Big,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Layout {
    pub size: u64,
    pub align: u64,
}

impl Layout {
    pub const fn new(size: u64, align: u64) -> Self {
        Self { size, align }
    }
}

/// Target layout can assign different pointer widths/alignments to different
/// address spaces. That is enough for host/device/shared memory models without
/// teaching ABI about CUDA, ROCm, XLA, or any particular runtime.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TargetDataLayout {
    pub endianness: Endianness,
    pub pointers: BTreeMap<AddressSpaceId, Layout>,
    pub integer_alignments: BTreeMap<u16, u64>,
    pub float_alignments: BTreeMap<u16, u64>,
}

impl TargetDataLayout {
    pub fn new(endianness: Endianness, default_pointer: Layout) -> Self {
        let mut pointers = BTreeMap::new();
        pointers.insert(AddressSpaceId::default_space(), default_pointer);
        Self {
            endianness,
            pointers,
            integer_alignments: BTreeMap::new(),
            float_alignments: BTreeMap::new(),
        }
    }

    pub fn with_pointer_layout(mut self, address_space: AddressSpaceId, layout: Layout) -> Self {
        self.pointers.insert(address_space, layout);
        self
    }

    pub fn with_integer_alignment(mut self, bits: u16, align: u64) -> Self {
        self.integer_alignments.insert(bits, align);
        self
    }

    pub fn with_float_alignment(mut self, bits: u16, align: u64) -> Self {
        self.float_alignments.insert(bits, align);
        self
    }

    pub fn pointer_layout(&self, address_space: &AddressSpaceId) -> Result<Layout, LayoutError> {
        self.pointers
            .get(address_space)
            .copied()
            .ok_or_else(|| LayoutError::UnknownAddressSpace(address_space.clone()))
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FieldLayout {
    pub name: String,
    pub offset: u64,
    pub layout: Layout,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RecordLayout {
    pub layout: Layout,
    pub fields: Vec<FieldLayout>,
}

pub fn layout_of(ty: &AbiType, target: &TargetDataLayout) -> Result<Layout, LayoutError> {
    match ty {
        AbiType::Unit => Ok(Layout::new(0, 1)),
        AbiType::Int(int) => integer_layout(*int, target),
        AbiType::Float(float) => float_layout(*float, target),
        AbiType::Pointer(pointer) => target.pointer_layout(&pointer.address_space),
        AbiType::Function(_) => target.pointer_layout(&AddressSpaceId::default_space()),
        AbiType::Array(array) => {
            let element = layout_of(&array.element, target)?;
            let size = element.size.checked_mul(array.length).ok_or(LayoutError::SizeOverflow)?;
            Ok(Layout::new(size, element.align))
        }
        AbiType::Record(record) => Ok(layout_record(record, target)?.layout),
        AbiType::Union(union) => {
            let mut size = 0_u64;
            let mut align = 1_u64;
            for field in &union.fields {
                let field_layout = layout_of(&field.ty, target)?;
                size = size.max(field_layout.size);
                align = align.max(field_layout.align);
            }
            Ok(Layout::new(align_up(size, align)?, align))
        }
        AbiType::Enum(enumeration) => integer_layout(enumeration.repr, target),
        AbiType::Resource(resource) => match &resource.repr {
            ResourceRepr::Pointer { address_space } => target.pointer_layout(address_space),
            ResourceRepr::Integer(int) => integer_layout(*int, target),
        },
        AbiType::Opaque(id) => Err(LayoutError::OpaqueByValue(id.to_string())),
    }
}

pub fn layout_record(
    record: &crate::RecordType,
    target: &TargetDataLayout,
) -> Result<RecordLayout, LayoutError> {
    match record.repr {
        RecordRepr::Transparent => layout_transparent_record(record, target),
        RecordRepr::C | RecordRepr::Packed => layout_aggregate_record(record, target),
    }
}

fn layout_aggregate_record(
    record: &crate::RecordType,
    target: &TargetDataLayout,
) -> Result<RecordLayout, LayoutError> {
    let packed = record.repr == RecordRepr::Packed;
    let mut offset = 0_u64;
    let mut record_align = 1_u64;
    let mut fields = Vec::with_capacity(record.fields.len());

    for field in &record.fields {
        let mut field_layout = layout_of(&field.ty, target)?;
        if packed {
            field_layout.align = 1;
        }

        offset = align_up(offset, field_layout.align)?;
        fields.push(FieldLayout {
            name: field.name.clone(),
            offset,
            layout: field_layout,
        });
        offset = offset.checked_add(field_layout.size).ok_or(LayoutError::SizeOverflow)?;
        record_align = record_align.max(field_layout.align);
    }

    let size = align_up(offset, record_align)?;
    Ok(RecordLayout { layout: Layout::new(size, record_align), fields })
}

fn layout_transparent_record(
    record: &crate::RecordType,
    target: &TargetDataLayout,
) -> Result<RecordLayout, LayoutError> {
    if record.fields.len() != 1 {
        return Err(LayoutError::InvalidTransparentRecord(record.id.to_string()));
    }

    let field = &record.fields[0];
    let layout = layout_of(&field.ty, target)?;
    Ok(RecordLayout {
        layout,
        fields: vec![FieldLayout { name: field.name.clone(), offset: 0, layout }],
    })
}

fn integer_layout(int: IntType, target: &TargetDataLayout) -> Result<Layout, LayoutError> {
    match int.width {
        IntWidth::Pointer => target.pointer_layout(&AddressSpaceId::default_space()),
        IntWidth::Fixed(bits) => {
            if bits == 0 || bits % 8 != 0 {
                return Err(LayoutError::InvalidIntegerWidth(bits));
            }
            let size = u64::from(bits / 8);
            let align = target.integer_alignments.get(&bits).copied()
                .ok_or(LayoutError::MissingIntegerAlignment(bits))?;
            Ok(Layout::new(size, align))
        }
    }
}

fn float_layout(float: FloatType, target: &TargetDataLayout) -> Result<Layout, LayoutError> {
    let bits = match float {
        FloatType::F16 | FloatType::BF16 => 16,
        FloatType::F32 => 32,
        FloatType::F64 => 64,
    };
    let align = target.float_alignments.get(&bits).copied()
        .ok_or(LayoutError::MissingFloatAlignment(bits))?;
    Ok(Layout::new(u64::from(bits / 8), align))
}

fn align_up(value: u64, align: u64) -> Result<u64, LayoutError> {
    if align == 0 || !align.is_power_of_two() {
        return Err(LayoutError::InvalidAlignment(align));
    }
    let mask = align - 1;
    value.checked_add(mask).map(|v| v & !mask).ok_or(LayoutError::SizeOverflow)
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum LayoutError {
    InvalidAlignment(u64),
    InvalidIntegerWidth(u16),
    MissingIntegerAlignment(u16),
    MissingFloatAlignment(u16),
    UnknownAddressSpace(AddressSpaceId),
    InvalidTransparentRecord(String),
    OpaqueByValue(String),
    SizeOverflow,
}

impl fmt::Display for LayoutError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidAlignment(v) => write!(f, "invalid ABI alignment `{v}`"),
            Self::InvalidIntegerWidth(v) => write!(f, "invalid ABI integer width `{v}`"),
            Self::MissingIntegerAlignment(v) => write!(f, "target does not define alignment for i{v}/u{v}"),
            Self::MissingFloatAlignment(v) => write!(f, "target does not define alignment for f{v}"),
            Self::UnknownAddressSpace(v) => write!(f, "target does not define pointer layout for address space `{v}`"),
            Self::InvalidTransparentRecord(v) => write!(f, "transparent record `{v}` must contain exactly one field"),
            Self::OpaqueByValue(v) => write!(f, "opaque ABI type `{v}` cannot cross the boundary by value"),
            Self::SizeOverflow => write!(f, "ABI layout size overflow"),
        }
    }
}

impl Error for LayoutError {}
