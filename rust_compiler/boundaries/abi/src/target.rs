use crate::CallingConvention;
use severian_target::{Architecture, OperatingSystem, TargetSpec};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Endianness {
    Little,
    Big,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ScalarLayout {
    pub size: u64,
    pub alignment: u32,
}

impl ScalarLayout {
    pub const fn new(size: u64, alignment: u32) -> Self {
        Self { size, alignment }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TargetDataLayout {
    pub endianness: Endianness,
    pub pointer: ScalarLayout,
    pub integer_alignments: Vec<(u16, u32)>,
    pub float_alignments: Vec<(u16, u32)>,
    pub aggregate_alignment: u32,
    pub stack_alignment: u32,
    pub machine_integer_bits: u16,
    pub machine_float_bits: u16,
}

impl TargetDataLayout {
    pub fn scalar(&self, bits: u16, float: bool) -> ScalarLayout {
        let bytes = u64::from(bits.div_ceil(8));
        let table = if float {
            &self.float_alignments
        } else {
            &self.integer_alignments
        };
        let alignment = table
            .iter()
            .find_map(|(width, alignment)| (*width >= bits).then_some(*alignment))
            .or_else(|| table.last().map(|(_, alignment)| *alignment))
            .unwrap_or_else(|| u32::try_from(bytes.next_power_of_two()).unwrap_or(u32::MAX));
        ScalarLayout::new(bytes, alignment)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AbiTarget {
    pub triple: String,
    pub architecture: Architecture,
    pub operating_system: OperatingSystem,
    pub data_layout: TargetDataLayout,
}

impl AbiTarget {
    pub fn derive(target: &TargetSpec) -> Self {
        Self::from_spec_and_pointer_bits(target, target.pointer_bits())
    }

    pub fn from_spec_and_pointer_bits(target: &TargetSpec, pointer_bits: u16) -> Self {
        let pointer_bytes = u64::from(pointer_bits.div_ceil(8));
        let pointer_alignment = u32::try_from(pointer_bytes).unwrap_or(u32::MAX);
        Self {
            triple: target.triple.clone(),
            architecture: target.architecture,
            operating_system: target.operating_system,
            data_layout: TargetDataLayout {
                endianness: Endianness::Little,
                pointer: ScalarLayout::new(pointer_bytes, pointer_alignment),
                integer_alignments: vec![(8, 1), (16, 2), (32, 4), (64, 8), (128, 16)],
                float_alignments: vec![(16, 2), (32, 4), (64, 8), (128, 16)],
                aggregate_alignment: pointer_alignment,
                stack_alignment: if pointer_bits >= 64 {
                    16
                } else {
                    pointer_alignment
                },
                machine_integer_bits: target.machine_integer_bits(),
                machine_float_bits: target.machine_float_bits(),
            },
        }
    }

    pub const fn native_c_convention(&self) -> CallingConvention {
        match (self.architecture, self.operating_system) {
            (Architecture::X86_64, OperatingSystem::Windows) => CallingConvention::Win64,
            (Architecture::X86_64, _) => CallingConvention::SysV64,
            (Architecture::Aarch64, _) => CallingConvention::Aapcs64,
            _ => CallingConvention::C,
        }
    }

    pub const fn resolve_convention(&self, convention: CallingConvention) -> CallingConvention {
        match convention {
            CallingConvention::C | CallingConvention::System => self.native_c_convention(),
            concrete => concrete,
        }
    }
}
