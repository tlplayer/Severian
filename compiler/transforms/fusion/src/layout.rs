//! Severian-owned GPU layout planning.
//!
//! The XOR-basis representation is translated from Triton's `LinearLayout`
//! (`third_party/triton-donor/include/triton/Tools/LinearLayout.h` and
//! `third_party/triton-donor/lib/Tools/LinearLayout.cpp`). The representation
//! is now ordinary Severian fusion data: it neither links to Triton nor emits
//! a Triton dialect.

use std::fmt;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LayoutDimension {
    pub name: String,
    pub size: u32,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LayoutInputDimension {
    pub name: String,
    /// One output vector for each power-of-two input bit.
    pub bases: Vec<Vec<u32>>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LinearLayout {
    inputs: Vec<LayoutInputDimension>,
    outputs: Vec<LayoutDimension>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LinearLayoutError {
    EmptyName,
    DuplicateInput(String),
    DuplicateOutput(String),
    NonPowerOfTwoInput {
        name: String,
        size: u32,
    },
    NonPowerOfTwoOutput {
        name: String,
        size: u32,
    },
    BasisRank {
        input: String,
        expected: usize,
        found: usize,
    },
    BasisOutOfBounds {
        input: String,
        output: String,
        value: u32,
        size: u32,
    },
    InputRank {
        expected: usize,
        found: usize,
    },
    InputOutOfBounds {
        input: String,
        value: u32,
        size: u32,
    },
    SizeOverflow(String),
    InvalidWarpSize(u32),
    InvalidWarpsPerBlock(u32),
    ElementCountOverflow(u64),
}

impl fmt::Display for LinearLayoutError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyName => formatter.write_str("linear-layout dimensions require names"),
            Self::DuplicateInput(name) => write!(formatter, "duplicate input dimension `{name}`"),
            Self::DuplicateOutput(name) => write!(formatter, "duplicate output dimension `{name}`"),
            Self::NonPowerOfTwoInput { name, size } => write!(
                formatter,
                "input dimension `{name}` has non-power-of-two size {size}"
            ),
            Self::NonPowerOfTwoOutput { name, size } => write!(
                formatter,
                "output dimension `{name}` has non-power-of-two size {size}"
            ),
            Self::BasisRank {
                input,
                expected,
                found,
            } => write!(
                formatter,
                "input `{input}` basis has output rank {found}, expected {expected}"
            ),
            Self::BasisOutOfBounds {
                input,
                output,
                value,
                size,
            } => write!(
                formatter,
                "input `{input}` basis coordinate {value} exceeds `{output}` size {size}"
            ),
            Self::InputRank { expected, found } => write!(
                formatter,
                "layout received {found} inputs, expected {expected}"
            ),
            Self::InputOutOfBounds { input, value, size } => write!(
                formatter,
                "layout input `{input}` value {value} exceeds size {size}"
            ),
            Self::SizeOverflow(input) => write!(formatter, "input `{input}` size overflows u32"),
            Self::InvalidWarpSize(size) => {
                write!(formatter, "warp size {size} is not a power of two")
            }
            Self::InvalidWarpsPerBlock(count) => write!(
                formatter,
                "warps-per-block {count} is not a nonzero power of two"
            ),
            Self::ElementCountOverflow(count) => write!(
                formatter,
                "element count {count} cannot be scheduled with 32-bit GPU dimensions"
            ),
        }
    }
}

impl std::error::Error for LinearLayoutError {}

impl LinearLayout {
    pub fn new(
        inputs: Vec<LayoutInputDimension>,
        outputs: Vec<LayoutDimension>,
    ) -> Result<Self, LinearLayoutError> {
        for (index, input) in inputs.iter().enumerate() {
            if input.name.is_empty() {
                return Err(LinearLayoutError::EmptyName);
            }
            if inputs[..index].iter().any(|found| found.name == input.name) {
                return Err(LinearLayoutError::DuplicateInput(input.name.clone()));
            }
        }
        for (index, output) in outputs.iter().enumerate() {
            if output.name.is_empty() {
                return Err(LinearLayoutError::EmptyName);
            }
            if !output.size.is_power_of_two() {
                return Err(LinearLayoutError::NonPowerOfTwoOutput {
                    name: output.name.clone(),
                    size: output.size,
                });
            }
            if outputs[..index]
                .iter()
                .any(|found| found.name == output.name)
            {
                return Err(LinearLayoutError::DuplicateOutput(output.name.clone()));
            }
        }
        for input in &inputs {
            for basis in &input.bases {
                if basis.len() != outputs.len() {
                    return Err(LinearLayoutError::BasisRank {
                        input: input.name.clone(),
                        expected: outputs.len(),
                        found: basis.len(),
                    });
                }
                for (coordinate, output) in basis.iter().zip(&outputs) {
                    if *coordinate >= output.size {
                        return Err(LinearLayoutError::BasisOutOfBounds {
                            input: input.name.clone(),
                            output: output.name.clone(),
                            value: *coordinate,
                            size: output.size,
                        });
                    }
                }
            }
        }
        Ok(Self { inputs, outputs })
    }

    pub fn inputs(&self) -> &[LayoutInputDimension] {
        &self.inputs
    }

    pub fn outputs(&self) -> &[LayoutDimension] {
        &self.outputs
    }

    pub fn input_size(&self, index: usize) -> Result<u32, LinearLayoutError> {
        let input = &self.inputs[index];
        1u32.checked_shl(u32::try_from(input.bases.len()).unwrap_or(u32::MAX))
            .ok_or_else(|| LinearLayoutError::SizeOverflow(input.name.clone()))
    }

    pub fn apply(&self, values: &[u32]) -> Result<Vec<u32>, LinearLayoutError> {
        if values.len() != self.inputs.len() {
            return Err(LinearLayoutError::InputRank {
                expected: self.inputs.len(),
                found: values.len(),
            });
        }
        let mut result = vec![0; self.outputs.len()];
        for (index, (input, value)) in self.inputs.iter().zip(values).enumerate() {
            let size = self.input_size(index)?;
            if *value >= size {
                return Err(LinearLayoutError::InputOutOfBounds {
                    input: input.name.clone(),
                    value: *value,
                    size,
                });
            }
            for (bit, basis) in input.bases.iter().enumerate() {
                if value & (1 << bit) == 0 {
                    continue;
                }
                for (coordinate, basis_coordinate) in result.iter_mut().zip(basis) {
                    *coordinate ^= basis_coordinate;
                }
            }
        }
        Ok(result)
    }
}

/// One-dimensional blocked schedule for a fused elementwise region. Shape
/// delinearization remains a later MLIR concern, so rank is data and does not
/// create a different schedule kind.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ElementwiseSchedule {
    pub element_count: u64,
    pub warp_size: u32,
    pub warps_per_block: u32,
    pub threads_per_block: u32,
    pub block_count: u32,
    pub layout: LinearLayout,
}

/// Translates Triton's default blocked-encoding policy: reverse logical order,
/// one element per thread, and a power-of-two lane/warp decomposition. Tail
/// elements are represented by the launch mask rather than by another kernel.
pub fn blocked_elementwise_schedule(
    element_count: u64,
    warp_size: u32,
    warps_per_block: u32,
) -> Result<ElementwiseSchedule, LinearLayoutError> {
    if !warp_size.is_power_of_two() {
        return Err(LinearLayoutError::InvalidWarpSize(warp_size));
    }
    if warps_per_block == 0 || !warps_per_block.is_power_of_two() {
        return Err(LinearLayoutError::InvalidWarpsPerBlock(warps_per_block));
    }
    let threads_per_block = warp_size
        .checked_mul(warps_per_block)
        .ok_or(LinearLayoutError::ElementCountOverflow(element_count))?;
    let block_count = element_count
        .div_ceil(u64::from(threads_per_block))
        .try_into()
        .map_err(|_| LinearLayoutError::ElementCountOverflow(element_count))?;
    let lane_bases = (0..warp_size.ilog2()).map(|bit| vec![1 << bit]).collect();
    let warp_bases = (0..warps_per_block.ilog2())
        .map(|bit| vec![warp_size << bit])
        .collect();
    let layout = LinearLayout::new(
        vec![
            LayoutInputDimension {
                name: "lane".into(),
                bases: lane_bases,
            },
            LayoutInputDimension {
                name: "warp".into(),
                bases: warp_bases,
            },
        ],
        vec![LayoutDimension {
            name: "element".into(),
            size: threads_per_block,
        }],
    )?;
    Ok(ElementwiseSchedule {
        element_count,
        warp_size,
        warps_per_block,
        threads_per_block,
        block_count,
        layout,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn blocked_schedule_maps_lanes_and_warps_without_overlap() {
        let schedule = blocked_elementwise_schedule(1_001, 32, 4).unwrap();
        assert_eq!(schedule.threads_per_block, 128);
        assert_eq!(schedule.block_count, 8);
        let mut offsets = std::collections::BTreeSet::new();
        for warp in 0..4 {
            for lane in 0..32 {
                offsets.insert(schedule.layout.apply(&[lane, warp]).unwrap()[0]);
            }
        }
        assert_eq!(offsets, (0..128).collect());
    }

    #[test]
    fn blocked_schedule_uses_one_masked_kernel_for_a_tail() {
        let schedule = blocked_elementwise_schedule(129, 64, 2).unwrap();
        assert_eq!(schedule.block_count, 2);
        assert_eq!(schedule.threads_per_block, 128);
    }
}
