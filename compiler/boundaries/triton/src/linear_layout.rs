//! Target-independent linear layouts ported from Triton's `LinearLayout`.
//!
//! Donor sources:
//! - `third_party/triton-donor/include/triton/Tools/LinearLayout.h`
//! - `third_party/triton-donor/lib/Tools/LinearLayout.cpp`

use std::fmt;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LayoutDimension {
    pub name: String,
    pub size: u32,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LayoutInputDimension {
    pub name: String,
    /// Each entry is the output vector for one power-of-two input bit.
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
}

impl fmt::Display for LinearLayoutError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyName => formatter.write_str("linear-layout dimensions require names"),
            Self::DuplicateInput(name) => write!(formatter, "duplicate input dimension `{name}`"),
            Self::DuplicateOutput(name) => write!(formatter, "duplicate output dimension `{name}`"),
            Self::NonPowerOfTwoInput { name, size } => {
                write!(
                    formatter,
                    "input dimension `{name}` has non-power-of-two size {size}"
                )
            }
            Self::NonPowerOfTwoOutput { name, size } => {
                write!(
                    formatter,
                    "output dimension `{name}` has non-power-of-two size {size}"
                )
            }
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
            Self::InputRank { expected, found } => {
                write!(
                    formatter,
                    "layout received {found} inputs, expected {expected}"
                )
            }
            Self::InputOutOfBounds { input, value, size } => write!(
                formatter,
                "layout input `{input}` value {value} exceeds size {size}"
            ),
            Self::SizeOverflow(input) => write!(formatter, "input `{input}` size overflows u32"),
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

    pub fn identity_1d(
        size: u32,
        input: impl Into<String>,
        output: impl Into<String>,
    ) -> Result<Self, LinearLayoutError> {
        let input = input.into();
        let output = output.into();
        if !size.is_power_of_two() {
            return Err(LinearLayoutError::NonPowerOfTwoOutput { name: output, size });
        }
        let bases = (0..size.ilog2()).map(|bit| vec![1 << bit]).collect();
        Self::new(
            vec![LayoutInputDimension { name: input, bases }],
            vec![LayoutDimension { name: output, size }],
        )
    }

    pub fn zeros_1d(
        input_size: u32,
        output_size: u32,
        input: impl Into<String>,
        output: impl Into<String>,
    ) -> Result<Self, LinearLayoutError> {
        let input = input.into();
        if !input_size.is_power_of_two() {
            return Err(LinearLayoutError::NonPowerOfTwoInput {
                name: input,
                size: input_size,
            });
        }
        let bases = (0..input_size.ilog2()).map(|_| vec![0]).collect();
        Self::new(
            vec![LayoutInputDimension { name: input, bases }],
            vec![LayoutDimension {
                name: output.into(),
                size: output_size,
            }],
        )
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identity_maps_each_input_to_itself() {
        let layout = LinearLayout::identity_1d(16, "register", "offset").unwrap();
        for value in 0..16 {
            assert_eq!(layout.apply(&[value]).unwrap(), [value]);
        }
    }

    #[test]
    fn zero_layout_represents_broadcast_storage() {
        let layout = LinearLayout::zeros_1d(8, 1, "lane", "scalar").unwrap();
        for lane in 0..8 {
            assert_eq!(layout.apply(&[lane]).unwrap(), [0]);
        }
    }

    #[test]
    fn xor_bases_reproduce_the_donor_swizzle_example() {
        let layout = LinearLayout::new(
            vec![
                LayoutInputDimension {
                    name: "thread".into(),
                    bases: vec![vec![1, 1], vec![2, 2]],
                },
                LayoutInputDimension {
                    name: "warp".into(),
                    bases: vec![vec![0, 1], vec![0, 2]],
                },
            ],
            vec![
                LayoutDimension {
                    name: "x".into(),
                    size: 4,
                },
                LayoutDimension {
                    name: "y".into(),
                    size: 4,
                },
            ],
        )
        .unwrap();
        assert_eq!(layout.apply(&[0, 0]).unwrap(), [0, 0]);
        assert_eq!(layout.apply(&[0, 3]).unwrap(), [0, 3]);
        assert_eq!(layout.apply(&[3, 0]).unwrap(), [3, 3]);
        assert_eq!(layout.apply(&[3, 3]).unwrap(), [3, 0]);
    }

    #[test]
    fn malformed_layouts_fail_before_scheduling() {
        let error = LinearLayout::new(
            vec![LayoutInputDimension {
                name: "lane".into(),
                bases: vec![vec![8]],
            }],
            vec![LayoutDimension {
                name: "offset".into(),
                size: 8,
            }],
        )
        .unwrap_err();
        assert!(matches!(error, LinearLayoutError::BasisOutOfBounds { .. }));
    }
}
