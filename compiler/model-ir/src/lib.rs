#![forbid(unsafe_code)]

use severian_dtype::DType;
use std::collections::HashSet;
use std::fmt;

macro_rules! identifier {
    ($name:ident) => {
        #[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
        pub struct $name(String);

        impl $name {
            pub fn new(value: impl Into<String>) -> Self {
                Self(value.into())
            }

            pub fn as_str(&self) -> &str {
                &self.0
            }
        }

        impl From<&str> for $name {
            fn from(value: &str) -> Self {
                Self::new(value)
            }
        }

        impl fmt::Display for $name {
            fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
                formatter.write_str(&self.0)
            }
        }
    };
}

identifier!(ArchitectureId);
identifier!(ComponentId);
identifier!(ProgramId);
identifier!(WeightId);
identifier!(StateId);
identifier!(CapabilityId);

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Dimension {
    Static(u64),
    Symbol(String),
    Dynamic,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct TensorSpec {
    pub dtype: DType,
    pub dimensions: Vec<Dimension>,
}

impl TensorSpec {
    pub fn new(dtype: DType, dimensions: impl Into<Vec<Dimension>>) -> Self {
        Self {
            dtype,
            dimensions: dimensions.into(),
        }
    }

    pub fn static_element_count(&self) -> Option<u64> {
        self.dimensions.iter().try_fold(1_u64, |count, dimension| {
            let Dimension::Static(size) = dimension else {
                return None;
            };
            count.checked_mul(*size)
        })
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum PortRole {
    Input,
    Output,
    StateInput(StateId),
    StateOutput(StateId),
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Port {
    pub name: String,
    pub tensor: TensorSpec,
    pub role: PortRole,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum ProgramRole {
    Forward,
    Prefill,
    Decode,
    Logits,
    Sample,
    CodecEncode,
    CodecDecode,
    MaskedGenerationStep,
    Custom(String),
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Program {
    pub id: ProgramId,
    pub role: ProgramRole,
    pub entry_point: String,
    pub ports: Vec<Port>,
    /// The model compiler uses this ordered list to bind immutable parameters.
    pub weights: Vec<WeightId>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ComputePolicy {
    pub compute: DType,
    pub accumulation: DType,
    pub output: DType,
}

impl ComputePolicy {
    pub const fn uniform(dtype: DType) -> Self {
        Self {
            compute: dtype,
            accumulation: dtype,
            output: dtype,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum WeightTransform {
    None,
    Transpose(Vec<usize>),
    Reshape(Vec<Dimension>),
    Cast(DType),
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct WeightSpec {
    pub id: WeightId,
    /// Name in the artifact. It intentionally differs from the stable logical id.
    pub artifact_name: String,
    pub storage: TensorSpec,
    pub policy: ComputePolicy,
    pub transform: WeightTransform,
    pub tied_to: Option<WeightId>,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct StateSpec {
    pub id: StateId,
    pub tensor: TensorSpec,
    pub per_layer: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ComponentPlan {
    pub id: ComponentId,
    pub architecture: ArchitectureId,
    pub programs: Vec<ProgramId>,
    pub weights: Vec<WeightId>,
    pub children: Vec<ComponentPlan>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModelPlan {
    pub architecture: ArchitectureId,
    pub revision: String,
    pub programs: Vec<Program>,
    pub weights: Vec<WeightSpec>,
    pub state: Vec<StateSpec>,
    pub capabilities: Vec<CapabilityId>,
    pub components: Vec<ComponentPlan>,
}

impl ModelPlan {
    pub fn program(&self, id: &ProgramId) -> Option<&Program> {
        self.programs.iter().find(|program| &program.id == id)
    }

    pub fn program_for(&self, role: &ProgramRole) -> Option<&Program> {
        self.programs.iter().find(|program| &program.role == role)
    }

    pub fn weight(&self, id: &WeightId) -> Option<&WeightSpec> {
        self.weights.iter().find(|weight| &weight.id == id)
    }

    pub fn supports(&self, capability: &str) -> bool {
        self.capabilities
            .iter()
            .any(|candidate| candidate.as_str() == capability)
    }

    pub fn validate(&self) -> Result<(), PlanError> {
        ensure_unique("program", self.programs.iter().map(|item| item.id.as_str()))?;
        ensure_unique("weight", self.weights.iter().map(|item| item.id.as_str()))?;
        ensure_unique("state", self.state.iter().map(|item| item.id.as_str()))?;
        ensure_unique(
            "capability",
            self.capabilities.iter().map(CapabilityId::as_str),
        )?;

        let programs = self
            .programs
            .iter()
            .map(|program| program.id.clone())
            .collect::<HashSet<_>>();
        let weights = self
            .weights
            .iter()
            .map(|weight| weight.id.clone())
            .collect::<HashSet<_>>();
        let states = self
            .state
            .iter()
            .map(|state| state.id.clone())
            .collect::<HashSet<_>>();

        for program in &self.programs {
            for weight in &program.weights {
                if !weights.contains(weight) {
                    return Err(PlanError::UnknownWeight {
                        owner: program.id.to_string(),
                        weight: weight.to_string(),
                    });
                }
            }
            for port in &program.ports {
                let state = match &port.role {
                    PortRole::StateInput(state) | PortRole::StateOutput(state) => Some(state),
                    PortRole::Input | PortRole::Output => None,
                };
                if state.is_some_and(|state| !states.contains(state)) {
                    return Err(PlanError::UnknownState {
                        owner: program.id.to_string(),
                        state: state.expect("checked above").to_string(),
                    });
                }
            }
        }

        for weight in &self.weights {
            if let Some(tied_to) = &weight.tied_to {
                if tied_to == &weight.id || !weights.contains(tied_to) {
                    return Err(PlanError::InvalidTiedWeight {
                        weight: weight.id.to_string(),
                        tied_to: tied_to.to_string(),
                    });
                }
            }
        }

        for component in &self.components {
            validate_component(component, &programs, &weights)?;
        }
        Ok(())
    }
}

fn ensure_unique<'a>(kind: &'static str, values: impl Iterator<Item = &'a str>) -> Result<(), PlanError> {
    let mut seen = HashSet::new();
    for value in values {
        if !seen.insert(value) {
            return Err(PlanError::DuplicateId {
                kind,
                id: value.to_owned(),
            });
        }
    }
    Ok(())
}

fn validate_component(
    component: &ComponentPlan,
    programs: &HashSet<ProgramId>,
    weights: &HashSet<WeightId>,
) -> Result<(), PlanError> {
    for program in &component.programs {
        if !programs.contains(program) {
            return Err(PlanError::UnknownProgram {
                owner: component.id.to_string(),
                program: program.to_string(),
            });
        }
    }
    for weight in &component.weights {
        if !weights.contains(weight) {
            return Err(PlanError::UnknownWeight {
                owner: component.id.to_string(),
                weight: weight.to_string(),
            });
        }
    }
    for child in &component.children {
        validate_component(child, programs, weights)?;
    }
    Ok(())
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PlanError {
    DuplicateId { kind: &'static str, id: String },
    UnknownProgram { owner: String, program: String },
    UnknownWeight { owner: String, weight: String },
    UnknownState { owner: String, state: String },
    InvalidTiedWeight { weight: String, tied_to: String },
}

impl fmt::Display for PlanError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::DuplicateId { kind, id } => write!(formatter, "duplicate {kind} id `{id}`"),
            Self::UnknownProgram { owner, program } => {
                write!(formatter, "`{owner}` references unknown program `{program}`")
            }
            Self::UnknownWeight { owner, weight } => {
                write!(formatter, "`{owner}` references unknown weight `{weight}`")
            }
            Self::UnknownState { owner, state } => {
                write!(formatter, "`{owner}` references unknown state `{state}`")
            }
            Self::InvalidTiedWeight { weight, tied_to } => {
                write!(formatter, "weight `{weight}` has invalid tie target `{tied_to}`")
            }
        }
    }
}

impl std::error::Error for PlanError {}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ShapeBinding {
    pub symbol: String,
    pub value: u64,
}

/// Stable identity for an executable specialization. Weight bytes are not part
/// of this key; callers add the resolved artifact digest as `artifact_digest`.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct CompileKey {
    pub architecture: ArchitectureId,
    pub revision: String,
    pub artifact_digest: String,
    pub program: ProgramId,
    pub backend: String,
    pub device: String,
    pub shapes: Vec<ShapeBinding>,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tensor() -> TensorSpec {
        TensorSpec::new(DType::BF16, vec![Dimension::Symbol("sequence".into())])
    }

    #[test]
    fn validates_multi_program_state_edges() {
        let state = StateId::from("kv");
        let plan = ModelPlan {
            architecture: ArchitectureId::from("qwen2"),
            revision: "main".into(),
            programs: vec![Program {
                id: ProgramId::from("decode"),
                role: ProgramRole::Decode,
                entry_point: "qwen2_decode".into(),
                ports: vec![Port {
                    name: "kv_in".into(),
                    tensor: tensor(),
                    role: PortRole::StateInput(state.clone()),
                }],
                weights: vec![WeightId::from("embed")],
            }],
            weights: vec![WeightSpec {
                id: WeightId::from("embed"),
                artifact_name: "model.embed_tokens.weight".into(),
                storage: tensor(),
                policy: ComputePolicy::uniform(DType::BF16),
                transform: WeightTransform::None,
                tied_to: None,
            }],
            state: vec![StateSpec {
                id: state,
                tensor: tensor(),
                per_layer: true,
            }],
            capabilities: vec![CapabilityId::from("text.generate")],
            components: vec![],
        };
        assert_eq!(plan.validate(), Ok(()));
        assert!(plan.supports("text.generate"));
    }

    #[test]
    fn rejects_missing_weight_bindings() {
        let plan = ModelPlan {
            architecture: ArchitectureId::from("qwen2"),
            revision: "main".into(),
            programs: vec![Program {
                id: ProgramId::from("forward"),
                role: ProgramRole::Forward,
                entry_point: "forward".into(),
                ports: vec![],
                weights: vec![WeightId::from("missing")],
            }],
            weights: vec![],
            state: vec![],
            capabilities: vec![],
            components: vec![],
        };
        assert!(matches!(plan.validate(), Err(PlanError::UnknownWeight { .. })));
    }
}

