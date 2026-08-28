use serde::{Deserialize, Serialize};
use severian_fusion::{FusionGraph, FusionNode, GraphError, NodeId};

pub const PROGRAM_MAGIC: [u8; 8] = *b"SEVJIT01";
pub const PROGRAM_VERSION: u32 = 1;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TensorJitTarget {
    Cpu,
    Amd,
    Nvidia,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TensorJitProgram {
    pub target: TensorJitTarget,
    pub architecture: String,
    pub nodes: Vec<FusionNode>,
    pub inputs: Vec<NodeId>,
    pub outputs: Vec<NodeId>,
}

#[derive(Debug)]
pub enum ProgramError {
    Header,
    Version(u32),
    Codec(String),
    Graph(GraphError),
}

impl std::fmt::Display for ProgramError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Header => formatter.write_str("invalid Tensor-JIT program header"),
            Self::Version(version) => {
                write!(formatter, "unsupported Tensor-JIT program version {version}")
            }
            Self::Codec(message) => formatter.write_str(message),
            Self::Graph(error) => error.fmt(formatter),
        }
    }
}

impl std::error::Error for ProgramError {}

impl TensorJitProgram {
    pub fn graph(&self) -> Result<FusionGraph, ProgramError> {
        FusionGraph::new(self.nodes.clone()).map_err(ProgramError::Graph)
    }

    pub fn encode(&self) -> Result<Vec<u8>, ProgramError> {
        let payload =
            bincode::serialize(self).map_err(|error| ProgramError::Codec(error.to_string()))?;
        let size = u32::try_from(payload.len()).map_err(|_| ProgramError::Header)?;
        let mut encoded = Vec::with_capacity(16 + payload.len());
        encoded.extend_from_slice(&PROGRAM_MAGIC);
        encoded.extend_from_slice(&PROGRAM_VERSION.to_le_bytes());
        encoded.extend_from_slice(&size.to_le_bytes());
        encoded.extend_from_slice(&payload);
        Ok(encoded)
    }

    pub fn decode(encoded: &[u8]) -> Result<Self, ProgramError> {
        if encoded.len() < 16 || encoded[..8] != PROGRAM_MAGIC {
            return Err(ProgramError::Header);
        }
        let version = u32::from_le_bytes(encoded[8..12].try_into().expect("fixed header"));
        if version != PROGRAM_VERSION {
            return Err(ProgramError::Version(version));
        }
        let size = u32::from_le_bytes(encoded[12..16].try_into().expect("fixed header")) as usize;
        if encoded.len() != 16 + size {
            return Err(ProgramError::Header);
        }
        let program: Self = bincode::deserialize(&encoded[16..])
            .map_err(|error| ProgramError::Codec(error.to_string()))?;
        program.graph()?;
        Ok(program)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use severian_fusion::{Dimension, ElementKind, FusionNode, NodeKind, Shape};

    #[test]
    fn program_round_trips_structural_identity_without_ranked_symbols() {
        let shape = Shape::typed([Dimension::Dynamic], ElementKind::BrainFloat, 16);
        let nodes = vec![
            FusionNode::structural(0, NodeKind::Parameter, [], shape.clone()),
            FusionNode::structural(1, NodeKind::Elementwise, [NodeId(0)], shape),
        ];
        let program = TensorJitProgram {
            target: TensorJitTarget::Cpu,
            architecture: "x86_64".into(),
            nodes,
            inputs: vec![NodeId(0)],
            outputs: vec![NodeId(1)],
        };
        let encoded = program.encode().unwrap();
        assert_eq!(TensorJitProgram::decode(&encoded).unwrap(), program);
        assert!(!String::from_utf8_lossy(&encoded).contains("rank1_bf16"));
    }
}
