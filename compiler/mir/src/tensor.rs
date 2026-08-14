use crate::ValueRef;
use severian_hir::{TensorElementType, TensorType};

mod resolve;

pub(crate) use resolve::{resolve_tensor_op, tensor_operands};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TensorOpId(pub u32);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TensorOperand {
    pub value: ValueRef,
    pub ty: TensorType,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ElementwiseKind {
    Add,
    Subtract,
    Multiply,
    Divide,
    Relu,
    Silu,
    Exp,
    Tanh,
    Rsqrt,
    Sigmoid,
    Gelu,
    Cosine,
    Sine,
    Where,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReductionKind {
    Sum,
    Mean,
    Maximum,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScalarKind {
    Add,
    Multiply,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScalarValue {
    Literal(u64),
    Operand(ValueRef),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NormalizationKind {
    Softmax,
    LayerNorm,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ElementwiseOp {
    pub kind: ElementwiseKind,
    pub inputs: Vec<TensorOperand>,
    pub result: TensorType,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MatmulOp {
    pub left: TensorOperand,
    pub right: TensorOperand,
    pub result: TensorType,
    pub accumulation: TensorElementType,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReductionOp {
    pub kind: ReductionKind,
    pub input: TensorOperand,
    pub axes: Vec<u64>,
    pub axes_known: bool,
    pub last_axis: bool,
    pub result: TensorType,
    pub accumulation: TensorElementType,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ReshapeOp {
    pub input: TensorOperand,
    pub result: TensorType,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TransposeOp {
    pub input: TensorOperand,
    pub permutation: Vec<u64>,
    pub permutation_known: bool,
    pub result: TensorType,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BroadcastOp {
    pub input: TensorOperand,
    pub dimensions: Vec<u64>,
    pub dimensions_known: bool,
    pub result: TensorType,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ScalarOp {
    pub kind: ScalarKind,
    pub input: TensorOperand,
    pub value: ScalarValue,
    pub result: TensorType,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NormalizationOp {
    pub kind: NormalizationKind,
    pub input: TensorOperand,
    /// Negative axes remain relative when the source tensor rank is dynamic.
    /// Backends that require a concrete axis resolve them once rank is known.
    pub axis: i64,
    pub epsilon: Option<ScalarValue>,
    pub result: TensorType,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GatherOp {
    pub table: TensorOperand,
    pub indices: TensorOperand,
    pub result: TensorType,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ConvertOp {
    pub input: TensorOperand,
    pub result: TensorType,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SliceIndex {
    Static(i64),
    Dynamic(ValueRef),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SliceOp {
    pub input: TensorOperand,
    pub starts: Vec<SliceIndex>,
    pub limits: Vec<SliceIndex>,
    pub strides: Vec<SliceIndex>,
    pub result: TensorType,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DynamicSliceOp {
    pub input: TensorOperand,
    pub starts: Vec<SliceIndex>,
    pub sizes: Vec<SliceIndex>,
    pub result: TensorType,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DynamicUpdateSliceOp {
    pub input: TensorOperand,
    pub update: TensorOperand,
    pub starts: Vec<SliceIndex>,
    pub dynamic_index: Option<TensorOperand>,
    pub axis: Option<u64>,
    pub result: TensorType,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ConcatenateOp {
    pub inputs: Vec<TensorOperand>,
    pub axis: u64,
    pub result: TensorType,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TensorOp {
    Elementwise(ElementwiseOp),
    Matmul(MatmulOp),
    Reduction(ReductionOp),
    Reshape(ReshapeOp),
    Transpose(TransposeOp),
    Broadcast(BroadcastOp),
    Scalar(ScalarOp),
    Normalization(NormalizationOp),
    Gather(GatherOp),
    Convert(ConvertOp),
    Slice(SliceOp),
    DynamicSlice(DynamicSliceOp),
    DynamicUpdateSlice(DynamicUpdateSliceOp),
    Concatenate(ConcatenateOp),
}

impl TensorOp {
    pub const fn name(&self) -> &'static str {
        match self {
            Self::Elementwise(op) => match op.kind {
                ElementwiseKind::Add => "elementwise.add",
                ElementwiseKind::Subtract => "elementwise.subtract",
                ElementwiseKind::Multiply => "elementwise.multiply",
                ElementwiseKind::Divide => "elementwise.divide",
                ElementwiseKind::Relu => "elementwise.relu",
                ElementwiseKind::Silu => "elementwise.silu",
                ElementwiseKind::Exp => "elementwise.exp",
                ElementwiseKind::Tanh => "elementwise.tanh",
                ElementwiseKind::Rsqrt => "elementwise.rsqrt",
                ElementwiseKind::Sigmoid => "elementwise.sigmoid",
                ElementwiseKind::Gelu => "elementwise.gelu",
                ElementwiseKind::Cosine => "elementwise.cosine",
                ElementwiseKind::Sine => "elementwise.sine",
                ElementwiseKind::Where => "elementwise.where",
            },
            Self::Matmul(_) => "matmul",
            Self::Reduction(op) => match op.kind {
                ReductionKind::Sum => "reduction.sum",
                ReductionKind::Mean => "reduction.mean",
                ReductionKind::Maximum => "reduction.maximum",
            },
            Self::Reshape(_) => "reshape",
            Self::Transpose(_) => "transpose",
            Self::Broadcast(_) => "broadcast",
            Self::Scalar(op) => match op.kind {
                ScalarKind::Add => "scalar.add",
                ScalarKind::Multiply => "scalar.multiply",
            },
            Self::Normalization(op) => match op.kind {
                NormalizationKind::Softmax => "normalization.softmax",
                NormalizationKind::LayerNorm => "normalization.layer_norm",
            },
            Self::Gather(_) => "gather",
            Self::Convert(_) => "convert",
            Self::Slice(_) => "slice",
            Self::DynamicSlice(_) => "dynamic_slice",
            Self::DynamicUpdateSlice(_) => "dynamic_update_slice",
            Self::Concatenate(_) => "concatenate",
        }
    }

    pub const fn result(&self) -> TensorType {
        match self {
            Self::Elementwise(op) => op.result,
            Self::Matmul(op) => op.result,
            Self::Reduction(op) => op.result,
            Self::Reshape(op) => op.result,
            Self::Transpose(op) => op.result,
            Self::Broadcast(op) => op.result,
            Self::Scalar(op) => op.result,
            Self::Normalization(op) => op.result,
            Self::Gather(op) => op.result,
            Self::Convert(op) => op.result,
            Self::Slice(op) => op.result,
            Self::DynamicSlice(op) => op.result,
            Self::DynamicUpdateSlice(op) => op.result,
            Self::Concatenate(op) => op.result,
        }
    }

    pub fn inputs(&self) -> Vec<TensorOperand> {
        match self {
            Self::Elementwise(op) => op.inputs.clone(),
            Self::Matmul(op) => vec![op.left, op.right],
            Self::Reduction(op) => vec![op.input],
            Self::Reshape(op) => vec![op.input],
            Self::Transpose(op) => vec![op.input],
            Self::Broadcast(op) => vec![op.input],
            Self::Scalar(op) => vec![op.input],
            Self::Normalization(op) => vec![op.input],
            Self::Gather(op) => vec![op.table, op.indices],
            Self::Convert(op) => vec![op.input],
            Self::Slice(op) => vec![op.input],
            Self::DynamicSlice(op) => vec![op.input],
            Self::DynamicUpdateSlice(op) => {
                let mut inputs = vec![op.input, op.update];
                inputs.extend(op.dynamic_index);
                inputs
            }
            Self::Concatenate(op) => op.inputs.clone(),
        }
    }
}
