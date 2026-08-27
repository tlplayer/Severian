use crate::{
    AttributeId, Attrs, CompilerId, Conversion, EffectSet, FloatFormat, IntegerWidth, IrContext,
    LoweringCapability, OpId, OperationDiagnostic, OperationInterface, OperationRegistry,
    PrimitiveRepresentation, RegisteredOperation, TyId, TypeContext, TypeId,
};
use std::fmt;

pub const FROM_ELEMENTS: OpId = OpId::named("tensor", "from_elements");
pub const CONVERT: OpId = OpId::named("tensor", "convert");
pub const ADD: OpId = OpId::named("tensor", "add");
pub const SUBTRACT: OpId = OpId::named("tensor", "subtract");
pub const MULTIPLY: OpId = OpId::named("tensor", "multiply");
pub const DIVIDE: OpId = OpId::named("tensor", "divide");
pub const REDUCE_SUM: OpId = OpId::named("tensor", "reduce_sum");
pub const REDUCE_SUM_AXIS: OpId = OpId::named("tensor", "reduce_sum_axis");
pub const MATMUL: OpId = OpId::named("tensor", "matmul");
pub const TRANSPOSE: OpId = OpId::named("tensor", "transpose");
pub const SLICE: OpId = OpId::named("tensor", "slice");
pub const MATERIALIZE: OpId = OpId::named("tensor", "materialize");
pub const SHAPE: OpId = OpId::named("tensor", "shape");
pub const STRIDES: OpId = OpId::named("tensor", "strides");
pub const VALUES: OpId = OpId::named("tensor", "values");
pub const RESHAPE: OpId = OpId::named("tensor", "reshape");
pub const PERMUTE: OpId = OpId::named("tensor", "permute");
pub const MEAN_LAST: OpId = OpId::named("tensor", "mean_last");
pub const RSQRT: OpId = OpId::named("tensor", "rsqrt");
pub const EXP: OpId = OpId::named("tensor", "exp");
pub const LOG: OpId = OpId::named("tensor", "log");
pub const TANH: OpId = OpId::named("tensor", "tanh");
pub const SILU: OpId = OpId::named("tensor", "silu");
pub const SOFTMAX_LAST: OpId = OpId::named("tensor", "softmax_last");
pub const GATHER: OpId = OpId::named("tensor", "gather");
pub const CONCATENATE: OpId = OpId::named("tensor", "concatenate");
pub const REPEAT: OpId = OpId::named("tensor", "repeat");
pub const ROPE: OpId = OpId::named("tensor", "rope");
pub const RELU: OpId = OpId::named("tensor", "relu");
pub const SCALE: OpId = OpId::named("tensor", "scale");
pub const LAYER_NORM: OpId = OpId::named("tensor", "layer_norm");
pub const RELU_BACKWARD: OpId = OpId::named("tensor", "relu_backward");
pub const SOFTMAX_BACKWARD: OpId = OpId::named("tensor", "softmax_backward");
pub const LAYER_NORM_BACKWARD: OpId = OpId::named("tensor", "layer_norm_backward");
pub const BACKWARD_MSE: OpId = OpId::named("tensor", "backward_mse");
pub const GRADIENT: OpId = OpId::named("tensor", "gradient");
pub const SGD: OpId = OpId::named("tensor", "sgd");
pub const ADD_SCALAR: OpId = OpId::named("tensor", "add_scalar");
pub const ELEMENT_TYPE: AttributeId = AttributeId::from_name("tensor.element_type");
pub const TARGET_ELEMENT_TYPE: AttributeId = AttributeId::from_name("tensor.target_element_type");
pub const RESULT_SHAPE: AttributeId = AttributeId::from_name("tensor.result_shape");

pub fn compiler_id() -> CompilerId {
    CompilerId::from_path("tensor.compiler.TensorCompiler")
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum TensorElementKind {
    SignedInteger(u16),
    UnsignedInteger(u16),
    Float8E4M3Fn,
    Float8E5M2,
    IeeeFloat(u16),
    BrainFloat16,
}

impl TensorElementKind {
    pub const ALL: [Self; 17] = [
        Self::SignedInteger(8),
        Self::SignedInteger(16),
        Self::SignedInteger(32),
        Self::SignedInteger(64),
        Self::SignedInteger(128),
        Self::UnsignedInteger(8),
        Self::UnsignedInteger(16),
        Self::UnsignedInteger(32),
        Self::UnsignedInteger(64),
        Self::UnsignedInteger(128),
        Self::Float8E4M3Fn,
        Self::Float8E5M2,
        Self::IeeeFloat(16),
        Self::BrainFloat16,
        Self::IeeeFloat(32),
        Self::IeeeFloat(64),
        Self::IeeeFloat(128),
    ];

    pub fn from_type(types: &TypeContext, element: TypeId) -> Option<Self> {
        match types.primitive(element)?.representation {
            PrimitiveRepresentation::Integer {
                bits: IntegerWidth::Fixed(bits),
                signed: true,
            } if matches!(bits, 8 | 16 | 32 | 64 | 128) => Some(Self::SignedInteger(bits)),
            PrimitiveRepresentation::Integer {
                bits: IntegerWidth::Fixed(bits),
                signed: false,
            } if matches!(bits, 8 | 16 | 32 | 64 | 128) => Some(Self::UnsignedInteger(bits)),
            PrimitiveRepresentation::Float {
                format: FloatFormat::Float8E4M3Fn,
            } => Some(Self::Float8E4M3Fn),
            PrimitiveRepresentation::Float {
                format: FloatFormat::Float8E5M2,
            } => Some(Self::Float8E5M2),
            PrimitiveRepresentation::Float {
                format: FloatFormat::Ieee(bits),
            } if matches!(bits, 16 | 32 | 64 | 128) => Some(Self::IeeeFloat(bits)),
            PrimitiveRepresentation::Float {
                format: FloatFormat::BrainFloat16,
            } => Some(Self::BrainFloat16),
            _ => None,
        }
    }

    pub fn storage_tag(self) -> u8 {
        match self {
            Self::SignedInteger(8) => 0,
            Self::SignedInteger(16) => 1,
            Self::SignedInteger(32) => 2,
            Self::SignedInteger(64) => 3,
            Self::SignedInteger(128) => 4,
            Self::UnsignedInteger(8) => 5,
            Self::UnsignedInteger(16) => 6,
            Self::UnsignedInteger(32) => 7,
            Self::UnsignedInteger(64) => 8,
            Self::UnsignedInteger(128) => 9,
            Self::Float8E4M3Fn => 10,
            Self::Float8E5M2 => 11,
            Self::IeeeFloat(16) => 12,
            Self::BrainFloat16 => 13,
            Self::IeeeFloat(32) => 14,
            Self::IeeeFloat(64) => 15,
            Self::IeeeFloat(128) => 16,
            _ => unreachable!("tensor element widths are validated at construction"),
        }
    }

    pub const fn bits(self) -> u16 {
        match self {
            Self::SignedInteger(bits) | Self::UnsignedInteger(bits) | Self::IeeeFloat(bits) => bits,
            Self::Float8E4M3Fn | Self::Float8E5M2 => 8,
            Self::BrainFloat16 => 16,
        }
    }

    pub const fn byte_width(self) -> u8 {
        (self.bits() / 8) as u8
    }

    pub const fn accumulation(self) -> Self {
        match self {
            Self::SignedInteger(8 | 16 | 32) => Self::SignedInteger(64),
            Self::UnsignedInteger(8 | 16 | 32) => Self::UnsignedInteger(64),
            Self::Float8E4M3Fn | Self::Float8E5M2 | Self::IeeeFloat(16) | Self::BrainFloat16 => {
                Self::IeeeFloat(32)
            }
            other => other,
        }
    }
}

pub fn element_storage_tag(types: &TypeContext, element: TypeId) -> Option<u8> {
    TensorElementKind::from_type(types, element).map(TensorElementKind::storage_tag)
}

#[derive(Clone)]
struct TensorOperationInterface {
    id: OpId,
    operands: usize,
    results: usize,
    capabilities: Vec<LoweringCapability>,
}

impl OperationInterface for TensorOperationInterface {
    fn infer_types(
        &self,
        _operands: &[TyId],
        _attributes: &Attrs,
    ) -> Result<Vec<TyId>, OperationDiagnostic> {
        Err(OperationDiagnostic {
            operation: self.id,
            message: "tensor result types are resolved by semantic generic inference".into(),
        })
    }

    fn verify(
        &self,
        operation: &RegisteredOperation,
        _context: &IrContext<'_>,
    ) -> Result<(), OperationDiagnostic> {
        if operation.operands.len() != self.operands || operation.results.len() != self.results {
            return Err(OperationDiagnostic {
                operation: self.id,
                message: format!(
                    "tensor operation expects {} operand(s) and {} result(s)",
                    self.operands, self.results
                ),
            });
        }
        Ok(())
    }

    fn effects(&self, _operation: &RegisteredOperation) -> EffectSet {
        EffectSet::ALLOCATE
    }

    fn canonicalize(&self, _operation: &RegisteredOperation) -> Option<crate::CanonicalRewrite> {
        None
    }

    fn lowering_capabilities(&self) -> &[LoweringCapability] {
        &self.capabilities
    }
}

pub fn install_operations(registry: &mut OperationRegistry) -> Result<(), OperationDiagnostic> {
    for (id, operands) in [
        (FROM_ELEMENTS, 2),
        (CONVERT, 1),
        (ADD, 2),
        (SUBTRACT, 2),
        (MULTIPLY, 2),
        (DIVIDE, 2),
        (REDUCE_SUM, 1),
        (REDUCE_SUM_AXIS, 2),
        (MATMUL, 2),
        (TRANSPOSE, 1),
        (SLICE, 4),
        (MATERIALIZE, 1),
        (SHAPE, 1),
        (STRIDES, 1),
        (VALUES, 1),
        (RESHAPE, 2),
        (PERMUTE, 2),
        (MEAN_LAST, 1),
        (RSQRT, 1),
        (EXP, 1),
        (LOG, 1),
        (TANH, 1),
        (SILU, 1),
        (SOFTMAX_LAST, 1),
        (GATHER, 2),
        (CONCATENATE, 3),
        (REPEAT, 2),
        (ROPE, 2),
        (RELU, 1),
        (SCALE, 2),
        (LAYER_NORM, 2),
        (RELU_BACKWARD, 2),
        (SOFTMAX_BACKWARD, 2),
        (LAYER_NORM_BACKWARD, 3),
        (BACKWARD_MSE, 1),
        (GRADIENT, 1),
        (SGD, 2),
        (ADD_SCALAR, 2),
    ] {
        registry.register(
            id,
            TensorOperationInterface {
                id,
                operands,
                results: 1,
                capabilities: vec![LoweringCapability::Compiler(compiler_id())],
            },
        )?;
    }
    Ok(())
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum TensorDimension {
    Dynamic,
    Known(u64),
}

impl TensorDimension {
    fn broadcast(self, other: Self) -> Option<Self> {
        match (self, other) {
            (Self::Known(left), Self::Known(right)) if left == right => Some(self),
            (Self::Known(1), right) => Some(right),
            (left, Self::Known(1)) => Some(left),
            (Self::Dynamic, _) | (_, Self::Dynamic) => Some(Self::Dynamic),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
pub struct TensorShape(pub Vec<TensorDimension>);

impl TensorShape {
    pub fn ranked(dimensions: impl IntoIterator<Item = u64>) -> Self {
        Self(dimensions.into_iter().map(TensorDimension::Known).collect())
    }

    pub fn dynamic(rank: usize) -> Self {
        Self(vec![TensorDimension::Dynamic; rank])
    }

    pub fn rank(&self) -> usize {
        self.0.len()
    }

    pub fn element_count(&self) -> Option<u64> {
        self.0.iter().try_fold(1u64, |count, dimension| {
            let TensorDimension::Known(dimension) = dimension else {
                return None;
            };
            count.checked_mul(*dimension)
        })
    }

    pub fn broadcast(&self, other: &Self) -> Result<Self, TensorError> {
        let rank = self.rank().max(other.rank());
        let mut dimensions = Vec::with_capacity(rank);
        for offset in 0..rank {
            let left = self
                .0
                .get(self.rank().wrapping_sub(offset + 1))
                .copied()
                .unwrap_or(TensorDimension::Known(1));
            let right = other
                .0
                .get(other.rank().wrapping_sub(offset + 1))
                .copied()
                .unwrap_or(TensorDimension::Known(1));
            dimensions.push(
                left.broadcast(right).ok_or_else(|| {
                    TensorError::IncompatibleBroadcast(self.clone(), other.clone())
                })?,
            );
        }
        dimensions.reverse();
        Ok(Self(dimensions))
    }

    pub fn matmul(&self, other: &Self) -> Result<Self, TensorError> {
        if self.rank() < 2 || other.rank() < 2 {
            return Err(TensorError::MatmulRequiresRankTwo);
        }
        let left_contract = self.0[self.rank() - 1];
        let right_contract = other.0[other.rank() - 2];
        if matches!(
            (left_contract, right_contract),
            (TensorDimension::Known(left), TensorDimension::Known(right)) if left != right
        ) {
            return Err(TensorError::IncompatibleContraction(
                left_contract,
                right_contract,
            ));
        }
        let batches = TensorShape(self.0[..self.rank() - 2].to_vec())
            .broadcast(&TensorShape(other.0[..other.rank() - 2].to_vec()))?;
        let mut result = batches.0;
        result.push(self.0[self.rank() - 2]);
        result.push(other.0[other.rank() - 1]);
        Ok(Self(result))
    }

    pub fn permute(&self, axes: &[usize]) -> Result<Self, TensorError> {
        if axes.len() != self.rank() {
            return Err(TensorError::InvalidPermutation);
        }
        let mut seen = vec![false; self.rank()];
        let mut dimensions = Vec::with_capacity(self.rank());
        for axis in axes {
            if *axis >= self.rank() || std::mem::replace(&mut seen[*axis], true) {
                return Err(TensorError::InvalidPermutation);
            }
            dimensions.push(self.0[*axis]);
        }
        Ok(Self(dimensions))
    }
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct TensorType {
    pub element: TypeId,
    pub shape: TensorShape,
}

impl TensorType {
    pub fn broadcast(&self, other: &Self) -> Result<Self, TensorError> {
        if self.element != other.element {
            return Err(TensorError::ElementTypeMismatch(
                self.element,
                other.element,
            ));
        }
        Ok(Self {
            element: self.element,
            shape: self.shape.broadcast(&other.shape)?,
        })
    }

    pub fn matmul(&self, other: &Self) -> Result<Self, TensorError> {
        if self.element != other.element {
            return Err(TensorError::ElementTypeMismatch(
                self.element,
                other.element,
            ));
        }
        Ok(Self {
            element: self.element,
            shape: self.shape.matmul(&other.shape)?,
        })
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TensorConversion {
    pub source: TensorType,
    pub target: TensorType,
    pub element: Conversion,
}

impl TypeContext {
    pub fn tensor_conversion(
        &self,
        source: &TensorType,
        target_element: TypeId,
    ) -> Option<TensorConversion> {
        let element = self.numeric_conversion(source.element, target_element)?;
        Some(TensorConversion {
            source: source.clone(),
            target: TensorType {
                element: target_element,
                shape: source.shape.clone(),
            },
            element,
        })
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TensorError {
    ElementTypeMismatch(TypeId, TypeId),
    IncompatibleBroadcast(TensorShape, TensorShape),
    MatmulRequiresRankTwo,
    IncompatibleContraction(TensorDimension, TensorDimension),
    InvalidPermutation,
}

impl fmt::Display for TensorError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{self:?}")
    }
}

impl std::error::Error for TensorError {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{install_primitives, ConversionKind, TypeContextBuilder};

    fn types() -> TypeContext {
        let mut types = TypeContextBuilder::new();
        install_primitives(&mut types).unwrap();
        types.build()
    }

    #[test]
    fn broadcasting_and_batched_matmul_share_one_shape_contract() {
        assert_eq!(
            TensorShape::ranked([2, 3, 4])
                .broadcast(&TensorShape::ranked([1, 4]))
                .unwrap(),
            TensorShape::ranked([2, 3, 4])
        );
        assert_eq!(
            TensorShape::ranked([5, 2, 3])
                .matmul(&TensorShape::ranked([1, 3, 7]))
                .unwrap(),
            TensorShape::ranked([5, 2, 7])
        );
        assert!(TensorShape::ranked([2, 3])
            .broadcast(&TensorShape::ranked([4, 3]))
            .is_err());
    }

    #[test]
    fn tensor_dtype_conversion_uses_the_scalar_conversion_matrix() {
        let types = types();
        let source = TensorType {
            element: types.resolve_name("f8e4m3fn").unwrap(),
            shape: TensorShape::dynamic(3),
        };
        let promoted = types
            .tensor_conversion(&source, types.resolve_name("f32").unwrap())
            .unwrap();
        assert_eq!(promoted.element.kind, ConversionKind::Promote);
        assert_eq!(promoted.target.shape, source.shape);
        let narrowed = types
            .tensor_conversion(&promoted.target, types.resolve_name("i8").unwrap())
            .unwrap();
        assert_eq!(narrowed.element.kind, ConversionKind::Lossy);
    }

    #[test]
    fn every_required_tensor_dtype_has_storage_and_accumulation_semantics() {
        let types = types();
        let names = [
            "i8", "i16", "i32", "i64", "i128", "u8", "u16", "u32", "u64", "u128", "f8e4m3fn",
            "f8e5m2", "f16", "bf16", "f32", "f64", "f128",
        ];
        for (tag, (name, kind)) in names.into_iter().zip(TensorElementKind::ALL).enumerate() {
            let element = types.resolve_name(name).unwrap();
            assert_eq!(TensorElementKind::from_type(&types, element), Some(kind));
            assert_eq!(element_storage_tag(&types, element), Some(tag as u8));
            assert!(kind.byte_width().is_power_of_two());
            assert!(TensorElementKind::ALL.contains(&kind.accumulation()));
        }
        assert_eq!(
            TensorElementKind::Float8E4M3Fn.accumulation(),
            TensorElementKind::IeeeFloat(32)
        );
        assert_eq!(
            TensorElementKind::SignedInteger(128).accumulation(),
            TensorElementKind::SignedInteger(128)
        );
        assert_eq!(
            TensorElementKind::IeeeFloat(128).accumulation(),
            TensorElementKind::IeeeFloat(128)
        );
    }

    #[test]
    fn tensor_operation_ids_are_backend_independent() {
        assert_eq!(ADD, OpId::named("tensor", "add"));
        assert_ne!(ADD, MATMUL);
    }
}
