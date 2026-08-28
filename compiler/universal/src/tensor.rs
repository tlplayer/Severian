use crate::{
    AttributeId, Attrs, CompilerId, Conversion, EffectSet, FloatFormat, IntegerWidth, IrContext,
    LoweringCapability, OpId, OperationDiagnostic, OperationInterface, OperationRegistry,
    PrimitiveRepresentation, RegisteredOperation, TyId, TypeContext, TypeId,
};
use std::fmt;

pub const ELEMENTWISE: OpId = OpId::named("tensor", "elementwise");
pub const REDUCE: OpId = OpId::named("tensor", "reduce");
pub const MATMUL: OpId = OpId::named("tensor", "matmul");
pub const RESHAPE_VIEW: OpId = OpId::named("tensor", "reshape_view");
pub const PERMUTE: OpId = OpId::named("tensor", "permute");
pub const SLICE: OpId = OpId::named("tensor", "slice");
pub const BROADCAST: OpId = OpId::named("tensor", "broadcast");
pub const GATHER: OpId = OpId::named("tensor", "gather");
pub const SCATTER: OpId = OpId::named("tensor", "scatter");
pub const CONCATENATE: OpId = OpId::named("tensor", "concatenate");
pub const CONVERT: OpId = OpId::named("tensor", "convert");
pub const STORAGE_VIEW: OpId = OpId::named("tensor", "storage_view");
pub const OPERATION_KIND: AttributeId = AttributeId::from_name("tensor.operation_kind");
pub const ELEMENT_TYPE: AttributeId = AttributeId::from_name("tensor.element_type");
pub const TARGET_ELEMENT_TYPE: AttributeId = AttributeId::from_name("tensor.target_element_type");
pub const RESULT_SHAPE: AttributeId = AttributeId::from_name("tensor.result_shape");
pub const REDUCTION_AXES: AttributeId = AttributeId::from_name("tensor.reduction_axes");

/// The small, structural tensor IR. Public tensor-library functions select a
/// variant through `OPERATION_KIND`; adding a library algorithm does not add
/// another universal operation identifier.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum TensorOp {
    Elementwise(ElementwiseOp),
    Reduce(ReductionOp),
    Matmul,
    ReshapeView(ReshapeViewOp),
    Permute(PermuteOp),
    Slice,
    Broadcast(BroadcastOp),
    Gather,
    Scatter,
    Concatenate,
    Convert,
    StorageView(StorageViewOp),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum ElementwiseOp {
    Add,
    Subtract,
    Multiply,
    Divide,
    Exp,
    Log,
    Tanh,
    Rsqrt,
    Relu,
    Scale,
    AddScalar,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum ReductionOp {
    Sum,
    SumAxis,
    MeanLast,
    MaxLast,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum ReshapeViewOp {
    Reshape,
    Materialize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum PermuteOp {
    Axes,
    Reverse,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum BroadcastOp {
    Like,
    Repeat,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum StorageViewOp {
    FromElements,
    FromAbi,
    Shape,
    Strides,
    Values,
}

impl TensorOp {
    pub const fn id(self) -> OpId {
        match self {
            Self::Elementwise(_) => ELEMENTWISE,
            Self::Reduce(_) => REDUCE,
            Self::Matmul => MATMUL,
            Self::ReshapeView(_) => RESHAPE_VIEW,
            Self::Permute(_) => PERMUTE,
            Self::Slice => SLICE,
            Self::Broadcast(_) => BROADCAST,
            Self::Gather => GATHER,
            Self::Scatter => SCATTER,
            Self::Concatenate => CONCATENATE,
            Self::Convert => CONVERT,
            Self::StorageView(_) => STORAGE_VIEW,
        }
    }

    pub const fn kind(self) -> Option<&'static str> {
        Some(match self {
            Self::Elementwise(ElementwiseOp::Add) => "add",
            Self::Elementwise(ElementwiseOp::Subtract) => "subtract",
            Self::Elementwise(ElementwiseOp::Multiply) => "multiply",
            Self::Elementwise(ElementwiseOp::Divide) => "divide",
            Self::Elementwise(ElementwiseOp::Exp) => "exp",
            Self::Elementwise(ElementwiseOp::Log) => "log",
            Self::Elementwise(ElementwiseOp::Tanh) => "tanh",
            Self::Elementwise(ElementwiseOp::Rsqrt) => "rsqrt",
            Self::Elementwise(ElementwiseOp::Relu) => "relu",
            Self::Elementwise(ElementwiseOp::Scale) => "scale",
            Self::Elementwise(ElementwiseOp::AddScalar) => "add_scalar",
            Self::Reduce(ReductionOp::Sum) => "sum",
            Self::Reduce(ReductionOp::SumAxis) => "sum_axis",
            Self::Reduce(ReductionOp::MeanLast) => "mean_last",
            Self::Reduce(ReductionOp::MaxLast) => "max_last",
            Self::ReshapeView(ReshapeViewOp::Reshape) => "reshape",
            Self::ReshapeView(ReshapeViewOp::Materialize) => "materialize",
            Self::Permute(PermuteOp::Axes) => "axes",
            Self::Permute(PermuteOp::Reverse) => "reverse",
            Self::Broadcast(BroadcastOp::Like) => "like",
            Self::Broadcast(BroadcastOp::Repeat) => "repeat",
            Self::StorageView(StorageViewOp::FromElements) => "from_elements",
            Self::StorageView(StorageViewOp::FromAbi) => "from_abi",
            Self::StorageView(StorageViewOp::Shape) => "shape",
            Self::StorageView(StorageViewOp::Strides) => "strides",
            Self::StorageView(StorageViewOp::Values) => "values",
            Self::Matmul
            | Self::Slice
            | Self::Gather
            | Self::Scatter
            | Self::Concatenate
            | Self::Convert => return None,
        })
    }

    pub fn apply(self, attributes: &mut Attrs) -> OpId {
        if let Some(kind) = self.kind() {
            attributes.insert(OPERATION_KIND, crate::AttrValue::String(kind.into()));
        }
        self.id()
    }

    pub fn decode(id: OpId, attributes: &Attrs) -> Option<Self> {
        let kind = match attributes.get(&OPERATION_KIND) {
            Some(crate::AttrValue::String(kind)) => Some(kind.as_str()),
            _ => None,
        };
        Some(match (id, kind) {
            (ELEMENTWISE, Some("add")) => Self::Elementwise(ElementwiseOp::Add),
            (ELEMENTWISE, Some("subtract")) => Self::Elementwise(ElementwiseOp::Subtract),
            (ELEMENTWISE, Some("multiply")) => Self::Elementwise(ElementwiseOp::Multiply),
            (ELEMENTWISE, Some("divide")) => Self::Elementwise(ElementwiseOp::Divide),
            (ELEMENTWISE, Some("exp")) => Self::Elementwise(ElementwiseOp::Exp),
            (ELEMENTWISE, Some("log")) => Self::Elementwise(ElementwiseOp::Log),
            (ELEMENTWISE, Some("tanh")) => Self::Elementwise(ElementwiseOp::Tanh),
            (ELEMENTWISE, Some("rsqrt")) => Self::Elementwise(ElementwiseOp::Rsqrt),
            (ELEMENTWISE, Some("relu")) => Self::Elementwise(ElementwiseOp::Relu),
            (ELEMENTWISE, Some("scale")) => Self::Elementwise(ElementwiseOp::Scale),
            (ELEMENTWISE, Some("add_scalar")) => Self::Elementwise(ElementwiseOp::AddScalar),
            (REDUCE, Some("sum")) => Self::Reduce(ReductionOp::Sum),
            (REDUCE, Some("sum_axis")) => Self::Reduce(ReductionOp::SumAxis),
            (REDUCE, Some("mean_last")) => Self::Reduce(ReductionOp::MeanLast),
            (REDUCE, Some("max_last")) => Self::Reduce(ReductionOp::MaxLast),
            (MATMUL, None) => Self::Matmul,
            (RESHAPE_VIEW, Some("reshape")) => Self::ReshapeView(ReshapeViewOp::Reshape),
            (RESHAPE_VIEW, Some("materialize")) => Self::ReshapeView(ReshapeViewOp::Materialize),
            (PERMUTE, Some("axes")) => Self::Permute(PermuteOp::Axes),
            (PERMUTE, Some("reverse")) => Self::Permute(PermuteOp::Reverse),
            (SLICE, None) => Self::Slice,
            (BROADCAST, Some("like")) => Self::Broadcast(BroadcastOp::Like),
            (BROADCAST, Some("repeat")) => Self::Broadcast(BroadcastOp::Repeat),
            (GATHER, None) => Self::Gather,
            (SCATTER, None) => Self::Scatter,
            (CONCATENATE, None) => Self::Concatenate,
            (CONVERT, None) => Self::Convert,
            (STORAGE_VIEW, Some("from_elements")) => Self::StorageView(StorageViewOp::FromElements),
            (STORAGE_VIEW, Some("from_abi")) => Self::StorageView(StorageViewOp::FromAbi),
            (STORAGE_VIEW, Some("shape")) => Self::StorageView(StorageViewOp::Shape),
            (STORAGE_VIEW, Some("strides")) => Self::StorageView(StorageViewOp::Strides),
            (STORAGE_VIEW, Some("values")) => Self::StorageView(StorageViewOp::Values),
            _ => return None,
        })
    }
}

/// Operations whose result preserves the element type selected by the first
/// tensor operand. Shape-changing operations may refine shape independently.
pub const TYPE_PRESERVING_OPERATIONS: &[OpId] = &[
    ELEMENTWISE,
    REDUCE,
    MATMUL,
    RESHAPE_VIEW,
    PERMUTE,
    SLICE,
    BROADCAST,
    GATHER,
    SCATTER,
    CONCATENATE,
];

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
    pub fn from_type(types: &TypeContext, element: TypeId) -> Option<Self> {
        match types.primitive(element)?.representation {
            PrimitiveRepresentation::Integer {
                bits: IntegerWidth::Fixed(bits),
                signed: true,
            } => Some(Self::SignedInteger(bits)),
            PrimitiveRepresentation::Integer {
                bits: IntegerWidth::Fixed(bits),
                signed: false,
            } => Some(Self::UnsignedInteger(bits)),
            PrimitiveRepresentation::Float {
                format: FloatFormat::Float8E4M3Fn,
            } => Some(Self::Float8E4M3Fn),
            PrimitiveRepresentation::Float {
                format: FloatFormat::Float8E5M2,
            } => Some(Self::Float8E5M2),
            PrimitiveRepresentation::Float {
                format: FloatFormat::Ieee(bits),
            } => Some(Self::IeeeFloat(bits)),
            PrimitiveRepresentation::Float {
                format: FloatFormat::BrainFloat16,
            } => Some(Self::BrainFloat16),
            _ => None,
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

#[derive(Clone)]
struct TensorOperationInterface {
    id: OpId,
    operands: std::ops::RangeInclusive<usize>,
    results: usize,
    capabilities: Vec<LoweringCapability>,
}

impl OperationInterface for TensorOperationInterface {
    fn infer_types(
        &self,
        operands: &[TyId],
        _attributes: &Attrs,
    ) -> Result<Vec<TyId>, OperationDiagnostic> {
        if TYPE_PRESERVING_OPERATIONS.contains(&self.id) {
            return operands
                .first()
                .copied()
                .map(|operand| vec![operand])
                .ok_or_else(|| OperationDiagnostic {
                    operation: self.id,
                    message: "type-preserving tensor operation requires a tensor operand".into(),
                });
        }
        Err(OperationDiagnostic {
            operation: self.id,
            message: "tensor result type requires an explicit element or non-tensor result type"
                .into(),
        })
    }

    fn verify(
        &self,
        operation: &RegisteredOperation,
        _context: &IrContext<'_>,
    ) -> Result<(), OperationDiagnostic> {
        if !self.operands.contains(&operation.operands.len())
            || operation.results.len() != self.results
        {
            return Err(OperationDiagnostic {
                operation: self.id,
                message: format!(
                    "tensor operation expects {}..={} operand(s) and {} result(s)",
                    self.operands.start(),
                    self.operands.end(),
                    self.results
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
        (ELEMENTWISE, 1..=3),
        (REDUCE, 1..=2),
        (MATMUL, 2..=2),
        (RESHAPE_VIEW, 1..=2),
        (PERMUTE, 1..=2),
        (SLICE, 4..=4),
        (BROADCAST, 2..=2),
        (GATHER, 2..=2),
        (SCATTER, 3..=3),
        (CONCATENATE, 3..=3),
        (CONVERT, 1..=1),
        (STORAGE_VIEW, 1..=2),
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
pub enum TensorShape {
    /// Rank is not known until execution. This is the shape of a source-level
    /// `Tensor[T]` annotation and lowers to MLIR's unranked tensor type.
    #[default]
    Unranked,
    /// Rank is known. Individual dimensions may remain dynamic.
    Ranked(Vec<TensorDimension>),
}

impl TensorShape {
    pub fn ranked(dimensions: impl IntoIterator<Item = u64>) -> Self {
        Self::Ranked(dimensions.into_iter().map(TensorDimension::Known).collect())
    }

    pub fn dynamic(rank: usize) -> Self {
        Self::Ranked(vec![TensorDimension::Dynamic; rank])
    }

    pub fn rank(&self) -> Option<usize> {
        match self {
            Self::Unranked => None,
            Self::Ranked(dimensions) => Some(dimensions.len()),
        }
    }

    pub fn dimensions(&self) -> Option<&[TensorDimension]> {
        match self {
            Self::Unranked => None,
            Self::Ranked(dimensions) => Some(dimensions),
        }
    }

    pub fn element_count(&self) -> Option<u64> {
        self.dimensions()?
            .iter()
            .try_fold(1u64, |count, dimension| {
                let TensorDimension::Known(dimension) = dimension else {
                    return None;
                };
                count.checked_mul(*dimension)
            })
    }

    pub fn broadcast(&self, other: &Self) -> Result<Self, TensorError> {
        let (Some(left_dimensions), Some(right_dimensions)) =
            (self.dimensions(), other.dimensions())
        else {
            return Ok(Self::Unranked);
        };
        let rank = left_dimensions.len().max(right_dimensions.len());
        let mut dimensions = Vec::with_capacity(rank);
        for offset in 0..rank {
            let left = left_dimensions
                .get(left_dimensions.len().wrapping_sub(offset + 1))
                .copied()
                .unwrap_or(TensorDimension::Known(1));
            let right = right_dimensions
                .get(right_dimensions.len().wrapping_sub(offset + 1))
                .copied()
                .unwrap_or(TensorDimension::Known(1));
            dimensions.push(
                left.broadcast(right).ok_or_else(|| {
                    TensorError::IncompatibleBroadcast(self.clone(), other.clone())
                })?,
            );
        }
        dimensions.reverse();
        Ok(Self::Ranked(dimensions))
    }

    pub fn matmul(&self, other: &Self) -> Result<Self, TensorError> {
        let (Some(left), Some(right)) = (self.dimensions(), other.dimensions()) else {
            return Ok(Self::Unranked);
        };
        if left.len() < 2 || right.len() < 2 {
            return Err(TensorError::MatmulRequiresRankTwo);
        }
        let left_contract = left[left.len() - 1];
        let right_contract = right[right.len() - 2];
        if matches!(
            (left_contract, right_contract),
            (TensorDimension::Known(left), TensorDimension::Known(right)) if left != right
        ) {
            return Err(TensorError::IncompatibleContraction(
                left_contract,
                right_contract,
            ));
        }
        let batches = TensorShape::Ranked(left[..left.len() - 2].to_vec())
            .broadcast(&TensorShape::Ranked(right[..right.len() - 2].to_vec()))?;
        let Self::Ranked(mut result) = batches else {
            return Ok(Self::Unranked);
        };
        result.push(left[left.len() - 2]);
        result.push(right[right.len() - 1]);
        Ok(Self::Ranked(result))
    }

    pub fn permute(&self, axes: &[usize]) -> Result<Self, TensorError> {
        let Some(source) = self.dimensions() else {
            return Ok(Self::Unranked);
        };
        if axes.len() != source.len() {
            return Err(TensorError::InvalidPermutation);
        }
        let mut seen = vec![false; source.len()];
        let mut dimensions = Vec::with_capacity(source.len());
        for axis in axes {
            if *axis >= source.len() || std::mem::replace(&mut seen[*axis], true) {
                return Err(TensorError::InvalidPermutation);
            }
            dimensions.push(source[*axis]);
        }
        Ok(Self::Ranked(dimensions))
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
    fn every_required_tensor_dtype_has_representation_and_accumulation_semantics() {
        let types = types();
        let kinds = [
            ("i8", TensorElementKind::SignedInteger(8)),
            ("i16", TensorElementKind::SignedInteger(16)),
            ("i32", TensorElementKind::SignedInteger(32)),
            ("i64", TensorElementKind::SignedInteger(64)),
            ("i128", TensorElementKind::SignedInteger(128)),
            ("u8", TensorElementKind::UnsignedInteger(8)),
            ("u16", TensorElementKind::UnsignedInteger(16)),
            ("u32", TensorElementKind::UnsignedInteger(32)),
            ("u64", TensorElementKind::UnsignedInteger(64)),
            ("u128", TensorElementKind::UnsignedInteger(128)),
            ("f8e4m3fn", TensorElementKind::Float8E4M3Fn),
            ("f8e5m2", TensorElementKind::Float8E5M2),
            ("f16", TensorElementKind::IeeeFloat(16)),
            ("bf16", TensorElementKind::BrainFloat16),
            ("f32", TensorElementKind::IeeeFloat(32)),
            ("f64", TensorElementKind::IeeeFloat(64)),
            ("f128", TensorElementKind::IeeeFloat(128)),
        ];
        for (name, kind) in kinds {
            let element = types.resolve_name(name).unwrap();
            assert_eq!(TensorElementKind::from_type(&types, element), Some(kind));
            assert!(kind.byte_width().is_power_of_two());
        }
        let f80 = types.resolve_name("f80").unwrap();
        assert_eq!(
            TensorElementKind::from_type(&types, f80),
            Some(TensorElementKind::IeeeFloat(80))
        );
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
    fn tensor_operation_ids_are_small_and_backend_independent() {
        let operations = [
            ELEMENTWISE,
            REDUCE,
            MATMUL,
            RESHAPE_VIEW,
            PERMUTE,
            SLICE,
            BROADCAST,
            GATHER,
            SCATTER,
            CONCATENATE,
            CONVERT,
            STORAGE_VIEW,
        ];
        assert_eq!(operations.len(), 12);
        assert_eq!(ELEMENTWISE, OpId::named("tensor", "elementwise"));
        assert_eq!(MATMUL, TensorOp::Matmul.id());
        let mut attributes = Attrs::new();
        let id = TensorOp::Elementwise(ElementwiseOp::Add).apply(&mut attributes);
        assert_eq!(id, ELEMENTWISE);
        assert_eq!(
            TensorOp::decode(id, &attributes),
            Some(TensorOp::Elementwise(ElementwiseOp::Add))
        );
    }

    #[test]
    fn every_tensor_operation_preserves_each_structural_element_type_generically() {
        let mut builder = TypeContextBuilder::new();
        crate::install_primitives(&mut builder).unwrap();
        let mut types = builder.build();
        let constructor = types
            .register_source_declaration("tensor.Tensor", "Tensor", 1)
            .unwrap();
        types.mark_tensor_constructor(constructor).unwrap();
        let mut registry = OperationRegistry::default();
        install_operations(&mut registry).unwrap();
        let element_names = [
            "i8", "i16", "i32", "i64", "i128", "u8", "u16", "u32", "u64", "u128", "f16", "f32",
            "f64", "f80", "f128",
        ];
        for name in element_names {
            let element = types.resolve_name(name).unwrap();
            let tensor = types
                .instantiate_tensor(constructor, element, TensorShape::ranked([2, 2]))
                .unwrap();
            for operation in TYPE_PRESERVING_OPERATIONS {
                let result = registry
                    .interface(*operation)
                    .unwrap()
                    .infer_types(&[tensor], &Attrs::new())
                    .unwrap();
                assert_eq!(result, [tensor], "{operation:?} erased Tensor[{name}]");
            }
        }
    }
}
