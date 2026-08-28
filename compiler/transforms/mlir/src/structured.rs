use crate::{
    type_spelling, LoweredTensorDimension, LoweredTensorElement, LoweredTensorShape, LoweredType,
    MlirError,
};
use std::collections::BTreeMap;
use std::fmt;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StructuredError(pub String);

impl fmt::Display for StructuredError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.0)
    }
}

impl std::error::Error for StructuredError {}

impl From<MlirError> for StructuredError {
    fn from(error: MlirError) -> Self {
        Self(error.to_string())
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum ValueType {
    Lowered(LoweredType),
    Index,
    ShapeTensor(usize),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Value {
    name: String,
    ty: ValueType,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SliceComponent {
    Static(i64),
    Dynamic(Value),
}

impl Value {
    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn lowered_type(&self) -> Option<&LoweredType> {
        match &self.ty {
            ValueType::Lowered(ty) => Some(ty),
            ValueType::Index | ValueType::ShapeTensor(_) => None,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AffineExpression {
    Dimension(usize),
    Constant(i64),
    FloorDiv { dimension: usize, divisor: u64 },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AffineMap {
    domain_rank: usize,
    results: Vec<AffineExpression>,
}

impl AffineMap {
    pub fn new(
        domain_rank: usize,
        results: Vec<AffineExpression>,
    ) -> Result<Self, StructuredError> {
        if results.iter().any(|expression| match expression {
            AffineExpression::Dimension(axis) => *axis >= domain_rank,
            AffineExpression::FloorDiv { dimension, divisor } => {
                *dimension >= domain_rank || *divisor == 0
            }
            AffineExpression::Constant(_) => false,
        }) {
            return Err(StructuredError(
                "affine map references an axis outside its domain".into(),
            ));
        }
        Ok(Self {
            domain_rank,
            results,
        })
    }

    pub fn identity(rank: usize) -> Self {
        Self {
            domain_rank: rank,
            results: (0..rank).map(AffineExpression::Dimension).collect(),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IteratorKind {
    Parallel,
    Reduction,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScalarBinaryOperation {
    AddFloat,
    AddInteger,
    SubtractFloat,
    SubtractInteger,
    MultiplyFloat,
    MultiplyInteger,
    DivideFloat,
    DivideSigned,
    DivideUnsigned,
    MaximumFloat,
    MaximumSigned,
    MaximumUnsigned,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScalarUnaryOperation {
    Exp,
    Log,
    Tanh,
    Rsqrt,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ScalarOperation {
    FloatConvert {
        result: String,
        operand: String,
        source: LoweredType,
        target: LoweredType,
    },
    Unary {
        result: String,
        operation: ScalarUnaryOperation,
        operand: String,
        ty: LoweredType,
    },
    Binary {
        result: String,
        operation: ScalarBinaryOperation,
        left: String,
        right: String,
        ty: LoweredType,
    },
    Yield {
        value: String,
        ty: LoweredType,
    },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GenericBody {
    arguments: Vec<(String, LoweredType)>,
    captures: Vec<Value>,
    operations: Vec<ScalarOperation>,
}

impl GenericBody {
    pub fn new(
        arguments: Vec<(String, LoweredType)>,
        operations: Vec<ScalarOperation>,
    ) -> Result<Self, StructuredError> {
        Self::with_captures(arguments, Vec::new(), operations)
    }

    pub fn with_captures(
        arguments: Vec<(String, LoweredType)>,
        captures: Vec<Value>,
        operations: Vec<ScalarOperation>,
    ) -> Result<Self, StructuredError> {
        let mut values = arguments.iter().cloned().collect::<BTreeMap<_, _>>();
        for capture in &captures {
            let ty = capture
                .lowered_type()
                .cloned()
                .ok_or_else(|| StructuredError("generic body cannot capture an index".into()))?;
            if values.insert(capture.name.clone(), ty).is_some() {
                return Err(StructuredError(format!(
                    "generic body capture `%{}` shadows a block argument",
                    capture.name
                )));
            }
        }
        let mut yielded = false;
        for (index, operation) in operations.iter().enumerate() {
            match operation {
                ScalarOperation::FloatConvert {
                    result,
                    operand,
                    source,
                    target,
                } => {
                    if yielded
                        || values.get(operand) != Some(source)
                        || !matches!(source, LoweredType::Float { .. })
                        || !matches!(target, LoweredType::Float { .. })
                    {
                        return Err(StructuredError(
                            "scalar floating conversion has inconsistent operands".into(),
                        ));
                    }
                    if values.insert(result.clone(), target.clone()).is_some() {
                        return Err(StructuredError(format!(
                            "scalar SSA value `%{result}` is defined twice"
                        )));
                    }
                }
                ScalarOperation::Unary {
                    result,
                    operand,
                    ty,
                    ..
                } => {
                    if yielded
                        || values.get(operand) != Some(ty)
                        || !matches!(ty, LoweredType::Float { .. })
                    {
                        return Err(StructuredError(
                            "scalar unary operation has an inconsistent floating operand".into(),
                        ));
                    }
                    if values.insert(result.clone(), ty.clone()).is_some() {
                        return Err(StructuredError(format!(
                            "scalar SSA value `%{result}` is defined twice"
                        )));
                    }
                }
                ScalarOperation::Binary {
                    result,
                    operation,
                    left,
                    right,
                    ty,
                } => {
                    if yielded || values.get(left) != Some(ty) || values.get(right) != Some(ty) {
                        return Err(StructuredError(
                            "scalar binary operation has inconsistent SSA operands".into(),
                        ));
                    }
                    if !scalar_binary_accepts(*operation, ty) {
                        return Err(StructuredError(
                            "scalar arithmetic operation is incompatible with its element type"
                                .into(),
                        ));
                    }
                    if values.insert(result.clone(), ty.clone()).is_some() {
                        return Err(StructuredError(format!(
                            "scalar SSA value `%{result}` is defined twice"
                        )));
                    }
                }
                ScalarOperation::Yield { value, ty } => {
                    if index + 1 != operations.len() || values.get(value) != Some(ty) {
                        return Err(StructuredError(
                            "linalg.yield must terminate the body with a defined value".into(),
                        ));
                    }
                    yielded = true;
                }
            }
        }
        if !yielded {
            return Err(StructuredError("linalg.generic body has no yield".into()));
        }
        Ok(Self {
            arguments,
            captures,
            operations,
        })
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum Operation {
    IndexConstant {
        result: Value,
        value: usize,
    },
    ScalarConstant {
        result: Value,
        literal: String,
    },
    FloatConvert {
        result: Value,
        source: Value,
        extend: bool,
    },
    TensorDim {
        result: Value,
        tensor: Value,
        axis: Value,
    },
    IndexCast {
        result: Value,
        source: Value,
    },
    IntegerToIndex {
        result: Value,
        source: Value,
    },
    IndexSubtract {
        result: Value,
        left: Value,
        right: Value,
    },
    IndexMultiply {
        result: Value,
        left: Value,
        right: Value,
    },
    IndexCeilDivideSigned {
        result: Value,
        left: Value,
        right: Value,
    },
    UnsignedToFloat {
        result: Value,
        source: Value,
    },
    TensorEmpty {
        result: Value,
        dynamic_sizes: Vec<Value>,
    },
    TensorCast {
        result: Value,
        source: Value,
    },
    ShapeTensorFromElements {
        result: Value,
        dimensions: Vec<Value>,
    },
    TensorReshape {
        result: Value,
        source: Value,
        shape: Value,
    },
    TensorExtractSlice {
        result: Value,
        source: Value,
        offsets: Vec<SliceComponent>,
        sizes: Vec<SliceComponent>,
        strides: Vec<SliceComponent>,
    },
    TensorConcat {
        result: Value,
        inputs: Vec<Value>,
        axis: usize,
    },
    TensorGather {
        result: Value,
        source: Value,
        indices: Value,
        dynamic_sizes: Vec<Value>,
        index_rank: usize,
    },
    TensorScatter {
        result: Value,
        source: Value,
        indices: Value,
        updates: Value,
        dynamic_sizes: Vec<Value>,
    },
    LinalgFill {
        result: Value,
        scalar: Value,
        output: Value,
    },
    LinalgGeneric {
        result: Value,
        inputs: Vec<Value>,
        output: Value,
        maps: Vec<AffineMap>,
        iterators: Vec<IteratorKind>,
        body: GenericBody,
    },
    Call {
        results: Vec<Value>,
        symbol: String,
        arguments: Vec<Value>,
    },
    Return(Vec<Value>),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Function {
    name: String,
    private: bool,
    parameters: Vec<Value>,
    results: Vec<LoweredType>,
    operations: Vec<Operation>,
}

#[derive(Debug, Clone)]
pub struct FunctionBuilder {
    function: Function,
    values: BTreeMap<String, ValueType>,
    terminated: bool,
}

impl FunctionBuilder {
    pub fn new(
        name: impl Into<String>,
        private: bool,
        parameters: Vec<(String, LoweredType)>,
        results: Vec<LoweredType>,
    ) -> Result<Self, StructuredError> {
        let name = name.into();
        legal_identifier(&name)?;
        let mut values = BTreeMap::new();
        let parameters = parameters
            .into_iter()
            .map(|(name, ty)| {
                legal_identifier(&name)?;
                let value = Value {
                    name: name.clone(),
                    ty: ValueType::Lowered(ty),
                };
                if values.insert(name.clone(), value.ty.clone()).is_some() {
                    return Err(StructuredError(format!(
                        "function parameter `%{name}` is defined twice"
                    )));
                }
                Ok(value)
            })
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Self {
            function: Function {
                name,
                private,
                parameters,
                results,
                operations: Vec::new(),
            },
            values,
            terminated: false,
        })
    }

    pub fn parameter(&self, index: usize) -> Result<Value, StructuredError> {
        self.function
            .parameters
            .get(index)
            .cloned()
            .ok_or_else(|| StructuredError(format!("function has no parameter at index {index}")))
    }

    pub fn index_constant(
        &mut self,
        name: impl Into<String>,
        value: usize,
    ) -> Result<Value, StructuredError> {
        let result = self.define(name, ValueType::Index)?;
        self.push(Operation::IndexConstant {
            result: result.clone(),
            value,
        })?;
        Ok(result)
    }

    pub fn scalar_constant(
        &mut self,
        name: impl Into<String>,
        literal: impl Into<String>,
        ty: LoweredType,
    ) -> Result<Value, StructuredError> {
        ensure_scalar(&ty)?;
        let result = self.define(name, ValueType::Lowered(ty))?;
        self.push(Operation::ScalarConstant {
            result: result.clone(),
            literal: literal.into(),
        })?;
        Ok(result)
    }

    pub fn float_convert(
        &mut self,
        name: impl Into<String>,
        source: &Value,
        target: LoweredType,
    ) -> Result<Value, StructuredError> {
        let Some(LoweredType::Float { format: source_format }) = source.lowered_type() else {
            return Err(StructuredError(
                "floating conversion requires a floating source".into(),
            ));
        };
        let LoweredType::Float { format: target_format } = &target else {
            return Err(StructuredError(
                "floating conversion requires a floating target".into(),
            ));
        };
        let source_bits = lowered_float_bits(*source_format);
        let target_bits = lowered_float_bits(*target_format);
        if source_bits == target_bits {
            return Err(StructuredError(
                "same-width floating formats require a structural bit conversion".into(),
            ));
        }
        let result = self.define(name, ValueType::Lowered(target))?;
        self.push(Operation::FloatConvert {
            result: result.clone(),
            source: source.clone(),
            extend: target_bits > source_bits,
        })?;
        Ok(result)
    }

    pub fn tensor_dim(
        &mut self,
        name: impl Into<String>,
        tensor: &Value,
        axis: &Value,
    ) -> Result<Value, StructuredError> {
        let rank = tensor_rank(tensor)?;
        if axis.ty != ValueType::Index {
            return Err(StructuredError(
                "tensor.dim axis must have index type".into(),
            ));
        }
        let result = self.define(name, ValueType::Index)?;
        let _ = rank;
        self.push(Operation::TensorDim {
            result: result.clone(),
            tensor: tensor.clone(),
            axis: axis.clone(),
        })?;
        Ok(result)
    }

    pub fn index_cast(
        &mut self,
        name: impl Into<String>,
        source: &Value,
        target: LoweredType,
    ) -> Result<Value, StructuredError> {
        if source.ty != ValueType::Index || !matches!(target, LoweredType::Integer { .. }) {
            return Err(StructuredError(
                "arith.index_cast requires an index and integer target".into(),
            ));
        }
        let result = self.define(name, ValueType::Lowered(target))?;
        self.push(Operation::IndexCast {
            result: result.clone(),
            source: source.clone(),
        })?;
        Ok(result)
    }

    pub fn integer_to_index(
        &mut self,
        name: impl Into<String>,
        source: &Value,
    ) -> Result<Value, StructuredError> {
        if !matches!(source.lowered_type(), Some(LoweredType::Integer { .. })) {
            return Err(StructuredError(
                "arith.index_cast requires an integer source".into(),
            ));
        }
        let result = self.define(name, ValueType::Index)?;
        self.push(Operation::IntegerToIndex {
            result: result.clone(),
            source: source.clone(),
        })?;
        Ok(result)
    }

    pub fn index_subtract(
        &mut self,
        name: impl Into<String>,
        left: &Value,
        right: &Value,
    ) -> Result<Value, StructuredError> {
        self.index_binary(name, left, right, false)
    }

    pub fn index_multiply(
        &mut self,
        name: impl Into<String>,
        left: &Value,
        right: &Value,
    ) -> Result<Value, StructuredError> {
        if left.ty != ValueType::Index || right.ty != ValueType::Index {
            return Err(StructuredError(
                "structured index arithmetic requires index operands".into(),
            ));
        }
        let result = self.define(name, ValueType::Index)?;
        self.push(Operation::IndexMultiply {
            result: result.clone(),
            left: left.clone(),
            right: right.clone(),
        })?;
        Ok(result)
    }

    pub fn index_ceil_divide_signed(
        &mut self,
        name: impl Into<String>,
        left: &Value,
        right: &Value,
    ) -> Result<Value, StructuredError> {
        self.index_binary(name, left, right, true)
    }

    fn index_binary(
        &mut self,
        name: impl Into<String>,
        left: &Value,
        right: &Value,
        divide: bool,
    ) -> Result<Value, StructuredError> {
        if left.ty != ValueType::Index || right.ty != ValueType::Index {
            return Err(StructuredError(
                "structured index arithmetic requires index operands".into(),
            ));
        }
        let result = self.define(name, ValueType::Index)?;
        self.push(if divide {
            Operation::IndexCeilDivideSigned {
                result: result.clone(),
                left: left.clone(),
                right: right.clone(),
            }
        } else {
            Operation::IndexSubtract {
                result: result.clone(),
                left: left.clone(),
                right: right.clone(),
            }
        })?;
        Ok(result)
    }

    pub fn unsigned_to_float(
        &mut self,
        name: impl Into<String>,
        source: &Value,
        target: LoweredType,
    ) -> Result<Value, StructuredError> {
        if !matches!(source.lowered_type(), Some(LoweredType::Integer { .. }))
            || !matches!(target, LoweredType::Float { .. })
        {
            return Err(StructuredError(
                "arith.uitofp requires an integer and floating target".into(),
            ));
        }
        let result = self.define(name, ValueType::Lowered(target))?;
        self.push(Operation::UnsignedToFloat {
            result: result.clone(),
            source: source.clone(),
        })?;
        Ok(result)
    }

    pub fn tensor_empty(
        &mut self,
        name: impl Into<String>,
        ty: LoweredType,
        dynamic_sizes: Vec<Value>,
    ) -> Result<Value, StructuredError> {
        let dynamic_count = match &ty {
            LoweredType::Tensor {
                shape: LoweredTensorShape::Ranked(dimensions),
                ..
            } => dimensions
                .iter()
                .filter(|dimension| **dimension == LoweredTensorDimension::Dynamic)
                .count(),
            _ => {
                return Err(StructuredError(
                    "tensor.empty requires a ranked tensor type".into(),
                ))
            }
        };
        if dynamic_sizes.len() != dynamic_count
            || dynamic_sizes
                .iter()
                .any(|value| value.ty != ValueType::Index)
        {
            return Err(StructuredError(
                "tensor.empty dynamic operands do not match its tensor type".into(),
            ));
        }
        let result = self.define(name, ValueType::Lowered(ty))?;
        self.push(Operation::TensorEmpty {
            result: result.clone(),
            dynamic_sizes,
        })?;
        Ok(result)
    }

    pub fn tensor_cast(
        &mut self,
        name: impl Into<String>,
        source: &Value,
        target: LoweredType,
    ) -> Result<Value, StructuredError> {
        let Some(LoweredType::Tensor {
            element: source_element,
            shape: source_shape,
        }) = source.lowered_type()
        else {
            return Err(StructuredError("tensor.cast source is not a tensor".into()));
        };
        let LoweredType::Tensor {
            element: target_element,
            shape: target_shape,
        } = &target
        else {
            return Err(StructuredError("tensor.cast target is not a tensor".into()));
        };
        if source_element != target_element
            || !tensor_cast_shapes_compatible(source_shape, target_shape)
        {
            return Err(StructuredError(
                "tensor.cast source and target contracts are incompatible".into(),
            ));
        }
        let result = self.define(name, ValueType::Lowered(target))?;
        self.push(Operation::TensorCast {
            result: result.clone(),
            source: source.clone(),
        })?;
        Ok(result)
    }

    pub fn shape_tensor_from_elements(
        &mut self,
        name: impl Into<String>,
        dimensions: Vec<Value>,
    ) -> Result<Value, StructuredError> {
        if dimensions.iter().any(|value| value.ty != ValueType::Index) {
            return Err(StructuredError(
                "shape tensor elements must have index type".into(),
            ));
        }
        let result = self.define(name, ValueType::ShapeTensor(dimensions.len()))?;
        self.push(Operation::ShapeTensorFromElements {
            result: result.clone(),
            dimensions,
        })?;
        Ok(result)
    }

    pub fn tensor_reshape(
        &mut self,
        name: impl Into<String>,
        source: &Value,
        shape: &Value,
        target: LoweredType,
    ) -> Result<Value, StructuredError> {
        let ValueType::ShapeTensor(result_rank) = &shape.ty else {
            return Err(StructuredError(
                "tensor.reshape shape must be a rank-one index tensor".into(),
            ));
        };
        if tensor_rank_from_type(&target) != Some(*result_rank) {
            return Err(StructuredError(
                "tensor.reshape shape length must equal the result rank".into(),
            ));
        }
        let source_element = tensor_scalar_type(
            source
                .lowered_type()
                .ok_or_else(|| StructuredError("tensor.reshape source is not lowered".into()))?,
        )?;
        if tensor_scalar_type(&target).ok().as_ref() != Some(&source_element) {
            return Err(StructuredError(
                "tensor.reshape must preserve its element type".into(),
            ));
        }
        let result = self.define(name, ValueType::Lowered(target))?;
        self.push(Operation::TensorReshape {
            result: result.clone(),
            source: source.clone(),
            shape: shape.clone(),
        })?;
        Ok(result)
    }

    pub fn tensor_extract_slice(
        &mut self,
        name: impl Into<String>,
        source: &Value,
        target: LoweredType,
        offsets: Vec<SliceComponent>,
        sizes: Vec<SliceComponent>,
        strides: Vec<SliceComponent>,
    ) -> Result<Value, StructuredError> {
        let rank = tensor_rank(source)?;
        if offsets.len() != rank || sizes.len() != rank || strides.len() != rank {
            return Err(StructuredError(
                "tensor.extract_slice parameters must match source rank".into(),
            ));
        }
        if strides
            .iter()
            .any(|stride| matches!(stride, SliceComponent::Static(value) if *value <= 0))
            || sizes
                .iter()
                .any(|size| matches!(size, SliceComponent::Static(value) if *value < 0))
        {
            return Err(StructuredError(
                "tensor.extract_slice requires non-negative sizes and positive strides".into(),
            ));
        }
        let target_rank = match &target {
            LoweredType::Tensor {
                shape: LoweredTensorShape::Ranked(dimensions),
                ..
            } => dimensions.len(),
            _ => {
                return Err(StructuredError(
                    "tensor.extract_slice target must be a ranked tensor".into(),
                ))
            }
        };
        if target_rank != rank {
            return Err(StructuredError(
                "tensor.extract_slice must preserve rank".into(),
            ));
        }
        let result = self.define(name, ValueType::Lowered(target))?;
        for component in offsets.iter().chain(&sizes).chain(&strides) {
            if let SliceComponent::Dynamic(value) = component {
                if value.ty != ValueType::Index
                    || self.values.get(&value.name) != Some(&ValueType::Index)
                {
                    return Err(StructuredError(
                        "dynamic tensor.extract_slice components must be defined index values"
                            .into(),
                    ));
                }
            }
        }
        self.push(Operation::TensorExtractSlice {
            result: result.clone(),
            source: source.clone(),
            offsets,
            sizes,
            strides,
        })?;
        Ok(result)
    }

    pub fn tensor_concat(
        &mut self,
        name: impl Into<String>,
        inputs: Vec<Value>,
        axis: usize,
        target: LoweredType,
    ) -> Result<Value, StructuredError> {
        let Some(first) = inputs.first() else {
            return Err(StructuredError(
                "tensor.concat requires at least one input".into(),
            ));
        };
        let rank = tensor_rank(first)?;
        if axis >= rank || inputs.iter().any(|input| tensor_rank(input) != Ok(rank)) {
            return Err(StructuredError(
                "tensor.concat axis and operand ranks are inconsistent".into(),
            ));
        }
        let element = tensor_scalar_type(
            first
                .lowered_type()
                .expect("tensor rank validation established a lowered type"),
        )?;
        if inputs.iter().any(|input| {
            input
                .lowered_type()
                .is_none_or(|ty| tensor_scalar_type(ty).ok().as_ref() != Some(&element))
        }) || tensor_rank_from_type(&target) != Some(rank)
            || tensor_scalar_type(&target).ok().as_ref() != Some(&element)
        {
            return Err(StructuredError(
                "tensor.concat operands and result must share rank and element type".into(),
            ));
        }
        let result = self.define(name, ValueType::Lowered(target))?;
        self.push(Operation::TensorConcat {
            result: result.clone(),
            inputs,
            axis,
        })?;
        Ok(result)
    }

    pub fn tensor_gather(
        &mut self,
        name: impl Into<String>,
        source: &Value,
        indices: &Value,
        dynamic_sizes: Vec<Value>,
        target: LoweredType,
    ) -> Result<Value, StructuredError> {
        let source_rank = tensor_rank(source)?;
        let index_rank = tensor_rank(indices)?;
        let target_rank = tensor_rank_from_type(&target).ok_or_else(|| {
            StructuredError("tensor.gather target must have known rank".into())
        })?;
        if source_rank == 0 || target_rank != index_rank + source_rank - 1 {
            return Err(StructuredError(
                "tensor.gather result rank must be index rank plus source rank minus one".into(),
            ));
        }
        if dynamic_sizes.iter().any(|value| value.ty != ValueType::Index) {
            return Err(StructuredError(
                "tensor.generate dynamic sizes must have index type".into(),
            ));
        }
        let result = self.define(name, ValueType::Lowered(target))?;
        self.push(Operation::TensorGather {
            result: result.clone(),
            source: source.clone(),
            indices: indices.clone(),
            dynamic_sizes,
            index_rank,
        })?;
        Ok(result)
    }

    pub fn tensor_scatter(
        &mut self,
        name: impl Into<String>,
        source: &Value,
        indices: &Value,
        updates: &Value,
        dynamic_sizes: Vec<Value>,
        target: LoweredType,
    ) -> Result<Value, StructuredError> {
        let source_rank = tensor_rank(source)?;
        if source_rank == 0
            || tensor_rank(indices)? != 1
            || tensor_rank(updates)? != source_rank
            || tensor_rank_from_type(&target) != Some(source_rank)
        {
            return Err(StructuredError(
                "tensor.scatter requires rank-one indices and rank-preserving updates".into(),
            ));
        }
        let source_element = tensor_scalar_type(
            source
                .lowered_type()
                .expect("tensor.scatter source is lowered"),
        )?;
        if tensor_scalar_type(
            updates
                .lowered_type()
                .expect("tensor.scatter updates are lowered"),
        )? != source_element
            || tensor_scalar_type(&target)? != source_element
        {
            return Err(StructuredError(
                "tensor.scatter source, updates, and result must share one element type".into(),
            ));
        }
        let result = self.define(name, ValueType::Lowered(target))?;
        self.push(Operation::TensorScatter {
            result: result.clone(),
            source: source.clone(),
            indices: indices.clone(),
            updates: updates.clone(),
            dynamic_sizes,
        })?;
        Ok(result)
    }

    pub fn linalg_fill(
        &mut self,
        name: impl Into<String>,
        scalar: &Value,
        output: &Value,
    ) -> Result<Value, StructuredError> {
        let output_ty = output
            .lowered_type()
            .ok_or_else(|| StructuredError("linalg.fill output is not a tensor".into()))?;
        let element = tensor_scalar_type(output_ty)?;
        if scalar.lowered_type() != Some(&element) {
            return Err(StructuredError(
                "linalg.fill scalar does not match its output element".into(),
            ));
        }
        let result = self.define(name, ValueType::Lowered(output_ty.clone()))?;
        self.push(Operation::LinalgFill {
            result: result.clone(),
            scalar: scalar.clone(),
            output: output.clone(),
        })?;
        Ok(result)
    }

    pub fn linalg_generic(
        &mut self,
        name: impl Into<String>,
        inputs: Vec<Value>,
        output: Value,
        maps: Vec<AffineMap>,
        iterators: Vec<IteratorKind>,
        body: GenericBody,
    ) -> Result<Value, StructuredError> {
        if maps.len() != inputs.len() + 1 {
            return Err(StructuredError(
                "linalg.generic requires one indexing map per input and output".into(),
            ));
        }
        if maps.iter().any(|map| map.domain_rank != iterators.len()) {
            return Err(StructuredError(
                "linalg.generic map domains must match its iterator count".into(),
            ));
        }
        for (value, map) in inputs.iter().chain([&output]).zip(&maps) {
            if tensor_rank(value)? != map.results.len() {
                return Err(StructuredError(
                    "linalg.generic map result rank does not match its tensor".into(),
                ));
            }
        }
        let argument_types = inputs
            .iter()
            .chain([&output])
            .map(|value| {
                value
                    .lowered_type()
                    .ok_or_else(|| StructuredError("linalg.generic value is not lowered".into()))
                    .and_then(tensor_scalar_type)
            })
            .collect::<Result<Vec<_>, _>>()?;
        if body
            .arguments
            .iter()
            .map(|(_, ty)| ty)
            .ne(argument_types.iter())
        {
            return Err(StructuredError(
                "linalg.generic block arguments do not match tensor elements".into(),
            ));
        }
        if body
            .captures
            .iter()
            .any(|capture| self.values.get(&capture.name) != Some(&capture.ty))
        {
            return Err(StructuredError(
                "linalg.generic body captures an undefined outer SSA value".into(),
            ));
        }
        let result_ty = output
            .lowered_type()
            .expect("tensor rank validation established a lowered type")
            .clone();
        let result = self.define(name, ValueType::Lowered(result_ty))?;
        self.push(Operation::LinalgGeneric {
            result: result.clone(),
            inputs,
            output,
            maps,
            iterators,
            body,
        })?;
        Ok(result)
    }

    pub fn call(
        &mut self,
        names: Vec<String>,
        symbol: impl Into<String>,
        arguments: Vec<Value>,
        result_types: Vec<LoweredType>,
    ) -> Result<Vec<Value>, StructuredError> {
        if names.len() != result_types.len() {
            return Err(StructuredError("func.call result arity mismatch".into()));
        }
        let results = names
            .into_iter()
            .zip(result_types)
            .map(|(name, ty)| self.define(name, ValueType::Lowered(ty)))
            .collect::<Result<Vec<_>, _>>()?;
        self.push(Operation::Call {
            results: results.clone(),
            symbol: symbol.into(),
            arguments,
        })?;
        Ok(results)
    }

    pub fn return_values(&mut self, values: Vec<Value>) -> Result<(), StructuredError> {
        let returned = values
            .iter()
            .map(|value| value.lowered_type().cloned())
            .collect::<Option<Vec<_>>>()
            .ok_or_else(|| StructuredError("function cannot return an index value".into()))?;
        if returned != self.function.results {
            return Err(StructuredError("function return signature mismatch".into()));
        }
        self.push(Operation::Return(values))?;
        self.terminated = true;
        Ok(())
    }

    pub fn finish(self) -> Result<Function, StructuredError> {
        if !self.terminated {
            return Err(StructuredError(format!(
                "function `{}` has no return",
                self.function.name
            )));
        }
        Ok(self.function)
    }

    fn define(&mut self, name: impl Into<String>, ty: ValueType) -> Result<Value, StructuredError> {
        let name = name.into();
        legal_identifier(&name)?;
        if self.values.insert(name.clone(), ty.clone()).is_some() {
            return Err(StructuredError(format!(
                "SSA value `%{name}` is defined twice"
            )));
        }
        Ok(Value { name, ty })
    }

    fn push(&mut self, operation: Operation) -> Result<(), StructuredError> {
        if self.terminated {
            return Err(StructuredError("operation follows function return".into()));
        }
        self.function.operations.push(operation);
        Ok(())
    }
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ModuleBuilder {
    declarations: Vec<(String, Vec<LoweredType>, Vec<LoweredType>)>,
    functions: Vec<Function>,
}

impl ModuleBuilder {
    pub fn declare_function(
        &mut self,
        name: impl Into<String>,
        parameters: Vec<LoweredType>,
        results: Vec<LoweredType>,
    ) -> Result<(), StructuredError> {
        let name = name.into();
        legal_identifier(&name)?;
        if self.functions.iter().any(|known| known.name == name)
            || self.declarations.iter().any(|known| known.0 == name)
        {
            return Err(StructuredError(format!(
                "MLIR function `{name}` is defined twice"
            )));
        }
        self.declarations.push((name, parameters, results));
        Ok(())
    }

    pub fn add_function(&mut self, function: Function) -> Result<(), StructuredError> {
        if self
            .functions
            .iter()
            .any(|known| known.name == function.name)
            || self
                .declarations
                .iter()
                .any(|known| known.0 == function.name)
        {
            return Err(StructuredError(format!(
                "MLIR function `{}` is defined twice",
                function.name
            )));
        }
        self.functions.push(function);
        Ok(())
    }

    pub fn print(&self) -> Result<String, StructuredError> {
        let mut output = "module {\n".to_owned();
        for (name, parameters, results) in &self.declarations {
            let parameters = parameters
                .iter()
                .map(type_spelling)
                .collect::<Result<Vec<_>, _>>()?
                .join(", ");
            output.push_str(&format!(
                "  func.func private @{name}({parameters}){}\n",
                print_result_signature(results)?
            ));
        }
        for function in &self.functions {
            print_function(&mut output, function)?;
        }
        output.push('}');
        Ok(output)
    }
}

fn print_function(output: &mut String, function: &Function) -> Result<(), StructuredError> {
    let visibility = if function.private { " private" } else { "" };
    let parameters = function
        .parameters
        .iter()
        .map(|value| Ok(format!("%{}: {}", value.name, print_value_type(&value.ty)?)))
        .collect::<Result<Vec<_>, StructuredError>>()?
        .join(", ");
    let results = print_result_signature(&function.results)?;
    output.push_str(&format!(
        "  func.func{visibility} @{}({parameters}){results} {{\n",
        function.name
    ));
    for operation in &function.operations {
        print_operation(output, operation)?;
    }
    output.push_str("  }\n");
    Ok(())
}

fn print_operation(output: &mut String, operation: &Operation) -> Result<(), StructuredError> {
    match operation {
        Operation::IndexConstant { result, value } => output.push_str(&format!(
            "    %{} = arith.constant {value} : index\n",
            result.name
        )),
        Operation::ScalarConstant { result, literal } => output.push_str(&format!(
            "    %{} = arith.constant {literal} : {}\n",
            result.name,
            print_value_type(&result.ty)?
        )),
        Operation::FloatConvert {
            result,
            source,
            extend,
        } => output.push_str(&format!(
            "    %{} = {} %{} : {} to {}\n",
            result.name,
            if *extend { "arith.extf" } else { "arith.truncf" },
            source.name,
            print_value_type(&source.ty)?,
            print_value_type(&result.ty)?
        )),
        Operation::TensorDim {
            result,
            tensor,
            axis,
        } => output.push_str(&format!(
            "    %{} = tensor.dim %{}, %{} : {}\n",
            result.name,
            tensor.name,
            axis.name,
            print_value_type(&tensor.ty)?
        )),
        Operation::IndexCast { result, source } => output.push_str(&format!(
            "    %{} = arith.index_cast %{} : index to {}\n",
            result.name,
            source.name,
            print_value_type(&result.ty)?
        )),
        Operation::IntegerToIndex { result, source } => output.push_str(&format!(
            "    %{} = arith.index_cast %{} : {} to index\n",
            result.name,
            source.name,
            print_value_type(&source.ty)?
        )),
        Operation::IndexSubtract {
            result,
            left,
            right,
        } => output.push_str(&format!(
            "    %{} = arith.subi %{}, %{} : index\n",
            result.name, left.name, right.name
        )),
        Operation::IndexMultiply {
            result,
            left,
            right,
        } => output.push_str(&format!(
            "    %{} = arith.muli %{}, %{} : index\n",
            result.name, left.name, right.name
        )),
        Operation::IndexCeilDivideSigned {
            result,
            left,
            right,
        } => output.push_str(&format!(
            "    %{} = arith.ceildivsi %{}, %{} : index\n",
            result.name, left.name, right.name
        )),
        Operation::UnsignedToFloat { result, source } => output.push_str(&format!(
            "    %{} = arith.uitofp %{} : {} to {}\n",
            result.name,
            source.name,
            print_value_type(&source.ty)?,
            print_value_type(&result.ty)?
        )),
        Operation::TensorEmpty {
            result,
            dynamic_sizes,
        } => output.push_str(&format!(
            "    %{} = tensor.empty({}) : {}\n",
            result.name,
            print_values(dynamic_sizes),
            print_value_type(&result.ty)?
        )),
        Operation::TensorCast { result, source } => output.push_str(&format!(
            "    %{} = tensor.cast %{} : {} to {}\n",
            result.name,
            source.name,
            print_value_type(&source.ty)?,
            print_value_type(&result.ty)?
        )),
        Operation::ShapeTensorFromElements { result, dimensions } => output.push_str(&format!(
            "    %{} = tensor.from_elements {} : {}\n",
            result.name,
            print_values(dimensions),
            print_value_type(&result.ty)?
        )),
        Operation::TensorReshape {
            result,
            source,
            shape,
        } => output.push_str(&format!(
            "    %{} = tensor.reshape %{}(%{}) : ({} , {}) -> {}\n",
            result.name,
            source.name,
            shape.name,
            print_value_type(&source.ty)?,
            print_value_type(&shape.ty)?,
            print_value_type(&result.ty)?
        )),
        Operation::TensorExtractSlice {
            result,
            source,
            offsets,
            sizes,
            strides,
        } => output.push_str(&format!(
            "    %{} = tensor.extract_slice %{}[{}] [{}] [{}] : {} to {}\n",
            result.name,
            source.name,
            print_slice_components(offsets),
            print_slice_components(sizes),
            print_slice_components(strides),
            print_value_type(&source.ty)?,
            print_value_type(&result.ty)?
        )),
        Operation::TensorConcat {
            result,
            inputs,
            axis,
        } => output.push_str(&format!(
            "    %{} = tensor.concat dim({axis}) {} : ({}) -> {}\n",
            result.name,
            print_values(inputs),
            inputs
                .iter()
                .map(|value| print_value_type(&value.ty))
                .collect::<Result<Vec<_>, _>>()?
                .join(", "),
            print_value_type(&result.ty)?
        )),
        Operation::TensorGather {
            result,
            source,
            indices,
            dynamic_sizes,
            index_rank,
        } => {
            let result_rank = tensor_rank_from_type(
                result
                    .lowered_type()
                    .expect("tensor.gather result is lowered"),
            )
            .expect("tensor.gather result is ranked");
            let coordinates = (0..result_rank)
                .map(|axis| format!("%gather_i{axis}"))
                .collect::<Vec<_>>();
            let block_arguments = coordinates
                .iter()
                .map(|coordinate| format!("{coordinate}: index"))
                .collect::<Vec<_>>()
                .join(", ");
            let index_coordinates = coordinates[..*index_rank].join(", ");
            let index_scalar = tensor_scalar_type(
                indices
                    .lowered_type()
                    .expect("tensor.gather indices are lowered"),
            )?;
            let result_scalar = tensor_scalar_type(
                result
                    .lowered_type()
                    .expect("tensor.gather result is lowered"),
            )?;
            output.push_str(&format!(
                "    %{} = tensor.generate {} {{\n    ^bb0({block_arguments}):\n      %gather_index_integer = tensor.extract %{}[{index_coordinates}] : {}\n      %gather_index = arith.index_cast %gather_index_integer : {} to index\n      %gather_value = tensor.extract %{}[{}] : {}\n      tensor.yield %gather_value : {}\n    }} : {}\n",
                result.name,
                print_values(dynamic_sizes),
                indices.name,
                print_value_type(&indices.ty)?,
                print_value_type(&ValueType::Lowered(index_scalar))?,
                source.name,
                std::iter::once("%gather_index".to_owned())
                    .chain(coordinates[*index_rank..].iter().cloned())
                    .collect::<Vec<_>>()
                    .join(", "),
                print_value_type(&source.ty)?,
                print_value_type(&ValueType::Lowered(result_scalar))?,
                print_value_type(&result.ty)?
            ));
        }
        Operation::TensorScatter {
            result,
            source,
            indices,
            updates,
            dynamic_sizes,
        } => {
            let result_rank = tensor_rank_from_type(
                result
                    .lowered_type()
                    .expect("tensor.scatter result is lowered"),
            )
            .expect("tensor.scatter result is ranked");
            let coordinates = (0..result_rank)
                .map(|axis| format!("%scatter_i{axis}"))
                .collect::<Vec<_>>();
            let block_arguments = coordinates
                .iter()
                .map(|coordinate| format!("{coordinate}: index"))
                .collect::<Vec<_>>()
                .join(", ");
            let index_scalar = tensor_scalar_type(
                indices
                    .lowered_type()
                    .expect("tensor.scatter indices are lowered"),
            )?;
            let result_scalar = tensor_scalar_type(
                result
                    .lowered_type()
                    .expect("tensor.scatter result is lowered"),
            )?;
            let trailing = coordinates[1..].join(", ");
            let update_coordinates = if trailing.is_empty() {
                "%scatter_update_index".to_owned()
            } else {
                format!("%scatter_update_index, {trailing}")
            };
            output.push_str(&format!(
                "    %{} = tensor.generate {} {{\n    ^bb0({block_arguments}):\n      %scatter_initial = tensor.extract %{}[{}] : {}\n      %scatter_axis = arith.constant 0 : index\n      %scatter_count = tensor.dim %{}, %scatter_axis : {}\n      %scatter_zero = arith.constant 0 : index\n      %scatter_one = arith.constant 1 : index\n      %scatter_selected = scf.for %scatter_update_index = %scatter_zero to %scatter_count step %scatter_one iter_args(%scatter_current = %scatter_initial) -> ({}) {{\n        %scatter_candidate_integer = tensor.extract %{}[%scatter_update_index] : {}\n        %scatter_candidate = arith.index_cast %scatter_candidate_integer : {} to index\n        %scatter_matches = arith.cmpi eq, %scatter_candidate, %scatter_i0 : index\n        %scatter_update = tensor.extract %{}[{update_coordinates}] : {}\n        %scatter_next = arith.select %scatter_matches, %scatter_update, %scatter_current : {}\n        scf.yield %scatter_next : {}\n      }}\n      tensor.yield %scatter_selected : {}\n    }} : {}\n",
                result.name,
                print_values(dynamic_sizes),
                source.name,
                coordinates.join(", "),
                print_value_type(&source.ty)?,
                indices.name,
                print_value_type(&indices.ty)?,
                print_value_type(&ValueType::Lowered(result_scalar.clone()))?,
                indices.name,
                print_value_type(&indices.ty)?,
                print_value_type(&ValueType::Lowered(index_scalar))?,
                updates.name,
                print_value_type(&updates.ty)?,
                print_value_type(&ValueType::Lowered(result_scalar.clone()))?,
                print_value_type(&ValueType::Lowered(result_scalar.clone()))?,
                print_value_type(&ValueType::Lowered(result_scalar))?,
                print_value_type(&result.ty)?
            ));
        }
        Operation::LinalgFill {
            result,
            scalar,
            output: destination,
        } => output.push_str(&format!(
            "    %{} = linalg.fill ins(%{} : {}) outs(%{} : {}) -> {}\n",
            result.name,
            scalar.name,
            print_value_type(&scalar.ty)?,
            destination.name,
            print_value_type(&destination.ty)?,
            print_value_type(&result.ty)?
        )),
        Operation::LinalgGeneric {
            result,
            inputs,
            output: destination,
            maps,
            iterators,
            body,
        } => {
            let maps = maps
                .iter()
                .map(print_affine_map)
                .collect::<Vec<_>>()
                .join(", ");
            let iterators = iterators
                .iter()
                .map(|iterator| match iterator {
                    IteratorKind::Parallel => "\"parallel\"",
                    IteratorKind::Reduction => "\"reduction\"",
                })
                .collect::<Vec<_>>()
                .join(", ");
            let input_values = print_values(inputs);
            let input_types = inputs
                .iter()
                .map(|value| print_value_type(&value.ty))
                .collect::<Result<Vec<_>, _>>()?
                .join(", ");
            output.push_str(&format!(
                "    %{} = linalg.generic {{indexing_maps = [{maps}], iterator_types = [{iterators}]}} ins({input_values} : {input_types}) outs(%{} : {}) {{\n",
                result.name,
                destination.name,
                print_value_type(&destination.ty)?,
            ));
            output.push_str("    ^bb0(");
            output.push_str(
                &body
                    .arguments
                    .iter()
                    .map(|(name, ty)| Ok(format!("%{name}: {}", type_spelling(ty)?)))
                    .collect::<Result<Vec<_>, StructuredError>>()?
                    .join(", "),
            );
            output.push_str("):\n");
            for scalar in &body.operations {
                match scalar {
                    ScalarOperation::FloatConvert {
                        result,
                        operand,
                        source,
                        target,
                    } => output.push_str(&format!(
                        "      %{result} = {} %{operand} : {} to {}\n",
                        if scalar_float_bits(target)? > scalar_float_bits(source)? {
                            "arith.extf"
                        } else {
                            "arith.truncf"
                        },
                        type_spelling(source)?,
                        type_spelling(target)?
                    )),
                    ScalarOperation::Unary {
                        result,
                        operation,
                        operand,
                        ty,
                    } => output.push_str(&format!(
                        "      %{result} = {} %{operand} : {}\n",
                        scalar_unary_name(*operation),
                        type_spelling(ty)?
                    )),
                    ScalarOperation::Binary {
                        result,
                        operation,
                        left,
                        right,
                        ty,
                    } => output.push_str(&format!(
                        "      %{result} = {} %{left}, %{right} : {}\n",
                        scalar_binary_name(*operation),
                        type_spelling(ty)?
                    )),
                    ScalarOperation::Yield { value, ty } => output.push_str(&format!(
                        "      linalg.yield %{value} : {}\n",
                        type_spelling(ty)?
                    )),
                }
            }
            output.push_str(&format!("    }} -> {}\n", print_value_type(&result.ty)?));
        }
        Operation::Call {
            results,
            symbol,
            arguments,
        } => {
            let assignment = if results.is_empty() {
                String::new()
            } else {
                format!("{} = ", print_values(results))
            };
            let arguments_types = arguments
                .iter()
                .map(|value| print_value_type(&value.ty))
                .collect::<Result<Vec<_>, _>>()?
                .join(", ");
            let result_types = results
                .iter()
                .map(|value| {
                    value
                        .lowered_type()
                        .cloned()
                        .ok_or_else(|| StructuredError("func.call returned index".into()))
                })
                .collect::<Result<Vec<_>, _>>()?;
            output.push_str(&format!(
                "    {assignment}func.call @{symbol}({}) : ({arguments_types}){}\n",
                print_values(arguments),
                print_result_signature(&result_types)?
            ));
        }
        Operation::Return(values) => {
            if values.is_empty() {
                output.push_str("    return\n");
            } else {
                let types = values
                    .iter()
                    .map(|value| print_value_type(&value.ty))
                    .collect::<Result<Vec<_>, _>>()?
                    .join(", ");
                output.push_str(&format!("    return {} : {types}\n", print_values(values)));
            }
        }
    }
    Ok(())
}

fn print_slice_components(components: &[SliceComponent]) -> String {
    components
        .iter()
        .map(|component| match component {
            SliceComponent::Static(value) => value.to_string(),
            SliceComponent::Dynamic(value) => format!("%{}", value.name),
        })
        .collect::<Vec<_>>()
        .join(", ")
}

fn tensor_rank(value: &Value) -> Result<usize, StructuredError> {
    match value.lowered_type() {
        Some(LoweredType::Tensor {
            shape: LoweredTensorShape::Ranked(dimensions),
            ..
        }) => Ok(dimensions.len()),
        _ => Err(StructuredError(
            "structured tensor operation requires known rank".into(),
        )),
    }
}

fn tensor_rank_from_type(ty: &LoweredType) -> Option<usize> {
    match ty {
        LoweredType::Tensor {
            shape: LoweredTensorShape::Ranked(dimensions),
            ..
        } => Some(dimensions.len()),
        _ => None,
    }
}

fn tensor_scalar_type(ty: &LoweredType) -> Result<LoweredType, StructuredError> {
    let LoweredType::Tensor { element, .. } = ty else {
        return Err(StructuredError("expected a tensor type".into()));
    };
    Ok(match element {
        LoweredTensorElement::Integer { bits, signed } => LoweredType::Integer {
            bits: *bits,
            signed: *signed,
        },
        LoweredTensorElement::Float { format } => LoweredType::Float { format: *format },
        LoweredTensorElement::Boolean => LoweredType::Boolean,
    })
}

fn tensor_cast_shapes_compatible(source: &LoweredTensorShape, target: &LoweredTensorShape) -> bool {
    match (source, target) {
        (LoweredTensorShape::Unranked, _) | (_, LoweredTensorShape::Unranked) => true,
        (LoweredTensorShape::Ranked(source), LoweredTensorShape::Ranked(target)) => {
            source.len() == target.len()
                && source.iter().zip(target).all(|(source, target)| {
                    source == target
                        || source == &LoweredTensorDimension::Dynamic
                        || target == &LoweredTensorDimension::Dynamic
                })
        }
    }
}

fn ensure_scalar(ty: &LoweredType) -> Result<(), StructuredError> {
    if matches!(
        ty,
        LoweredType::Integer { .. } | LoweredType::Float { .. } | LoweredType::Boolean
    ) {
        Ok(())
    } else {
        Err(StructuredError("expected an MLIR scalar type".into()))
    }
}

fn print_result_signature(results: &[LoweredType]) -> Result<String, StructuredError> {
    Ok(match results {
        [] => String::new(),
        [result] => format!(" -> {}", type_spelling(result)?),
        results => format!(
            " -> ({})",
            results
                .iter()
                .map(type_spelling)
                .collect::<Result<Vec<_>, _>>()?
                .join(", ")
        ),
    })
}

fn print_value_type(ty: &ValueType) -> Result<String, StructuredError> {
    match ty {
        ValueType::Lowered(ty) => Ok(type_spelling(ty)?),
        ValueType::Index => Ok("index".into()),
        ValueType::ShapeTensor(rank) => Ok(format!("tensor<{rank}xindex>")),
    }
}

fn print_values(values: &[Value]) -> String {
    values
        .iter()
        .map(|value| format!("%{}", value.name))
        .collect::<Vec<_>>()
        .join(", ")
}

fn print_affine_map(map: &AffineMap) -> String {
    let domain = (0..map.domain_rank)
        .map(|axis| format!("d{axis}"))
        .collect::<Vec<_>>()
        .join(", ");
    let results = map
        .results
        .iter()
        .map(|expression| match expression {
            AffineExpression::Dimension(axis) => format!("d{axis}"),
            AffineExpression::Constant(value) => value.to_string(),
            AffineExpression::FloorDiv { dimension, divisor } => {
                format!("d{dimension} floordiv {divisor}")
            }
        })
        .collect::<Vec<_>>()
        .join(", ");
    format!("affine_map<({domain}) -> ({results})>")
}

fn scalar_binary_name(operation: ScalarBinaryOperation) -> &'static str {
    match operation {
        ScalarBinaryOperation::AddFloat => "arith.addf",
        ScalarBinaryOperation::AddInteger => "arith.addi",
        ScalarBinaryOperation::SubtractFloat => "arith.subf",
        ScalarBinaryOperation::SubtractInteger => "arith.subi",
        ScalarBinaryOperation::MultiplyFloat => "arith.mulf",
        ScalarBinaryOperation::MultiplyInteger => "arith.muli",
        ScalarBinaryOperation::DivideFloat => "arith.divf",
        ScalarBinaryOperation::DivideSigned => "arith.divsi",
        ScalarBinaryOperation::DivideUnsigned => "arith.divui",
        ScalarBinaryOperation::MaximumFloat => "arith.maximumf",
        ScalarBinaryOperation::MaximumSigned => "arith.maxsi",
        ScalarBinaryOperation::MaximumUnsigned => "arith.maxui",
    }
}

fn scalar_unary_name(operation: ScalarUnaryOperation) -> &'static str {
    match operation {
        ScalarUnaryOperation::Exp => "math.exp",
        ScalarUnaryOperation::Log => "math.log",
        ScalarUnaryOperation::Tanh => "math.tanh",
        ScalarUnaryOperation::Rsqrt => "math.rsqrt",
    }
}

fn lowered_float_bits(format: crate::LoweredFloatFormat) -> u16 {
    match format {
        crate::LoweredFloatFormat::Float8E4M3Fn | crate::LoweredFloatFormat::Float8E5M2 => 8,
        crate::LoweredFloatFormat::BrainFloat16 => 16,
        crate::LoweredFloatFormat::Ieee(bits) => bits,
    }
}

fn scalar_float_bits(ty: &LoweredType) -> Result<u16, StructuredError> {
    let LoweredType::Float { format } = ty else {
        return Err(StructuredError("expected a scalar floating type".into()));
    };
    Ok(lowered_float_bits(*format))
}

fn scalar_binary_accepts(operation: ScalarBinaryOperation, ty: &LoweredType) -> bool {
    match operation {
        ScalarBinaryOperation::AddFloat
        | ScalarBinaryOperation::SubtractFloat
        | ScalarBinaryOperation::MultiplyFloat
        | ScalarBinaryOperation::DivideFloat
        | ScalarBinaryOperation::MaximumFloat => matches!(ty, LoweredType::Float { .. }),
        ScalarBinaryOperation::DivideSigned | ScalarBinaryOperation::MaximumSigned => {
            matches!(ty, LoweredType::Integer { signed: true, .. })
        }
        ScalarBinaryOperation::DivideUnsigned | ScalarBinaryOperation::MaximumUnsigned => {
            matches!(
                ty,
                LoweredType::Integer { signed: false, .. } | LoweredType::Boolean
            )
        }
        ScalarBinaryOperation::AddInteger
        | ScalarBinaryOperation::SubtractInteger
        | ScalarBinaryOperation::MultiplyInteger => {
            matches!(ty, LoweredType::Integer { .. } | LoweredType::Boolean)
        }
    }
}

fn legal_identifier(name: &str) -> Result<(), StructuredError> {
    if name.is_empty()
        || !name
            .chars()
            .all(|character| character.is_ascii_alphanumeric() || character == '_')
    {
        Err(StructuredError(format!(
            "`{name}` is not a legal structured MLIR identifier"
        )))
    } else {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::LoweredFloatFormat;

    fn matrix() -> LoweredType {
        LoweredType::Tensor {
            element: LoweredTensorElement::Float {
                format: LoweredFloatFormat::Ieee(32),
            },
            shape: LoweredTensorShape::Ranked(vec![
                LoweredTensorDimension::Known(2),
                LoweredTensorDimension::Known(3),
            ]),
        }
    }

    #[test]
    fn malformed_linalg_map_fails_during_structural_construction() {
        let tensor = matrix();
        let scalar = LoweredType::Float {
            format: LoweredFloatFormat::Ieee(32),
        };
        let mut function = FunctionBuilder::new(
            "entry",
            false,
            vec![
                ("left".into(), tensor.clone()),
                ("right".into(), tensor.clone()),
            ],
            vec![tensor.clone()],
        )
        .unwrap();
        let left = function.parameter(0).unwrap();
        let right = function.parameter(1).unwrap();
        let empty = function.tensor_empty("empty", tensor, Vec::new()).unwrap();
        let body = GenericBody::new(
            vec![
                ("lhs".into(), scalar.clone()),
                ("rhs".into(), scalar.clone()),
                ("unused".into(), scalar.clone()),
            ],
            vec![
                ScalarOperation::Binary {
                    result: "sum".into(),
                    operation: ScalarBinaryOperation::AddFloat,
                    left: "lhs".into(),
                    right: "rhs".into(),
                    ty: scalar.clone(),
                },
                ScalarOperation::Yield {
                    value: "sum".into(),
                    ty: scalar,
                },
            ],
        )
        .unwrap();
        let error = function
            .linalg_generic(
                "result",
                vec![left, right],
                empty,
                vec![
                    AffineMap::new(2, vec![AffineExpression::Dimension(0)]).unwrap(),
                    AffineMap::identity(2),
                    AffineMap::identity(2),
                ],
                vec![IteratorKind::Parallel; 2],
                body,
            )
            .unwrap_err();
        assert!(error
            .to_string()
            .contains("map result rank does not match its tensor"));
    }

    #[test]
    fn scalar_opcode_type_mismatch_fails_before_printing() {
        let integer = LoweredType::Integer {
            bits: 32,
            signed: true,
        };
        let error = GenericBody::new(
            vec![
                ("left".into(), integer.clone()),
                ("right".into(), integer.clone()),
            ],
            vec![
                ScalarOperation::Binary {
                    result: "bad".into(),
                    operation: ScalarBinaryOperation::AddFloat,
                    left: "left".into(),
                    right: "right".into(),
                    ty: integer.clone(),
                },
                ScalarOperation::Yield {
                    value: "bad".into(),
                    ty: integer,
                },
            ],
        )
        .unwrap_err();
        assert!(error
            .to_string()
            .contains("incompatible with its element type"));
    }
}
