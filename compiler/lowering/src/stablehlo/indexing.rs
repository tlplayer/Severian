use severian_hir::TensorType;

use super::{ops::list, MlirValue, StableHloEmitter};
use crate::tensor::tensor_type;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StableHloComparison {
    Eq,
    Ne,
    Ge,
    Gt,
    Le,
    Lt,
}

impl StableHloComparison {
    fn attribute(self) -> &'static str {
        match self {
            Self::Eq => "EQ",
            Self::Ne => "NE",
            Self::Ge => "GE",
            Self::Gt => "GT",
            Self::Le => "LE",
            Self::Lt => "LT",
        }
    }
}

impl StableHloEmitter {
    pub fn compare(
        &mut self,
        lhs: &MlirValue,
        rhs: &MlirValue,
        comparison: StableHloComparison,
        predicate_mlir_type: &str,
    ) -> MlirValue {
        let result = self.fresh();
        self.line(format!(
            "{result} = \"stablehlo.compare\"({}, {}) {{comparison_direction = #stablehlo<comparison_direction {}>}} : ({}, {}) -> {predicate_mlir_type}",
            lhs.name, rhs.name, comparison.attribute(), lhs.ty, rhs.ty,
        ));
        MlirValue::new(result, predicate_mlir_type)
    }

    pub fn concatenate(
        &mut self,
        inputs: &[MlirValue],
        dimension: u64,
        result_type: TensorType,
    ) -> MlirValue {
        assert!(!inputs.is_empty(), "concatenate requires at least one input");
        let result = self.fresh();
        let result_ty = tensor_type(result_type);
        let operands = inputs.iter().map(|value| value.name.as_str()).collect::<Vec<_>>().join(", ");
        let input_types = inputs.iter().map(|value| value.ty.as_str()).collect::<Vec<_>>().join(", ");
        self.line(format!(
            "{result} = \"stablehlo.concatenate\"({operands}) {{dimension = {dimension} : i64}} : ({input_types}) -> {result_ty}"
        ));
        MlirValue::from_tensor(result, result_type)
    }

    pub fn slice(
        &mut self,
        input: &MlirValue,
        starts: &[u64],
        limits: &[u64],
        strides: &[u64],
        result_type: TensorType,
    ) -> MlirValue {
        assert_eq!(starts.len(), limits.len());
        assert_eq!(starts.len(), strides.len());
        let result = self.fresh();
        let result_ty = tensor_type(result_type);
        self.line(format!(
            "{result} = \"stablehlo.slice\"({}) {{start_indices = array<i64: {}>, limit_indices = array<i64: {}>, strides = array<i64: {}>}} : ({}) -> {result_ty}",
            input.name, list(starts), list(limits), list(strides), input.ty,
        ));
        MlirValue::from_tensor(result, result_type)
    }

    pub fn dynamic_slice(
        &mut self,
        input: &MlirValue,
        starts: &[MlirValue],
        slice_sizes: &[u64],
        result_type: TensorType,
    ) -> MlirValue {
        let result = self.fresh();
        let result_ty = tensor_type(result_type);
        let mut operands = vec![input.name.as_str()];
        operands.extend(starts.iter().map(|value| value.name.as_str()));
        let mut input_types = vec![input.ty.as_str()];
        input_types.extend(starts.iter().map(|value| value.ty.as_str()));
        self.line(format!(
            "{result} = \"stablehlo.dynamic_slice\"({}) {{slice_sizes = array<i64: {}>}} : ({}) -> {result_ty}",
            operands.join(", "), list(slice_sizes), input_types.join(", "),
        ));
        MlirValue::from_tensor(result, result_type)
    }

    pub fn dynamic_update_slice(
        &mut self,
        input: &MlirValue,
        update: &MlirValue,
        starts: &[MlirValue],
        result_type: TensorType,
    ) -> MlirValue {
        let result = self.fresh();
        let result_ty = tensor_type(result_type);
        let mut operands = vec![input.name.as_str(), update.name.as_str()];
        operands.extend(starts.iter().map(|value| value.name.as_str()));
        let mut input_types = vec![input.ty.as_str(), update.ty.as_str()];
        input_types.extend(starts.iter().map(|value| value.ty.as_str()));
        self.line(format!(
            "{result} = \"stablehlo.dynamic_update_slice\"({}) : ({}) -> {result_ty}",
            operands.join(", "), input_types.join(", "),
        ));
        MlirValue::from_tensor(result, result_type)
    }

    pub fn select(
        &mut self,
        predicate: &MlirValue,
        on_true: &MlirValue,
        on_false: &MlirValue,
        result_type: TensorType,
    ) -> MlirValue {
        let result = self.fresh();
        let result_ty = tensor_type(result_type);
        self.line(format!(
            "{result} = \"stablehlo.select\"({}, {}, {}) : ({}, {}, {}) -> {result_ty}",
            predicate.name, on_true.name, on_false.name,
            predicate.ty, on_true.ty, on_false.ty,
        ));
        MlirValue::from_tensor(result, result_type)
    }

    pub fn reverse(
        &mut self,
        input: &MlirValue,
        dimensions: &[u64],
        result_type: TensorType,
    ) -> MlirValue {
        let result = self.fresh();
        let result_ty = tensor_type(result_type);
        self.line(format!(
            "{result} = \"stablehlo.reverse\"({}) {{dimensions = array<i64: {}>}} : ({}) -> {result_ty}",
            input.name, list(dimensions), input.ty,
        ));
        MlirValue::from_tensor(result, result_type)
    }

    pub fn pad(
        &mut self,
        input: &MlirValue,
        padding_value: &MlirValue,
        edge_low: &[u64],
        edge_high: &[u64],
        interior: &[u64],
        result_type: TensorType,
    ) -> MlirValue {
        assert_eq!(edge_low.len(), edge_high.len());
        assert_eq!(edge_low.len(), interior.len());
        let result = self.fresh();
        let result_ty = tensor_type(result_type);
        self.line(format!(
            "{result} = \"stablehlo.pad\"({}, {}) {{edge_padding_low = array<i64: {}>, edge_padding_high = array<i64: {}>, interior_padding = array<i64: {}>}} : ({}, {}) -> {result_ty}",
            input.name, padding_value.name, list(edge_low), list(edge_high),
            list(interior), input.ty, padding_value.ty,
        ));
        MlirValue::from_tensor(result, result_type)
    }

    pub fn get_dimension_size(
        &mut self,
        input: &MlirValue,
        dimension: u64,
        result_type: TensorType,
    ) -> MlirValue {
        let result = self.fresh();
        let result_ty = tensor_type(result_type);
        self.line(format!(
            "{result} = \"stablehlo.get_dimension_size\"({}) {{dimension = {dimension} : i64}} : ({}) -> {result_ty}",
            input.name, input.ty,
        ));
        MlirValue::from_tensor(result, result_type)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn gather(
        &mut self,
        operand: &MlirValue,
        start_indices: &MlirValue,
        offset_dims: &[u64],
        collapsed_slice_dims: &[u64],
        start_index_map: &[u64],
        index_vector_dim: u64,
        slice_sizes: &[u64],
        result_type: TensorType,
    ) -> MlirValue {
        let result = self.fresh();
        let result_ty = tensor_type(result_type);
        self.line(format!(
            concat!(
                "{result} = \"stablehlo.gather\"({}, {}) {{",
                "dimension_numbers = #stablehlo.gather<offset_dims = [{}], ",
                "collapsed_slice_dims = [{}], start_index_map = [{}], ",
                "index_vector_dim = {}>, slice_sizes = array<i64: {}>, ",
                "indices_are_sorted = false}} : ({}, {}) -> {result_ty}"
            ),
            operand.name, start_indices.name, list(offset_dims),
            list(collapsed_slice_dims), list(start_index_map), index_vector_dim,
            list(slice_sizes), operand.ty, start_indices.ty,
        ));
        MlirValue::from_tensor(result, result_type)
    }
}


/// Embedding lookup is an ordinary gather over the vocabulary dimension.
pub fn embedding_lookup(
    emitter: &mut StableHloEmitter,
    embedding_table: &MlirValue,
    token_indices: &MlirValue,
    index_rank: u64,
    _vocabulary_size: u64,
    embedding_size: u64,
    result_type: TensorType,
) -> MlirValue {
    let offset_dims = [index_rank];
    emitter.gather(
        embedding_table,
        token_indices,
        &offset_dims,
        &[0],
        &[0],
        index_rank,
        &[1, embedding_size],
        result_type,
    )
}
