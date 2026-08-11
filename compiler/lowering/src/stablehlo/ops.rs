use severian_hir::{TensorElementType, TensorType};

use super::scalar_tensor;
use crate::tensor::tensor_type;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MlirValue {
    pub name: String,
    pub ty: String,
    pub tensor: Option<TensorType>,
}

impl MlirValue {
    pub fn new(name: impl Into<String>, ty: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            ty: ty.into(),
            tensor: None,
        }
    }

    pub fn from_tensor(name: impl Into<String>, tensor: TensorType) -> Self {
        Self {
            name: name.into(),
            ty: tensor_type(tensor),
            tensor: Some(tensor),
        }
    }

    pub fn tensor_type(&self) -> Option<TensorType> {
        self.tensor
    }
}

#[derive(Debug, Default)]
pub struct StableHloEmitter {
    text: String,
    next_value: usize,
}

impl StableHloEmitter {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn into_string(self) -> String {
        self.text
    }

    pub fn as_str(&self) -> &str {
        &self.text
    }

    pub(crate) fn fresh(&mut self) -> String {
        let value = format!("%hlo{}", self.next_value);
        self.next_value += 1;
        value
    }

    pub(crate) fn line(&mut self, line: impl AsRef<str>) {
        self.text.push_str("    ");
        self.text.push_str(line.as_ref());
        self.text.push('\n');
    }

    pub fn constant_scalar(&mut self, literal: &str, result_type: TensorType) -> MlirValue {
        let result = self.fresh();
        let ty = tensor_type(result_type);
        self.line(format!(
            "{result} = stablehlo.constant dense<{literal}> : {ty}"
        ));
        MlirValue::from_tensor(result, result_type)
    }

    pub fn scalar(&mut self, value: &str, element: TensorElementType) -> MlirValue {
        self.constant_scalar(value, scalar_tensor(element))
    }

    pub fn splat(&mut self, value: &str, result_type: TensorType) -> MlirValue {
        let scalar = self.scalar(value, result_type.element);
        if result_type.rank == Some(0) {
            scalar
        } else {
            self.broadcast_in_dim(&scalar, &[], result_type)
        }
    }

    pub fn add(&mut self, lhs: &MlirValue, rhs: &MlirValue, result_type: TensorType) -> MlirValue {
        self.binary("add", lhs, rhs, result_type)
    }

    pub fn subtract(
        &mut self,
        lhs: &MlirValue,
        rhs: &MlirValue,
        result_type: TensorType,
    ) -> MlirValue {
        self.binary("subtract", lhs, rhs, result_type)
    }

    pub fn multiply(
        &mut self,
        lhs: &MlirValue,
        rhs: &MlirValue,
        result_type: TensorType,
    ) -> MlirValue {
        self.binary("multiply", lhs, rhs, result_type)
    }

    pub fn divide(
        &mut self,
        lhs: &MlirValue,
        rhs: &MlirValue,
        result_type: TensorType,
    ) -> MlirValue {
        self.binary("divide", lhs, rhs, result_type)
    }

    pub fn maximum(
        &mut self,
        lhs: &MlirValue,
        rhs: &MlirValue,
        result_type: TensorType,
    ) -> MlirValue {
        self.binary("maximum", lhs, rhs, result_type)
    }

    pub fn minimum(
        &mut self,
        lhs: &MlirValue,
        rhs: &MlirValue,
        result_type: TensorType,
    ) -> MlirValue {
        self.binary("minimum", lhs, rhs, result_type)
    }

    fn binary(
        &mut self,
        op: &str,
        lhs: &MlirValue,
        rhs: &MlirValue,
        result_type: TensorType,
    ) -> MlirValue {
        let result = self.fresh();
        let ty = tensor_type(result_type);
        self.line(format!(
            "{result} = stablehlo.{op} {}, {} : {ty}",
            lhs.name, rhs.name
        ));
        MlirValue::from_tensor(result, result_type)
    }

    pub fn negate(&mut self, input: &MlirValue, result_type: TensorType) -> MlirValue {
        let result = self.fresh();
        let ty = tensor_type(result_type);
        self.line(format!("{result} = stablehlo.negate {} : {ty}", input.name));
        MlirValue::from_tensor(result, result_type)
    }

    pub fn exponential(&mut self, input: &MlirValue, result_type: TensorType) -> MlirValue {
        self.unary("exponential", input, result_type)
    }

    pub fn cosine(&mut self, input: &MlirValue, result_type: TensorType) -> MlirValue {
        self.unary("cosine", input, result_type)
    }

    pub fn sine(&mut self, input: &MlirValue, result_type: TensorType) -> MlirValue {
        self.unary("sine", input, result_type)
    }

    pub fn tanh(&mut self, input: &MlirValue, result_type: TensorType) -> MlirValue {
        self.unary("tanh", input, result_type)
    }

    pub fn rsqrt(&mut self, input: &MlirValue, result_type: TensorType) -> MlirValue {
        self.unary("rsqrt", input, result_type)
    }

    pub fn logistic(&mut self, input: &MlirValue, result_type: TensorType) -> MlirValue {
        self.unary("logistic", input, result_type)
    }

    pub fn sqrt(&mut self, input: &MlirValue, result_type: TensorType) -> MlirValue {
        self.unary("sqrt", input, result_type)
    }

    fn unary(&mut self, operation: &str, input: &MlirValue, result_type: TensorType) -> MlirValue {
        let result = self.fresh();
        let ty = tensor_type(result_type);
        self.line(format!(
            "{result} = \"stablehlo.{operation}\"({}) : ({}) -> {ty}",
            input.name, input.ty,
        ));
        MlirValue::from_tensor(result, result_type)
    }

    pub fn reshape(&mut self, input: &MlirValue, result_type: TensorType) -> MlirValue {
        let result = self.fresh();
        let ty = tensor_type(result_type);
        self.line(format!(
            "{result} = stablehlo.reshape {} : ({}) -> {ty}",
            input.name, input.ty
        ));
        MlirValue::from_tensor(result, result_type)
    }

    pub fn transpose(
        &mut self,
        input: &MlirValue,
        permutation: &[u64],
        result_type: TensorType,
    ) -> MlirValue {
        let result = self.fresh();
        let ty = tensor_type(result_type);
        let permutation = dense_i64(permutation);
        self.line(format!(
            "{result} = stablehlo.transpose {} , dims = {permutation} : ({}) -> {ty}",
            input.name, input.ty
        ));
        MlirValue::from_tensor(result, result_type)
    }

    pub fn broadcast_in_dim(
        &mut self,
        input: &MlirValue,
        dimensions: &[u64],
        result_type: TensorType,
    ) -> MlirValue {
        let result = self.fresh();
        let ty = tensor_type(result_type);
        let dimensions = dense_i64(dimensions);
        self.line(format!(
            "{result} = stablehlo.broadcast_in_dim {} , dims = {dimensions} : ({}) -> {ty}",
            input.name, input.ty
        ));
        MlirValue::from_tensor(result, result_type)
    }

    pub fn dot_general(
        &mut self,
        lhs: &MlirValue,
        rhs: &MlirValue,
        lhs_batching: &[u64],
        rhs_batching: &[u64],
        lhs_contracting: &[u64],
        rhs_contracting: &[u64],
        result_type: TensorType,
    ) -> MlirValue {
        let result = self.fresh();
        let ty = tensor_type(result_type);

        self.line(format!(
            concat!(
                "{result} = \"stablehlo.dot_general\"({}, {}) {{",
                "dot_dimension_numbers = #stablehlo.dot<",
                "lhs_batching_dimensions = [{}], ",
                "rhs_batching_dimensions = [{}], ",
                "lhs_contracting_dimensions = [{}], ",
                "rhs_contracting_dimensions = [{}]>, ",
                "precision_config = [#stablehlo<precision DEFAULT>, ",
                "#stablehlo<precision DEFAULT>]}} : ({}, {}) -> {ty}"
            ),
            lhs.name,
            rhs.name,
            list(lhs_batching),
            list(rhs_batching),
            list(lhs_contracting),
            list(rhs_contracting),
            lhs.ty,
            rhs.ty,
            result = result,
            ty = ty,
        ));

        MlirValue::from_tensor(result, result_type)
    }

    pub fn convert(&mut self, input: &MlirValue, result_type: TensorType) -> MlirValue {
        let result = self.fresh();
        let ty = tensor_type(result_type);
        self.line(format!(
            "{result} = stablehlo.convert {} : ({}) -> {ty}",
            input.name, input.ty
        ));
        MlirValue::from_tensor(result, result_type)
    }

    pub fn custom_call(
        &mut self,
        target: &str,
        args: &[MlirValue],
        result_type: TensorType,
    ) -> MlirValue {
        let result = self.fresh();
        let ty = tensor_type(result_type);
        let operands = args
            .iter()
            .map(|arg| arg.name.as_str())
            .collect::<Vec<_>>()
            .join(", ");
        let input_types = args
            .iter()
            .map(|arg| arg.ty.as_str())
            .collect::<Vec<_>>()
            .join(", ");

        self.line(format!(
            "{result} = stablehlo.custom_call @\"{target}\"({operands}) {{api_version = 0 : i32}} : ({input_types}) -> {ty}"
        ));

        MlirValue::from_tensor(result, result_type)
    }
}

fn dense_i64(values: &[u64]) -> String {
    format!("[{}]", list(values))
}

pub(crate) fn list(values: &[u64]) -> String {
    values
        .iter()
        .map(u64::to_string)
        .collect::<Vec<_>>()
        .join(", ")
}
