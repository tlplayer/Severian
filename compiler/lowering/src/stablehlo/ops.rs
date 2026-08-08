use severian_hir::TensorType;

use crate::tensor::tensor_type;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MlirValue {
    pub name: String,
    pub ty: String,
}

impl MlirValue {
    pub fn new(name: impl Into<String>, ty: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            ty: ty.into(),
        }
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

    fn fresh(&mut self) -> String {
        let value = format!("%hlo{}", self.next_value);
        self.next_value += 1;
        value
    }

    fn line(&mut self, line: impl AsRef<str>) {
        self.text.push_str("    ");
        self.text.push_str(line.as_ref());
        self.text.push('\n');
    }

    pub fn constant_scalar(
        &mut self,
        literal: &str,
        result_type: TensorType,
    ) -> MlirValue {
        let result = self.fresh();
        let ty = tensor_type(result_type);
        self.line(format!(
            "{result} = stablehlo.constant dense<{literal}> : {ty}"
        ));
        MlirValue::new(result, ty)
    }

    pub fn add(
        &mut self,
        lhs: &MlirValue,
        rhs: &MlirValue,
        result_type: TensorType,
    ) -> MlirValue {
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
        MlirValue::new(result, ty)
    }

    pub fn negate(
        &mut self,
        input: &MlirValue,
        result_type: TensorType,
    ) -> MlirValue {
        let result = self.fresh();
        let ty = tensor_type(result_type);
        self.line(format!(
            "{result} = stablehlo.negate {} : {ty}",
            input.name
        ));
        MlirValue::new(result, ty)
    }

    pub fn reshape(
        &mut self,
        input: &MlirValue,
        result_type: TensorType,
    ) -> MlirValue {
        let result = self.fresh();
        let ty = tensor_type(result_type);
        self.line(format!(
            "{result} = stablehlo.reshape {} : ({}) -> {ty}",
            input.name, input.ty
        ));
        MlirValue::new(result, ty)
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
        MlirValue::new(result, ty)
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
        MlirValue::new(result, ty)
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
                "{result} = stablehlo.dot_general {}, {}, ",
                "batching_dims = [lhs = {}, rhs = {}], ",
                "contracting_dims = [lhs = {}, rhs = {}] : ",
                "({}, {}) -> {ty}"
            ),
            lhs.name,
            rhs.name,
            list(lhs_batching),
            list(rhs_batching),
            list(lhs_contracting),
            list(rhs_contracting),
            lhs.ty,
            rhs.ty,
        ));

        MlirValue::new(result, ty)
    }

    pub fn convert(
        &mut self,
        input: &MlirValue,
        result_type: TensorType,
    ) -> MlirValue {
        let result = self.fresh();
        let ty = tensor_type(result_type);
        self.line(format!(
            "{result} = stablehlo.convert {} : ({}) -> {ty}",
            input.name, input.ty
        ));
        MlirValue::new(result, ty)
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

        MlirValue::new(result, ty)
    }
}

fn dense_i64(values: &[u64]) -> String {
    format!(
        "dense<[{}]> : tensor<{}xi64>",
        list(values),
        values.len()
    )
}

fn list(values: &[u64]) -> String {
    values
        .iter()
        .map(u64::to_string)
        .collect::<Vec<_>>()
        .join(", ")
}
