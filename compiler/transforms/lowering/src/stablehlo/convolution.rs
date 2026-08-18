use severian_hir::TensorType;

use super::{ops::list, MlirValue, StableHloEmitter};
use crate::tensor::tensor_type;

/// Emits a two-dimensional NCHW/OIHW convolution. Group counts are explicit,
/// so depthwise and grouped convolution use the same StableHLO operation.
#[allow(clippy::too_many_arguments)]
pub fn conv2d_nchw(
    emitter: &mut StableHloEmitter,
    input: &MlirValue,
    kernel: &MlirValue,
    strides: [u64; 2],
    lhs_dilation: [u64; 2],
    rhs_dilation: [u64; 2],
    padding_low_high: [[u64; 2]; 2],
    feature_group_count: u64,
    batch_group_count: u64,
    result_type: TensorType,
) -> MlirValue {
    let result = emitter.fresh();
    let result_ty = tensor_type(result_type);
    let padding = format!(
        "dense<[[{}, {}], [{}, {}]]> : tensor<2x2xi64>",
        padding_low_high[0][0],
        padding_low_high[0][1],
        padding_low_high[1][0],
        padding_low_high[1][1],
    );
    emitter.line(format!(
        concat!(
            "{result} = \"stablehlo.convolution\"({}, {}) {{",
            "window_strides = array<i64: {}>, padding = {}, ",
            "lhs_dilation = array<i64: {}>, rhs_dilation = array<i64: {}>, ",
            "window_reversal = array<i1: false, false>, ",
            "dimension_numbers = #stablehlo.conv<[b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1]>, ",
            "feature_group_count = {} : i64, batch_group_count = {} : i64, ",
            "precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]",
            "}} : ({}, {}) -> {result_ty}"
        ),
        input.name,
        kernel.name,
        list(&strides),
        padding,
        list(&lhs_dilation),
        list(&rhs_dilation),
        feature_group_count,
        batch_group_count,
        input.ty,
        kernel.ty,
        result = result,
        result_ty = result_ty,
    ));
    MlirValue::from_tensor(result, result_type)
}
