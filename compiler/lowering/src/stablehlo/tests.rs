use super::*;
use crate::stablehlo::{
    attention::{pre_norm_swiglu_transformer_block, AttentionTypes, TransformerBlockTypes},
    normalization::{last_axis_reduced_type, reduced_axis_type, softmax_axis, softmax_last_axis},
};
use severian_hir::{TensorDimension, TensorElementType};

fn tensor(rank: u8) -> TensorType {
    TensorType {
        element: TensorElementType::F32,
        rank: Some(rank),
        dimensions: [TensorDimension::Dynamic; 8],
    }
}

#[test]
fn softmax_is_a_stablehlo_reduction_graph() {
    let input_type = tensor(4);
    let reduced_type = last_axis_reduced_type(input_type).unwrap();
    let input = argument("%input", input_type);
    let mut emitter = StableHloEmitter::new();
    softmax_last_axis(&mut emitter, &input, input_type, reduced_type);
    let text = emitter.as_str();
    assert!(text.contains("stablehlo.reduce"));
    assert!(text.contains("stablehlo.maximum"));
    assert!(text.contains("stablehlo.exponential"));
    assert!(text.contains("stablehlo.divide"));
    assert!(!text.contains("custom_call"));
}

#[test]
fn softmax_axis_preserves_every_non_reduced_dimension() {
    let input_type = tensor(3);
    let reduced_type = reduced_axis_type(input_type, 1).unwrap();
    let input = argument("%input", input_type);
    let mut emitter = StableHloEmitter::new();
    softmax_axis(&mut emitter, &input, input_type, reduced_type, 1);
    let text = emitter.as_str();
    assert!(text.contains("dimensions = array<i64: 1>"));
    assert!(text.contains("dims = [0, 2]"));
    assert!(!text.contains("custom_call"));
}

#[test]
fn pre_norm_swiglu_block_contains_attention_mlp_norms_and_residuals() {
    let model = tensor(3);
    let reduced_model = tensor(2);
    let weight = tensor(2);
    let norm_weight = tensor(1);
    let mask = argument("%mask", tensor(4));
    let input = argument("%input", model);
    let norm_a = argument("%norm_a", norm_weight);
    let norm_b = argument("%norm_b", norm_weight);
    let wq = argument("%wq", weight);
    let wk = argument("%wk", weight);
    let wv = argument("%wv", weight);
    let wo = argument("%wo", weight);
    let gate = argument("%gate", weight);
    let up = argument("%up", weight);
    let down = argument("%down", weight);
    let types = TransformerBlockTypes {
        model,
        reduced_model,
        attention: AttentionTypes {
            projected: tensor(3),
            projected_4d: tensor(4),
            qkv: tensor(4),
            key_transposed: tensor(4),
            scores: tensor(4),
            reduced_scores: tensor(3),
            context: tensor(4),
            context_transposed: tensor(4),
            merged_context: tensor(3),
            output: model,
        },
        mlp_intermediate: tensor(3),
    };
    let mut emitter = StableHloEmitter::new();
    pre_norm_swiglu_transformer_block(
        &mut emitter,
        &input,
        &norm_a,
        &wq,
        &wk,
        &wv,
        &wo,
        &mask,
        &norm_b,
        &gate,
        &up,
        &down,
        types,
        4096,
        128,
        1e-5,
    );
    let text = emitter.as_str();
    assert_eq!(text.matches("stablehlo.dot_general").count(), 9);
    assert!(text.matches("stablehlo.reduce").count() >= 4);
    assert!(text.contains("stablehlo.rsqrt"));
    assert!(text.contains("stablehlo.logistic"));
    assert!(!text.contains("custom_call"));
}
