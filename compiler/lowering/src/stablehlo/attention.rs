use severian_hir::TensorType;

use super::{
    linear::{linear_last_dimension, llama_mlp},
    normalization::{rms_norm, softmax_last_axis},
    MlirValue, StableHloEmitter,
};

#[derive(Debug, Clone, Copy)]
pub struct AttentionTypes {
    pub projected: TensorType,
    pub projected_4d: TensorType,
    pub qkv: TensorType,
    pub key_transposed: TensorType,
    pub scores: TensorType,
    pub reduced_scores: TensorType,
    pub context: TensorType,
    pub context_transposed: TensorType,
    pub merged_context: TensorType,
    pub output: TensorType,
}

/// Full-sequence multi-head self-attention. The mask is an additive mask in
/// score space (zero for visible positions and negative infinity otherwise).
pub fn full_self_attention(
    emitter: &mut StableHloEmitter,
    input: &MlirValue,
    query_weight: &MlirValue,
    key_weight: &MlirValue,
    value_weight: &MlirValue,
    output_weight: &MlirValue,
    causal_mask: &MlirValue,
    types: AttentionTypes,
    head_dimension: u64,
) -> MlirValue {
    let query = linear_last_dimension(emitter, input, query_weight, types.projected);
    let key = linear_last_dimension(emitter, input, key_weight, types.projected);
    let value = linear_last_dimension(emitter, input, value_weight, types.projected);

    let query = emitter.reshape(&query, types.projected_4d);
    let key = emitter.reshape(&key, types.projected_4d);
    let value = emitter.reshape(&value, types.projected_4d);
    let query = emitter.transpose(&query, &[0, 2, 1, 3], types.qkv);
    let key = emitter.transpose(&key, &[0, 2, 1, 3], types.qkv);
    let value = emitter.transpose(&value, &[0, 2, 1, 3], types.qkv);
    let key = emitter.transpose(&key, &[0, 1, 3, 2], types.key_transposed);

    let scores = emitter.dot_general(&query, &key, &[0, 1], &[0, 1], &[3], &[2], types.scores);
    let scale = emitter.splat(&(head_dimension as f64).sqrt().to_string(), types.scores);
    let scores = emitter.divide(&scores, &scale, types.scores);
    let scores = emitter.add(&scores, causal_mask, types.scores);
    let probabilities = softmax_last_axis(emitter, &scores, types.scores, types.reduced_scores);
    let context = emitter.dot_general(
        &probabilities,
        &value,
        &[0, 1],
        &[0, 1],
        &[3],
        &[2],
        types.context,
    );
    let context = emitter.transpose(&context, &[0, 2, 1, 3], types.context_transposed);
    let context = emitter.reshape(&context, types.merged_context);
    linear_last_dimension(emitter, &context, output_weight, types.output)
}

#[derive(Debug, Clone, Copy)]
pub struct TransformerBlockTypes {
    pub model: TensorType,
    pub reduced_model: TensorType,
    pub attention: AttentionTypes,
    pub mlp_intermediate: TensorType,
}

/// A pre-norm Llama transformer block: RMSNorm, attention, residual,
/// RMSNorm, SwiGLU MLP, residual. Every operation is ordinary StableHLO.
#[allow(clippy::too_many_arguments)]
pub fn llama_transformer_block(
    emitter: &mut StableHloEmitter,
    input: &MlirValue,
    attention_norm_weight: &MlirValue,
    query_weight: &MlirValue,
    key_weight: &MlirValue,
    value_weight: &MlirValue,
    attention_output_weight: &MlirValue,
    causal_mask: &MlirValue,
    ffn_norm_weight: &MlirValue,
    gate_weight: &MlirValue,
    up_weight: &MlirValue,
    down_weight: &MlirValue,
    types: TransformerBlockTypes,
    hidden_size: u64,
    head_dimension: u64,
    epsilon: f64,
) -> MlirValue {
    let normalized = rms_norm(
        emitter,
        input,
        attention_norm_weight,
        types.model,
        types.reduced_model,
        hidden_size,
        epsilon,
    );
    let attention = full_self_attention(
        emitter,
        &normalized,
        query_weight,
        key_weight,
        value_weight,
        attention_output_weight,
        causal_mask,
        types.attention,
        head_dimension,
    );
    let attention_residual = emitter.add(input, &attention, types.model);
    let normalized = rms_norm(
        emitter,
        &attention_residual,
        ffn_norm_weight,
        types.model,
        types.reduced_model,
        hidden_size,
        epsilon,
    );
    let mlp = llama_mlp(
        emitter,
        &normalized,
        gate_weight,
        up_weight,
        down_weight,
        types.mlp_intermediate,
        types.model,
    );
    emitter.add(&attention_residual, &mlp, types.model)
}
