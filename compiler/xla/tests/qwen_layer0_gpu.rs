use severian_xla::{
    CompileOptions, HostBuffer, PjrtClient, PjrtPlugin, SafeTensorStore,
    StableHloModule, XlaClient,
};
use std::path::PathBuf;

const TOKEN_ID: usize = 42;
const HIDDEN: usize = 2048;

const LAYER0_Q: &str = r#"
module {
  func.func @main(
      %embedding: tensor<151936x2048xbf16>,
      %norm_weight: tensor<2048xbf16>,
      %q_weight: tensor<2048x2048xbf16>,
      %q_bias: tensor<2048xbf16>,
      %indices: tensor<1x1xi64>
  ) -> tensor<1x1x2048xbf16> {
    %hidden = "stablehlo.gather"(%embedding, %indices) {
      dimension_numbers = #stablehlo.gather<offset_dims = [2], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 2>,
      slice_sizes = array<i64: 1, 2048>, indices_are_sorted = false
    } : (tensor<151936x2048xbf16>, tensor<1x1xi64>) -> tensor<1x1x2048xbf16>
    %hidden_f32 = stablehlo.convert %hidden : (tensor<1x1x2048xbf16>) -> tensor<1x1x2048xf32>
    %square = stablehlo.multiply %hidden_f32, %hidden_f32 : tensor<1x1x2048xf32>
    %zero = stablehlo.constant dense<0.0> : tensor<f32>
    %sum = "stablehlo.reduce"(%square, %zero) ({
      ^bb0(%left: tensor<f32>, %right: tensor<f32>):
        %combined = "stablehlo.add"(%left, %right) : (tensor<f32>, tensor<f32>) -> tensor<f32>
        "stablehlo.return"(%combined) : (tensor<f32>) -> ()
    }) {dimensions = array<i64: 2>} : (tensor<1x1x2048xf32>, tensor<f32>) -> tensor<1x1xf32>
    %count_scalar = stablehlo.constant dense<2048.0> : tensor<f32>
    %count = stablehlo.broadcast_in_dim %count_scalar, dims = [] : (tensor<f32>) -> tensor<1x1xf32>
    %variance = stablehlo.divide %sum, %count : tensor<1x1xf32>
    %epsilon_scalar = stablehlo.constant dense<0.000001> : tensor<f32>
    %epsilon = stablehlo.broadcast_in_dim %epsilon_scalar, dims = [] : (tensor<f32>) -> tensor<1x1xf32>
    %stabilized = stablehlo.add %variance, %epsilon : tensor<1x1xf32>
    %inverse = "stablehlo.rsqrt"(%stabilized) : (tensor<1x1xf32>) -> tensor<1x1xf32>
    %inverse_b = stablehlo.broadcast_in_dim %inverse, dims = [0, 1] : (tensor<1x1xf32>) -> tensor<1x1x2048xf32>
    %normalized_f32 = stablehlo.multiply %hidden_f32, %inverse_b : tensor<1x1x2048xf32>
    %normalized = stablehlo.convert %normalized_f32 : (tensor<1x1x2048xf32>) -> tensor<1x1x2048xbf16>
    %norm_weight_b = stablehlo.broadcast_in_dim %norm_weight, dims = [2] : (tensor<2048xbf16>) -> tensor<1x1x2048xbf16>
    %normed = stablehlo.multiply %normalized, %norm_weight_b : tensor<1x1x2048xbf16>
    %q = "stablehlo.dot_general"(%normed, %q_weight) {
      dot_dimension_numbers = #stablehlo.dot<lhs_batching_dimensions = [], rhs_batching_dimensions = [], lhs_contracting_dimensions = [2], rhs_contracting_dimensions = [1]>,
      precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]
    } : (tensor<1x1x2048xbf16>, tensor<2048x2048xbf16>) -> tensor<1x1x2048xbf16>
    %q_bias_b = stablehlo.broadcast_in_dim %q_bias, dims = [2] : (tensor<2048xbf16>) -> tensor<1x1x2048xbf16>
    %result = stablehlo.add %q, %q_bias_b : tensor<1x1x2048xbf16>
    return %result : tensor<1x1x2048xbf16>
  }
}
"#;

fn bf16_to_f32(bytes: &[u8], index: usize) -> f32 {
    let bits = u16::from_ne_bytes([bytes[index * 2], bytes[index * 2 + 1]]);
    f32::from_bits(u32::from(bits) << 16)
}

fn rounded_bf16(value: f32) -> f32 {
    let bits = value.to_bits();
    let rounded = bits.wrapping_add(0x7fff + ((bits >> 16) & 1));
    f32::from_bits(rounded & 0xffff_0000)
}

#[test]
fn real_qwen_layer0_rmsnorm_and_q_projection_execute_on_amd_gpu(
) -> Result<(), Box<dyn std::error::Error>> {
    let model = std::env::var_os("SEVERIAN_QWEN_MODEL")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("benchmarks/inference/models/Qwen2.5-3B-Instruct"));
    let store = SafeTensorStore::open(&model)?;
    let embedding_source = store.get("model.embed_tokens.weight")?;
    let norm_source = store.get("model.layers.0.input_layernorm.weight")?;
    let q_source = store.get("model.layers.0.self_attn.q_proj.weight")?;
    let bias_source = store.get("model.layers.0.self_attn.q_proj.bias")?;

    let embedding_row = &embedding_source.bytes()
        [TOKEN_ID * HIDDEN * 2..(TOKEN_ID + 1) * HIDDEN * 2];
    let hidden = (0..HIDDEN)
        .map(|index| bf16_to_f32(embedding_row, index))
        .collect::<Vec<_>>();
    let variance = hidden.iter().map(|value| value * value).sum::<f32>() / HIDDEN as f32;
    let inverse = (variance + 1e-6).sqrt().recip();
    let normed = (0..HIDDEN)
        .map(|index| {
            let normalized = rounded_bf16(hidden[index] * inverse);
            rounded_bf16(normalized * bf16_to_f32(norm_source.bytes(), index))
        })
        .collect::<Vec<_>>();
    let reference = (0..HIDDEN)
        .map(|output| {
            let row = output * HIDDEN;
            let sum = (0..HIDDEN).fold(0.0f32, |sum, input| {
                sum + normed[input] * bf16_to_f32(q_source.bytes(), row + input)
            });
            rounded_bf16(sum + bf16_to_f32(bias_source.bytes(), output))
        })
        .collect::<Vec<_>>();

    let plugin = PjrtPlugin::load_rocm()?;
    let client = PjrtClient::new(plugin)?;
    let device = client.amd_gpu_device()?;
    println!("AMD device description: {}", device.description);
    let xla = XlaClient::new(client);
    let executable = xla.compile(
        &StableHloModule::from_text(LAYER0_Q),
        &CompileOptions::default(),
    )?;
    let embedding = store.upload_bf16(xla.pjrt(), "model.embed_tokens.weight", Some(&device))?;
    let norm = store.upload_bf16(
        xla.pjrt(), "model.layers.0.input_layernorm.weight", Some(&device),
    )?;
    let q_weight = store.upload_bf16(
        xla.pjrt(), "model.layers.0.self_attn.q_proj.weight", Some(&device),
    )?;
    let q_bias = store.upload_bf16(
        xla.pjrt(), "model.layers.0.self_attn.q_proj.bias", Some(&device),
    )?;
    let indices = xla.upload_to(HostBuffer::from_i64([1, 1], &[TOKEN_ID as i64])?, &device)?;
    for buffer in [&embedding, &norm, &q_weight, &q_bias, &indices] {
        assert!(!buffer.is_on_cpu()?);
        assert!(buffer.is_on_device(&device)?);
    }
    let mut outputs = executable.execute(
        &[&embedding, &norm, &q_weight, &q_bias, &indices],
        &device,
    )?;
    let output = outputs.remove(0);
    assert!(!output.is_on_cpu()?);
    assert!(output.is_on_device(&device)?);
    let bytes = output.to_host_bytes()?;
    let actual = (0..HIDDEN)
        .map(|index| bf16_to_f32(&bytes, index))
        .collect::<Vec<_>>();
    let errors = actual.iter().zip(&reference)
        .map(|(actual, expected)| (actual - expected).abs())
        .collect::<Vec<_>>();
    let max_error = errors.iter().copied().fold(0.0f32, f32::max);
    let mean_error = errors.iter().sum::<f32>() / errors.len() as f32;
    println!("layer0 Q projection max_abs_error={max_error} mean_abs_error={mean_error}");
    assert!(max_error < 0.5, "layer 0 Q max error is {max_error}");
    assert!(mean_error < 0.05, "layer 0 Q mean error is {mean_error}");
    Ok(())
}
