use severian_xla::{
    Buffer, CompileOptions, HostBuffer, PjrtClient, PjrtPlugin, SafeTensorStore,
    StableHloModule, XlaClient,
};
use std::path::PathBuf;

const TOKEN_ID: i64 = 42;

struct LayerBuffers {
    input_norm: Buffer,
    q_weight: Buffer,
    q_bias: Buffer,
    k_weight: Buffer,
    k_bias: Buffer,
    v_weight: Buffer,
    v_bias: Buffer,
    o_weight: Buffer,
    post_norm: Buffer,
    gate_weight: Buffer,
    up_weight: Buffer,
    down_weight: Buffer,
}

fn layer_module() -> &'static str {
    r#"
module {
  func.func @main(
    %hidden: tensor<1x1x2048xbf16>, %input_norm: tensor<2048xbf16>,
    %qw: tensor<2048x2048xbf16>, %qb: tensor<2048xbf16>,
    %kw: tensor<256x2048xbf16>, %kb: tensor<256xbf16>,
    %vw: tensor<256x2048xbf16>, %vb: tensor<256xbf16>,
    %ow: tensor<2048x2048xbf16>, %post_norm: tensor<2048xbf16>,
    %gatew: tensor<11008x2048xbf16>, %upw: tensor<11008x2048xbf16>,
    %downw: tensor<2048x11008xbf16>
  ) -> tensor<1x1x2048xbf16> {
    %h32 = stablehlo.convert %hidden : (tensor<1x1x2048xbf16>) -> tensor<1x1x2048xf32>
    %sq = stablehlo.multiply %h32, %h32 : tensor<1x1x2048xf32>
    %z = stablehlo.constant dense<0.0> : tensor<f32>
    %sum = "stablehlo.reduce"(%sq, %z) ({
      ^bb0(%a: tensor<f32>, %b: tensor<f32>):
        %c = "stablehlo.add"(%a, %b) : (tensor<f32>, tensor<f32>) -> tensor<f32>
        "stablehlo.return"(%c) : (tensor<f32>) -> ()
    }) {dimensions = array<i64: 2>} : (tensor<1x1x2048xf32>, tensor<f32>) -> tensor<1x1xf32>
    %count0 = stablehlo.constant dense<2048.0> : tensor<f32>
    %count = stablehlo.broadcast_in_dim %count0, dims = [] : (tensor<f32>) -> tensor<1x1xf32>
    %var = stablehlo.divide %sum, %count : tensor<1x1xf32>
    %eps0 = stablehlo.constant dense<0.000001> : tensor<f32>
    %eps = stablehlo.broadcast_in_dim %eps0, dims = [] : (tensor<f32>) -> tensor<1x1xf32>
    %vare = stablehlo.add %var, %eps : tensor<1x1xf32>
    %inv = "stablehlo.rsqrt"(%vare) : (tensor<1x1xf32>) -> tensor<1x1xf32>
    %invb = stablehlo.broadcast_in_dim %inv, dims = [0, 1] : (tensor<1x1xf32>) -> tensor<1x1x2048xf32>
    %n32 = stablehlo.multiply %h32, %invb : tensor<1x1x2048xf32>
    %n16a = stablehlo.convert %n32 : (tensor<1x1x2048xf32>) -> tensor<1x1x2048xbf16>
    %inwb = stablehlo.broadcast_in_dim %input_norm, dims = [2] : (tensor<2048xbf16>) -> tensor<1x1x2048xbf16>
    %n16 = stablehlo.multiply %n16a, %inwb : tensor<1x1x2048xbf16>

    %q0 = "stablehlo.dot_general"(%n16, %qw) {dot_dimension_numbers = #stablehlo.dot<lhs_batching_dimensions = [], rhs_batching_dimensions = [], lhs_contracting_dimensions = [2], rhs_contracting_dimensions = [1]>} : (tensor<1x1x2048xbf16>, tensor<2048x2048xbf16>) -> tensor<1x1x2048xbf16>
    %k0 = "stablehlo.dot_general"(%n16, %kw) {dot_dimension_numbers = #stablehlo.dot<lhs_batching_dimensions = [], rhs_batching_dimensions = [], lhs_contracting_dimensions = [2], rhs_contracting_dimensions = [1]>} : (tensor<1x1x2048xbf16>, tensor<256x2048xbf16>) -> tensor<1x1x256xbf16>
    %v0 = "stablehlo.dot_general"(%n16, %vw) {dot_dimension_numbers = #stablehlo.dot<lhs_batching_dimensions = [], rhs_batching_dimensions = [], lhs_contracting_dimensions = [2], rhs_contracting_dimensions = [1]>} : (tensor<1x1x2048xbf16>, tensor<256x2048xbf16>) -> tensor<1x1x256xbf16>
    %qbb = stablehlo.broadcast_in_dim %qb, dims = [2] : (tensor<2048xbf16>) -> tensor<1x1x2048xbf16>
    %kbb = stablehlo.broadcast_in_dim %kb, dims = [2] : (tensor<256xbf16>) -> tensor<1x1x256xbf16>
    %vbb = stablehlo.broadcast_in_dim %vb, dims = [2] : (tensor<256xbf16>) -> tensor<1x1x256xbf16>
    %q1 = stablehlo.add %q0, %qbb : tensor<1x1x2048xbf16>
    %k1 = stablehlo.add %k0, %kbb : tensor<1x1x256xbf16>
    %v1 = stablehlo.add %v0, %vbb : tensor<1x1x256xbf16>
    %qr = stablehlo.reshape %q1 : (tensor<1x1x2048xbf16>) -> tensor<1x1x16x128xbf16>
    %kr = stablehlo.reshape %k1 : (tensor<1x1x256xbf16>) -> tensor<1x1x2x128xbf16>
    %vr = stablehlo.reshape %v1 : (tensor<1x1x256xbf16>) -> tensor<1x1x2x128xbf16>
    %qt = stablehlo.transpose %qr, dims = [0, 2, 1, 3] : (tensor<1x1x16x128xbf16>) -> tensor<1x16x1x128xbf16>
    %kt = stablehlo.transpose %kr, dims = [0, 2, 1, 3] : (tensor<1x1x2x128xbf16>) -> tensor<1x2x1x128xbf16>
    %vt = stablehlo.transpose %vr, dims = [0, 2, 1, 3] : (tensor<1x1x2x128xbf16>) -> tensor<1x2x1x128xbf16>

    // Position zero RoPE: emit the ordinary f32 graph even though cos=1/sin=0.
    %qf = stablehlo.convert %qt : (tensor<1x16x1x128xbf16>) -> tensor<1x16x1x128xf32>
    %kf = stablehlo.convert %kt : (tensor<1x2x1x128xbf16>) -> tensor<1x2x1x128xf32>
    %one = stablehlo.constant dense<1.0> : tensor<f32>
    %zero = stablehlo.constant dense<0.0> : tensor<f32>
    %qcos = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<1x16x1x128xf32>
    %qsin = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<1x16x1x128xf32>
    %kcos = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<1x2x1x128xf32>
    %ksin = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<1x2x1x128xf32>
    %qd = stablehlo.multiply %qf, %qcos : tensor<1x16x1x128xf32>
    %qz = stablehlo.multiply %qf, %qsin : tensor<1x16x1x128xf32>
    %qrope = stablehlo.add %qd, %qz : tensor<1x16x1x128xf32>
    %kd = stablehlo.multiply %kf, %kcos : tensor<1x2x1x128xf32>
    %kz = stablehlo.multiply %kf, %ksin : tensor<1x2x1x128xf32>
    %krope = stablehlo.add %kd, %kz : tensor<1x2x1x128xf32>

    %ke = stablehlo.reshape %krope : (tensor<1x2x1x128xf32>) -> tensor<1x2x1x1x128xf32>
    %ve = stablehlo.reshape %vt : (tensor<1x2x1x128xbf16>) -> tensor<1x2x1x1x128xbf16>
    %keb = stablehlo.broadcast_in_dim %ke, dims = [0, 1, 2, 3, 4] : (tensor<1x2x1x1x128xf32>) -> tensor<1x2x8x1x128xf32>
    %veb = stablehlo.broadcast_in_dim %ve, dims = [0, 1, 2, 3, 4] : (tensor<1x2x1x1x128xbf16>) -> tensor<1x2x8x1x128xbf16>
    %kg = stablehlo.reshape %keb : (tensor<1x2x8x1x128xf32>) -> tensor<1x16x1x128xf32>
    %vg16 = stablehlo.reshape %veb : (tensor<1x2x8x1x128xbf16>) -> tensor<1x16x1x128xbf16>
    %vg = stablehlo.convert %vg16 : (tensor<1x16x1x128xbf16>) -> tensor<1x16x1x128xf32>
    %ktt = stablehlo.transpose %kg, dims = [0, 1, 3, 2] : (tensor<1x16x1x128xf32>) -> tensor<1x16x128x1xf32>
    %scores0 = "stablehlo.dot_general"(%qrope, %ktt) {dot_dimension_numbers = #stablehlo.dot<lhs_batching_dimensions = [0, 1], rhs_batching_dimensions = [0, 1], lhs_contracting_dimensions = [3], rhs_contracting_dimensions = [2]>} : (tensor<1x16x1x128xf32>, tensor<1x16x128x1xf32>) -> tensor<1x16x1x1xf32>
    %scale0 = stablehlo.constant dense<0.08838834764831845> : tensor<f32>
    %scale = stablehlo.broadcast_in_dim %scale0, dims = [] : (tensor<f32>) -> tensor<1x16x1x1xf32>
    %scores = stablehlo.multiply %scores0, %scale : tensor<1x16x1x1xf32>
    %neg_inf = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %mx = "stablehlo.reduce"(%scores, %neg_inf) ({
      ^bb0(%a: tensor<f32>, %b: tensor<f32>):
        %c = "stablehlo.maximum"(%a, %b) : (tensor<f32>, tensor<f32>) -> tensor<f32>
        "stablehlo.return"(%c) : (tensor<f32>) -> ()
    }) {dimensions = array<i64: 3>} : (tensor<1x16x1x1xf32>, tensor<f32>) -> tensor<1x16x1xf32>
    %mxb = stablehlo.broadcast_in_dim %mx, dims = [0, 1, 2] : (tensor<1x16x1xf32>) -> tensor<1x16x1x1xf32>
    %shift = stablehlo.subtract %scores, %mxb : tensor<1x16x1x1xf32>
    %ex = "stablehlo.exponential"(%shift) : (tensor<1x16x1x1xf32>) -> tensor<1x16x1x1xf32>
    %den = "stablehlo.reduce"(%ex, %z) ({
      ^bb0(%a: tensor<f32>, %b: tensor<f32>):
        %c = "stablehlo.add"(%a, %b) : (tensor<f32>, tensor<f32>) -> tensor<f32>
        "stablehlo.return"(%c) : (tensor<f32>) -> ()
    }) {dimensions = array<i64: 3>} : (tensor<1x16x1x1xf32>, tensor<f32>) -> tensor<1x16x1xf32>
    %denb = stablehlo.broadcast_in_dim %den, dims = [0, 1, 2] : (tensor<1x16x1xf32>) -> tensor<1x16x1x1xf32>
    %prob = stablehlo.divide %ex, %denb : tensor<1x16x1x1xf32>
    %ctx = "stablehlo.dot_general"(%prob, %vg) {dot_dimension_numbers = #stablehlo.dot<lhs_batching_dimensions = [0, 1], rhs_batching_dimensions = [0, 1], lhs_contracting_dimensions = [3], rhs_contracting_dimensions = [2]>} : (tensor<1x16x1x1xf32>, tensor<1x16x1x128xf32>) -> tensor<1x16x1x128xf32>
    %ctx16 = stablehlo.convert %ctx : (tensor<1x16x1x128xf32>) -> tensor<1x16x1x128xbf16>
    %ct = stablehlo.transpose %ctx16, dims = [0, 2, 1, 3] : (tensor<1x16x1x128xbf16>) -> tensor<1x1x16x128xbf16>
    %cm = stablehlo.reshape %ct : (tensor<1x1x16x128xbf16>) -> tensor<1x1x2048xbf16>
    %attn = "stablehlo.dot_general"(%cm, %ow) {dot_dimension_numbers = #stablehlo.dot<lhs_batching_dimensions = [], rhs_batching_dimensions = [], lhs_contracting_dimensions = [2], rhs_contracting_dimensions = [1]>} : (tensor<1x1x2048xbf16>, tensor<2048x2048xbf16>) -> tensor<1x1x2048xbf16>
    %res1 = stablehlo.add %hidden, %attn : tensor<1x1x2048xbf16>

    %r32 = stablehlo.convert %res1 : (tensor<1x1x2048xbf16>) -> tensor<1x1x2048xf32>
    %rsq = stablehlo.multiply %r32, %r32 : tensor<1x1x2048xf32>
    %rsum = "stablehlo.reduce"(%rsq, %z) ({
      ^bb0(%a: tensor<f32>, %b: tensor<f32>):
        %c = "stablehlo.add"(%a, %b) : (tensor<f32>, tensor<f32>) -> tensor<f32>
        "stablehlo.return"(%c) : (tensor<f32>) -> ()
    }) {dimensions = array<i64: 2>} : (tensor<1x1x2048xf32>, tensor<f32>) -> tensor<1x1xf32>
    %rvar = stablehlo.divide %rsum, %count : tensor<1x1xf32>
    %rvare = stablehlo.add %rvar, %eps : tensor<1x1xf32>
    %rinv = "stablehlo.rsqrt"(%rvare) : (tensor<1x1xf32>) -> tensor<1x1xf32>
    %rinvb = stablehlo.broadcast_in_dim %rinv, dims = [0, 1] : (tensor<1x1xf32>) -> tensor<1x1x2048xf32>
    %rn32 = stablehlo.multiply %r32, %rinvb : tensor<1x1x2048xf32>
    %rn16a = stablehlo.convert %rn32 : (tensor<1x1x2048xf32>) -> tensor<1x1x2048xbf16>
    %pnw = stablehlo.broadcast_in_dim %post_norm, dims = [2] : (tensor<2048xbf16>) -> tensor<1x1x2048xbf16>
    %rn16 = stablehlo.multiply %rn16a, %pnw : tensor<1x1x2048xbf16>
    %gate = "stablehlo.dot_general"(%rn16, %gatew) {dot_dimension_numbers = #stablehlo.dot<lhs_batching_dimensions = [], rhs_batching_dimensions = [], lhs_contracting_dimensions = [2], rhs_contracting_dimensions = [1]>} : (tensor<1x1x2048xbf16>, tensor<11008x2048xbf16>) -> tensor<1x1x11008xbf16>
    %up = "stablehlo.dot_general"(%rn16, %upw) {dot_dimension_numbers = #stablehlo.dot<lhs_batching_dimensions = [], rhs_batching_dimensions = [], lhs_contracting_dimensions = [2], rhs_contracting_dimensions = [1]>} : (tensor<1x1x2048xbf16>, tensor<11008x2048xbf16>) -> tensor<1x1x11008xbf16>
    %sig = "stablehlo.logistic"(%gate) : (tensor<1x1x11008xbf16>) -> tensor<1x1x11008xbf16>
    %silu = stablehlo.multiply %gate, %sig : tensor<1x1x11008xbf16>
    %gated = stablehlo.multiply %silu, %up : tensor<1x1x11008xbf16>
    %mlp = "stablehlo.dot_general"(%gated, %downw) {dot_dimension_numbers = #stablehlo.dot<lhs_batching_dimensions = [], rhs_batching_dimensions = [], lhs_contracting_dimensions = [2], rhs_contracting_dimensions = [1]>} : (tensor<1x1x11008xbf16>, tensor<2048x11008xbf16>) -> tensor<1x1x2048xbf16>
    %result = stablehlo.add %res1, %mlp : tensor<1x1x2048xbf16>
    return %result : tensor<1x1x2048xbf16>
  }
}
"#
}

const EMBED: &str = r#"
module { func.func @main(%table: tensor<151936x2048xbf16>, %ids: tensor<1x1xi64>) -> tensor<1x1x2048xbf16> {
  %r = "stablehlo.gather"(%table, %ids) {dimension_numbers = #stablehlo.gather<offset_dims = [2], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 2>, slice_sizes = array<i64: 1, 2048>, indices_are_sorted = false} : (tensor<151936x2048xbf16>, tensor<1x1xi64>) -> tensor<1x1x2048xbf16>
  return %r : tensor<1x1x2048xbf16>
} }
"#;

const FINAL_LOGITS: &str = r#"
module { func.func @main(%hidden: tensor<1x1x2048xbf16>, %norm: tensor<2048xbf16>, %embedding: tensor<151936x2048xbf16>) -> tensor<1x1x151936xbf16> {
  %h32 = stablehlo.convert %hidden : (tensor<1x1x2048xbf16>) -> tensor<1x1x2048xf32>
  %sq = stablehlo.multiply %h32, %h32 : tensor<1x1x2048xf32>
  %z = stablehlo.constant dense<0.0> : tensor<f32>
  %sum = "stablehlo.reduce"(%sq, %z) ({ ^bb0(%a: tensor<f32>, %b: tensor<f32>): %c = "stablehlo.add"(%a, %b) : (tensor<f32>, tensor<f32>) -> tensor<f32> "stablehlo.return"(%c) : (tensor<f32>) -> () }) {dimensions = array<i64: 2>} : (tensor<1x1x2048xf32>, tensor<f32>) -> tensor<1x1xf32>
  %c0 = stablehlo.constant dense<2048.0> : tensor<f32>
  %c = stablehlo.broadcast_in_dim %c0, dims = [] : (tensor<f32>) -> tensor<1x1xf32>
  %v = stablehlo.divide %sum, %c : tensor<1x1xf32>
  %e0 = stablehlo.constant dense<0.000001> : tensor<f32>
  %e = stablehlo.broadcast_in_dim %e0, dims = [] : (tensor<f32>) -> tensor<1x1xf32>
  %ve = stablehlo.add %v, %e : tensor<1x1xf32>
  %i = "stablehlo.rsqrt"(%ve) : (tensor<1x1xf32>) -> tensor<1x1xf32>
  %ib = stablehlo.broadcast_in_dim %i, dims = [0, 1] : (tensor<1x1xf32>) -> tensor<1x1x2048xf32>
  %n32 = stablehlo.multiply %h32, %ib : tensor<1x1x2048xf32>
  %n16a = stablehlo.convert %n32 : (tensor<1x1x2048xf32>) -> tensor<1x1x2048xbf16>
  %nw = stablehlo.broadcast_in_dim %norm, dims = [2] : (tensor<2048xbf16>) -> tensor<1x1x2048xbf16>
  %n16 = stablehlo.multiply %n16a, %nw : tensor<1x1x2048xbf16>
  %logits = "stablehlo.dot_general"(%n16, %embedding) {dot_dimension_numbers = #stablehlo.dot<lhs_batching_dimensions = [], rhs_batching_dimensions = [], lhs_contracting_dimensions = [2], rhs_contracting_dimensions = [1]>} : (tensor<1x1x2048xbf16>, tensor<151936x2048xbf16>) -> tensor<1x1x151936xbf16>
  return %logits : tensor<1x1x151936xbf16>
} }
"#;

fn bf16(bytes: &[u8], index: usize) -> f32 {
    let bits = u16::from_ne_bytes([bytes[index * 2], bytes[index * 2 + 1]]);
    f32::from_bits(u32::from(bits) << 16)
}

#[test]
fn full_qwen_next_token_executes_on_amd_gpu() -> Result<(), Box<dyn std::error::Error>> {
    let model = std::env::var_os("SEVERIAN_QWEN_MODEL").map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("benchmarks/inference/models/Qwen2.5-3B-Instruct"));
    let store = SafeTensorStore::open(model)?;
    let plugin = PjrtPlugin::load_rocm()?;
    let client = PjrtClient::new(plugin)?;
    let device = client.amd_gpu_device()?;
    println!("AMD device description: {}", device.description);
    let xla = XlaClient::new(client);
    let embed_exe = xla.compile(&StableHloModule::from_text(EMBED), &CompileOptions::default())?;
    let layer_exe = xla.compile(&StableHloModule::from_text(layer_module()), &CompileOptions::default())?;
    let final_exe = xla.compile(&StableHloModule::from_text(FINAL_LOGITS), &CompileOptions::default())?;

    let embedding = store.upload_bf16(xla.pjrt(), "model.embed_tokens.weight", Some(&device))?;
    let final_norm = store.upload_bf16(xla.pjrt(), "model.norm.weight", Some(&device))?;
    let ids = xla.upload_to(HostBuffer::from_i64([1, 1], &[TOKEN_ID])?, &device)?;
    let mut layers = Vec::with_capacity(36);
    for layer in 0..36 {
        let prefix = format!("model.layers.{layer}");
        let upload = |suffix: &str| {
            store.upload_bf16(xla.pjrt(), &format!("{prefix}.{suffix}"), Some(&device))
        };
        layers.push(LayerBuffers {
            input_norm: upload("input_layernorm.weight")?,
            q_weight: upload("self_attn.q_proj.weight")?, q_bias: upload("self_attn.q_proj.bias")?,
            k_weight: upload("self_attn.k_proj.weight")?, k_bias: upload("self_attn.k_proj.bias")?,
            v_weight: upload("self_attn.v_proj.weight")?, v_bias: upload("self_attn.v_proj.bias")?,
            o_weight: upload("self_attn.o_proj.weight")?,
            post_norm: upload("post_attention_layernorm.weight")?,
            gate_weight: upload("mlp.gate_proj.weight")?, up_weight: upload("mlp.up_proj.weight")?,
            down_weight: upload("mlp.down_proj.weight")?,
        });
        println!("uploaded layer {layer}");
    }
    assert!(!embedding.is_on_cpu()? && embedding.is_on_device(&device)?);
    for layer in &layers {
        for buffer in [&layer.input_norm, &layer.q_weight, &layer.q_bias, &layer.k_weight,
            &layer.k_bias, &layer.v_weight, &layer.v_bias, &layer.o_weight, &layer.post_norm,
            &layer.gate_weight, &layer.up_weight, &layer.down_weight] {
            assert!(!buffer.is_on_cpu()? && buffer.is_on_device(&device)?);
        }
    }
    println!("all 36 layers GPU resident");

    let mut hidden = embed_exe.execute(&[&embedding, &ids], &device)?.remove(0);
    for (index, layer) in layers.iter().enumerate() {
        hidden = layer_exe.execute(&[&hidden, &layer.input_norm, &layer.q_weight, &layer.q_bias,
            &layer.k_weight, &layer.k_bias, &layer.v_weight, &layer.v_bias, &layer.o_weight,
            &layer.post_norm, &layer.gate_weight, &layer.up_weight, &layer.down_weight], &device)?.remove(0);
        assert!(!hidden.is_on_cpu()? && hidden.is_on_device(&device)?);
        println!("executed layer {index}");
    }
    let logits = final_exe.execute(&[&hidden, &final_norm, &embedding], &device)?.remove(0);
    assert!(!logits.is_on_cpu()? && logits.is_on_device(&device)?);
    let bytes = logits.to_host_bytes()?;
    let (token, value) = (0..151936usize)
        .map(|index| (index, bf16(&bytes, index)))
        .max_by(|left, right| left.1.total_cmp(&right.1))
        .unwrap();
    println!("input_token={TOKEN_ID} next_token={token} max_logit={value}");
    assert!(value.is_finite());
    Ok(())
}
