use severian_driver::compile_path;
use severian_lowering::stablehlo::lower_entry;
use severian_xla::{
    Buffer, CompileOptions, HostBuffer, PjrtClient, PjrtPlugin, SafeTensorStore,
    StableHloModule, XlaClient,
};
use std::path::{Path, PathBuf};

const TOKEN_ID: i64 = 42;
const EXPECTED_TOKEN_ID: usize = 25852;

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

fn bf16(bytes: &[u8], index: usize) -> f32 {
    let bits = u16::from_ne_bytes([bytes[index * 2], bytes[index * 2 + 1]]);
    f32::from_bits(u32::from(bits) << 16)
}

#[test]
fn full_qwen_next_token_is_compiled_from_severian_and_executes_on_amd_gpu(
) -> Result<(), Box<dyn std::error::Error>> {
    let workspace = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let compilation = compile_path(
        &workspace.join("benchmarks/inference/severian/qwen_kernels.sev"),
    )?;
    let entry = |name: &str| {
        compilation
            .optimized_hir
            .functions
            .iter()
            .find(|function| function.name == name)
            .unwrap_or_else(|| panic!("missing Severian tensor entry `{name}`"))
            .id
    };
    let embedding_hlo = lower_entry(&compilation.optimized_hir, entry("embeddingKernel"))?;
    let layer_hlo = lower_entry(&compilation.optimized_hir, entry("qwen2LayerKernel"))?;
    let logits_hlo = lower_entry(&compilation.optimized_hir, entry("logitsKernel"))?;

    for operation in [
        "stablehlo.dot_general",
        "stablehlo.reduce",
        "stablehlo.rsqrt",
        "stablehlo.exponential",
        "stablehlo.transpose",
        "stablehlo.broadcast_in_dim",
    ] {
        assert!(layer_hlo.as_str().contains(operation), "missing {operation}");
    }
    assert!(!layer_hlo.as_str().contains("custom_call"));
    assert!(embedding_hlo.as_str().contains("stablehlo.gather"));

    let model = std::env::var_os("SEVERIAN_QWEN_MODEL")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            workspace.join("benchmarks/inference/models/Qwen2.5-3B-Instruct")
        });
    let store = SafeTensorStore::open(model)?;
    let plugin = PjrtPlugin::load_rocm()?;
    let client = PjrtClient::new(plugin)?;
    let device = client.amd_gpu_device()?;
    println!("AMD device description: {}", device.description);
    let xla = XlaClient::new(client);
    let options = CompileOptions::default();
    let embed_exe = xla.compile(
        &StableHloModule::from_text(embedding_hlo.as_str()),
        &options,
    )?;
    let layer_exe = xla.compile(
        &StableHloModule::from_text(layer_hlo.as_str()),
        &options,
    )?;
    let final_exe = xla.compile(
        &StableHloModule::from_text(logits_hlo.as_str()),
        &options,
    )?;

    let embedding =
        store.upload_bf16(xla.pjrt(), "model.embed_tokens.weight", Some(&device))?;
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
            q_weight: upload("self_attn.q_proj.weight")?,
            q_bias: upload("self_attn.q_proj.bias")?,
            k_weight: upload("self_attn.k_proj.weight")?,
            k_bias: upload("self_attn.k_proj.bias")?,
            v_weight: upload("self_attn.v_proj.weight")?,
            v_bias: upload("self_attn.v_proj.bias")?,
            o_weight: upload("self_attn.o_proj.weight")?,
            post_norm: upload("post_attention_layernorm.weight")?,
            gate_weight: upload("mlp.gate_proj.weight")?,
            up_weight: upload("mlp.up_proj.weight")?,
            down_weight: upload("mlp.down_proj.weight")?,
        });
    }
    assert!(!embedding.is_on_cpu()? && embedding.is_on_device(&device)?);
    for layer in &layers {
        for buffer in [
            &layer.input_norm,
            &layer.q_weight,
            &layer.q_bias,
            &layer.k_weight,
            &layer.k_bias,
            &layer.v_weight,
            &layer.v_bias,
            &layer.o_weight,
            &layer.post_norm,
            &layer.gate_weight,
            &layer.up_weight,
            &layer.down_weight,
        ] {
            assert!(!buffer.is_on_cpu()? && buffer.is_on_device(&device)?);
        }
    }
    println!("all 36 layers GPU resident");

    let mut hidden = embed_exe
        .execute(&[&embedding, &ids], &device)?
        .remove(0);
    for (index, layer) in layers.iter().enumerate() {
        hidden = layer_exe
            .execute(
                &[
                    &hidden,
                    &layer.input_norm,
                    &layer.q_weight,
                    &layer.q_bias,
                    &layer.k_weight,
                    &layer.k_bias,
                    &layer.v_weight,
                    &layer.v_bias,
                    &layer.o_weight,
                    &layer.post_norm,
                    &layer.gate_weight,
                    &layer.up_weight,
                    &layer.down_weight,
                ],
                &device,
            )?
            .remove(0);
        assert!(!hidden.is_on_cpu()? && hidden.is_on_device(&device)?);
        println!("executed layer {index}");
    }
    let logits = final_exe
        .execute(&[&hidden, &final_norm, &embedding], &device)?
        .remove(0);
    assert!(!logits.is_on_cpu()? && logits.is_on_device(&device)?);
    let bytes = logits.to_host_bytes()?;
    let (token, value) = (0..151936usize)
        .map(|index| (index, bf16(&bytes, index)))
        .max_by(|left, right| left.1.total_cmp(&right.1))
        .unwrap();
    println!("input_token={TOKEN_ID} next_token={token} max_logit={value}");
    assert!(value.is_finite());
    assert_eq!(token, EXPECTED_TOKEN_ID);
    Ok(())
}
