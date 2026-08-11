use severian_driver::compile_path;
use severian_lowering::stablehlo::lower_entry;
use severian_xla::{
    Buffer, CompileOptions, HostBuffer, PjrtClient, PjrtPlugin, SafeTensorStore, StableHloModule,
    XlaClient,
};
use std::path::{Path, PathBuf};

const PROMPT_IDS: [i64; 5] = [49, 19696, 525, 2518, 11];
const EXPECTED_TOKEN_ID: usize = 348;
const PREFILL_CAPACITY: usize = 32;

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

fn rope_values(cosine: bool) -> Vec<f32> {
    let mut values = Vec::with_capacity(PREFILL_CAPACITY * 128);
    for position in 0..PREFILL_CAPACITY {
        let frequencies = (0..64)
            .map(|index| {
                let angle = position as f32 * 1_000_000.0_f32.powf(-(index as f32) / 64.0);
                if cosine {
                    angle.cos()
                } else {
                    angle.sin()
                }
            })
            .collect::<Vec<_>>();
        values.extend_from_slice(&frequencies);
        values.extend_from_slice(&frequencies);
    }
    values
}

#[test]
fn full_qwen_next_token_is_compiled_from_severian_and_executes_on_amd_gpu(
) -> Result<(), Box<dyn std::error::Error>> {
    let workspace = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let compilation =
        compile_path(&workspace.join("benchmarks/inference/severian/qwen_kernels.sev"))?;
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
        assert!(
            layer_hlo.as_str().contains(operation),
            "missing {operation}"
        );
    }
    assert!(!layer_hlo.as_str().contains("custom_call"));
    assert!(embedding_hlo.as_str().contains("stablehlo.gather"));

    let model = std::env::var_os("SEVERIAN_QWEN_MODEL")
        .map(PathBuf::from)
        .unwrap_or_else(|| workspace.join("benchmarks/inference/models/Qwen2.5-3B-Instruct"));
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
    let layer_exe = xla.compile(&StableHloModule::from_text(layer_hlo.as_str()), &options)?;
    let final_exe = xla.compile(&StableHloModule::from_text(logits_hlo.as_str()), &options)?;

    let embedding = store.upload_bf16(xla.pjrt(), "model.embed_tokens.weight", Some(&device))?;
    let final_norm = store.upload_bf16(xla.pjrt(), "model.norm.weight", Some(&device))?;
    let mut padded_ids = PROMPT_IDS.to_vec();
    padded_ids.resize(PREFILL_CAPACITY, 0);
    let ids = xla.upload_to(
        HostBuffer::from_i64([1, PREFILL_CAPACITY as i64], &padded_ids)?,
        &device,
    )?;
    let cosine = xla.upload_to(
        HostBuffer::from_f32([1, 1, PREFILL_CAPACITY as i64, 128], &rope_values(true))?,
        &device,
    )?;
    let sine = xla.upload_to(
        HostBuffer::from_f32([1, 1, PREFILL_CAPACITY as i64, 128], &rope_values(false))?,
        &device,
    )?;
    let mask_values = (0..PREFILL_CAPACITY)
        .flat_map(|query| {
            (0..PREFILL_CAPACITY).map(move |key| {
                if query < PROMPT_IDS.len() && key <= query {
                    0.0
                } else {
                    -1.0e30
                }
            })
        })
        .collect::<Vec<_>>();
    let mask = xla.upload_to(
        HostBuffer::from_f32(
            [1, 1, PREFILL_CAPACITY as i64, PREFILL_CAPACITY as i64],
            &mask_values,
        )?,
        &device,
    )?;
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

    let mut hidden = embed_exe.execute(&[&embedding, &ids], &device)?.remove(0);
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
                    &cosine,
                    &sine,
                    &mask,
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
    let final_row = (PROMPT_IDS.len() - 1) * 151936;
    let (token, value) = (0..151936usize)
        .map(|index| (index, bf16(&bytes, final_row + index)))
        .max_by(|left, right| left.1.total_cmp(&right.1))
        .unwrap();
    println!("prompt_ids={PROMPT_IDS:?} next_token={token} max_logit={value}");
    assert!(value.is_finite());
    assert_eq!(token, EXPECTED_TOKEN_ID);
    Ok(())
}
