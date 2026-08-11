use severian_xla::{
    CompileOptions, HostBuffer, PjrtClient, PjrtPlugin, SafeTensorStore,
    StableHloModule, XlaClient,
};
use std::path::PathBuf;

const TOKEN_ID: usize = 42;
const HIDDEN_SIZE: usize = 2048;

const EMBEDDING_GATHER: &str = r#"
module {
  func.func @main(
      %table: tensor<151936x2048xbf16>,
      %indices: tensor<1x1xi64>
  ) -> tensor<1x1x2048xbf16> {
    %result = "stablehlo.gather"(%table, %indices) {
      dimension_numbers = #stablehlo.gather<
        offset_dims = [2],
        collapsed_slice_dims = [0],
        start_index_map = [0],
        index_vector_dim = 2>,
      slice_sizes = array<i64: 1, 2048>,
      indices_are_sorted = false
    } : (tensor<151936x2048xbf16>, tensor<1x1xi64>) -> tensor<1x1x2048xbf16>
    return %result : tensor<1x1x2048xbf16>
  }
}
"#;

#[test]
fn real_qwen_embedding_is_gathered_on_amd_gpu() -> Result<(), Box<dyn std::error::Error>> {
    let model = std::env::var_os("SEVERIAN_QWEN_MODEL")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            PathBuf::from("benchmarks/inference/models/Qwen2.5-3B-Instruct")
        });
    let store = SafeTensorStore::open(&model)?;
    let source = store.get("model.embed_tokens.weight")?;
    assert_eq!(source.entry().shape, [151936, HIDDEN_SIZE as i64]);

    let row_bytes = HIDDEN_SIZE * 2;
    let row_start = TOKEN_ID * row_bytes;
    let expected = source.bytes()[row_start..row_start + row_bytes].to_vec();

    let plugin = PjrtPlugin::load_rocm()?;
    let client = PjrtClient::new(plugin)?;
    let device = client.amd_gpu_device()?;
    println!("AMD device description: {}", device.description);
    let xla = XlaClient::new(client);
    let executable = xla.compile(
        &StableHloModule::from_text(EMBEDDING_GATHER),
        &CompileOptions::default(),
    )?;

    let table = store.upload_bf16(xla.pjrt(), "model.embed_tokens.weight", Some(&device))?;
    let indices = xla.upload_to(
        HostBuffer::from_i64([1, 1], &[TOKEN_ID as i64])?,
        &device,
    )?;
    assert!(!table.is_on_cpu()?, "embedding table is resident on the CPU");
    assert!(!indices.is_on_cpu()?, "token IDs are resident on the CPU");
    assert!(table.is_on_device(&device)?);
    assert!(indices.is_on_device(&device)?);

    let mut outputs = executable.execute(&[&table, &indices], &device)?;
    assert_eq!(outputs.len(), 1);
    let output = outputs.remove(0);
    assert!(!output.is_on_cpu()?, "embedding output is resident on the CPU");
    assert!(output.is_on_device(&device)?);
    let actual = output.to_host_bytes()?;
    assert_eq!(actual, expected, "GPU gather differs from checkpoint BF16 row");
    println!(
        "gathered token {TOKEN_ID}: {} BF16 values / {} bytes matched checkpoint",
        HIDDEN_SIZE,
        actual.len(),
    );
    Ok(())
}
