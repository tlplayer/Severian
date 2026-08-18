use severian_xla::{
    CompileOptions, HostBuffer, PjrtClient, PjrtPlugin, StableHloModule, XlaClient, XlaError,
};

const MATMUL: &str = r#"
module {
  func.func @main(
      %lhs: tensor<2x2xf32>,
      %rhs: tensor<2x2xf32>
  ) -> tensor<2x2xf32> {
    %result = "stablehlo.dot_general"(%lhs, %rhs) {
      dot_dimension_numbers = #stablehlo.dot<
        lhs_batching_dimensions = [],
        rhs_batching_dimensions = [],
        lhs_contracting_dimensions = [1],
        rhs_contracting_dimensions = [0]
      >,
      precision_config = [
        #stablehlo<precision DEFAULT>,
        #stablehlo<precision DEFAULT>
      ]
    } : (
        tensor<2x2xf32>,
        tensor<2x2xf32>
    ) -> tensor<2x2xf32>

    return %result : tensor<2x2xf32>
  }
}
"#;

#[test]
fn stablehlo_matmul_executes_on_amd_gpu() -> Result<(), Box<dyn std::error::Error>> {
    let plugin = match PjrtPlugin::load_rocm() {
        Ok(plugin) => plugin,
        Err(XlaError::PluginLoad(message))
            if std::env::var_os("SEVERIAN_ROCM_PJRT_PLUGIN").is_none()
                && message.contains("not found") =>
        {
            eprintln!("skipping AMD GPU integration test: {message}");
            return Ok(());
        }
        Err(error) => return Err(error.into()),
    };
    let client = PjrtClient::new(plugin)?;
    let platform = client.platform_name()?;
    let addressable_count = client.addressable_devices()?.len();
    let device = client.amd_gpu_device()?;

    println!("PJRT platform name: {platform}");
    println!("AMD device description: {}", device.description);
    println!("addressable device count: {addressable_count}");

    let xla = XlaClient::new(client);
    let mut compile_options = CompileOptions::default();
    compile_options.portable_artifact = false;
    let executable = xla.compile(&StableHloModule::from_text(MATMUL), &compile_options)?;
    println!("StableHLO compile success");

    let a = xla.upload_to(
        HostBuffer::from_f32([2, 2], &[1.0, 2.0, 3.0, 4.0])?,
        &device,
    )?;
    let b = xla.upload_to(
        HostBuffer::from_f32([2, 2], &[5.0, 6.0, 7.0, 8.0])?,
        &device,
    )?;

    println!("input A IsOnCpu: {}", a.is_on_cpu()?);
    println!("input B IsOnCpu: {}", b.is_on_cpu()?);
    assert!(!a.is_on_cpu()?, "input A is resident on the CPU");
    assert!(!b.is_on_cpu()?, "input B is resident on the CPU");
    assert!(a.is_on_device(&device)?, "input A is on a different device");
    assert!(b.is_on_device(&device)?, "input B is on a different device");

    let mut outputs = executable.execute(&[&a, &b], &device)?;
    println!("PJRT execute success");
    assert_eq!(outputs.len(), 1, "matmul must return exactly one buffer");
    let output = outputs.remove(0);
    println!("output IsOnCpu: {}", output.is_on_cpu()?);
    assert!(!output.is_on_cpu()?, "output is resident on the CPU");
    assert!(
        output.is_on_device(&device)?,
        "output is on a different device"
    );

    let result = output.to_f32()?;
    println!("result: {result:?}");
    let expected = [19.0, 22.0, 43.0, 50.0];
    assert_eq!(result.len(), expected.len());
    for (actual, expected) in result.iter().zip(expected) {
        assert!((actual - expected).abs() <= 1e-5, "{actual} != {expected}");
    }
    Ok(())
}
