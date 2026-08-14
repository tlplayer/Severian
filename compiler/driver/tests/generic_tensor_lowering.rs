use severian_driver::compile_dependency_path;
use severian_lowering::stablehlo::lower_entry;
use std::path::Path;

#[test]
fn axis_softmax_and_where_lower_as_generic_stablehlo() {
    std::thread::Builder::new()
        .name("generic-tensor-lowering".into())
        .stack_size(16 * 1024 * 1024)
        .spawn(|| {
            let workspace = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
            let fixture = workspace.join("compiler/driver/tests/fixtures/generic_tensor_ops");
            let compilation =
                compile_dependency_path(&fixture.join("ops.sev"), &fixture.join("package.toml"))
                    .unwrap();
            let entry = compilation
                .optimized_hir
                .functions
                .iter()
                .find(|function| {
                    function.native_symbol.is_none()
                        && function
                            .decorators
                            .iter()
                            .any(|decorator| decorator.package == "tensor")
                })
                .expect("missing generic tensor lowering fixture")
                .id;
            let module = lower_entry(&compilation.optimized_hir, entry).unwrap();
            let stablehlo = module.as_str();

            assert!(stablehlo.contains("stablehlo.reduce"));
            assert!(stablehlo.contains("dimensions = array<i64: 1>"));
            assert!(stablehlo.contains("stablehlo.select"));
            assert!(!stablehlo.contains("custom_call"));
        })
        .unwrap()
        .join()
        .unwrap();
}
