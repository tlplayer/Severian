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
            assert!(stablehlo.contains("stablehlo.reshape"));
            assert!(stablehlo.contains("stablehlo.add"));
            assert!(!stablehlo.contains("custom_call"));
        })
        .unwrap()
        .join()
        .unwrap();
}

#[test]
fn add_and_reshape_have_no_dtype_specific_escape_symbols() {
    let workspace = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let mut paths = vec![workspace.join("compiler/hir/src/tensor.rs")];
    collect_severian_sources(&workspace.join("library"), &mut paths);
    collect_severian_sources(&workspace.join("benchmarks"), &mut paths);
    let source = paths
        .into_iter()
        .map(|path| std::fs::read_to_string(path).unwrap())
        .collect::<String>();
    let forbidden = [
        format!("{}{}", "__sev_tensor_", "bf16_add"),
        format!("{}{}", "__sev_tensor_", "f32_add"),
        format!("{}{}", "__sev_tensor_", "bf16_reshape"),
        format!("{}{}", "__sev_tensor_", "f32_reshape"),
        format!("{}{}", "add_bf", "_16"),
        format!("{}{}", "add_f", "_32"),
        format!("{}{}", "reshape_bf", "_16"),
        format!("{}{}", "reshape_f", "_32"),
    ];
    for symbol in forbidden {
        assert!(
            !contains_identifier(&source, &symbol),
            "obsolete tensor API `{symbol}`"
        );
    }
}

fn contains_identifier(source: &str, symbol: &str) -> bool {
    source.match_indices(symbol).any(|(start, _)| {
        let identifier = |character: Option<char>| {
            character.is_some_and(|character| character == '_' || character.is_alphanumeric())
        };
        !identifier(source[..start].chars().next_back())
            && !identifier(source[start + symbol.len()..].chars().next())
    })
}

fn collect_severian_sources(directory: &Path, output: &mut Vec<std::path::PathBuf>) {
    for entry in std::fs::read_dir(directory).unwrap() {
        let path = entry.unwrap().path();
        if path.is_dir() {
            collect_severian_sources(&path, output);
        } else if path.extension().is_some_and(|extension| extension == "sev") {
            output.push(path);
        }
    }
}
