use severian_driver::{compile_native_tests, compile_path, compile_source};
use std::path::{Path, PathBuf};
use std::process::Command;

fn examples_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../docs/examples")
}

fn severian_files(directory: &Path) -> Vec<PathBuf> {
    let mut files = std::fs::read_dir(directory)
        .unwrap()
        .map(|entry| entry.unwrap().path())
        .filter(|path| path.extension().is_some_and(|extension| extension == "sev"))
        .collect::<Vec<_>>();
    files.sort();
    files
}

#[test]
fn checks_all_concurrency_examples_through_the_frontend() {
    let directory = examples_root().join("08-concurrency");
    for fixture in severian_files(&directory) {
        compile_path(&fixture).unwrap_or_else(|error| panic!("{}: {error}", fixture.display()));
    }
}

#[test]
fn native_calls_require_an_explicit_declaration_and_lower_to_its_abi_symbol() {
    let compilation = compile_source(concat!(
        "unsafe:\n",
        "    extern(\"__sev_regex_matches\") def matches(\n",
        "        value: string,\n",
        "        pattern: string,\n",
        "    ) -> bool\n",
        "\n",
        "def main():\n",
        "    print(matches(\"severian\", \"sev.*\"))\n",
    ))
    .unwrap();

    assert!(compilation
        .mlir
        .as_str()
        .contains("llvm.func @__sev_regex_matches"));
    assert!(compilation
        .mlir
        .as_str()
        .contains("llvm.call @__sev_regex_matches"));

    let error =
        compile_source("def main():\n    print(runtime.fileRead(\"missing.txt\"))\n").unwrap_err();
    assert!(error.to_string().contains("runtime"));
}

#[test]
fn ranked_tensor_example_emits_real_linalg_kernels() {
    let fixture = examples_root().join("22-ranked-tensors/main.sev");
    let compilation = compile_path(&fixture).unwrap();
    let mlir = compilation.mlir.as_str();

    assert!(mlir.contains("linalg.generic"));
    assert!(mlir.contains("linalg.matmul"));
    assert!(mlir.contains("@__sev_linalg_sum"));
    assert!(mlir.contains("llvm.emit_c_interface"));
    assert!(mlir.contains("llvm.call @__sev_tensor_matmul"));
    assert!(mlir.contains("llvm.call @__sev_tensor_add"));
    assert!(mlir.contains("llvm.call @__sev_tensor_relu"));
}

#[test]
fn fuses_stacked_model_activations_without_user_optimization_syntax() {
    let fixture = examples_root().join("21-parallel-kernels/main.sev");
    let compilation = compile_path(&fixture).unwrap();
    let mlir = compilation.mlir.as_str();
    let forward = mlir
        .split("llvm.func @__sev_fn_forward(")
        .nth(1)
        .unwrap()
        .split("llvm.func @main(")
        .next()
        .unwrap();
    assert!(forward.contains("llvm.call @__sev_fused_activations"));
    assert!(forward.contains("severian_fusion = \"automatic\""));
    assert!(!forward.contains("llvm.call @activation"));
    assert!(!forward.contains("llvm.call @tanhActivation"));
    assert!(!forward.contains("llvm.call @swishActivation"));
}

#[test]
fn does_not_fuse_user_functions_that_only_share_activation_names() {
    let compilation = compile_source(
        "def relu(X: list[float]) -> list[float]:\n    return X\n\ndef swish(X: list[float]) -> list[float]:\n    return X\n\ndef main():\n    print(swish(relu([1.0])))\n",
    )
    .unwrap();
    let main = compilation
        .mlir
        .as_str()
        .split("llvm.func @main(")
        .nth(1)
        .unwrap();
    assert!(!main.contains("llvm.call @__sev_fused_activations"));
}

#[test]
fn chained_class_filter_calls_the_method_instead_of_the_list_intrinsic() {
    let compilation = compile_source(concat!(
        "class Dataset:\n",
        "    value: int\n",
        "\n",
        "    def filter(predicate: Function[int, bool]) -> Dataset:\n",
        "        return Dataset(value)\n",
        "\n",
        "def dataset() -> Dataset:\n",
        "    return Dataset(1)\n",
        "\n",
        "def main():\n",
        "    filtered = dataset().filter(|value| value > 0)\n",
        "    print(filtered.value)\n",
    ))
    .unwrap();

    let main = compilation
        .mlir
        .as_str()
        .split("llvm.func @main(")
        .nth(1)
        .unwrap();
    assert!(main.contains("llvm.call @__sev_method_Dataset_filter"));
    assert!(!main.contains("llvm.call @__sev_unbox_ptr"));
}

#[test]
fn class_methods_take_precedence_over_collection_intrinsic_names() {
    let compilation = compile_source(concat!(
        "class Buffer:\n",
        "    values: list[int]\n",
        "\n",
        "    def append(value: int):\n",
        "        values.append(value)\n",
        "\n",
        "    def pop() -> int:\n",
        "        return values.pop()\n",
        "\n",
        "    def to_list() -> list[int]:\n",
        "        return values\n",
        "\n",
        "def main():\n",
        "    buffer = Buffer([])\n",
        "    buffer.append(4)\n",
        "    values = buffer.to_list()\n",
        "    print(buffer.pop())\n",
        "    print(size(values))\n",
    ))
    .unwrap();

    let main = compilation
        .mlir
        .as_str()
        .split("llvm.func @main(")
        .nth(1)
        .unwrap();
    for method in ["append", "pop", "to_list"] {
        assert!(
            main.contains(&format!("llvm.call @__sev_method_Buffer_{method}")),
            "missing class dispatch for {method}: {main}"
        );
    }
}

#[test]
fn typed_map_membership_lowers_to_a_runtime_lookup() {
    let compilation = compile_source(concat!(
        "def contains(values: map[string, int], key: string) -> bool:\n",
        "    return key in values\n",
        "\n",
        "def missing(values: map[string, int], key: string) -> bool:\n",
        "    return key not in values\n",
        "\n",
        "def remove(values: map[string, int], key: string) -> int:\n",
        "    return values.pop(key, -1)\n",
    ))
    .unwrap();

    assert_eq!(
        compilation
            .mlir
            .as_str()
            .matches("llvm.call @__sev_map_contains")
            .count(),
        2
    );
    assert!(compilation
        .mlir
        .as_str()
        .contains("llvm.call @__sev_map_pop"));
}

#[test]
fn compiles_server_syntax_and_propagated_file_errors() {
    let root = examples_root();
    for fixture in [
        "17-servers/01-simple-server.sev",
        "17-servers/02-chat-server.sev",
        "17-servers/03-map-reduce-server.sev",
        "bugs/errors/ignored_error/fixed.sev",
    ] {
        compile_path(&root.join(fixture)).unwrap_or_else(|error| panic!("{fixture}: {error}"));
    }
}

#[test]
fn lowers_locked_method_tasks_to_workers_and_awaits_each_tuple_member() {
    let fixture = examples_root().join("bugs/threads/data_race/fixed.sev");
    let compilation = compile_path(&fixture).unwrap();
    let main = compilation
        .mlir
        .as_str()
        .split("llvm.func @main(")
        .nth(1)
        .unwrap();

    assert_eq!(
        main.matches("llvm.call @__sev_task_spawn_Counter_increment")
            .count(),
        2
    );
    assert_eq!(main.matches("llvm.call @__sev_task_await_unit").count(), 2);
}

#[test]
fn builds_and_executes_the_native_inference_node_vertically() {
    let package = examples_root().join("27-inference-orchestrator");
    let build = Command::new(env!("CARGO_BIN_EXE_sev"))
        .arg("build")
        .current_dir(&package)
        .output()
        .unwrap();
    assert!(
        build.status.success(),
        "{}",
        String::from_utf8_lossy(&build.stderr)
    );
    let build_log = String::from_utf8_lossy(&build.stdout);
    for package_name in ["tensor", "model", "network", "orchestrator"] {
        assert!(
            build_log.contains(&format!("Built {package_name} ->")),
            "missing {package_name} library build in:\n{build_log}"
        );
    }

    let executable = package.join("target/debug/inference-node-example");
    let output = Command::new(&executable).output().unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert_eq!(
        String::from_utf8(output.stdout).unwrap(),
        std::fs::read_to_string(package.join("main.stdout")).unwrap()
    );

    let compilation = compile_path(&package.join("main.sev")).unwrap();
    let test_executable =
        std::env::temp_dir().join(format!("severian-inference-tests-{}", std::process::id()));
    assert_eq!(
        compile_native_tests(&compilation, &test_executable).unwrap(),
        3
    );
    let tests = Command::new(&test_executable).output().unwrap();
    let _ = std::fs::remove_file(test_executable);
    assert!(
        tests.status.success(),
        "{}",
        String::from_utf8_lossy(&tests.stderr)
    );
    assert_eq!(String::from_utf8(tests.stdout).unwrap(), "3 passed\n");
}

#[test]
fn builds_model_submodules_and_runs_the_transformer_container_example() {
    let package = examples_root().join("28-transformer-container");
    let build = Command::new(env!("CARGO_BIN_EXE_sev"))
        .arg("build")
        .current_dir(&package)
        .output()
        .unwrap();
    assert!(
        build.status.success(),
        "{}",
        String::from_utf8_lossy(&build.stderr)
    );
    let build_log = String::from_utf8_lossy(&build.stdout);
    let model = build_log.find("Built model ->").unwrap();
    let neuralnet = build_log.find("Built model.neuralnet ->").unwrap();
    let application = build_log
        .find("Built transformer-container-example ->")
        .unwrap();
    assert!(model < neuralnet && neuralnet < application);

    let executable = package.join("target/debug/transformer-container-example");
    let output = Command::new(&executable).output().unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert_eq!(
        String::from_utf8(output.stdout).unwrap(),
        std::fs::read_to_string(package.join("main.stdout")).unwrap()
    );

    let compilation = compile_path(&package.join("main.sev")).unwrap();
    let test_executable = std::env::temp_dir().join(format!(
        "severian-transformer-container-tests-{}",
        std::process::id()
    ));
    assert_eq!(
        compile_native_tests(&compilation, &test_executable).unwrap(),
        4
    );
    let tests = Command::new(&test_executable).output().unwrap();
    let _ = std::fs::remove_file(test_executable);
    assert!(
        tests.status.success(),
        "{}",
        String::from_utf8_lossy(&tests.stderr)
    );
    assert_eq!(
        String::from_utf8(tests.stdout).unwrap(),
        format!(
            "{}4 passed\n",
            std::fs::read_to_string(package.join("main.stdout")).unwrap()
        )
    );
}

#[test]
fn builds_and_runs_the_operating_system_laboratory_vertically() {
    let package = examples_root().join("../lab/operating_system");
    let build = Command::new(env!("CARGO_BIN_EXE_sev"))
        .arg("build")
        .current_dir(&package)
        .output()
        .unwrap();
    assert!(
        build.status.success(),
        "{}",
        String::from_utf8_lossy(&build.stderr)
    );
    let build_log = String::from_utf8_lossy(&build.stdout);
    let platform = build_log.find("Built platform ->").unwrap();
    let kernel = build_log.find("Built kernel ->").unwrap();
    let application = build_log.find("Built operating-system-lab ->").unwrap();
    assert!(platform < kernel && kernel < application);

    let executable = package.join("target/debug/operating-system-lab");
    let output = Command::new(&executable).output().unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert_eq!(
        String::from_utf8(output.stdout).unwrap(),
        std::fs::read_to_string(package.join("main.stdout")).unwrap()
    );

    let compilation = compile_path(&package.join("main.sev")).unwrap();
    assert!(compilation.mlir.as_str().contains("__sev_collection_clone"));
    let test_executable = std::env::temp_dir().join(format!(
        "severian-operating-system-tests-{}",
        std::process::id()
    ));
    assert_eq!(
        compile_native_tests(&compilation, &test_executable).unwrap(),
        5
    );
    let tests = Command::new(&test_executable).output().unwrap();
    let _ = std::fs::remove_file(test_executable);
    assert!(
        tests.status.success(),
        "{}",
        String::from_utf8_lossy(&tests.stderr)
    );
    assert_eq!(
        String::from_utf8(tests.stdout).unwrap(),
        format!(
            "{}5 passed\n",
            std::fs::read_to_string(package.join("main.stdout")).unwrap()
        )
    );
}
