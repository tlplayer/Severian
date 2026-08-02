use severian_driver::{compile_path, compile_source, run, run_tests};
use severian_hir::TestMode;
use std::path::{Path, PathBuf};

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
fn checks_and_tests_frontend_example_directories() {
    let root = examples_root();
    let directories = [
        "00-getting-started",
        "01-values-control",
        "02-functions-modules",
        "03-collections-iteration",
        "04-classes-traits",
        "05-ownership-borrowing",
        "06-results-patterns",
        "07-generics-constraints",
    ];

    let mut compiled = 0;
    let mut severian_tests = 0;
    for directory in directories {
        for fixture in severian_files(&root.join(directory)) {
            let compilation = compile_path(&fixture)
                .unwrap_or_else(|error| panic!("{}: {error}", fixture.display()));
            severian_tests += run_tests(&compilation.hir, |_| {})
                .unwrap_or_else(|error| panic!("{}: {error}", fixture.display()));
            compiled += 1;
        }
    }

    assert_eq!(compiled, 27);
    assert_eq!(severian_tests, 11);
}

#[test]
fn checks_all_concurrency_examples_through_the_frontend() {
    let directory = examples_root().join("08-concurrency");
    for fixture in severian_files(&directory) {
        compile_path(&fixture).unwrap_or_else(|error| panic!("{}: {error}", fixture.display()));
    }
}

#[test]
fn runs_channel_switches_generated_defaults_and_unsafe_addressing() {
    let root = examples_root();

    let channel_switch = compile_path(&root.join("08-concurrency/08-channel-switch.sev")).unwrap();
    let mut output = Vec::new();
    run(&channel_switch.hir, |line| output.push(line.to_owned())).unwrap();
    assert_eq!(output, ["message: hello", "command: refresh"]);

    let generated_defaults =
        compile_path(&root.join("13-method-mutation/02-mutation-contract-placeholder.sev"))
            .unwrap();
    assert_eq!(run_tests(&generated_defaults.hir, |_| {}).unwrap(), 1);

    let unsafe_addressing =
        compile_path(&root.join("09-systems-unsafe/01-isolated-pointer.sev")).unwrap();
    assert_eq!(run_tests(&unsafe_addressing.hir, |_| {}).unwrap(), 2);

    let enum_basics = compile_path(&root.join("12-enums-aliases/01-enum-basics.sev")).unwrap();
    assert_eq!(run_tests(&enum_basics.hir, |_| {}).unwrap(), 1);
}

#[test]
fn compiles_and_classifies_the_test_gallery() {
    let directory = examples_root().join("15-tests");
    let mut modes = Vec::new();
    let mut tests = 0;

    for fixture in severian_files(&directory) {
        let compilation =
            compile_path(&fixture).unwrap_or_else(|error| panic!("{}: {error}", fixture.display()));
        tests += run_tests(&compilation.hir, |_| {})
            .unwrap_or_else(|error| panic!("{}: {error}", fixture.display()));
        for function in &compilation.hir.functions {
            for test in &function.tests {
                modes.extend(test.modes.iter().copied());
            }
        }
    }

    assert_eq!(tests, 8);
    assert!(modes.contains(&TestMode::Property));
    assert!(modes.contains(&TestMode::Bench));
    assert!(modes.contains(&TestMode::Chaos));
    assert!(modes.contains(&TestMode::Integration));
}

#[test]
fn resolves_path_dependencies_from_severian_manifests() {
    let root = examples_root().join("14-packages");
    let library = compile_path(&root.join("geometry/src/lib.sev")).unwrap();
    assert_eq!(run_tests(&library.hir, |_| {}).unwrap(), 1);

    let application = compile_path(&root.join("app/src/main.sev")).unwrap();
    let mut output = Vec::new();
    run(&application.hir, |line| output.push(line.to_owned())).unwrap();
    assert_eq!(output, ["5"]);
}

#[test]
fn resolves_model_decorator_symbols_to_package_functions() {
    let fixture = examples_root().join("20-model-symbols/main.sev");
    let compilation = compile_path(&fixture).unwrap();
    let mut output = Vec::new();
    run(&compilation.hir, |line| output.push(line.to_owned())).unwrap();
    assert_eq!(
        output,
        ["[0, 0, 3]", "[0.5]", "[0, 0, 0, 0, 0, 0, 0, 0, 1]",]
    );
}

#[test]
fn fuses_stacked_model_activations_without_user_optimization_syntax() {
    let fixture = examples_root().join("21-parallel-kernels/main.sev");
    let compilation = compile_path(&fixture).unwrap();
    let mlir = compilation.mlir.as_str();
    let forward = mlir
        .split("llvm.func @forward(")
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
