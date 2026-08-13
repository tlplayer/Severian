use std::{
    path::PathBuf,
    process::Command,
    time::{SystemTime, UNIX_EPOCH},
};

fn fixture() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../benchmarks/popcorn/vectorsum_v2/kernel.sev")
}

fn temporary_file(name: &str) -> PathBuf {
    let nonce = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    std::env::temp_dir().join(format!("severian-{name}-{}-{nonce}", std::process::id()))
}

#[test]
fn inspect_explains_automatic_backend_selection() {
    let output = Command::new(env!("CARGO_BIN_EXE_sev"))
        .args(["kernel", "inspect"])
        .arg(fixture())
        .args([
            "--entry",
            "reduction_sum",
            "--backend",
            "auto",
            "--target",
            "cuda:sm_90",
        ])
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("operation: reduction.sum"));
    assert!(stdout.contains("target: cuda:sm_90"));
    assert!(stdout.contains("requested backend: auto"));
    assert!(stdout.contains("selected backend: triton"));
    assert!(stdout.contains("fallback: xla"));
    assert!(!stdout.contains("popcorn"));
}

#[test]
fn emit_writes_a_standalone_triton_module() {
    let artifact = temporary_file("reduction.ttir.mlir");
    let output = Command::new(env!("CARGO_BIN_EXE_sev"))
        .args(["kernel", "emit"])
        .arg(fixture())
        .args([
            "--entry",
            "reduction_sum",
            "--backend",
            "triton",
            "--output",
        ])
        .arg(&artifact)
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let source = std::fs::read_to_string(&artifact).unwrap();
    assert!(source.contains("tt.func public @reduction_sum"));
    assert!(source.contains("severian_operation = \"reduction.sum\""));
    assert!(source.contains("tensor<256xf32>"));
    assert!(source.contains("tt.atomic_rmw fadd"));
    assert!(!source.contains("import torch"));
    assert!(!source.contains("import triton"));
    assert!(!source.contains("custom_kernel"));
    assert!(!source.contains("leaderboard"));
    let _ = std::fs::remove_file(artifact);
}

#[test]
fn xla_fallback_emits_stablehlo_from_kernel_ir() {
    let artifact = temporary_file("reduction.stablehlo.mlir");
    let output = Command::new(env!("CARGO_BIN_EXE_sev"))
        .args(["kernel", "emit"])
        .arg(fixture())
        .args(["--entry", "reduction_sum", "--backend", "xla", "--output"])
        .arg(&artifact)
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let source = std::fs::read_to_string(&artifact).unwrap();
    assert!(source.contains("stablehlo.reduce"));
    assert!(source.contains("tensor<?xf32>"));
    assert!(source.contains("-> tensor<f32>"));
    let _ = std::fs::remove_file(artifact);
}

#[test]
fn direct_kernel_invocation_compiles_the_selected_artifact() {
    let target = fixture()
        .parent()
        .unwrap()
        .join("target/debug/reduction_sum.ttir.mlir");
    let _ = std::fs::remove_file(&target);
    let output = Command::new(env!("CARGO_BIN_EXE_sev"))
        .arg(fixture())
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(target.is_file());
    assert!(String::from_utf8_lossy(&output.stdout).contains("Compiled 1 function(s)"));
    let _ = std::fs::remove_file(target);
}
