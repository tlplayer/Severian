use std::{
    path::PathBuf,
    process::Command,
    time::{SystemTime, UNIX_EPOCH},
};

fn vector_sum_fixture() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../docs/examples/26-problems/12-reduction-sum-gpu.sev")
}

#[test]
fn exports_a_submit_ready_popcorn_vector_sum() {
    let nonce = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let root = std::env::temp_dir().join(format!(
        "severian-popcorn-export-{}-{nonce}",
        std::process::id()
    ));
    std::fs::create_dir_all(&root).unwrap();
    let submission = root.join("submission.py");
    let output = Command::new(env!("CARGO_BIN_EXE_sev"))
        .args(["kernel", "export", "popcorn"])
        .arg(vector_sum_fixture())
        .args([
            "--entry",
            "reductionSum",
            "--leaderboard",
            "vectorsum_v2",
            "--gpu",
            "A100",
            "--output",
        ])
        .arg(&submission)
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );

    let source = std::fs::read_to_string(&submission).unwrap();
    assert!(source.starts_with("#!POPCORN leaderboard vectorsum_v2\n#!POPCORN gpu A100\n"));
    assert!(source.contains("SEVERIAN_ENTRY = \"reductionSum\""));
    assert!(source.contains("SEVERIAN_OPERATION = \"tensor.reduce_sum\""));
    assert!(source.contains("@triton.jit"));
    assert!(source.contains("tl.atomic_add(output_pointer, partial)"));
    assert!(source.contains("custom_kernel = torch.compile"));
    assert!(!source.contains("subprocess"));
    assert!(!source.contains(".cpu()"));

    let syntax = Command::new("python3")
        .args(["-m", "py_compile"])
        .arg(&submission)
        .output()
        .unwrap();
    assert!(
        syntax.status.success(),
        "{}",
        String::from_utf8_lossy(&syntax.stderr)
    );
    std::fs::remove_dir_all(root).unwrap();
}

#[test]
fn rejects_unimplemented_popcorn_problem_contracts() {
    let output = Command::new(env!("CARGO_BIN_EXE_sev"))
        .args(["kernel", "export", "popcorn"])
        .arg(vector_sum_fixture())
        .args(["--leaderboard", "matmul_v2"])
        .output()
        .unwrap();
    assert!(!output.status.success());
    assert!(String::from_utf8_lossy(&output.stderr)
        .contains("current exporter supports `vectorsum_v2`"));
}
