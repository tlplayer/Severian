use std::{
    path::PathBuf,
    process::Command,
    time::{SystemTime, UNIX_EPOCH},
};

fn temporary_directory() -> PathBuf {
    let nonce = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("system time must follow the Unix epoch")
        .as_nanos();
    std::env::temp_dir().join(format!("severian-emit-cli-{}-{nonce}", std::process::id()))
}

#[test]
fn top_level_emit_reaches_every_active_compiler_representation() {
    let directory = temporary_directory();
    std::fs::create_dir_all(&directory).unwrap();
    let source = directory.join("tensor_add.sev");
    std::fs::write(
        &source,
        "import tensor\n\n@tensor\ndef add(left: Tensor[f64, dynamic], right: Tensor[f64, dynamic]) -> Tensor[f64, dynamic]:\n    return tensor.rankedAdd(left, right)\n",
    )
    .unwrap();

    for (stage, artifact) in [
        ("hir", "tensor_add.hir"),
        ("mir", "tensor_add.mir"),
        ("mlir", "tensor_add.mlir"),
        ("llvm", "tensor_add.ll"),
        ("asm", "tensor_add.s"),
        ("stablehlo", "tensor_add.stablehlo.mlir"),
    ] {
        let output = Command::new(env!("CARGO_BIN_EXE_sev"))
            .args(["--emit", stage])
            .arg(&source)
            .output()
            .unwrap();
        assert!(
            output.status.success(),
            "sev --emit {stage} failed: {}",
            String::from_utf8_lossy(&output.stderr)
        );
        let artifact = directory.join("target/debug").join(artifact);
        assert!(artifact.is_file(), "missing {}", artifact.display());
        assert!(artifact.metadata().unwrap().len() > 0);
    }

    let _ = std::fs::remove_dir_all(directory);
}
