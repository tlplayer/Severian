use std::{
    path::PathBuf,
    process::Command,
    time::{SystemTime, UNIX_EPOCH},
};

#[test]
fn requested_standard_library_surface_executes_natively() {
    let nonce = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let root = std::env::temp_dir().join(format!(
        "severian-standard-library-{}-{nonce}",
        std::process::id()
    ));
    std::fs::create_dir_all(&root).unwrap();
    let source = root.join("main.sev");
    std::fs::write(
        &source,
        r#"import core
import list
import map
import set
import string
import math
import random
import path
import json
import regex
import time
import process
import environment
import tensor

def main():
    numbers := [3, 1, 2]
    list.sort(numbers)
    assert(numbers == [1, 2, 3])
    assert(len(numbers) == 3)
    assert(size(numbers) == 3)
    assert(numbers.len() == 3)
    assert(numbers.size() == 3)
    assert(bits(numbers) == bytes(numbers) * 8)
    assert(numbers.bits() == numbers.bytes() * 8)
    assert(capacity(numbers) >= len(numbers))
    assert(core.sum(numbers) == 6)
    assert(string.strip("  Severian  ") == "Severian")
    assert(math.floor(math.sqrt(9.0)) == 3)
    random.seed(7)
    assert(random.randint(1, 1) == 1)
    mapping := {"answer": 42}
    assert(map.get(mapping, "answer", 0) == 42)
    entries := {1, 2}
    set.add(entries, 3)
    assert(3 in entries)
    assert(regex.search("[0-9]+", "release-42"))
    assert(path.basename("/tmp/example.txt") == "example.txt")
    assert(environment.set("SEVERIAN_LIBRARY_TEST", "ready"))
    assert(environment.get("SEVERIAN_LIBRARY_TEST") == "ready")
    assert(environment.remove("SEVERIAN_LIBRARY_TEST"))
    assert(json.dumps([1, 2, 3]) == "[1,2,3]")
    value = tensor.ones([2, 2])
    assert(len(value) == 4)
    assert(value.size() == 4)
    assert(value.bytes() == 32)
    assert(value.bits() == 256)
    assert(tensor.mean(value) == 1.0)
    assert(time.monotonic() > 0.0)
    assert(process.run("true") == 0)
"#,
    )
    .unwrap();

    let output = Command::new(env!("CARGO_BIN_EXE_sev"))
        .arg("run")
        .arg(&source)
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "stdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    std::fs::remove_dir_all(root).unwrap();
}

#[test]
fn all_requested_packages_are_workspace_members() {
    let workspace = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../library/package.toml");
    let manifest = std::fs::read_to_string(workspace).unwrap();
    for package in [
        "core",
        "list",
        "map",
        "set",
        "string",
        "math",
        "random",
        "file",
        "path",
        "json",
        "regex",
        "time",
        "process",
        "environment",
        "http",
        "network",
        "logging",
        "tensor",
    ] {
        assert!(
            manifest.contains(&format!("\"{package}\"")),
            "missing {package}"
        );
    }
}
