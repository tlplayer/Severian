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
    std::env::temp_dir().join(format!("severian-lint-cli-{}-{nonce}", std::process::id()))
}

#[test]
fn lint_reports_and_safely_fixes_role_based_names() {
    let directory = temporary_directory();
    std::fs::create_dir_all(&directory).unwrap();
    let source = directory.join("main.sev");
    std::fs::write(
        &source,
        "def LoadModel(modelPath: string):\n    hiddenState = modelPath\n    return hiddenState\n",
    )
    .unwrap();

    let report = Command::new(env!("CARGO_BIN_EXE_sev"))
        .args(["lint"])
        .arg(&source)
        .output()
        .unwrap();
    assert!(report.status.success());
    let standard_error = String::from_utf8_lossy(&report.stderr);
    assert!(standard_error.contains("warning[N002]"));
    assert!(standard_error.contains("warning[N001]"));

    let fix = Command::new(env!("CARGO_BIN_EXE_sev"))
        .args(["lint", "--fix"])
        .arg(&source)
        .output()
        .unwrap();
    assert!(fix.status.success());
    assert_eq!(
        std::fs::read_to_string(&source).unwrap(),
        "def load_model(model_path: string):\n    hidden_state = model_path\n    return hidden_state\n"
    );

    let _ = std::fs::remove_dir_all(directory);
}

#[test]
fn lint_rejects_adjacent_module_unsafe_blocks() {
    let directory = temporary_directory();
    std::fs::create_dir_all(&directory).unwrap();
    let source = directory.join("main.sev");
    std::fs::write(
        &source,
        concat!(
            "unsafe:\n",
            "    native(\"first\") def first() -> int\n",
            "\n",
            "unsafe:\n",
            "    native(\"second\") def second() -> int\n",
        ),
    )
    .unwrap();

    let report = Command::new(env!("CARGO_BIN_EXE_sev"))
        .args(["lint"])
        .arg(&source)
        .output()
        .unwrap();

    assert!(!report.status.success());
    let standard_error = String::from_utf8_lossy(&report.stderr);
    assert!(standard_error.contains("error[N012]"));
    assert!(standard_error.contains("one cohesive `unsafe:` block"));

    let _ = std::fs::remove_dir_all(directory);
}

#[test]
fn camel_case_compatibility_remains_callable_but_is_linted() {
    let directory = temporary_directory();
    std::fs::create_dir_all(&directory).unwrap();
    let source = directory.join("main.sev");
    std::fs::write(
        &source,
        concat!(
            "class Point:\n",
            "    x: int\n",
            "\n",
            "    def getX() -> int:\n",
            "        return x\n",
            "\n",
            "def main():\n",
            "    point = Point(7)\n",
            "    print(point.getX())\n",
        ),
    )
    .unwrap();

    let lint = Command::new(env!("CARGO_BIN_EXE_sev"))
        .args(["lint"])
        .arg(&source)
        .output()
        .unwrap();
    assert!(lint.status.success());
    let standard_error = String::from_utf8_lossy(&lint.stderr);
    assert!(standard_error.contains("warning[N002]"));
    assert!(standard_error.contains("`getX` should be `get_x`"));

    let run = Command::new(env!("CARGO_BIN_EXE_sev"))
        .arg(&source)
        .output()
        .unwrap();
    assert!(run.status.success());
    assert_eq!(String::from_utf8_lossy(&run.stdout), "7\n");

    let _ = std::fs::remove_dir_all(directory);
}

#[test]
fn lint_reports_and_fixes_contract_layout() {
    let directory = temporary_directory();
    std::fs::create_dir_all(&directory).unwrap();
    let source = directory.join("main.sev");
    std::fs::write(
        &source,
        concat!(
            "class Range:\n",
            "    low: int with { low >= 0, }\n",
            "    high: int with { high > low, high < 100 }\n",
            "\n",
            "def positive(value: int) -> int with\n",
            "{\n",
            "    value > 0,\n",
            "}:\n",
            "    return value\n",
        ),
    )
    .unwrap();

    let report = Command::new(env!("CARGO_BIN_EXE_sev"))
        .args(["lint"])
        .arg(&source)
        .output()
        .unwrap();
    assert!(report.status.success());
    assert!(String::from_utf8_lossy(&report.stderr).contains("lint::contract-layout"));

    let fix = Command::new(env!("CARGO_BIN_EXE_sev"))
        .args(["lint", "--fix"])
        .arg(&source)
        .output()
        .unwrap();
    assert!(fix.status.success());
    assert_eq!(
        std::fs::read_to_string(&source).unwrap(),
        concat!(
            "class Range:\n",
            "    low: int with { low >= 0 }\n",
            "    high: int with\n",
            "    {\n",
            "        high > low,\n",
            "        high < 100,\n",
            "    }\n",
            "\n",
            "def positive(value: int) -> int with { value > 0 }:\n",
            "    return value\n",
        )
    );

    let _ = std::fs::remove_dir_all(directory);
}
