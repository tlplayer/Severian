use std::{
    path::PathBuf,
    process::Command,
    time::{SystemTime, UNIX_EPOCH},
};

fn source(label: &str, contents: &str) -> PathBuf {
    let nonce = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let directory = std::env::temp_dir().join(format!(
        "severian-conversion-{label}-{}-{nonce}",
        std::process::id()
    ));
    std::fs::create_dir_all(&directory).unwrap();
    let path = directory.join("main.sev");
    std::fs::write(&path, contents).unwrap();
    path
}

#[test]
fn int_converts_numeric_values_and_base_ten_strings() {
    let path = source(
        "int",
        "def main():\n    print(int(\"1\"))\n    print(int(\" -42 \"))\n    print(int(3.9))\n    print(int(true))\n",
    );
    let output = Command::new(env!("CARGO_BIN_EXE_sev"))
        .arg(&path)
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert_eq!(String::from_utf8_lossy(&output.stdout), "1\n-42\n3\n1\n");
    std::fs::remove_dir_all(path.parent().unwrap()).unwrap();
}

#[test]
fn string_formats_primitives_collections_data_objects_and_tensors() {
    let path = source(
        "string",
        concat!(
            "import data\n",
            "import tensor\n\n",
            "def main():\n",
            "    print(string(1))\n",
            "    print(string(0.125))\n",
            "    print(string(1.0))\n",
            "    print(string(true))\n",
            "    print(string([\"Ada\", \"Grace\"]))\n",
            "    print(string({\"count\": 2}))\n",
            "    print(string(data.Data([\"name\"], [[\"Ada\"], [\"Grace\"]])))\n",
            "    value = tensor.tensor([1.0, 2.5, 3.0, 4.0], [2, 2])\n",
            "    print(string(value))\n",
        ),
    );
    let output = Command::new(env!("CARGO_BIN_EXE_sev"))
        .arg(&path)
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert_eq!(
        String::from_utf8_lossy(&output.stdout),
        concat!(
            "1\n",
            "0.125\n",
            "1.0\n",
            "true\n",
            "[\"Ada\", \"Grace\"]\n",
            "{\"count\": 2}\n",
            "Data(columns_data=[\"name\"], rows_data=[[\"Ada\"], [\"Grace\"]])\n",
            "Tensor(shape=[2, 2], values=[1.0, 2.5, 3.0, 4.0])\n",
        )
    );
    std::fs::remove_dir_all(path.parent().unwrap()).unwrap();
}
