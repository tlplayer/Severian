use severian_package::{
    load_embedded_official_interfaces, load_manifest_native_units, load_official_interfaces,
    EmbeddedOfficialPackage,
};
use std::path::PathBuf;
use std::time::{SystemTime, UNIX_EPOCH};

fn parse(source: &str) -> severian_ast::Module {
    let tokens = severian_lexer::lex(source).unwrap();
    severian_parser::parse(&tokens).unwrap()
}

fn library_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../library")
}

fn temporary_package(name: &str) -> PathBuf {
    let nonce = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let root = std::env::temp_dir().join(format!(
        "severian-native-manifest-{name}-{}-{nonce}",
        std::process::id()
    ));
    std::fs::create_dir_all(root.join("native/include")).unwrap();
    std::fs::write(
        root.join("native/provider.c"),
        "int provider(void) { return 0; }\n",
    )
    .unwrap();
    root
}

#[test]
fn keeps_tensor_fusion_contracts_out_of_the_model_package() {
    let module = parse("import model\nimport tensor\n");
    let interfaces = load_official_interfaces(&module, &library_root()).unwrap();
    let model = interfaces
        .iter()
        .find(|interface| interface.name == "model")
        .unwrap();

    assert!(model.compiler.symbols.is_empty());
    assert!(model.compiler.fusion_aliases.is_empty());
    assert!(model.compiler.graph_rules.is_empty());
    let tensor = interfaces
        .iter()
        .find(|interface| interface.name == "tensor")
        .expect("tensor interface is loaded alongside model");
    assert!(tensor.compiler.fusion_rules.iter().any(|rule| {
        rule.function == "tensor.relu"
            && rule.runtime_symbol == "__sev_fused_activations"
            && rule.opcode == 1
    }));
}

#[test]
fn loads_only_reachable_official_interfaces() {
    let module = parse("import tensor\nimport application_package\n");
    let interfaces = load_official_interfaces(&module, &library_root()).unwrap();

    assert_eq!(
        interfaces
            .iter()
            .map(|interface| interface.name.as_str())
            .collect::<Vec<_>>(),
        [
            "csv",
            "data",
            "file",
            "json",
            "os",
            "path",
            "platform",
            "regex",
            "src.expression",
            "src.file_binary",
            "src.file_csv",
            "src.file_interface",
            "src.file_json",
            "src.file_mp3",
            "src.file_text",
            "src.file_wav",
            "src.file_yaml",
            "src.query",
            "src.schema",
            "src.sql",
            "tensor",
            "yaml",
        ]
    );
}

#[test]
fn loads_transitive_packages_from_compiler_embedded_sources() {
    let module = parse("import alpha\n");
    let packages = [
        EmbeddedOfficialPackage {
            name: "alpha",
            manifest: "[package]\nname = \"alpha\"\nversion = \"1.0.0\"\n",
            source: "import \"src/alpha_widget.sev\" as alpha_widget\nimport beta\n\ndef alpha_value() -> int:\n    return beta.beta_value()\n",
            modules: &[severian_package::EmbeddedOfficialModule {
                path: "src/alpha_widget.sev",
                source: "class Widget:\n    value: int\n",
            }],
            native_assets: &[],
        },
        EmbeddedOfficialPackage {
            name: "beta",
            manifest: "[package]\nname = \"beta\"\nversion = \"1.0.0\"\n",
            source: "def beta_value() -> int:\n    return 42\n",
            modules: &[],
            native_assets: &[],
        },
    ];

    let interfaces = load_embedded_official_interfaces(&module, &packages).unwrap();

    assert_eq!(
        interfaces
            .iter()
            .map(|interface| interface.name.as_str())
            .collect::<Vec<_>>(),
        ["alpha", "beta", "src.alpha_widget"]
    );
    assert!(interfaces[0].source_path.ends_with("alpha/src/lib.sev"));
    assert_eq!(interfaces[2].export_package.as_deref(), Some("alpha"));
}

#[test]
fn native_units_are_declarative_targeted_and_package_scoped() {
    let root = temporary_package("valid");
    let manifest = root.join("package.toml");
    std::fs::write(
        &manifest,
        "[package]\nname = \"native-test\"\nversion = \"0.1.0\"\n\n[[ffi.c]]\nname = \"posix\"\nabi = \"c-v1\"\ntargets = [\"linux\", \"macos\"]\nsources = [\"native/provider.c\"]\ninclude = [\"native/include\"]\nlibraries = [\"ssl\"]\n",
    )
    .unwrap();

    let units = load_manifest_native_units(&manifest).unwrap();
    assert_eq!(units.len(), 1);
    assert_eq!(units[0].package, "native-test");
    assert!(units[0].sources[0].is_absolute());
    assert_eq!(units[0].libraries, ["ssl"]);
    std::fs::remove_dir_all(root).unwrap();
}

#[test]
fn native_units_reject_package_supplied_compiler_flags() {
    let root = temporary_package("flags");
    let manifest = root.join("package.toml");
    std::fs::write(
        &manifest,
        "[package]\nname = \"native-test\"\nversion = \"0.1.0\"\n\n[[ffi.c]]\nname = \"posix\"\nabi = \"c-v1\"\ntargets = [\"linux\"]\nsources = [\"native/provider.c\"]\nflags = [\"-march=native\"]\n",
    )
    .unwrap();

    let error = load_manifest_native_units(&manifest).unwrap_err();
    assert!(error
        .to_string()
        .contains("compiler and linker flags are not allowed"));
    std::fs::remove_dir_all(root).unwrap();
}
