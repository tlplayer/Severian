use severian_package::{load_official_interfaces, GraphOperation};
use std::path::PathBuf;

fn parse(source: &str) -> severian_ast::Module {
    let tokens = severian_lexer::lex(source).unwrap();
    severian_parser::parse(&tokens).unwrap()
}

fn library_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../library")
}

#[test]
fn loads_package_owned_symbols_and_fusion_contracts() {
    let module = parse("import model\nimport tensor\n");
    let interfaces = load_official_interfaces(&module, &library_root()).unwrap();
    let model = interfaces
        .iter()
        .find(|interface| interface.name == "model")
        .unwrap();

    assert_eq!(model.compiler.symbols["Relu"], "reluList");
    assert!(model
        .compiler
        .fusion_aliases
        .iter()
        .any(|alias| { alias.function == "model.reluList" && alias.target == "tensor.relu" }));
    assert!(model.compiler.graph_rules.iter().any(|rule| {
        rule.function == "model.graphMatmul" && rule.operation == GraphOperation::Matmul
    }));
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

    assert_eq!(interfaces.len(), 1);
    assert_eq!(interfaces[0].name, "tensor");
}
