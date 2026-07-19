use severian_mlir::Module;

#[test]
fn preserves_emitted_text() {
    let module = Module::new("module {}\n".into());
    assert_eq!(module.as_str(), "module {}\n");
}
