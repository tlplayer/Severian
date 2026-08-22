use super::*;
use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};

static NEXT: AtomicUsize = AtomicUsize::new(0);

fn temporary() -> PathBuf {
    let path = std::env::temp_dir().join(format!(
        "severian-semantic-package-{}-{}",
        std::process::id(),
        NEXT.fetch_add(1, Ordering::Relaxed)
    ));
    std::fs::create_dir_all(&path).unwrap();
    path
}

#[test]
fn qualified_imported_overloads_are_checked_in_the_package_namespace() {
    let root = temporary();
    std::fs::write(
        root.join("b.sev"),
        "def choose(value: i32) -> i32:\n    return value\ndef choose(value: i64) -> i64:\n    return value\n",
    )
    .unwrap();
    std::fs::write(
        root.join("a.sev"),
        "import \"b.sev\" as b\ndef selected(value: i32) -> i32:\n    return b.choose(value)\n",
    )
    .unwrap();
    let graph = severian_modules::resolve(&root.join("a.sev")).unwrap();
    let universal = severian_bootstrap::load().unwrap();
    let typed = analyze_package(&graph, &universal).unwrap();
    assert_eq!(typed.hir.modules.len(), 2);
    assert_eq!(
        typed
            .hir
            .modules
            .iter()
            .map(|module| module.functions.len())
            .sum::<usize>(),
        3
    );
    std::fs::remove_dir_all(root).unwrap();
}

#[test]
fn declaration_identity_does_not_depend_on_unrelated_source_items() {
    let root = temporary();
    let source = root.join("identity.sev");
    std::fs::write(&source, "def kept(value: i32) -> i32:\n    return value\n").unwrap();
    let universal = severian_bootstrap::load().unwrap();
    let first = analyze_package(&severian_modules::resolve(&source).unwrap(), &universal).unwrap();
    let kept = |program: &TypedProgram| {
        program
            .index
            .definitions
            .values()
            .find(|definition| definition.name == "kept")
            .unwrap()
            .id
    };
    let before = kept(&first);
    let lowered_before = first.hir.modules[0]
        .functions
        .iter()
        .find(|function| function.name == "kept")
        .unwrap()
        .id;
    std::fs::write(
        &source,
        "def unrelated():\n    pass\ndef kept(value: i32) -> i32:\n    return value\n",
    )
    .unwrap();
    let second = analyze_package(&severian_modules::resolve(&source).unwrap(), &universal).unwrap();
    assert_eq!(before, kept(&second));
    assert_eq!(
        lowered_before,
        second.hir.modules[0]
            .functions
            .iter()
            .find(|function| function.name == "kept")
            .unwrap()
            .id
    );
    std::fs::remove_dir_all(root).unwrap();
}

#[test]
fn imported_generic_overload_is_specialized_after_declaration_collection() {
    let root = temporary();
    std::fs::write(
        root.join("b.sev"),
        "def choose[T](value: T) -> T:\n    return value\ndef choose(value: string) -> string:\n    return value\n",
    )
    .unwrap();
    std::fs::write(
        root.join("a.sev"),
        "import \"b.sev\" as b\ndef selected(value: i32) -> i32:\n    return b.choose(value)\n",
    )
    .unwrap();
    let graph = severian_modules::resolve(&root.join("a.sev")).unwrap();
    let universal = severian_bootstrap::load().unwrap();
    let typed = analyze_package(&graph, &universal).unwrap();
    let generic = typed
        .index
        .definitions
        .values()
        .find(|definition| {
            matches!(
                &definition.kind,
                DefKind::Function(function) if !function.type_parameters.is_empty()
            )
        })
        .unwrap();
    let DefKind::Function(signature) = &generic.kind else {
        unreachable!()
    };
    assert_eq!(signature.type_parameters, ["T"]);
    assert_eq!(
        typed
            .hir
            .modules
            .iter()
            .map(|module| module.functions.len())
            .sum::<usize>(),
        3
    );
    std::fs::remove_dir_all(root).unwrap();
}

#[test]
fn uncalled_generic_declarations_are_indexed_without_forcing_a_body_instance() {
    let root = temporary();
    let source = root.join("generic.sev");
    std::fs::write(
        &source,
        "def identity[T](value: T) -> T:\n    return value\n",
    )
    .unwrap();
    let graph = severian_modules::resolve(&source).unwrap();
    let universal = severian_bootstrap::load().unwrap();
    let typed = analyze_package(&graph, &universal).unwrap();
    assert_eq!(typed.index.definitions.len(), 1);
    assert!(typed.hir.modules[0].functions.is_empty());
    std::fs::remove_dir_all(root).unwrap();
}

#[test]
fn declaration_only_module_cycles_can_resolve_mutually_recursive_bodies() {
    let root = temporary();
    std::fs::write(
        root.join("a.sev"),
        "import \"b.sev\" as b\ndef a() -> int:\n    return b.b()\n",
    )
    .unwrap();
    std::fs::write(
        root.join("b.sev"),
        "import \"a.sev\" as a\ndef b() -> int:\n    return a.a()\n",
    )
    .unwrap();
    let graph = severian_modules::resolve(&root.join("a.sev")).unwrap();
    let universal = severian_bootstrap::load().unwrap();
    let typed = analyze_package(&graph, &universal).unwrap();
    assert_eq!(
        typed
            .index
            .definitions
            .values()
            .filter(|definition| matches!(definition.kind, DefKind::Function(_)))
            .count(),
        2
    );
    assert_eq!(typed.hir.modules.len(), 2);
    std::fs::remove_dir_all(root).unwrap();
}
