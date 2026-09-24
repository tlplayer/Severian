use super::*;
use crate::package::{analyze_package, collect_declarations, resolve_imports};
use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};
static NEXT: AtomicUsize = AtomicUsize::new(0);
struct Fixture(PathBuf);
impl Fixture {
    fn new(files: &[(&str, &str)]) -> Self {
        let path = std::env::temp_dir().join(format!(
            "sev-narrow-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::create_dir_all(&path).unwrap();
        for (name, text) in files {
            std::fs::write(path.join(name), text).unwrap();
        }
        Self(path)
    }
    fn graph(&self) -> ModuleGraph {
        severian_modules::resolve(&self.0.join("main.sev")).unwrap()
    }
}
impl Drop for Fixture {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}
fn narrowed(graph: &ModuleGraph) -> (ProgramIndex, ImportPlan) {
    let mut index = collect_declarations(graph).unwrap();
    let plan = resolve_required_imports(graph, &index, collect_import_requirements(graph));
    apply_import_plan(&mut index, &plan);
    (index, plan)
}
#[test]
fn wildcard_variant_use_retains_its_enum_through_reexports() {
    let f = Fixture::new(&[
        ("types.sev", "enum Kind:\n    IntegerKind\n    FloatKind\nenum Unused:\n    NotUsed\nclass Scalar:\n    kind: Kind\ndef accept(value: Scalar) -> Scalar:\n    return value\n"),
        ("lib.sev", "import * from \"types.sev\"\n"),
        ("main.sev", "import * from \"lib.sev\"\ndef run() -> Scalar:\n    return accept(Scalar(IntegerKind))\n"),
    ]);
    let graph = f.graph();
    let typed = analyze_package(&graph, &severian_bootstrap::load().unwrap()).unwrap();
    let scope = &typed.index.modules[&graph.modules.last().unwrap().id].scope.bindings;
    assert!(scope.contains_key("Kind"));
    assert!(!scope.contains_key("Unused"));
}

#[test]
fn automatic_build_narrows_wildcards_and_keeps_overloads() {
    let f=Fixture::new(&[("lib.sev","def needed(value: int) -> int:\n    return value\ndef needed(value: string) -> string:\n    return value\ndef unused() -> int:\n    return 9\n"),
        ("main.sev","import * from \"lib.sev\"\ndef run() -> int:\n    return needed(3)\n")]);
    let graph = f.graph();
    let typed = analyze_package(&graph, &severian_bootstrap::load().unwrap()).unwrap();
    let scope = &typed.index.modules[&graph.modules.last().unwrap().id]
        .scope
        .bindings;
    assert!(matches!(scope.get("needed"),Some(Resolution::OverloadSet(ids)) if ids.len()==2));
    assert!(!scope.contains_key("unused"));
    assert!(std::fs::read_to_string(f.0.join("main.sev"))
        .unwrap()
        .contains("import *"));
}
#[test]
fn formatted_strings_and_binding_order_are_ast_requirements() {
    let f=Fixture::new(&[("lib.sev","def needed() -> int:\n    return 1\ndef other() -> int:\n    return 2\ndef unused() -> int:\n    return 3\n"),
        ("main.sev","import * from \"lib.sev\"\ndef run() -> string:\n    text = f\"{needed()}\"\n    other = other()\n    return text\n")]);
    let graph = f.graph();
    let (index, _) = narrowed(&graph);
    let scope = &index.modules[&graph.modules.last().unwrap().id]
        .scope
        .bindings;
    assert!(scope.contains_key("needed"));
    assert!(scope.contains_key("other"));
    assert!(!scope.contains_key("unused"));
}
#[test]
fn lexical_scopes_do_not_hide_uses_outside_a_branch_or_lambda() {
    let f=Fixture::new(&[("lib.sev","def needed() -> int:\n    return 1\ndef unused() -> int:\n    return 2\n"),
        ("main.sev","import * from \"lib.sev\"\ndef run(unused: int) -> int:\n    if true:\n        needed = 1\n    local = lambda needed: needed\n    return needed()\n")]);
    let graph = f.graph();
    let (index, _) = narrowed(&graph);
    let scope = &index.modules[&graph.modules.last().unwrap().id]
        .scope
        .bindings;
    assert!(scope.contains_key("needed"));
    assert!(!scope.contains_key("unused"));
}
#[test]
fn diamond_cycles_and_reexports_only_propagate_demanded_names() {
    let f=Fixture::new(&[("leaf.sev","def needed() -> int:\n    return 1\ndef unused() -> int:\n    return 2\n"),
        ("a.sev","import * from \"b.sev\"\nimport * from \"leaf.sev\"\n"),
        ("b.sev","import * from \"a.sev\"\nimport * from \"leaf.sev\"\n"),
        ("main.sev","import * from \"a.sev\"\nimport * from \"b.sev\"\ndef run() -> int:\n    return needed()\n")]);
    let mut graph = f.graph();
    let (index, _) = narrowed(&graph);
    for m in &graph.modules {
        if !m.path.ends_with("leaf.sev") {
            assert!(!index.exports[&m.id].contains_key("unused"));
        }
    }
    let root = graph.modules.last().unwrap().id;
    assert!(matches!(
        index.modules[&root].scope.bindings.get("needed"),
        Some(Resolution::Def(_))
    ));
    graph.modules.reverse();
    let (reverse, _) = narrowed(&graph);
    assert_eq!(index.exports, reverse.exports);
    assert_eq!(index.modules, reverse.modules);
}
#[test]
fn namespace_reexports_demand_members_in_the_original_module() {
    let f = Fixture::new(&[
        (
            "leaf.sev",
            "def needed() -> int:\n    return 1\ndef unused() -> int:\n    return 2\n",
        ),
        ("facade.sev", "import * from \"leaf.sev\" as api\n"),
        (
            "main.sev",
            "from \"facade.sev\" import api\ndef run() -> int:\n    return api.needed()\n",
        ),
    ]);
    let graph = f.graph();
    analyze_package(&graph, &severian_bootstrap::load().unwrap()).unwrap();
}
#[test]
fn ambiguity_is_not_resolved_by_picking_the_first_provider() {
    let f=Fixture::new(&[("a.sev","class Value:\n    x: int\n"),("b.sev","class Value:\n    x: int\n"),
        ("main.sev","import * from \"a.sev\"\nimport * from \"b.sev\"\ndef run(value: Value) -> Value:\n    return value\n")]);
    let graph = f.graph();
    let (index, _) = narrowed(&graph);
    assert!(
        matches!(index.modules[&graph.modules.last().unwrap().id].scope.bindings.get("Value"),Some(Resolution::Ambiguous(ids)) if ids.len()==2)
    );
}
#[test]
fn unused_import_keeps_the_dependency_and_initializer() {
    let f = Fixture::new(&[
        (
            "lib.sev",
            "seed = 7\ndef unused() -> int:\n    return seed\n",
        ),
        (
            "main.sev",
            "import * from \"lib.sev\"\ndef run() -> int:\n    return 1\n",
        ),
    ]);
    let graph = f.graph();
    let typed = analyze_package(&graph, &severian_bootstrap::load().unwrap()).unwrap();
    assert_eq!(typed.hir.modules.len(), 2);
    assert!(!typed.index.modules[&graph.modules.last().unwrap().id]
        .scope
        .bindings
        .contains_key("seed"));
}
#[test]
fn export_growth_is_bounded_by_requested_names() {
    let mut library = String::new();
    for i in 0..100 {
        library.push_str(&format!("def value{i}() -> int:\n    return {i}\n"));
    }
    let f = Fixture::new(&[
        ("leaf.sev", &library),
        ("a.sev", "import * from \"leaf.sev\"\n"),
        ("b.sev", "import * from \"a.sev\"\n"),
        (
            "main.sev",
            "import * from \"b.sev\"\ndef run() -> int:\n    return value0()\n",
        ),
    ]);
    let graph = f.graph();
    let (index, plan) = narrowed(&graph);
    let mut legacy = collect_declarations(&graph).unwrap();
    resolve_imports(&graph, &mut legacy);
    assert_eq!(legacy.exports.values().map(|e| e.len()).sum::<usize>(), 401);
    assert_eq!(index.exports.values().map(|e| e.len()).sum::<usize>(), 104);
    assert_eq!(plan.imported_bindings, 3);
}

#[test]
fn source_import_loader_and_selection_parse() {
    let repository = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../..");
    for name in ["source.sev", "imports.sev", "../../tests/imports.sev"] {
        let path = repository
            .join("sev_compiler/frontend/modules/modules/src")
            .join(name);
        let text = std::fs::read_to_string(&path).unwrap();
        let source = severian_source::SourceFile::virtual_source(&path, text);
        severian_parser::parse(&severian_lexer::scan(&source).unwrap()).unwrap();
    }
}

#[test]
fn extensible_import_spelling_still_uses_demand_resolution() {
    let f = Fixture::new(&[
        (
            "package.json",
            "{\"language\":{\"explicit-imports\":false}}",
        ),
        (
            "lib.sev",
            "def needed() -> int:\n    return 1\ndef unused() -> int:\n    return 2\n",
        ),
        (
            "main.sev",
            "import * from \"lib.sev\"\ndef run() -> int:\n    return needed()\n",
        ),
    ]);
    let graph = f.graph();
    let (index, _) = narrowed(&graph);
    assert!(!index.modules[&graph.modules.last().unwrap().id]
        .scope
        .bindings
        .contains_key("unused"));
}

#[test]
fn narrowed_namespace_with_only_a_constructor_is_not_a_tensor_receiver() {
    let f = Fixture::new(&[
        ("lib.sev", "class Problem:\n    code: int\ndef unused() -> int:\n    return 9\n"),
        ("main.sev", "import * from \"lib.sev\" as diagnostics\ndef run() -> diagnostics.Problem:\n    return diagnostics.Problem(7)\n"),
    ]);
    let graph = f.graph();
    let typed = analyze_package(&graph, &severian_bootstrap::load().unwrap()).unwrap();
    severian_mir::build(&typed.hir).unwrap();
}
