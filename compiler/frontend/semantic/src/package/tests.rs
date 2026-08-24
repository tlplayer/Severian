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
fn imported_classes_and_list_results_keep_package_wide_types() {
    let root = temporary();
    std::fs::write(
        root.join("filesystem.sev"),
        "class Metadata:\n    path: string\n    size: int\ndef stat(path: string) -> Metadata:\n    return Metadata(path, 8)\ndef entries() -> list[string]:\n    return []\n",
    )
    .unwrap();
    std::fs::write(
        root.join("app.sev"),
        "import \"filesystem.sev\" as filesystem\ndef selected() -> int:\n    information = filesystem.stat(\"/tmp/example\")\n    values = filesystem.entries()\n    assert(size(values) == 0)\n    return information.size\n",
    )
    .unwrap();
    let graph = severian_modules::resolve(&root.join("app.sev")).unwrap();
    let universal = severian_bootstrap::load().unwrap();
    let typed = analyze_package(&graph, &universal).unwrap();
    assert_eq!(typed.hir.modules.len(), 2);
    assert_eq!(
        typed
            .hir
            .modules
            .iter()
            .flat_map(|module| &module.classes)
            .count(),
        2
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
fn mutable_global_reassignment_is_not_indexed_as_a_second_declaration() {
    let root = temporary();
    let source = root.join("mutable-global.sev");
    std::fs::write(&source, "value := \"one\"\nvalue = \"two\"\n").unwrap();
    let universal = severian_bootstrap::load().unwrap();
    let typed = analyze_package(&severian_modules::resolve(&source).unwrap(), &universal).unwrap();
    let module = &typed.hir.modules[0];
    assert_eq!(module.bindings.len(), 2);
    assert_eq!(module.bindings[0].variable, module.bindings[1].variable);
    assert!(module.bindings.iter().all(|binding| binding.mutable));
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
fn one_generic_definition_collects_multiple_ordered_instances() {
    let root = temporary();
    let source = root.join("instances.sev");
    std::fs::write(
        &source,
        "def identity[T](value: T) -> T:\n    return value\ndef number() -> int:\n    return identity(42)\ndef text() -> string:\n    return identity(\"sev\")\n",
    )
    .unwrap();
    let universal = severian_bootstrap::load().unwrap();
    let typed = analyze_package(&severian_modules::resolve(&source).unwrap(), &universal).unwrap();
    let generic = typed
        .index
        .definitions
        .values()
        .find(|definition| definition.name == "identity")
        .unwrap()
        .id;
    let instances = typed.hir.modules[0]
        .functions
        .iter()
        .filter(|function| function.definition == generic)
        .collect::<Vec<_>>();
    assert_eq!(instances.len(), 2);
    assert_ne!(instances[0].id, instances[1].id);
    assert_ne!(instances[0].substitution, instances[1].substitution);
    std::fs::remove_dir_all(root).unwrap();
}

#[test]
fn source_operator_traits_constrain_generic_instances() {
    let root = temporary();
    let source = root.join("ordered.sev");
    std::fs::write(
        &source,
        "trait Ordered:\n    operator <(right: Self) -> bool\ndef minimum[T: Ordered](left: T, right: T) -> T:\n    if left < right:\n        return left\n    return right\ndef selected() -> int:\n    return minimum(2, 1)\n",
    )
    .unwrap();
    let universal = severian_bootstrap::load().unwrap();
    analyze_package(&severian_modules::resolve(&source).unwrap(), &universal).unwrap();

    std::fs::write(
        &source,
        "trait Ordered:\n    operator <(right: Self) -> bool\ndef minimum[T: Ordered](left: T, right: T) -> T:\n    return left\ndef selected() -> bool:\n    return minimum(true, false)\n",
    )
    .unwrap();
    let error =
        analyze_package(&severian_modules::resolve(&source).unwrap(), &universal).unwrap_err();
    assert_eq!(error.code, "E000217");
    assert!(error.message.contains("bool"));
    assert!(error.message.contains("Ordered"));
    std::fs::remove_dir_all(root).unwrap();
}

#[test]
fn source_method_traits_authorize_operators_for_each_generic_instance() {
    let root = temporary();
    let source = root.join("numeric.sev");
    std::fs::write(
        &source,
        "trait Numeric:\n    def add(other: Self) -> Self\n    def multiply(other: Self) -> Self\n    def less_than(other: Self) -> bool\ndef affine[T: Numeric](x: T, scale: T, bias: T) -> T:\n    y := x * scale\n    y = y + bias\n    if y < bias:\n        return bias\n    return y\ndef integer() -> int:\n    return affine(4, 3, 2)\ndef floating() -> f64:\n    return affine(4.0, 0.5, 1.0)\n",
    )
    .unwrap();
    let universal = severian_bootstrap::load().unwrap();
    let typed = analyze_package(&severian_modules::resolve(&source).unwrap(), &universal).unwrap();
    let generic = typed
        .index
        .definitions
        .values()
        .find(|definition| definition.name == "affine")
        .unwrap()
        .id;
    let instances = typed.hir.modules[0]
        .functions
        .iter()
        .filter(|function| function.definition == generic)
        .collect::<Vec<_>>();
    assert_eq!(instances.len(), 2);
    assert_ne!(instances[0].substitution, instances[1].substitution);
    assert_eq!(typed.hir.modules[0].traits[0].name, "Numeric");
    std::fs::remove_dir_all(root).unwrap();
}

#[test]
fn trait_typed_parameters_specialize_to_source_classes() {
    let root = temporary();
    let source = root.join("drawable.sev");
    std::fs::write(
        &source,
        "trait Named:\n    def name() -> string\ntrait Drawable:\n    Named\n    def draw()\nclass Button: Drawable\n    label: string\n    def name() -> string:\n        return label\n    def draw():\n        pass\ndef render(item: Drawable):\n    observed = item.name()\n    item.draw()\ndef main():\n    button = Button(\"Save\")\n    render(button)\n",
    )
    .unwrap();
    let universal = severian_bootstrap::load().unwrap();
    let typed = analyze_package(&severian_modules::resolve(&source).unwrap(), &universal).unwrap();
    let render = typed.index.definitions.values().find(|definition| {
        definition.name == "render" && matches!(definition.kind, DefKind::Function(_))
    });
    let render = render.unwrap();
    let instances = typed.hir.modules[0]
        .functions
        .iter()
        .filter(|function| function.definition == render.id)
        .collect::<Vec<_>>();
    assert_eq!(instances.len(), 1);
    assert!(!instances[0].substitution.0.is_empty());
    severian_mir::build(&typed.hir).unwrap();

    std::fs::write(
        &source,
        "trait Drawable:\n    def draw()\nclass Label:\n    text: string\ndef render(item: Drawable):\n    item.draw()\ndef main():\n    label = Label(\"Save\")\n    render(label)\n",
    )
    .unwrap();
    let error =
        analyze_package(&severian_modules::resolve(&source).unwrap(), &universal).unwrap_err();
    assert_eq!(error.code, "E000217");
    assert!(error.message.contains("Label"));
    assert!(error.message.contains("Drawable"));
    std::fs::remove_dir_all(root).unwrap();
}

#[test]
fn concrete_overloads_rank_ahead_of_generic_fallbacks() {
    let root = temporary();
    let source = root.join("ranking.sev");
    std::fs::write(
        &source,
        "def encode(value: int) -> int:\n    return 1\ndef encode[T](value: T) -> int:\n    return 2\ndef selected() -> int:\n    return encode(42)\n",
    )
    .unwrap();
    let universal = severian_bootstrap::load().unwrap();
    analyze_package(&severian_modules::resolve(&source).unwrap(), &universal).unwrap();
    std::fs::remove_dir_all(root).unwrap();
}

#[test]
fn generic_bodies_require_operator_capabilities_before_instantiation() {
    let root = temporary();
    let source = root.join("body-constraints.sev");
    std::fs::write(
        &source,
        "def smaller[T](left: T, right: T) -> bool:\n    return left < right\n",
    )
    .unwrap();
    let universal = severian_bootstrap::load().unwrap();
    let error =
        analyze_package(&severian_modules::resolve(&source).unwrap(), &universal).unwrap_err();
    assert_eq!(error.code, "E000219");

    std::fs::write(
        &source,
        "def smaller[T: Integer[T]](left: T, right: T) -> bool:\n    return left < right\ndef selected() -> bool:\n    return smaller(1, 2)\n",
    )
    .unwrap();
    analyze_package(&severian_modules::resolve(&source).unwrap(), &universal).unwrap();
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
