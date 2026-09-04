use super::*;
use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};

static NEXT: AtomicUsize = AtomicUsize::new(0);

#[test]
fn generic_parameter_kinds_keep_types_dimensions_and_shapes_separate() {
    let source = severian_source::SourceFile::virtual_source(
        "shape-kinds.sev",
        "def contract[T: TensorElement, B: Dim, Batch: Shape, *Tail: Dim](value: T) -> T:\n    return value\n",
    );
    let tokens = severian_lexer::scan(&source).unwrap();
    let ast = severian_parser::parse(&tokens).unwrap();
    let severian_ast::Item::Function(function) = &ast.items[0] else {
        panic!("expected function")
    };
    let parameters = generic_parameters(&function.type_parameters, &function.constraints);
    assert_eq!(parameters.len(), 4);
    assert_eq!(
        parameters[0].kind,
        severian_universal::GenericParamKind::Type
    );
    assert_eq!(
        parameters[1].kind,
        severian_universal::GenericParamKind::Dimension
    );
    assert_eq!(
        parameters[2].kind,
        severian_universal::GenericParamKind::Shape
    );
    assert_eq!(
        parameters[3].kind,
        severian_universal::GenericParamKind::Shape
    );
    assert!(parameters[3].variadic);
}

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
fn primitive_class_collapses_to_the_universal_identity_and_installs_operators() {
    let source = severian_source::SourceFile::virtual_source(
        "bool.sev",
        "class bool: Primitive :\n    default_literal: bool = true\n    operator &(right: bool) -> bool:\n        return right\n",
    );
    let ast = severian_parser::parse(&severian_lexer::scan(&source).unwrap()).unwrap();
    let graph = severian_modules::ModuleGraph {
        modules: vec![severian_modules::ResolvedModule {
            id: severian_modules::ModuleId(1),
            path: PathBuf::from("bool.sev"),
            source,
            package: severian_modules::PackageId(0),
            ast,
            imports: Vec::new(),
        }],
    };
    let universal = severian_bootstrap::load().unwrap();
    let universal_bool = universal.types.resolve_name("bool").unwrap();
    let typed = analyze_package(&graph, &universal).unwrap();

    assert_eq!(typed.types.resolve_name("bool"), Some(universal_bool));
    assert_eq!(
        typed
            .types
            .definitions()
            .filter(|definition| definition.name == "bool")
            .count(),
        1
    );
    assert!(typed.types.supports_binary(
        severian_universal::BinaryOperator::BitwiseAnd,
        universal_bool
    ));
    assert!(typed
        .hir
        .modules
        .iter()
        .flat_map(|module| &module.classes)
        .all(|class| class.id != universal_bool));
}

#[test]
fn injected_prelude_declarations_are_local_and_not_reexported() {
    let mut source = severian_source::SourceFile::virtual_source(
        "bootstrap-prelude.sev",
        "def print(value: string) -> i32\n",
    );
    source.id = severian_source::SourceId(u32::MAX);
    let ast = severian_parser::parse(&severian_lexer::scan(&source).unwrap()).unwrap();
    let module = severian_modules::ModuleId(1);
    let graph = severian_modules::ModuleGraph {
        modules: vec![severian_modules::ResolvedModule {
            id: module,
            path: PathBuf::from("bootstrap-prelude.sev"),
            source,
            package: severian_modules::PackageId(0),
            ast,
            imports: Vec::new(),
        }],
    };

    let index = collect_declarations(&graph).unwrap();
    assert!(index.modules[&module].scope.bindings.contains_key("print"));
    assert!(!index.exports[&module].contains_key("print"));
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
fn imported_trait_namespace_dispatch_keeps_the_package_qualifier() {
    let root = temporary();
    std::fs::write(
        root.join("models.sev"),
        "trait Model:\n    @model\n    def load(name: string) -> string with { (name) -> bool }\ndef load_tiny(name: string) -> string:\n    return name\nclass Tiny: Model\n    def load(name: string) -> string with { name == \"tiny\" }:\n        return load_tiny(name)\n",
    )
    .unwrap();
    std::fs::write(
        root.join("app.sev"),
        "import \"models.sev\" as ai\ndef selected() -> string:\n    return ai.model.load(\"tiny\")\n",
    )
    .unwrap();
    let graph = severian_modules::resolve(&root.join("app.sev")).unwrap();
    let universal = severian_bootstrap::load().unwrap();
    let typed = analyze_package(&graph, &universal).unwrap();
    severian_mir::build(&typed.hir).unwrap();
    std::fs::remove_dir_all(root).unwrap();
}

#[test]
fn extensions_survive_package_indexing_and_lowering() {
    let root = temporary();
    std::fs::write(
        root.join("extensions.sev"),
        "class Counter:\n    value: int\n    def get() -> int:\n        return value\n\nextend Counter:\n    def reset() -> Counter:\n        return Counter(0)\n\n@combinatorics\nextend set[T]:\n    operator +(other: set[T]) -> set[T]:\n        result := self\n        for value in other:\n            result.add(value)\n        return result\n\n@combinatorics(+)\ndef union(left: set[int], right: set[int]) -> set[int]:\n    return left + right\n\ndef selected() -> int:\n    counter := Counter(1).reset()\n    values := union({1, 2}, {2, 3})\n    return counter.get()\n",
    )
    .unwrap();
    let graph = severian_modules::resolve(&root.join("extensions.sev")).unwrap();
    let universal = severian_bootstrap::load().unwrap();
    let typed = analyze_package(&graph, &universal).unwrap();
    severian_mir::build(&typed.hir).unwrap();
    std::fs::remove_dir_all(root).unwrap();
}

#[test]
fn qualified_imported_types_resolve_in_annotations_and_constructors() {
    let root = temporary();
    std::fs::write(
        root.join("model.sev"),
        "class Item:\n    value: i32\ndef make(value: i32) -> Item:\n    return Item(value)\n",
    )
    .unwrap();
    std::fs::write(
        root.join("app.sev"),
        "import \"model.sev\" as model\ndef read(item: model.Item) -> i32:\n    return item.value\ndef build(value: i32) -> model.Item:\n    return model.Item(value)\ndef selected() -> i32:\n    return read(model.make(7))\n",
    )
    .unwrap();
    let graph = severian_modules::resolve(&root.join("app.sev")).unwrap();
    let universal = severian_bootstrap::load().unwrap();
    let typed = analyze_package(&graph, &universal).unwrap();
    severian_mir::build(&typed.hir).unwrap();
    std::fs::remove_dir_all(root).unwrap();
}

#[test]
fn transitively_reachable_class_methods_keep_origin_types_and_functions() {
    let root = temporary();
    std::fs::write(
        root.join("storage.sev"),
        "class Store:\n    handle: int\ndef close_store(store: Store) -> bool:\n    return store.handle != 0\n",
    )
    .unwrap();
    std::fs::write(
        root.join("codec.sev"),
        "import \"storage.sev\" as storage\nclass Decoder:\n    parameters: storage.Store\n    def close() -> bool:\n        return storage.close_store(parameters)\n",
    )
    .unwrap();
    std::fs::write(
        root.join("model.sev"),
        "import \"codec.sev\" as codec\nclass Model:\n    decoder: codec.Decoder\n    def close() -> bool:\n        return decoder.close()\n",
    )
    .unwrap();
    std::fs::write(
        root.join("app.sev"),
        "import \"model.sev\" as model\ndef selected(value: model.Model) -> bool:\n    return value.close()\n",
    )
    .unwrap();
    let graph = severian_modules::resolve(&root.join("app.sev")).unwrap();
    let universal = severian_bootstrap::load().unwrap();
    let typed = analyze_package(&graph, &universal).unwrap();
    severian_mir::build(&typed.hir).unwrap();
    std::fs::remove_dir_all(root).unwrap();
}

#[test]
fn package_signatures_preserve_named_parameters_and_defaults() {
    let root = temporary();
    std::fs::write(
        root.join("math.sev"),
        "def scale(value: float, factor: float = 2.0) -> float:\n    return value * factor\n",
    )
    .unwrap();
    std::fs::write(
        root.join("app.sev"),
        "import \"math.sev\" as math\ndef local(value: float, factor: float = 3.0) -> float:\n    return value * factor\ndef selected() -> float:\n    first = local(value=4.0)\n    return math.scale(value=first)\n",
    )
    .unwrap();
    let graph = severian_modules::resolve(&root.join("app.sev")).unwrap();
    let universal = severian_bootstrap::load().unwrap();
    let typed = analyze_package(&graph, &universal).unwrap();
    severian_mir::build(&typed.hir).unwrap();
    std::fs::remove_dir_all(root).unwrap();
}

#[test]
fn imported_union_parameters_preserve_members_and_accept_injections() {
    let root = temporary();
    std::fs::write(
        root.join("convert.sev"),
        "def to_float(value: string | int | float) -> float:\n    return float(value)\n",
    )
    .unwrap();
    std::fs::write(
        root.join("app.sev"),
        "import \"convert.sev\" as convert\ndef selected() -> float:\n    return convert.to_float(\"4.5\") + convert.to_float(4) + convert.to_float(4.5)\n",
    )
    .unwrap();
    let graph = severian_modules::resolve(&root.join("app.sev")).unwrap();
    let universal = severian_bootstrap::load().unwrap();
    let typed = analyze_package(&graph, &universal).unwrap();
    severian_mir::build(&typed.hir).unwrap();
    std::fs::remove_dir_all(root).unwrap();
}

#[test]
fn custom_contract_errors_flow_into_fallible_else_handlers() {
    let root = temporary();
    std::fs::write(
        root.join("fallible.sev"),
        "class DivideError: Error\n    message: string\ndef divide(value: f64, divisor: f64) -> f64 | DivideError with { divisor != 0.0 -> DivideError(\"zero\") }:\n    return value / divisor\ntest:\n    divide(1, 0) else error:\n        assert(error == DivideError)\n",
    )
    .unwrap();
    let graph = severian_modules::resolve(&root.join("fallible.sev")).unwrap();
    let universal = severian_bootstrap::load().unwrap();
    let typed = analyze_package_with_context(
        &graph,
        &universal,
        PackageAnalysisContext {
            test_package: Some(graph.modules[0].package),
        },
    )
    .unwrap();
    severian_mir::build(&typed.hir).unwrap();
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
fn explicit_type_arguments_specialize_imported_and_nested_generic_calls() {
    let root = temporary();
    std::fs::write(
        root.join("generic.sev"),
        "def identity[T](value: T) -> T:\n    return value\ndef forward[T](value: T) -> T:\n    return identity[T](value)\n",
    )
    .unwrap();
    std::fs::write(
        root.join("app.sev"),
        "import \"generic.sev\" as generic\ndef selected(value: i32) -> i32:\n    return generic.forward[i32](value)\n",
    )
    .unwrap();
    let graph = severian_modules::resolve(&root.join("app.sev")).unwrap();
    let universal = severian_bootstrap::load().unwrap();
    let typed = analyze_package(&graph, &universal).unwrap();
    severian_mir::build(&typed.hir).unwrap();
    let substitutions = typed
        .hir
        .modules
        .iter()
        .flat_map(|module| &module.functions)
        .filter(|function| !function.substitution.0.is_empty())
        .count();
    assert_eq!(substitutions, 2);
    std::fs::remove_dir_all(root).unwrap();
}

#[test]
fn complete_generic_body_specialization_rewrites_every_type_application_position() {
    let root = temporary();
    let source = root.join("complete.sev");
    std::fs::write(
        &source,
        "def complete[T](value: T = factory[T]()) -> T:\n    local: Box[T] = Box[T](forward[list[T]](value))\n    return local\n",
    )
    .unwrap();
    let graph = severian_modules::resolve(&source).unwrap();
    let function = graph.modules[0]
        .ast
        .items
        .iter()
        .find_map(|item| match item {
            severian_ast::Item::Function(function) => Some(function),
            _ => None,
        })
        .unwrap();
    let substitution = [("T".into(), "bf16".into())]
        .into_iter()
        .collect::<super::generic::Substitution>();
    let specialized = super::generic::specialize_function(function, &substitution);

    assert_eq!(
        specialized.parameters[0].annotation.simple_name(),
        Some("bf16")
    );
    assert_eq!(specialized.result.simple_name(), Some("bf16"));
    let default = specialized.parameters[0].default.as_ref().unwrap();
    let severian_ast::ExpressionKind::Call { callee, .. } = &default.kind else {
        panic!("default must remain a call")
    };
    let severian_ast::ExpressionKind::TypeApplication { arguments, .. } = &callee.kind else {
        panic!("default callee must remain a type application")
    };
    assert_eq!(arguments[0].simple_name(), Some("bf16"));

    let body = specialized.body.as_ref().unwrap();
    let severian_ast::Statement::Binding(local) = &body[0] else {
        panic!("first statement must remain a binding")
    };
    let (name, annotation_arguments) = local.annotation.as_ref().unwrap().named_parts().unwrap();
    assert_eq!(name, "Box");
    assert_eq!(annotation_arguments[0].simple_name(), Some("bf16"));
    let severian_ast::ExpressionKind::Call { callee, arguments } = &local.value.kind else {
        panic!("local initializer must remain a constructor call")
    };
    let severian_ast::ExpressionKind::TypeApplication {
        arguments: class_arguments,
        ..
    } = &callee.kind
    else {
        panic!("constructor must retain its type application")
    };
    assert_eq!(class_arguments[0].simple_name(), Some("bf16"));
    let severian_ast::ExpressionKind::Call {
        callee: nested_callee,
        ..
    } = &arguments[0].value.kind
    else {
        panic!("constructor argument must remain the nested call")
    };
    let severian_ast::ExpressionKind::TypeApplication {
        arguments: nested_arguments,
        ..
    } = &nested_callee.kind
    else {
        panic!("nested call must retain its type application")
    };
    let (name, list_arguments) = nested_arguments[0].named_parts().unwrap();
    assert_eq!(name, "list");
    assert_eq!(list_arguments[0].simple_name(), Some("bf16"));
    std::fs::remove_dir_all(root).unwrap();
}

#[test]
fn package_dependency_generic_call_keeps_definition_and_substitution_in_hir_and_mir() {
    let root = temporary();
    let tensor_root = root.join("tensor");
    let app_root = root.join("app");
    std::fs::create_dir_all(&tensor_root).unwrap();
    std::fs::create_dir_all(&app_root).unwrap();
    let tensor_source = tensor_root.join("lib.sev");
    let app_source = app_root.join("main.sev");
    std::fs::write(
        &tensor_source,
        "class Tensor[T]:\n    value: T\ndef relay[T](value: T) -> T:\n    return value\ndef load[T](entry: T) -> Tensor[T]:\n    local: T = relay[T](entry)\n    return Tensor[T](local)\n",
    )
    .unwrap();
    std::fs::write(
        &app_source,
        "import tensor\ndef selected(entry: bf16) -> tensor.Tensor[bf16]:\n    return tensor.load[bf16](entry)\n",
    )
    .unwrap();
    let app = severian_modules::PackageId(0);
    let tensor = severian_modules::PackageId(1);
    let packages = severian_modules::PackageGraph {
        root: app,
        packages: std::collections::BTreeMap::from([
            (
                app,
                severian_modules::ResolvedPackage {
                    id: app,
                    root: app_root.clone(),
                    library: app_source.clone(),
                    dependencies: std::collections::BTreeMap::from([("tensor".into(), tensor)]),
                },
            ),
            (
                tensor,
                severian_modules::ResolvedPackage {
                    id: tensor,
                    root: tensor_root,
                    library: tensor_source,
                    dependencies: std::collections::BTreeMap::new(),
                },
            ),
        ]),
    };
    let graph = severian_modules::resolve_with_packages(&app_source, &packages).unwrap();
    let universal = severian_bootstrap::load().unwrap();
    let typed = analyze_package(&graph, &universal).unwrap();
    let load_definition = typed
        .index
        .definitions
        .values()
        .find(|definition| {
            definition.name == "load" && definition.id.package == u128::from(tensor.0)
        })
        .unwrap();
    let DefKind::Function(load_interface) = &load_definition.kind else {
        panic!("tensor.load must be indexed as a function")
    };
    assert!(load_interface.generic_body.is_some());
    let relay_definition = typed
        .index
        .definitions
        .values()
        .find(|definition| {
            definition.name == "relay" && definition.id.package == u128::from(tensor.0)
        })
        .unwrap();

    let bf16 = typed.types.resolve_name("bf16").unwrap();
    let expected =
        severian_universal::Substitution::new([(severian_universal::GenericParamId(0), bf16)]);
    let selected = typed
        .hir
        .modules
        .iter()
        .flat_map(|module| &module.functions)
        .find(|function| function.name == "selected")
        .unwrap();
    let body = selected.body.as_ref().unwrap();
    let severian_hir::Statement::Return(Some(call)) = &body.statements[0] else {
        panic!("selected must return the tensor.load call")
    };
    let severian_hir::ExpressionKind::Call { callee, .. } = &call.kind else {
        panic!("selected return must remain a call")
    };
    let severian_hir::Callee::Direct {
        function,
        substitution,
        ..
    } = callee
    else {
        panic!("tensor.load must be a direct generic call")
    };
    assert_eq!(*function, load_definition.id);
    assert_eq!(*substitution, expected);
    assert_eq!(substitution.0.len(), 1);

    let load_instance = typed
        .hir
        .modules
        .iter()
        .flat_map(|module| &module.functions)
        .find(|function| function.definition == load_definition.id)
        .unwrap();
    assert_eq!(load_instance.name, "load");
    assert_eq!(load_instance.substitution, expected);
    let load_module = typed
        .hir
        .modules
        .iter()
        .find(|module| {
            module
                .functions
                .iter()
                .any(|function| function.id == load_instance.id)
        })
        .unwrap();
    let load_body = load_instance.body.as_ref().unwrap();
    let severian_hir::Statement::Binding(local) = load_body.statements[0] else {
        panic!("specialized dependency body must retain its local")
    };
    let local = load_module
        .bindings
        .iter()
        .find(|binding| binding.id == local)
        .unwrap();
    let severian_hir::ExpressionKind::Call { callee, .. } = &local.value.kind else {
        panic!("dependency local must retain its nested relay call")
    };
    let severian_hir::Callee::Direct {
        function,
        substitution,
        ..
    } = callee
    else {
        panic!("nested relay must remain a direct generic call")
    };
    assert_eq!(*function, relay_definition.id);
    assert_eq!(*substitution, expected);

    let mir = severian_mir::build(&typed.hir).unwrap();
    let selected = mir
        .functions
        .iter()
        .find(|function| function.name == "selected")
        .unwrap();
    let body = selected.body.as_ref().unwrap();
    let (function, substitution) = body
        .blocks
        .iter()
        .find_map(|block| match &block.terminator {
            severian_mir::Terminator::Call {
                callee:
                    severian_mir::Callee::Direct {
                        function,
                        substitution,
                        ..
                    },
                ..
            } => Some((*function, substitution.clone())),
            _ => None,
        })
        .unwrap();
    assert_eq!(function, load_definition.id);
    assert_eq!(substitution, expected);
    std::fs::remove_dir_all(root).unwrap();
}

#[test]
fn every_outer_instance_discovers_its_nested_generic_instance() {
    let root = temporary();
    let source = root.join("instances.sev");
    std::fs::write(
        &source,
        "def inner[T](value: T) -> T:\n    return value\ndef outer[T](value: T) -> T:\n    return inner[T](value)\ndef selected_integer(value: i32) -> i32:\n    return outer[i32](value)\ndef selected_boolean(value: bool) -> bool:\n    return outer[bool](value)\n",
    )
    .unwrap();
    let graph = severian_modules::resolve(&source).unwrap();
    let universal = severian_bootstrap::load().unwrap();
    let typed = analyze_package(&graph, &universal).unwrap();
    let inner = typed
        .index
        .definitions
        .values()
        .find(|definition| definition.name == "inner")
        .unwrap();
    let instances = typed
        .hir
        .modules
        .iter()
        .flat_map(|module| &module.functions)
        .filter(|function| function.definition == inner.id)
        .collect::<Vec<_>>();
    assert_eq!(instances.len(), 2);
    assert!(instances
        .iter()
        .all(|function| function.name == "inner" && function.substitution.0.len() == 1));
    assert_ne!(instances[0].substitution, instances[1].substitution);
    severian_mir::build(&typed.hir).unwrap();
    std::fs::remove_dir_all(root).unwrap();
}

#[test]
fn explicit_type_application_selects_the_matching_generic_arity_and_overload() {
    let root = temporary();
    let source = root.join("overloads.sev");
    std::fs::write(
        &source,
        "def choose[T](value: T) -> T:\n    return value\ndef choose[T, U](value: T, fallback: U) -> U:\n    return fallback\ndef selected(value: i32, fallback: bool) -> bool:\n    return choose[i32, bool](value, fallback)\n",
    )
    .unwrap();
    let graph = severian_modules::resolve(&source).unwrap();
    let universal = severian_bootstrap::load().unwrap();
    let typed = analyze_package(&graph, &universal).unwrap();
    let chosen = typed
        .index
        .definitions
        .values()
        .find(|definition| {
            matches!(
                &definition.kind,
                DefKind::Function(function)
                    if definition.name == "choose" && function.type_parameters.len() == 2
            )
        })
        .unwrap();
    let selected = typed.hir.modules[0]
        .functions
        .iter()
        .find(|function| function.name == "selected")
        .unwrap();
    let body = selected.body.as_ref().unwrap();
    let severian_hir::Statement::Return(Some(call)) = &body.statements[0] else {
        panic!("selected must return the applied generic call")
    };
    let severian_hir::ExpressionKind::Call {
        callee:
            severian_hir::Callee::Direct {
                function,
                substitution,
                ..
            },
        ..
    } = &call.kind
    else {
        panic!("ordinary TypeApplication must lower to a direct HIR call")
    };
    assert_eq!(*function, chosen.id);
    assert_eq!(substitution.0.len(), 2);
    assert_eq!(
        substitution.get(severian_universal::GenericParamId(0)),
        typed.types.resolve_name("i32")
    );
    assert_eq!(
        substitution.get(severian_universal::GenericParamId(1)),
        typed.types.resolve_name("bool")
    );
    std::fs::remove_dir_all(root).unwrap();
}

#[test]
fn explicit_type_application_reports_generic_arity_before_call_lowering() {
    let root = temporary();
    let source = root.join("arity.sev");
    std::fs::write(
        &source,
        "def identity[T](value: T) -> T:\n    return value\ndef selected(value: i32) -> i32:\n    return identity[i32, bool](value)\n",
    )
    .unwrap();
    let universal = severian_bootstrap::load().unwrap();
    let error =
        analyze_package(&severian_modules::resolve(&source).unwrap(), &universal).unwrap_err();
    assert_eq!(error.code, "E000206");
    assert!(error.message.contains("expects 1 generic type argument"));
    assert!(error.message.contains("received 2"));
    std::fs::remove_dir_all(root).unwrap();
}

#[test]
fn explicit_type_application_requires_resolvable_type_arguments() {
    let root = temporary();
    let source = root.join("unresolved.sev");
    std::fs::write(
        &source,
        "def identity[T](value: T) -> T:\n    return value\ndef selected(value: i32) -> i32:\n    return identity[MissingType](value)\n",
    )
    .unwrap();
    let universal = severian_bootstrap::load().unwrap();
    let error =
        analyze_package(&severian_modules::resolve(&source).unwrap(), &universal).unwrap_err();
    assert_eq!(error.code, "E000204");
    assert!(error.message.contains("MissingType"));
    std::fs::remove_dir_all(root).unwrap();
}

#[test]
fn explicit_type_ids_filter_other_materialized_instances() {
    let root = temporary();
    let source = root.join("instance-filter.sev");
    std::fs::write(
        &source,
        "def select[T](value: i32) -> i32:\n    return value\ndef materialize_integer(value: i32) -> i32:\n    return select[i32](value)\ndef materialize_boolean(value: i32) -> i32:\n    return select[bool](value)\ndef chosen(value: i32) -> i32:\n    return select[i32](value)\n",
    )
    .unwrap();
    let graph = severian_modules::resolve(&source).unwrap();
    let universal = severian_bootstrap::load().unwrap();
    let typed = analyze_package(&graph, &universal).unwrap();
    let chosen = typed.hir.modules[0]
        .functions
        .iter()
        .find(|function| function.name == "chosen")
        .unwrap();
    let severian_hir::Statement::Return(Some(call)) = &chosen.body.as_ref().unwrap().statements[0]
    else {
        panic!("chosen must return the explicitly applied call")
    };
    let severian_hir::ExpressionKind::Call {
        callee: severian_hir::Callee::Direct { substitution, .. },
        ..
    } = &call.kind
    else {
        panic!("chosen must lower to a direct generic call")
    };
    assert_eq!(substitution.0.len(), 1);
    assert_eq!(
        substitution.get(severian_universal::GenericParamId(0)),
        typed.types.resolve_name("i32")
    );
    std::fs::remove_dir_all(root).unwrap();
}

#[test]
fn qualified_generic_class_application_stays_a_constructor_outcome() {
    let root = temporary();
    let dependency_root = root.join("dependency");
    let app_root = root.join("app");
    std::fs::create_dir_all(&dependency_root).unwrap();
    std::fs::create_dir_all(&app_root).unwrap();
    let dependency_source = dependency_root.join("lib.sev");
    let app_source = app_root.join("main.sev");
    std::fs::write(&dependency_source, "class Box[T]:\n    value: T\n").unwrap();
    std::fs::write(
        &app_source,
        "import dependency\ndef selected(value: i32) -> dependency.Box[i32]:\n    return dependency.Box[i32](value)\n",
    )
    .unwrap();
    let app = severian_modules::PackageId(0);
    let dependency = severian_modules::PackageId(1);
    let packages = severian_modules::PackageGraph {
        root: app,
        packages: std::collections::BTreeMap::from([
            (
                app,
                severian_modules::ResolvedPackage {
                    id: app,
                    root: app_root,
                    library: app_source.clone(),
                    dependencies: std::collections::BTreeMap::from([(
                        "dependency".into(),
                        dependency,
                    )]),
                },
            ),
            (
                dependency,
                severian_modules::ResolvedPackage {
                    id: dependency,
                    root: dependency_root,
                    library: dependency_source,
                    dependencies: std::collections::BTreeMap::new(),
                },
            ),
        ]),
    };
    let graph = severian_modules::resolve_with_packages(&app_source, &packages).unwrap();
    let universal = severian_bootstrap::load().unwrap();
    let typed = analyze_package(&graph, &universal).unwrap();
    let selected = typed
        .hir
        .modules
        .iter()
        .flat_map(|module| &module.functions)
        .find(|function| function.name == "selected")
        .unwrap();
    let body = selected.body.as_ref().unwrap();
    let severian_hir::Statement::Return(Some(value)) = &body.statements[0] else {
        panic!("selected must return the qualified constructor")
    };
    assert!(matches!(
        value.kind,
        severian_hir::ExpressionKind::Aggregate { .. }
    ));
    severian_mir::build(&typed.hir).unwrap();
    std::fs::remove_dir_all(root).unwrap();
}

#[test]
fn semantic_lowering_has_no_load_named_branch() {
    let semantic = include_str!("../lib.rs");
    assert!(!semantic.contains("\"load\""));
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
fn ownership_unary_arguments_specialize_nested_generic_calls() {
    let root = temporary();
    let source = root.join("nested-generic-copy.sev");
    std::fs::write(
        &source,
        "def identity[T](value: T) -> T:\n    return value\ndef selected(value: i32) -> i32:\n    return identity(copy value)\n",
    )
    .unwrap();
    let graph = severian_modules::resolve(&source).unwrap();
    let universal = severian_bootstrap::load().unwrap();
    let typed = analyze_package(&graph, &universal).unwrap();
    let identity = typed
        .index
        .definitions
        .values()
        .find(|definition| definition.name == "identity")
        .unwrap()
        .id;
    assert!(typed.hir.modules[0]
        .functions
        .iter()
        .any(|function| function.definition == identity));
    std::fs::remove_dir_all(root).unwrap();
}

#[test]
fn concrete_class_methods_request_generic_function_specializations() {
    let root = temporary();
    let source = root.join("class-generic-call.sev");
    std::fs::write(
        &source,
        "def identity[T](value: T) -> T:\n    return value\nclass Runner:\n    value: i32\n    def run() -> i32:\n        return identity(copy value)\n",
    )
    .unwrap();
    let graph = severian_modules::resolve(&source).unwrap();
    let universal = severian_bootstrap::load().unwrap();
    let typed = analyze_package(&graph, &universal).unwrap();
    let identity = typed
        .index
        .definitions
        .values()
        .find(|definition| definition.name == "identity")
        .unwrap()
        .id;
    assert!(typed.hir.modules[0]
        .functions
        .iter()
        .any(|function| function.definition == identity));
    std::fs::remove_dir_all(root).unwrap();
}

#[test]
fn method_call_results_contribute_generic_type_evidence() {
    let root = temporary();
    let source = root.join("method-result-generic-call.sev");
    std::fs::write(
        &source,
        "class Source:\n    def value() -> i32:\n        return 1\ndef convert[A, B](source: A, target: B) -> B:\n    return target\ndef selected(source: Source, target: f64) -> f64:\n    return convert(source.value(), target)\n",
    )
    .unwrap();
    let graph = severian_modules::resolve(&source).unwrap();
    let universal = severian_bootstrap::load().unwrap();
    let typed = analyze_package(&graph, &universal).unwrap();
    let convert = typed
        .index
        .definitions
        .values()
        .find(|definition| definition.name == "convert")
        .unwrap()
        .id;
    assert!(typed.hir.modules[0]
        .functions
        .iter()
        .any(|function| function.definition == convert));
    std::fs::remove_dir_all(root).unwrap();
}

#[test]
fn generic_results_flow_through_bindings_into_nested_generic_calls() {
    let root = temporary();
    let source = root.join("nested-generic-result.sev");
    std::fs::write(
        &source,
        "class Box[T]:\n    value: T\ndef normalize[T](input: Box[T]) -> Box[T]:\n    return input\ndef attend[T](input: Box[T]) -> Box[T]:\n    return input\ndef layer[T](input: Box[T]) -> Box[T]:\n    normalized = normalize(copy input)\n    return attend(normalized)\ndef run(input: Box[f32]) -> Box[f32]:\n    return layer(input)\n",
    )
    .unwrap();
    let graph = severian_modules::resolve(&source).unwrap();
    let universal = severian_bootstrap::load().unwrap();
    let typed = analyze_package(&graph, &universal).unwrap();
    for name in ["normalize", "attend", "layer"] {
        let definition = typed
            .index
            .definitions
            .values()
            .find(|definition| definition.name == name)
            .unwrap()
            .id;
        assert!(typed.hir.modules[0]
            .functions
            .iter()
            .any(|function| function.definition == definition));
    }
    std::fs::remove_dir_all(root).unwrap();
}

#[test]
fn irrelevant_unknown_arguments_do_not_erase_generic_result_types() {
    let root = temporary();
    let source = root.join("partial-generic-evidence.sev");
    std::fs::write(
        &source,
        "class Box[T]:\n    value: T\ndef scale[T](input: Box[T], amount: float) -> Box[T]:\n    return input\ndef consume[T](input: Box[T]) -> Box[T]:\n    return input\ndef run[T](input: Box[T], width: int) -> Box[T]:\n    scaled = scale(input, 1.0 / float(width))\n    return consume(scaled)\ndef selected(input: Box[f32]) -> Box[f32]:\n    return run(input, 2)\n",
    )
    .unwrap();
    let graph = severian_modules::resolve(&source).unwrap();
    let universal = severian_bootstrap::load().unwrap();
    let typed = analyze_package(&graph, &universal).unwrap();
    let consume = typed
        .index
        .definitions
        .values()
        .find(|definition| definition.name == "consume")
        .unwrap()
        .id;
    assert!(typed.hir.modules[0]
        .functions
        .iter()
        .any(|function| function.definition == consume));
    std::fs::remove_dir_all(root).unwrap();
}

#[test]
fn hooks_wrap_loop_returns_and_ownership_reaches_a_fixed_point() {
    let root = temporary();
    let source = root.join("hook-loop.sev");
    std::fs::write(
        &source,
        "class HookContext:\n    function: string\n    result: int\n    error: string\ntrait Monitor:\n    @monitor\n    def monitor(context: HookContext) -> None with context\nclass Metric: Monitor\n    def monitor(context: HookContext) -> None with context:\n        with context:\n            print(\"enter\", context.function)\n        without context:\n            print(\"exit\", context.function)\n@monitor\ndef search(items: list[int], item: int) -> int:\n    for value in items:\n        if value == item:\n            return value\n    throw Error(\"missing\")\n",
    )
    .unwrap();
    let universal = severian_bootstrap::load().unwrap();
    let typed = analyze_package(&severian_modules::resolve(&source).unwrap(), &universal).unwrap();
    let search = typed.hir.modules[0]
        .functions
        .iter()
        .find(|function| function.name == "search")
        .unwrap();
    let body = search.body.as_ref().unwrap();
    assert!(matches!(
        body.statements[0],
        severian_hir::Statement::Binding(_)
    ));
    assert!(matches!(
        body.statements[1],
        severian_hir::Statement::FieldSet { .. }
    ));
    let mut mir = severian_mir::build(&typed.hir).unwrap();
    severian_mir::run_required_pipeline(&mut mir, &universal).unwrap();
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
fn implicit_variadic_generics_specialize_from_test_calls() {
    let root = temporary();
    let source = root.join("variadic.sev");
    std::fs::write(
        &source,
        "def print_values(values: T...):\n    for value in values:\n        print(value)\n\ntest with integ:\n    print_values(\"answer\", 42, true)\n",
    )
    .unwrap();
    let universal = severian_bootstrap::load().unwrap();
    let typed = analyze_package(&severian_modules::resolve(&source).unwrap(), &universal).unwrap();
    severian_mir::build(&typed.hir).unwrap();
    assert!(typed.hir.modules[0]
        .functions
        .iter()
        .any(|function| function.name == "print_values"));
    std::fs::remove_dir_all(root).unwrap();
}

#[test]
fn generic_map_literals_infer_nested_arguments_and_lower_pair_iteration() {
    let root = temporary();
    let source = root.join("map-sum.sev");
    std::fs::write(
        &source,
        "trait Hash:\n    def hash() -> usize\ntrait Equal:\n    def equal(other: Self) -> bool\ntrait Number:\n    def zero() -> Self\n    def add(other: Self) -> Self\ndef sum_values[K: Hash + Equal, V: Number](values: map[K, V]) -> V:\n    total := V.zero()\n    for _, value in values:\n        total = total.add(value)\n    return total\ndef selected() -> int:\n    counts = {\"first\": 34, \"second\": 8}\n    return sum_values(counts)\n",
    )
    .unwrap();
    let universal = severian_bootstrap::load().unwrap();
    let typed = analyze_package(&severian_modules::resolve(&source).unwrap(), &universal).unwrap();
    assert!(typed.hir.modules[0]
        .classes
        .iter()
        .any(|class| class.name == "map[string, int]"));
    severian_mir::build(&typed.hir).unwrap();
    std::fs::remove_dir_all(root).unwrap();
}

#[test]
fn comparison_result_expectations_do_not_poison_generic_operands() {
    let root = temporary();
    let source = root.join("map-sum-comparison.sev");
    std::fs::write(
        &source,
        "trait Hash:\n    def hash() -> usize\ntrait Equal:\n    def equal(other: Self) -> bool\ntrait Number:\n    def zero() -> Self\n    def add(other: Self) -> Self\ndef sum_values[K: Hash + Equal, V: Number](values: map[K, V]) -> V:\n    total := V.zero()\n    for _, value in values:\n        total = total.add(value)\n    return total\ntest:\n    counts = {\"first\": 34, \"second\": 8}\n    assert(sum_values(counts) == 42)\n",
    )
    .unwrap();
    let universal = severian_bootstrap::load().unwrap();
    let typed = analyze_package(&severian_modules::resolve(&source).unwrap(), &universal).unwrap();
    assert!(typed.hir.modules[0]
        .functions
        .iter()
        .any(|function| function.name == "sum_values"));
    std::fs::remove_dir_all(root).unwrap();
}

#[test]
fn incompatible_generic_inferences_are_reported_at_the_call() {
    let root = temporary();
    let source = root.join("generic-conflict.sev");
    std::fs::write(
        &source,
        "def identity[T](value: T) -> T:\n    return value\ndef selected() -> string:\n    return identity(1)\n",
    )
    .unwrap();
    let universal = severian_bootstrap::load().unwrap();
    let graph = severian_modules::resolve(&source).unwrap();
    let source_file = graph.modules[0].source.clone();
    let error = analyze_package(&graph, &universal).unwrap_err();
    assert_eq!(error.code, "E000217");
    assert!(error.message.contains("conflicting inferences for `T`"));
    assert!(error.message.contains("`string` and `int`"));
    assert_eq!(
        source_file
            .location(error.span.unwrap().start)
            .unwrap()
            .line,
        4
    );
    assert!(error
        .notes
        .iter()
        .any(|note| note.contains("first inferred as `string`")));
    std::fs::remove_dir_all(root).unwrap();
}

#[test]
fn generic_constraint_failures_include_call_and_declaration_frames() {
    let root = temporary();
    let source = root.join("generic-constraint-origin.sev");
    std::fs::write(
        &source,
        "trait Number:\n    def add(other: Self) -> Self\ndef keep[V: Number](value: V) -> V:\n    return value\ndef selected() -> bool:\n    return keep(true)\n",
    )
    .unwrap();
    let universal = severian_bootstrap::load().unwrap();
    let graph = severian_modules::resolve(&source).unwrap();
    let source_file = graph.modules[0].source.clone();
    let error = analyze_package(&graph, &universal).unwrap_err();
    assert_eq!(error.code, "E000217");
    assert!(error.message.contains("keep[V=bool]"));
    assert_eq!(
        source_file
            .location(error.span.unwrap().start)
            .unwrap()
            .line,
        6
    );
    assert_eq!(error.additional.len(), 1);
    assert_eq!(
        source_file
            .location(error.additional[0].span.unwrap().start)
            .unwrap()
            .line,
        3
    );
    assert!(error.additional[0]
        .message
        .contains("`V` must satisfy `Number`"));
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

#[test]
fn enum_names_are_package_types_in_signatures() {
    let root = temporary();
    let source = root.join("enum.sev");
    std::fs::write(
        &source,
        "enum Result:\n    Value(value: int)\n    Message(value: string)\n    Empty\ndef unwrap(result: Result) -> int:\n    match result:\n        Value:\n            return value\n        Message:\n            return 0\n        Empty:\n            return 0\ndef selected() -> int:\n    return unwrap(Value(7))\n",
    )
    .unwrap();
    let universal = severian_bootstrap::load().unwrap();
    let typed = analyze_package(&severian_modules::resolve(&source).unwrap(), &universal).unwrap();
    severian_mir::build(&typed.hir).unwrap();
    std::fs::remove_dir_all(root).unwrap();
}

#[test]
fn package_class_fields_resolve_sibling_class_types() {
    let root = temporary();
    let source = root.join("ids.sev");
    std::fs::write(
        &source,
        "class DeclarationId:\n    value: u128\nclass PrimitiveId:\n    declaration: DeclarationId\ndef wrap(value: DeclarationId) -> PrimitiveId:\n    return PrimitiveId(value)\n",
    )
    .unwrap();
    let universal = severian_bootstrap::load().unwrap();
    let typed = analyze_package(&severian_modules::resolve(&source).unwrap(), &universal).unwrap();
    severian_mir::build(&typed.hir).unwrap();
    std::fs::remove_dir_all(root).unwrap();
}

#[test]
fn package_class_fields_survive_diamond_relative_imports() {
    let root = temporary();
    std::fs::write(
        root.join("ids.sev"),
        "class DeclarationId:\n    value: u128\nclass PrimitiveId:\n    declaration: DeclarationId\n",
    )
    .unwrap();
    std::fs::write(
        root.join("operator.sev"),
        "import \"ids.sev\"\nclass Signature:\n    declaration: DeclarationId\n",
    )
    .unwrap();
    std::fs::write(
        root.join("lib.sev"),
        "import \"ids.sev\"\nimport \"operator.sev\"\n",
    )
    .unwrap();
    let universal = severian_bootstrap::load().unwrap();
    let typed = analyze_package(
        &severian_modules::resolve(&root.join("lib.sev")).unwrap(),
        &universal,
    )
    .unwrap();
    severian_mir::build(&typed.hir).unwrap();
    std::fs::remove_dir_all(root).unwrap();
}

#[test]
fn package_class_fields_resolve_qualified_classes_inside_lists() {
    let root = temporary();
    std::fs::write(root.join("query.sev"), "class Step:\n    name: string\n").unwrap();
    std::fs::write(
        root.join("lib.sev"),
        "import \"query.sev\" as query\nclass Data:\n    steps: list[query.Step]\n",
    )
    .unwrap();
    let universal = severian_bootstrap::load().unwrap();
    let typed = analyze_package(
        &severian_modules::resolve(&root.join("lib.sev")).unwrap(),
        &universal,
    )
    .unwrap();
    severian_mir::build(&typed.hir).unwrap();
    std::fs::remove_dir_all(root).unwrap();
}

#[test]
fn tensor_intrinsics_do_not_consume_other_package_namespaces_as_receivers() {
    let root = temporary();
    std::fs::write(
        root.join("paths.sev"),
        "def exists(value: string) -> bool:\n    return value != \"\"\n",
    )
    .unwrap();
    std::fs::write(
        root.join("tensor.sev"),
        "class Tensor[T]:\n    handle: string\ndef tensor(values: list[float], shape: list[int]) -> Tensor[f64]:\n    return Tensor[f64](\"\")\n",
    )
    .unwrap();
    std::fs::write(
        root.join("lib.sev"),
        "import \"paths.sev\" as path\nimport \"tensor.sev\" as tensor\ndef ready(value: string) -> bool:\n    return path.exists(value)\n",
    )
    .unwrap();
    let universal = severian_bootstrap::load().unwrap();
    let typed = analyze_package(
        &severian_modules::resolve(&root.join("lib.sev")).unwrap(),
        &universal,
    )
    .unwrap();
    severian_mir::build(&typed.hir).unwrap();
    std::fs::remove_dir_all(root).unwrap();
}

#[test]
fn source_class_constructors_infer_generic_arguments_and_self_methods() {
    let root = temporary();
    let source = root.join("class-generic-self.sev");
    std::fs::write(
        &source,
        "trait Term:\n    def replace(other: Self) -> Self\nclass Concrete: Term\n    value: i64\n    def replace(other: Self) -> Self:\n        return Concrete(other.value)\ndef apply[T: Term](left: T, right: T) -> T:\n    return left.replace(right)\ntest:\n    result := apply(Concrete(1), Concrete(42))\n    assert(result.value == 42)\n",
    )
    .unwrap();
    let universal = severian_bootstrap::load().unwrap();
    let typed = analyze_package(&severian_modules::resolve(&source).unwrap(), &universal).unwrap();
    severian_mir::build(&typed.hir).unwrap();
    std::fs::remove_dir_all(root).unwrap();
}

#[test]
fn generic_class_fields_resolve_source_enum_types() {
    let root = temporary();
    let source = root.join("generic-enum-field.sev");
    std::fs::write(
        &source,
        "enum OperationKind:\n    Add\n    Multiply\nclass Symbol:\n    id: u64\nclass Binding[Y]:\n    symbol: Y\n    operation: OperationKind\ntest:\n    binding := Binding[Symbol](Symbol(1), Add)\n",
    )
    .unwrap();
    let universal = severian_bootstrap::load().unwrap();
    let typed = analyze_package(&severian_modules::resolve(&source).unwrap(), &universal).unwrap();
    severian_mir::build(&typed.hir).unwrap();
    std::fs::remove_dir_all(root).unwrap();
}
