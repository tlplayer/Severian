use serde_json::{json, Value};
use severian_hir::FunctionId;
use severian_mir::{Callee, CfgBody, CfgStatement, Operand, Place, PlaceBase, Rvalue, Terminator};
use severian_modules::ModuleGraph;
use severian_source::{SourceFile, Span};
use severian_universal::{DefId, TypeContext, TypeId};
use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::Path;

const AGENT_IR_VERSION: u32 = 1;

pub(crate) fn write(
    output: &Path,
    package: &str,
    root: &Path,
    graph: &ModuleGraph,
    hir: &severian_hir::Program,
    mir: &severian_mir::Module,
    types: &TypeContext,
) -> Result<(), String> {
    let graphs = output.join("graphs");
    fs::create_dir_all(&graphs)
        .map_err(|error| format!("could not create {}: {error}", graphs.display()))?;
    let source_root = common_source_root(root, graph);
    let root_functions = graph
        .modules
        .last()
        .map(|module| ast_function_names(&module.ast))
        .unwrap_or_default();
    let root_types = graph
        .modules
        .last()
        .map(|module| ast_type_names(&module.ast))
        .unwrap_or_default();

    let module_names = graph
        .modules
        .iter()
        .map(|module| display_path(&source_root, &module.path))
        .collect::<Vec<_>>();
    let module_ids = graph
        .modules
        .iter()
        .zip(&module_names)
        .map(|(module, name)| (module.id, module_identifier(package, name, module.id.0)))
        .collect::<BTreeMap<_, _>>();
    let function_ids = function_ids(package, hir, &module_names);
    let function_by_instance = hir
        .modules
        .iter()
        .flat_map(|module| &module.functions)
        .filter_map(|function| {
            function_ids
                .get(&function.id)
                .cloned()
                .map(|id| (function.id, id))
        })
        .collect::<BTreeMap<_, _>>();
    let function_by_definition = hir
        .modules
        .iter()
        .flat_map(|module| &module.functions)
        .filter_map(|function| {
            function_ids
                .get(&function.id)
                .cloned()
                .map(|id| (function.definition, id))
        })
        .collect::<BTreeMap<_, _>>();

    let mut declarations = Vec::new();
    let mut symbols = BTreeMap::<String, Value>::new();
    let mut type_edges = Vec::new();
    let mut call_edges = Vec::new();
    let mut reference_edges = Vec::new();
    let mut test_records = Vec::new();
    let mut public_api = Vec::new();
    let root_module_index = graph.modules.len().saturating_sub(1);

    for (module_index, hir_module) in hir.modules.iter().enumerate() {
        let module_name = module_names
            .get(module_index)
            .cloned()
            .unwrap_or_else(|| format!("module-{module_index}"));
        let module_id = graph
            .modules
            .get(module_index)
            .and_then(|resolved| module_ids.get(&resolved.id))
            .cloned()
            .unwrap_or_else(|| format!("M:{package}:module-{module_index}"));
        let source_functions = graph
            .modules
            .get(module_index)
            .map(|module| ast_function_names(&module.ast))
            .unwrap_or_default();
        for function in &hir_module.functions {
            let id = function_ids[&function.id].clone();
            symbols.insert(
                id.clone(),
                json!({"id": id, "kind": "F", "name": function.name}),
            );
            let source_owned = graph.modules.get(module_index).is_some_and(|module| {
                function.definition.package == u128::from(module.package.0)
                    && function.definition.module == module.id.0
                    && !function.name.starts_with("__")
                    && source_functions.contains(function.name.as_str())
            });
            if !source_owned {
                continue;
            }
            let argument_records = function
                .parameters
                .iter()
                .map(|argument| {
                    let type_id = type_identifier(types, argument.contract.ty);
                    type_edges.push(json!({"from": id, "relationship": "accepts", "to": type_id}));
                    json!({
                        "id": format!("A:{}:{}", id.trim_start_matches("F:"), argument.name),
                        "kind": "A",
                        "name": argument.name,
                        "type": type_id,
                        "modifiers": argument.contract.modifiers.iter().map(|modifier| modifier.name.as_str()).collect::<Vec<_>>(),
                    })
                })
                .collect::<Vec<_>>();
            let result = type_identifier(types, function.result.ty);
            type_edges.push(json!({"from": id, "relationship": "returns", "to": result}));
            let interface = format!(
                "{}({:?})->{}:{:?}:{:?}",
                function.name,
                function
                    .parameters
                    .iter()
                    .map(|parameter| parameter.contract.ty)
                    .collect::<Vec<_>>(),
                function.result.ty.0,
                function.compile_route,
                function.call_type,
            );
            let body = mir
                .functions
                .iter()
                .find(|candidate| candidate.id == function.id)
                .and_then(|candidate| candidate.body.as_ref());
            let semantic = semantic_fingerprint(function, body);
            declarations.push(json!({
                "id": id,
                "kind": "F",
                "name": function.name,
                "qualified_name": format!("{}.{}", module_name, function.name),
                "definition": def_identifier(function.definition),
                "module": module_id,
                "source": function_source(
                    graph,
                    module_index,
                    &function.name,
                    function.parameters.len(),
                    body,
                    &module_name,
                ),
                "signature": {"arguments": argument_records, "result": result},
                "generic_parameters": function.generic_parameters.iter().map(|parameter| format!("{parameter:?}")).collect::<Vec<_>>(),
                "compile_route": format!("{:?}", function.compile_route),
                "call_type": format!("{:?}", function.call_type),
                "effects": function_effects(mir, function.id),
                "throws": function_throws(mir, function.id),
                "source_hash": source_hash_for_module(graph, module_index),
                "semantic_hash": stable_hash_hex(semantic.as_bytes()),
                "interface_hash": stable_hash_hex(interface.as_bytes()),
            }));
            if module_index == root_module_index
                && matches!(function.call_type, severian_hir::CallType::Severian)
                && root_functions.contains(function.name.as_str())
            {
                public_api.push(function_ids[&function.id].clone());
            }
        }
        let source_types = graph
            .modules
            .get(module_index)
            .map(|module| ast_type_names(&module.ast))
            .unwrap_or_default();
        for class in &hir_module.classes {
            if !source_types.contains(class.name.as_str()) {
                continue;
            }
            let type_id = type_identifier(types, class.id);
            let id = format!(
                "D:{}:class:{}",
                module_id.trim_start_matches("M:"),
                class.id.0
            );
            type_edges.push(json!({"from": id, "relationship": "declares", "to": type_id}));
            let fields = class
                .fields
                .iter()
                .map(|field| {
                    let field_type = type_identifier(types, field.ty);
                    type_edges.push(json!({"from": id, "relationship": "contains", "to": field_type, "field": field.name}));
                    json!({"name": field.name, "type": field_type})
                })
                .collect::<Vec<_>>();
            declarations.push(json!({
                "id": id,
                "kind": "D",
                "declaration_kind": "class",
                "name": class.name,
                "module": module_id,
                "source": {"file": module_name},
                "fields": fields,
                "semantic_hash": stable_hash_hex(format!("{class:?}").as_bytes()),
            }));
            symbols.insert(
                id.clone(),
                json!({"id": id, "kind": "D", "name": class.name}),
            );
            if module_index == root_module_index && root_types.contains(class.name.as_str()) {
                public_api.push(type_id);
            }
        }
        let source_traits = graph
            .modules
            .get(module_index)
            .map(|module| ast_trait_names(&module.ast))
            .unwrap_or_default();
        for declaration in &hir_module.traits {
            if !source_traits.contains(declaration.name.as_str()) {
                continue;
            }
            let id = format!("C:{}", def_identifier(declaration.definition));
            declarations.push(json!({
                "id": id,
                "kind": "C",
                "declaration_kind": "trait",
                "name": declaration.name,
                "module": module_id,
                "source": {"file": module_name},
                "methods": declaration.methods.iter().map(|method| format!("{method:?}")).collect::<Vec<_>>(),
                "semantic_hash": stable_hash_hex(format!("{declaration:?}").as_bytes()),
            }));
            symbols.insert(
                id.clone(),
                json!({"id": id, "kind": "C", "name": declaration.name}),
            );
        }
        for test in &hir_module.tests {
            let id = format!(
                "test:{package}:{}:{}",
                module_label(&module_name),
                test.name
            );
            let tested_function = function_by_instance
                .get(&test.function)
                .cloned()
                .unwrap_or_else(|| format!("F:{:032x}", test.function.0));
            test_records.push(json!({
                "id": id,
                "kind": "test",
                "name": test.name,
                "module": module_id,
                "function": tested_function,
                "modes": test.modes.iter().map(|mode| mode.name()).collect::<Vec<_>>(),
                "expectations": test.expectations.iter().map(|expectation| format!("{expectation:?}")).collect::<Vec<_>>(),
            }));
            symbols.insert(
                id.clone(),
                json!({"id": id, "kind": "test", "name": test.name}),
            );
        }
    }

    collect_mir_edges(
        mir,
        &function_by_instance,
        &function_by_definition,
        &mut symbols,
        &mut call_edges,
        &mut reference_edges,
    );
    link_tests(&mut test_records, &call_edges, &mut reference_edges);

    let type_records = types
        .definitions()
        .map(|definition| {
            let id = type_identifier(types, definition.id);
            symbols.entry(id.clone()).or_insert_with(|| {
                json!({"id": id, "kind": "T", "name": definition.name})
            });
            json!({
                "id": id,
                "kind": "T",
                "name": definition.name,
                "path": definition.path,
                "definition_kind": format!("{:?}", definition.kind),
                "parameter_count": definition.parameter_count,
                "interface_hash": stable_hash_hex(format!("{}:{}:{:?}", definition.path, definition.parameter_count, definition.kind).as_bytes()),
            })
        })
        .collect::<Vec<_>>();

    let source_map = graph
        .modules
        .iter()
        .map(|module| {
            json!({
                "id": format!("source:{}", module.source.id.0),
                "module": module_ids[&module.id],
                "file": display_path(&source_root, &module.path),
                "bytes": module.source.text.len(),
                "lines": module.source.line_count(),
                "source_hash": stable_hash_hex(module.source.text.as_bytes()),
            })
        })
        .collect::<Vec<_>>();
    let dependency_edges = graph
        .modules
        .iter()
        .flat_map(|module| {
            let from = module_ids[&module.id].clone();
            let module_ids = &module_ids;
            module.imports.iter().map(move |import| {
                json!({
                    "from": from,
                    "relationship": "imports",
                    "to": module_ids.get(&import.module).cloned().unwrap_or_else(|| format!("M:{:032x}", import.module.0)),
                    "source": {"start": import.span.start, "end": import.span.end},
                })
            })
        })
        .collect::<Vec<_>>();
    let dependencies = dependency_edges
        .iter()
        .filter_map(|edge| edge.get("to").and_then(Value::as_str))
        .map(str::to_owned)
        .collect::<BTreeSet<_>>();
    let ownership_edges = ownership_edges(mir, &function_by_instance);
    add_dependency_hashes(
        &mut declarations,
        [&call_edges, &type_edges, &reference_edges],
    );
    let entrypoints = mir
        .entry
        .and_then(|entry| function_by_instance.get(&entry).cloned())
        .into_iter()
        .collect::<Vec<_>>();
    let package_record = json!({
        "agent_ir": AGENT_IR_VERSION,
        "package": package,
        "entrypoints": entrypoints,
        "modules": graph.modules.iter().map(|module| module_ids[&module.id].clone()).collect::<Vec<_>>(),
        "public_api": public_api,
        "dependencies": dependencies,
        "counts": {
            "symbols": symbols.len(),
            "declarations": declarations.len(),
            "types": type_records.len(),
            "tests": test_records.len(),
        },
    });

    write_json(output.join("package.json"), &package_record)?;
    write_jsonl(output.join("symbols.jsonl"), symbols.values())?;
    write_jsonl(output.join("declarations.jsonl"), declarations.iter())?;
    write_jsonl(output.join("types.jsonl"), type_records.iter())?;
    write_jsonl(output.join("tests.jsonl"), test_records.iter())?;
    write_jsonl(
        output.join("diagnostics.jsonl"),
        std::iter::empty::<&Value>(),
    )?;
    write_json(output.join("source-map.json"), &source_map)?;
    write_json(graphs.join("calls.json"), &call_edges)?;
    write_json(graphs.join("dependencies.json"), &dependency_edges)?;
    write_json(graphs.join("ownership.json"), &ownership_edges)?;
    write_json(graphs.join("types.json"), &type_edges)?;
    write_json(graphs.join("references.json"), &reference_edges)?;
    Ok(())
}

fn function_ids(
    package: &str,
    hir: &severian_hir::Program,
    module_names: &[String],
) -> BTreeMap<FunctionId, String> {
    let mut ids = BTreeMap::new();
    for (module_index, hir_module) in hir.modules.iter().enumerate() {
        let module_name = module_names
            .get(module_index)
            .map(|name| module_label(name))
            .unwrap_or_else(|| format!("module-{module_index}"));
        for function in &hir_module.functions {
            ids.insert(
                function.id,
                format!(
                    "F:{package}:{module_name}.{}@{:032x}",
                    function.name, function.id.0
                ),
            );
        }
    }
    ids
}

fn collect_mir_edges(
    mir: &severian_mir::Module,
    by_instance: &BTreeMap<FunctionId, String>,
    by_definition: &BTreeMap<DefId, String>,
    symbols: &mut BTreeMap<String, Value>,
    calls: &mut Vec<Value>,
    references: &mut Vec<Value>,
) {
    for function in &mir.functions {
        let Some(body) = &function.body else { continue };
        let caller = by_instance
            .get(&function.id)
            .cloned()
            .unwrap_or_else(|| format!("F:{:032x}", function.id.0));
        collect_body_edges(
            body,
            &caller,
            by_instance,
            by_definition,
            symbols,
            calls,
            references,
        );
    }
}

fn link_tests(tests: &mut [Value], calls: &[Value], references: &mut Vec<Value>) {
    for test in tests {
        let Some(test_id) = test.get("id").and_then(Value::as_str).map(str::to_owned) else {
            continue;
        };
        let Some(wrapper) = test
            .get("function")
            .and_then(Value::as_str)
            .map(str::to_owned)
        else {
            continue;
        };
        let tested = calls
            .iter()
            .filter(|edge| edge.get("from").and_then(Value::as_str) == Some(wrapper.as_str()))
            .filter_map(|edge| edge.get("to").and_then(Value::as_str))
            .filter(|target| *target != wrapper)
            .filter(|target| !is_internal_function_id(target))
            .map(str::to_owned)
            .collect::<BTreeSet<_>>();
        for target in &tested {
            references.push(json!({"from": test_id, "relationship": "tests", "to": target}));
        }
        if let Some(object) = test.as_object_mut() {
            object.insert("tests".into(), json!(tested));
        }
    }
}

fn collect_body_edges(
    body: &CfgBody,
    caller: &str,
    by_instance: &BTreeMap<FunctionId, String>,
    by_definition: &BTreeMap<DefId, String>,
    symbols: &mut BTreeMap<String, Value>,
    calls: &mut Vec<Value>,
    references: &mut Vec<Value>,
) {
    for block in &body.blocks {
        for statement in &block.statements {
            if let CfgStatement::Operation { id, .. } = statement {
                let operation = format!("O:{:032x}:{:032x}", id.dialect.0, id.operation.0);
                symbols
                    .entry(operation.clone())
                    .or_insert_with(|| json!({"id": operation, "kind": "O", "name": operation}));
                references
                    .push(json!({"from": caller, "relationship": "constructs", "to": operation}));
            }
        }
        match &block.terminator {
            Terminator::Call { callee, .. } | Terminator::Spawn { callee, .. } => {
                let target = callee_identifier(callee, by_instance, by_definition);
                calls.push(json!({"from": caller, "relationship": "calls", "to": target, "block": format!("B:{}:{}", caller.trim_start_matches("F:"), block.id.0)}));
            }
            _ => {}
        }
    }
}

fn callee_identifier(
    callee: &Callee,
    by_instance: &BTreeMap<FunctionId, String>,
    by_definition: &BTreeMap<DefId, String>,
) -> String {
    match callee {
        Callee::Direct {
            instance: Some(instance),
            ..
        } => by_instance
            .get(instance)
            .cloned()
            .unwrap_or_else(|| format!("F:{:032x}", instance.0)),
        Callee::Direct { function, .. } => by_definition
            .get(function)
            .cloned()
            .unwrap_or_else(|| format!("F:{}", def_identifier(*function))),
        Callee::Method { implementation, .. } => by_definition
            .get(implementation)
            .cloned()
            .unwrap_or_else(|| format!("F:{}", def_identifier(*implementation))),
        Callee::Constructor { type_def, .. } => format!("D:{}", def_identifier(*type_def)),
        Callee::Intrinsic(operation) => format!(
            "O:{:032x}:{:032x}",
            operation.dialect.0, operation.operation.0
        ),
        Callee::FunctionValue(_) => "F:<dynamic>".into(),
    }
}

fn function_effects(mir: &severian_mir::Module, id: FunctionId) -> Vec<&'static str> {
    let Some(body) = mir
        .functions
        .iter()
        .find(|function| function.id == id)
        .and_then(|function| function.body.as_ref())
    else {
        return Vec::new();
    };
    let mut effects = BTreeSet::new();
    for block in &body.blocks {
        for statement in &block.statements {
            match statement {
                CfgStatement::Assign(_, Rvalue::BorrowShared(_)) => {
                    effects.insert("memory.read");
                }
                CfgStatement::Assign(_, Rvalue::BorrowExclusive(_)) | CfgStatement::Drop(_) => {
                    effects.insert("memory.write");
                }
                CfgStatement::Operation { .. } => {
                    effects.insert("operation");
                }
                _ => {}
            }
        }
        if matches!(
            block.terminator,
            Terminator::Spawn { .. } | Terminator::SpawnFieldUpdate { .. }
        ) {
            effects.insert("async.spawn");
        }
    }
    effects.into_iter().collect()
}

fn function_throws(mir: &severian_mir::Module, id: FunctionId) -> Vec<&'static str> {
    let throws = mir
        .functions
        .iter()
        .find(|function| function.id == id)
        .and_then(|function| function.body.as_ref())
        .is_some_and(|body| {
            body.blocks
                .iter()
                .any(|block| matches!(block.terminator, Terminator::Throw(_)))
        });
    if throws {
        vec!["E:<dynamic>"]
    } else {
        Vec::new()
    }
}

fn ownership_edges(
    mir: &severian_mir::Module,
    by_instance: &BTreeMap<FunctionId, String>,
) -> Vec<Value> {
    let mut edges = Vec::new();
    for function in &mir.functions {
        let Some(body) = &function.body else { continue };
        let owner = by_instance
            .get(&function.id)
            .cloned()
            .unwrap_or_else(|| format!("F:{:032x}", function.id.0));
        for block in &body.blocks {
            for statement in &block.statements {
                match statement {
                    CfgStatement::Assign(_, Rvalue::BorrowShared(place)) => {
                        push_ownership(&mut edges, &owner, "borrows", place)
                    }
                    CfgStatement::Assign(_, Rvalue::BorrowExclusive(place)) => {
                        push_ownership(&mut edges, &owner, "borrows_mut", place)
                    }
                    CfgStatement::Drop(place) => push_ownership(&mut edges, &owner, "drops", place),
                    CfgStatement::Assign(_, rvalue) => {
                        collect_rvalue_moves(&mut edges, &owner, rvalue)
                    }
                    _ => {}
                }
            }
        }
    }
    edges
}

fn collect_rvalue_moves(edges: &mut Vec<Value>, owner: &str, rvalue: &Rvalue) {
    let operands = match rvalue {
        Rvalue::Use(operand)
        | Rvalue::Await { task: operand }
        | Rvalue::Convert { operand, .. } => vec![operand],
        Rvalue::Unary { operand, .. } => vec![operand],
        Rvalue::Binary { left, right, .. } => vec![left, right],
        Rvalue::Aggregate { fields, .. } | Rvalue::Variant { fields, .. } => fields.iter().collect(),
        Rvalue::BorrowShared(_)
        | Rvalue::BorrowExclusive(_)
        | Rvalue::AddressOf(_) => Vec::new(),
    };
    for operand in operands {
        if let Operand::Move(place) = operand {
            push_ownership(edges, owner, "moves", place);
        }
    }
}

fn push_ownership(edges: &mut Vec<Value>, owner: &str, relationship: &str, place: &Place) {
    let base = match place.base {
        PlaceBase::Local(local) => {
            format!("V:{}:local:{}", owner.trim_start_matches("F:"), local.0)
        }
        PlaceBase::Global(global) => format!("V:global:{}", global.0),
    };
    edges.push(json!({"from": owner, "relationship": relationship, "to": base, "projection": format!("{:?}", place.projection)}));
}

fn source_hash_for_module(graph: &ModuleGraph, index: usize) -> String {
    graph
        .modules
        .get(index)
        .map(|module| stable_hash_hex(module.source.text.as_bytes()))
        .unwrap_or_else(|| stable_hash_hex(&[]))
}

fn function_source(
    graph: &ModuleGraph,
    module_index: usize,
    name: &str,
    parameter_count: usize,
    body: Option<&CfgBody>,
    file_name: &str,
) -> Value {
    let Some(module) = graph.modules.get(module_index) else {
        return json!({"file": file_name});
    };
    let source = &module.source;
    if let Some(span) = ast_function_span(&module.ast, name, parameter_count) {
        return source_span(source, file_name, span);
    }
    let Some(body) = body else {
        return json!({"file": file_name});
    };
    let spans = body
        .locals
        .iter()
        .filter_map(|local| local.span)
        .chain(
            body.blocks
                .iter()
                .flat_map(|block| block.statement_spans.iter().copied().flatten()),
        )
        .chain(body.blocks.iter().filter_map(|block| block.terminator_span))
        .filter(|span| span.source == source.id)
        .collect::<Vec<_>>();
    let Some(start) = spans.iter().map(|span| span.start).min() else {
        return json!({"file": file_name});
    };
    let end = spans.iter().map(|span| span.end).max().unwrap_or(start);
    source_span(source, file_name, Span::new(source.id, start, end))
}

fn ast_function_span(
    ast: &severian_ast::Module,
    name: &str,
    parameter_count: usize,
) -> Option<Span> {
    let matches = |function: &severian_ast::FunctionDeclaration| {
        (function.name == name && function.parameters.len() == parameter_count)
            .then_some(function.span)
    };
    for item in &ast.items {
        let span = match item {
            severian_ast::Item::Function(function) => matches(function),
            severian_ast::Item::Class(class) => class
                .constructors
                .iter()
                .chain(&class.methods)
                .find_map(matches),
            severian_ast::Item::Trait(declaration) => declaration.methods.iter().find_map(matches),
            severian_ast::Item::Extension(extension) => extension.methods.iter().find_map(matches),
            _ => None,
        };
        if span.is_some() {
            return span;
        }
    }
    None
}

fn source_span(source: &SourceFile, file_name: &str, span: Span) -> Value {
    let start = source.location(span.start);
    let end = source.location(span.end);
    json!({
        "file": file_name,
        "start": span.start,
        "end": span.end,
        "start_line": start.map(|location| location.line),
        "start_column": start.map(|location| location.column),
        "end_line": end.map(|location| location.line),
        "end_column": end.map(|location| location.column),
    })
}

fn semantic_fingerprint(
    function: &severian_hir::FunctionDeclaration,
    body: Option<&CfgBody>,
) -> String {
    let mut parts = vec![
        function.name.clone(),
        format!("parameters:{:?}", function.parameters),
        format!("result:{:?}", function.result),
        format!("generics:{:?}", function.generic_parameters),
        format!("route:{:?}", function.compile_route),
        format!("call:{:?}", function.call_type),
    ];
    if let Some(body) = body {
        parts.push(format!(
            "entry:{:?}:return:{:?}",
            body.entry, body.return_type
        ));
        for block in &body.blocks {
            parts.push(format!(
                "block:{:?}:execution:{:?}:parameters:{:?}",
                block.id, block.execution, block.parameters
            ));
            for statement in &block.statements {
                parts.push(match statement {
                    CfgStatement::Assert {
                        condition, message, ..
                    } => format!("Assert({condition:?},{message:?})"),
                    CfgStatement::Coverage(point) => format!(
                        "Coverage({:?},{},{:?})",
                        point.kind, point.ordinal, point.key
                    ),
                    other => format!("{other:?}"),
                });
            }
            parts.push(format!("terminator:{:?}", block.terminator));
        }
    }
    parts.join("\n")
}

fn add_dependency_hashes<'a, const N: usize>(
    declarations: &mut [Value],
    graphs: [&'a Vec<Value>; N],
) {
    for declaration in declarations {
        let Some(id) = declaration.get("id").and_then(Value::as_str) else {
            continue;
        };
        let mut dependencies = graphs
            .iter()
            .flat_map(|graph| graph.iter())
            .filter(|edge| edge.get("from").and_then(Value::as_str) == Some(id))
            .filter_map(|edge| edge.get("to").and_then(Value::as_str))
            .collect::<Vec<_>>();
        dependencies.sort_unstable();
        dependencies.dedup();
        if let Some(object) = declaration.as_object_mut() {
            object.insert(
                "dependency_hash".into(),
                json!(stable_hash_hex(dependencies.join("\n").as_bytes())),
            );
        }
    }
}

fn type_identifier(types: &TypeContext, id: TypeId) -> String {
    types
        .definition(id)
        .map(|definition| format!("T:{}", definition.path))
        .unwrap_or_else(|| format!("T:#{}", id.0))
}

fn module_identifier(package: &str, path: &str, module_id: u128) -> String {
    format!("M:{package}:{}@{module_id:032x}", module_label(path))
}

fn module_label(path: &str) -> String {
    path.trim_end_matches(".sev").replace(['/', '\\'], ".")
}

fn is_internal_function_id(id: &str) -> bool {
    id.rsplit(':')
        .next()
        .and_then(|name| name.split('@').next())
        .is_some_and(|name| name.starts_with("__"))
}

fn ast_function_names(ast: &severian_ast::Module) -> BTreeSet<String> {
    let mut names = BTreeSet::new();
    for item in &ast.items {
        match item {
            severian_ast::Item::Function(function) => {
                names.insert(function.name.clone());
            }
            severian_ast::Item::Class(class) => {
                names.extend(
                    class
                        .constructors
                        .iter()
                        .chain(&class.methods)
                        .map(|function| function.name.clone()),
                );
                names.extend(class.operators.iter().filter_map(|operator| {
                    operator.operator.standard_spelling().map(str::to_owned)
                }));
            }
            severian_ast::Item::Trait(declaration) => {
                names.extend(
                    declaration
                        .methods
                        .iter()
                        .map(|function| function.name.clone()),
                );
                names.extend(declaration.operators.iter().filter_map(|operator| {
                    operator.operator.standard_spelling().map(str::to_owned)
                }));
            }
            severian_ast::Item::Extension(extension) => {
                names.extend(
                    extension
                        .methods
                        .iter()
                        .map(|function| function.name.clone()),
                );
                names.extend(extension.operators.iter().filter_map(|operator| {
                    operator.operator.standard_spelling().map(str::to_owned)
                }));
            }
            _ => {}
        }
    }
    names
}

fn ast_type_names(ast: &severian_ast::Module) -> BTreeSet<String> {
    ast.items
        .iter()
        .filter_map(|item| match item {
            severian_ast::Item::Class(class) => Some(class.name.clone()),
            severian_ast::Item::Enum(enumeration) => Some(enumeration.name.clone()),
            severian_ast::Item::Type(declaration) => Some(declaration.name.clone()),
            _ => None,
        })
        .collect()
}

fn ast_trait_names(ast: &severian_ast::Module) -> BTreeSet<String> {
    ast.items
        .iter()
        .filter_map(|item| match item {
            severian_ast::Item::Trait(declaration) => Some(declaration.name.clone()),
            _ => None,
        })
        .collect()
}

fn common_source_root(root: &Path, graph: &ModuleGraph) -> std::path::PathBuf {
    let original = fs::canonicalize(root).unwrap_or_else(|_| root.to_owned());
    let mut common = original.clone();
    for module in &graph.modules {
        while !module.path.starts_with(&common) {
            let Some(parent) = common.parent() else {
                return original;
            };
            if parent.parent().is_none() {
                return original;
            }
            common = parent.to_owned();
        }
    }
    common
}

fn display_path(root: &Path, path: &Path) -> String {
    path.strip_prefix(root)
        .unwrap_or(path)
        .to_string_lossy()
        .replace('\\', "/")
}

fn def_identifier(id: DefId) -> String {
    format!("{:032x}:{:032x}:{}", id.package, id.module, id.declaration)
}

fn stable_hash_hex(bytes: &[u8]) -> String {
    const OFFSET: u128 = 0x6c62_272e_07bb_0142_62b8_2175_6295_c58d;
    const PRIME: u128 = 0x0000_0000_0100_0000_0000_0000_0000_013b;
    let mut hash = OFFSET;
    for byte in bytes {
        hash ^= u128::from(*byte);
        hash = hash.wrapping_mul(PRIME);
    }
    format!("{hash:032x}")
}

fn write_json(path: impl AsRef<Path>, value: &impl serde::Serialize) -> Result<(), String> {
    let path = path.as_ref();
    let mut text = serde_json::to_string_pretty(value)
        .map_err(|error| format!("could not encode {}: {error}", path.display()))?;
    text.push('\n');
    fs::write(path, text).map_err(|error| format!("could not write {}: {error}", path.display()))
}

fn write_jsonl<'a>(
    path: impl AsRef<Path>,
    values: impl IntoIterator<Item = &'a Value>,
) -> Result<(), String> {
    let path = path.as_ref();
    let mut text = String::new();
    for value in values {
        text.push_str(
            &serde_json::to_string(value)
                .map_err(|error| format!("could not encode {}: {error}", path.display()))?,
        );
        text.push('\n');
    }
    fs::write(path, text).map_err(|error| format!("could not write {}: {error}", path.display()))
}
