use crate::{stablehlo, tensor};
use severian_hir::{
    AssignmentOp, BinaryOp, BindingId, BindingRef, Class, ComprehensionClause, Expression,
    Function, FunctionId, Instruction, MatchPattern, OwnershipOp, Program, SwitchArm,
    TaskPlacement, TypeDefinitionId, TypeKind, UnaryOp, ValueType,
};
use severian_mlir::Module;
use std::cell::{Cell, RefCell};
use std::collections::{HashMap, HashSet};
use std::fmt::Write;
use std::rc::Rc;

struct LoweringEnvironment<'a> {
    globals: &'a [severian_hir::Global],
    classes: &'a [Class],
    strings: &'a [String],
    function_returns: &'a HashMap<FunctionId, ValueType>,
    function_params: &'a HashMap<FunctionId, Vec<ValueType>>,
    function_return_classes: &'a HashMap<FunctionId, TypeDefinitionId>,
    method_return_classes: &'a HashMap<(TypeDefinitionId, String), TypeDefinitionId>,
    closure_definitions: &'a Rc<RefCell<String>>,
    next_closure: &'a Rc<Cell<usize>>,
    function_closures: &'a Rc<RefCell<HashMap<FunctionId, String>>>,
    native_symbols: &'a HashMap<FunctionId, String>,
    sources: &'a severian_hir::SourceMap,
    trait_registries: &'a std::collections::BTreeMap<String, severian_hir::TraitRegistryDefinition>,
}

fn lower_function(function: &Function, environment: &LoweringEnvironment<'_>, output: &mut String) {
    let is_main = function.name == "main";
    write!(
        output,
        "  llvm.func @{}(",
        source_function_symbol(&function.name)
    )
    .unwrap();
    for (index, param) in function.params.iter().enumerate() {
        if index > 0 {
            output.push_str(", ");
        }
        write!(output, "%arg_{index}: {}", mlir_type(param.ty)).unwrap();
    }
    output.push(')');
    if is_main {
        output.push_str(" -> i32");
    } else if function.return_type != ValueType::Unit {
        write!(output, " -> {}", mlir_type(function.return_type)).unwrap();
    }
    output.push_str(" {\n");

    let mut context = LowerContext {
        output,
        strings: environment.strings,
        function_returns: environment.function_returns,
        function_params: environment.function_params,
        function_return_classes: environment.function_return_classes,
        method_return_classes: environment.method_return_classes,
        closure_definitions: Rc::clone(environment.closure_definitions),
        next_closure: Rc::clone(environment.next_closure),
        function_closures: Rc::clone(environment.function_closures),
        native_symbols: environment.native_symbols,
        sources: environment.sources,
        trait_registries: environment.trait_registries,
        classes: environment.classes,
        field_object: None,
        field_names: HashSet::new(),
        field_types: HashMap::new(),
        field_classes: HashMap::new(),
        object_classes: HashMap::new(),
        object_class_ids: HashMap::new(),
        receiver_types: HashMap::new(),
        declared_return: function.return_type,
        task_results: HashMap::new(),
        channel_types: HashMap::new(),
        variables: function
            .params
            .iter()
            .enumerate()
            .map(|(index, param)| (param.name.id, (format!("%arg_{index}"), param.ty)))
            .collect(),
        next_value: 0,
        next_block: 0,
        terminated: false,
        loop_targets: Vec::new(),
        is_main,
        closure_callback: false,
        placement: TaskPlacement::Default,
        active_hir_id: None,
        active_expression_type: None,
    };
    for (index, parameter) in function.params.iter().enumerate() {
        if let Some(receiver) = &parameter.receiver {
            let value = format!("%arg_{index}");
            context
                .object_classes
                .insert(value.clone(), receiver.name.clone());
            context.receiver_types.insert(value, receiver.clone());
        }
    }
    for global in environment.globals {
        let value = context.lower_expression(&global.value);
        context.variables.insert(global.name.id, value);
    }
    context.lower_instructions(&function.instructions);
    if !context.terminated {
        if is_main {
            let success = context.fresh_value();
            writeln!(
                context.output,
                "    {success} = llvm.mlir.constant(0 : i32) : i32"
            )
            .unwrap();
            writeln!(context.output, "    llvm.return {success} : i32").unwrap();
        } else if function.return_type == ValueType::Unit {
            context.output.push_str("    llvm.return\n");
        } else {
            let value = context.fresh_value();
            match function.return_type {
                ValueType::Int => writeln!(context.output, "    {value} = llvm.mlir.constant(0 : i64) : i64\n    llvm.return {value} : i64").unwrap(),
                ValueType::Float => writeln!(context.output, "    {value} = llvm.mlir.constant(0.0 : f64) : f64\n    llvm.return {value} : f64").unwrap(),
                ValueType::Bool => writeln!(context.output, "    {value} = llvm.mlir.constant(0 : i1) : i1\n    llvm.return {value} : i1").unwrap(),
                _ => writeln!(context.output, "    {value} = llvm.mlir.zero : !llvm.ptr\n    llvm.return {value} : !llvm.ptr").unwrap(),
            }
        }
    }
    context.output.push_str("  }\n");
}

fn lower_class_function(
    class: &Class,
    function: &Function,
    symbol: &str,
    environment: &LoweringEnvironment<'_>,
    output: &mut String,
) {
    write!(output, "  llvm.func @{symbol}(%self: !llvm.ptr").unwrap();
    for (index, param) in function.params.iter().enumerate() {
        write!(output, ", %arg_{index}: {}", mlir_type(param.ty)).unwrap();
    }
    output.push(')');
    if function.return_type != ValueType::Unit {
        write!(output, " -> {}", mlir_type(function.return_type)).unwrap();
    }
    output.push_str(" {\n");

    let mut context = LowerContext {
        output,
        strings: environment.strings,
        function_returns: environment.function_returns,
        function_params: environment.function_params,
        function_return_classes: environment.function_return_classes,
        method_return_classes: environment.method_return_classes,
        closure_definitions: Rc::clone(environment.closure_definitions),
        next_closure: Rc::clone(environment.next_closure),
        function_closures: Rc::clone(environment.function_closures),
        native_symbols: environment.native_symbols,
        sources: environment.sources,
        trait_registries: environment.trait_registries,
        classes: environment.classes,
        field_object: Some("%self".into()),
        field_names: class.fields.iter().cloned().collect(),
        field_types: class
            .fields
            .iter()
            .cloned()
            .zip(class.field_types.iter().copied())
            .collect(),
        field_classes: class
            .fields
            .iter()
            .cloned()
            .zip(class.field_classes.iter().cloned())
            .filter_map(|(field, class)| class.map(|class| (field, class)))
            .collect(),
        object_classes: HashMap::from([("%self".into(), class.name.clone())]),
        object_class_ids: HashMap::from([("%self".into(), class.id)]),
        receiver_types: HashMap::new(),
        task_results: HashMap::new(),
        channel_types: HashMap::new(),
        variables: function
            .params
            .iter()
            .enumerate()
            .map(|(index, param)| (param.name.id, (format!("%arg_{index}"), param.ty)))
            .collect(),
        next_value: 0,
        next_block: 0,
        terminated: false,
        loop_targets: Vec::new(),
        is_main: false,
        closure_callback: false,
        declared_return: function.return_type,
        placement: TaskPlacement::Default,
        active_hir_id: None,
        active_expression_type: None,
    };
    for (index, parameter) in function.params.iter().enumerate() {
        if let Some(receiver) = &parameter.receiver {
            let value = format!("%arg_{index}");
            context
                .object_classes
                .insert(value.clone(), receiver.name.clone());
            context.receiver_types.insert(value, receiver.clone());
        }
    }
    for global in environment.globals {
        let value = context.lower_expression(&global.value);
        context.variables.insert(global.name.id, value);
    }
    context.lower_instructions(&function.instructions);
    if !context.terminated {
        if function.return_type == ValueType::Unit {
            context.output.push_str("    llvm.return\n");
        } else {
            context.output.push_str("    llvm.unreachable\n");
        }
    }
    context.output.push_str("  }\n");
}

#[derive(Clone)]
struct LoopTarget {
    break_block: usize,
    continue_block: usize,
    carried: Vec<(BindingId, ValueType)>,
    index: Option<String>,
}

struct LowerContext<'a> {
    output: &'a mut String,
    strings: &'a [String],
    function_returns: &'a HashMap<FunctionId, ValueType>,
    function_params: &'a HashMap<FunctionId, Vec<ValueType>>,
    function_return_classes: &'a HashMap<FunctionId, TypeDefinitionId>,
    method_return_classes: &'a HashMap<(TypeDefinitionId, String), TypeDefinitionId>,
    closure_definitions: Rc<RefCell<String>>,
    next_closure: Rc<Cell<usize>>,
    function_closures: Rc<RefCell<HashMap<FunctionId, String>>>,
    native_symbols: &'a HashMap<FunctionId, String>,
    sources: &'a severian_hir::SourceMap,
    trait_registries: &'a std::collections::BTreeMap<String, severian_hir::TraitRegistryDefinition>,
    classes: &'a [Class],
    field_object: Option<String>,
    field_names: HashSet<String>,
    field_types: HashMap<String, ValueType>,
    field_classes: HashMap<String, String>,
    object_classes: HashMap<String, String>,
    object_class_ids: HashMap<String, TypeDefinitionId>,
    receiver_types: HashMap<String, severian_hir::ReceiverType>,
    task_results: HashMap<String, ValueType>,
    channel_types: HashMap<String, ValueType>,
    variables: HashMap<BindingId, (String, ValueType)>,
    next_value: usize,
    next_block: usize,
    terminated: bool,
    is_main: bool,
    closure_callback: bool,
    declared_return: ValueType,
    placement: TaskPlacement,
    loop_targets: Vec<LoopTarget>,
    active_hir_id: Option<severian_hir::HirId>,
    active_expression_type: Option<ValueType>,
}

mod bridge;
mod collect;
mod collection;
mod control_flow;
mod expression;
mod ffi_shim;
mod instruction;
mod module;
mod operator;
mod types;
mod value;

pub use bridge::{native_bridge_source, rocm_bridge_source};
use collect::*;
pub use module::lower;
use types::*;
