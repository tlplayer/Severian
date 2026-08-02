#![forbid(unsafe_code)]

use severian_hir::{
    AssignmentOp, BinaryOp, Class, Expression, Function, Instruction, MatchPattern, Program,
    SwitchArm, TaskPlacement, UnaryOp, ValueType,
};
use severian_mlir::Module;
use std::collections::{HashMap, HashSet};
use std::fmt::Write;

pub fn lower(program: &Program) -> Module {
    let mut strings = Vec::new();
    for class in &program.classes {
        strings.push(class.name.clone());
        strings.extend(class.fields.iter().cloned());
        for function in class.methods.iter().chain(&class.constructors) {
            collect_strings(&function.instructions, &mut strings);
        }
    }
    for global in &program.globals {
        collect_expression_strings(&global.value, &mut strings);
    }
    for function in &program.functions {
        collect_strings(&function.instructions, &mut strings);
    }

    let mut output = String::from("module {\n");
    for (index, value) in strings.iter().enumerate() {
        writeln!(
            output,
            "  llvm.mlir.global internal constant @__sev_str_{index}(\"{}\\00\")",
            escape_string(value)
        )
        .unwrap();
    }
    output.push_str(concat!(
        "  llvm.mlir.global internal constant @__sev_fmt_int(\"%ld\\0A\\00\")\n",
        "  llvm.mlir.global internal constant @__sev_fmt_float(\"%.15g\\0A\\00\")\n",
        "  llvm.mlir.global internal constant @__sev_bool_true(\"true\\00\")\n",
        "  llvm.mlir.global internal constant @__sev_bool_false(\"false\\00\")\n",
        "  llvm.func @puts(!llvm.ptr) -> i32\n",
        "  llvm.func @printf(!llvm.ptr, ...) -> i32\n\n",
        "  llvm.func @snprintf(!llvm.ptr, i64, !llvm.ptr, ...) -> i32\n",
        "  llvm.func @malloc(i64) -> !llvm.ptr\n",
        "  llvm.func @abort()\n",
        "  llvm.func @strtod(!llvm.ptr, !llvm.ptr) -> f64\n\n",
        "  llvm.func @strlen(!llvm.ptr) -> i64\n\n",
        "  llvm.func @__sev_box_i64(i64) -> !llvm.ptr\n",
        "  llvm.func @__sev_box_f64(f64) -> !llvm.ptr\n",
        "  llvm.func @__sev_box_bool(i1) -> !llvm.ptr\n",
        "  llvm.func @__sev_box_string(!llvm.ptr) -> !llvm.ptr\n",
        "  llvm.func @__sev_box_collection(!llvm.ptr) -> !llvm.ptr\n",
        "  llvm.func @__sev_unbox_i64(!llvm.ptr) -> i64\n",
        "  llvm.func @__sev_unbox_f64(!llvm.ptr) -> f64\n",
        "  llvm.func @__sev_unbox_bool(!llvm.ptr) -> i1\n",
        "  llvm.func @__sev_unbox_string(!llvm.ptr) -> !llvm.ptr\n",
        "  llvm.func @__sev_unbox_ptr(!llvm.ptr) -> !llvm.ptr\n",
        "  llvm.func @__sev_value_add(!llvm.ptr, !llvm.ptr) -> !llvm.ptr\n",
        "  llvm.func @__sev_value_sub(!llvm.ptr, !llvm.ptr) -> !llvm.ptr\n",
        "  llvm.func @__sev_value_mul(!llvm.ptr, !llvm.ptr) -> !llvm.ptr\n",
        "  llvm.func @__sev_value_div(!llvm.ptr, !llvm.ptr) -> !llvm.ptr\n",
        "  llvm.func @__sev_value_equal(!llvm.ptr, !llvm.ptr) -> i1\n",
        "  llvm.func @__sev_value_less(!llvm.ptr, !llvm.ptr) -> i1\n",
        "  llvm.func @__sev_collection_new(i64) -> !llvm.ptr\n",
        "  llvm.func @__sev_collection_push(!llvm.ptr, !llvm.ptr)\n",
        "  llvm.func @__sev_collection_get(!llvm.ptr, i64) -> !llvm.ptr\n",
        "  llvm.func @__sev_collection_set(!llvm.ptr, i64, !llvm.ptr)\n",
        "  llvm.func @__sev_collection_size(!llvm.ptr) -> i64\n",
        "  llvm.func @__sev_collection_equal(!llvm.ptr, !llvm.ptr) -> i1\n",
        "  llvm.func @__sev_collection_reversed(!llvm.ptr) -> !llvm.ptr\n",
        "  llvm.func @__sev_set_contains(!llvm.ptr, !llvm.ptr) -> i1\n",
        "  llvm.func @__sev_map_new() -> !llvm.ptr\n",
        "  llvm.func @__sev_map_insert(!llvm.ptr, !llvm.ptr, !llvm.ptr)\n",
        "  llvm.func @__sev_map_get(!llvm.ptr, !llvm.ptr) -> !llvm.ptr\n",
        "  llvm.func @__sev_map_size(!llvm.ptr) -> i64\n",
        "  llvm.func @__sev_map_key_at(!llvm.ptr, i64) -> !llvm.ptr\n",
        "  llvm.func @__sev_map_value_at(!llvm.ptr, i64) -> !llvm.ptr\n",
        "  llvm.func @__sev_print_value(!llvm.ptr)\n",
        "  llvm.func @__sev_print_collection(!llvm.ptr)\n\n",
        "  llvm.func @__sev_print_value_inline(!llvm.ptr)\n",
        "  llvm.func @__sev_print_space()\n",
        "  llvm.func @__sev_print_newline()\n",
        "  llvm.func @__sev_object_new(!llvm.ptr) -> !llvm.ptr\n",
        "  llvm.func @__sev_object_set(!llvm.ptr, !llvm.ptr, !llvm.ptr)\n",
        "  llvm.func @__sev_object_get(!llvm.ptr, !llvm.ptr) -> !llvm.ptr\n",
        "  llvm.func @__sev_object_is(!llvm.ptr, !llvm.ptr) -> i1\n",
        "  llvm.func @__sev_matrix_new(i64, i64, f64) -> !llvm.ptr\n",
        "  llvm.func @__sev_matrix_shape(!llvm.ptr) -> !llvm.ptr\n",
        "  llvm.func @__sev_matrix_multiply(!llvm.ptr, !llvm.ptr) -> !llvm.ptr\n",
        "  llvm.func @__sev_matrix_scale(!llvm.ptr, f64) -> !llvm.ptr\n",
        "  llvm.func @__sev_matrix_equal(!llvm.ptr, !llvm.ptr) -> i1\n",
        "  llvm.func @__sev_matrix_print(!llvm.ptr)\n",
        "  llvm.func @__sev_dispatch_draw(!llvm.ptr)\n\n",
        "  llvm.func @__sev_variant_new(!llvm.ptr, !llvm.ptr) -> !llvm.ptr\n",
        "  llvm.func @__sev_variant_is(!llvm.ptr, !llvm.ptr) -> i1\n",
        "  llvm.func @__sev_variant_field(!llvm.ptr) -> !llvm.ptr\n",
        "  llvm.func @__sev_print_variant(!llvm.ptr)\n\n",
        "  llvm.func @__sev_builtin_read(!llvm.ptr) -> !llvm.ptr\n",
        "  llvm.func @__sev_builtin_http_get(!llvm.ptr) -> !llvm.ptr\n",
        "  llvm.func @__sev_builtin_int_parse(!llvm.ptr) -> !llvm.ptr\n",
        "  llvm.func @__sev_builtin_file_write(!llvm.ptr, !llvm.ptr) -> !llvm.ptr\n\n",
        "  llvm.func @__sev_regex_matches(!llvm.ptr, !llvm.ptr) -> i1\n\n",
        "  llvm.func @llvm.sqrt.f64(f64) -> f64\n\n",
    ));

    let task_specs = task_specs(program);
    let uses_channels = uses_channels(program);
    let mut await_types = HashSet::new();
    for task in &task_specs {
        write!(output, "  llvm.func @__sev_task_spawn_{}(", task.function).unwrap();
        for (index, ty) in task.params.iter().enumerate() {
            if index > 0 {
                output.push_str(", ");
            }
            output.push_str(mlir_type(*ty));
        }
        output.push_str(") -> !llvm.ptr\n");
        await_types.insert(task.return_type);
    }
    for ty in await_types {
        if ty == ValueType::Unit {
            output.push_str("  llvm.func @__sev_task_await_unit(!llvm.ptr)\n");
        } else {
            writeln!(
                output,
                "  llvm.func @__sev_task_await_{}(!llvm.ptr) -> {}",
                task_type_suffix(ty),
                mlir_type(ty)
            )
            .unwrap();
        }
    }
    if uses_channels {
        output.push_str(concat!(
            "  llvm.func @__sev_channel_create(i64) -> !llvm.ptr\n",
            "  llvm.func @__sev_channel_send_ptr_async(!llvm.ptr, !llvm.ptr) -> !llvm.ptr\n",
            "  llvm.func @__sev_channel_receive_ptr(!llvm.ptr) -> !llvm.ptr\n",
        ));
        if !task_specs
            .iter()
            .any(|task| task.return_type == ValueType::Unit)
        {
            output.push_str("  llvm.func @__sev_task_await_unit(!llvm.ptr)\n");
        }
    }
    if !task_specs.is_empty() || uses_channels {
        output.push('\n');
    }

    let mut function_returns = program
        .functions
        .iter()
        .map(|function| (function.name.clone(), function.return_type))
        .collect::<HashMap<_, _>>();
    let mut function_params = program
        .functions
        .iter()
        .map(|function| {
            (
                function.name.clone(),
                function
                    .params
                    .iter()
                    .map(|parameter| parameter.ty)
                    .collect(),
            )
        })
        .collect::<HashMap<_, Vec<_>>>();
    for class in &program.classes {
        for method in &class.methods {
            function_returns.insert(
                format!("{}_{}", class.name, method.name),
                method.return_type,
            );
            function_params.insert(
                format!("{}_{}", class.name, method.name),
                method.params.iter().map(|parameter| parameter.ty).collect(),
            );
        }
    }
    for class in &program.classes {
        for constructor in &class.constructors {
            lower_class_function(
                class,
                constructor,
                &format!("{}_ctor_{}", class.name, constructor.params.len()),
                &program.classes,
                &strings,
                &function_returns,
                &function_params,
                &mut output,
            );
        }
        for method in &class.methods {
            lower_class_function(
                class,
                method,
                &format!("{}_{}", class.name, method.name),
                &program.classes,
                &strings,
                &function_returns,
                &function_params,
                &mut output,
            );
        }
    }
    for function in &program.functions {
        lower_function(
            function,
            &program.globals,
            &program.classes,
            &strings,
            &function_returns,
            &function_params,
            &mut output,
        );
    }
    output.push_str("}\n");
    Module::new(output)
}

fn lower_function(
    function: &Function,
    globals: &[severian_hir::Global],
    classes: &[Class],
    strings: &[String],
    function_returns: &HashMap<String, ValueType>,
    function_params: &HashMap<String, Vec<ValueType>>,
    output: &mut String,
) {
    let is_main = function.name == "main";
    write!(output, "  llvm.func @{}(", function.name).unwrap();
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
        strings,
        function_returns,
        function_params,
        classes,
        field_object: None,
        field_names: HashSet::new(),
        object_classes: HashMap::new(),
        declared_return: function.return_type,
        task_results: HashMap::new(),
        channel_types: HashMap::new(),
        variables: function
            .params
            .iter()
            .enumerate()
            .map(|(index, param)| (param.name.clone(), (format!("%arg_{index}"), param.ty)))
            .collect(),
        next_value: 0,
        next_block: 0,
        terminated: false,
        is_main,
    };
    for global in globals {
        let value = context.lower_expression(&global.value);
        context.variables.insert(global.name.clone(), value);
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
    classes: &[Class],
    strings: &[String],
    function_returns: &HashMap<String, ValueType>,
    function_params: &HashMap<String, Vec<ValueType>>,
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
        strings,
        function_returns,
        function_params,
        classes,
        field_object: Some("%self".into()),
        field_names: class.fields.iter().cloned().collect(),
        object_classes: HashMap::from([("%self".into(), class.name.clone())]),
        task_results: HashMap::new(),
        channel_types: HashMap::new(),
        variables: function
            .params
            .iter()
            .enumerate()
            .map(|(index, param)| (param.name.clone(), (format!("%arg_{index}"), param.ty)))
            .collect(),
        next_value: 0,
        next_block: 0,
        terminated: false,
        is_main: false,
        declared_return: function.return_type,
    };
    context.lower_instructions(&function.instructions);
    if !context.terminated && function.return_type == ValueType::Unit {
        context.output.push_str("    llvm.return\n");
    }
    context.output.push_str("  }\n");
}

struct LowerContext<'a> {
    output: &'a mut String,
    strings: &'a [String],
    function_returns: &'a HashMap<String, ValueType>,
    function_params: &'a HashMap<String, Vec<ValueType>>,
    classes: &'a [Class],
    field_object: Option<String>,
    field_names: HashSet<String>,
    object_classes: HashMap<String, String>,
    task_results: HashMap<String, ValueType>,
    channel_types: HashMap<String, ValueType>,
    variables: HashMap<String, (String, ValueType)>,
    next_value: usize,
    next_block: usize,
    terminated: bool,
    is_main: bool,
    declared_return: ValueType,
}

impl LowerContext<'_> {
    fn lower_instructions(&mut self, instructions: &[Instruction]) {
        for instruction in instructions {
            if self.terminated {
                break;
            }
            match instruction {
                Instruction::Let { name, value } => {
                    let lowered = self.lower_expression(value);
                    self.variables.insert(name.clone(), lowered);
                }
                Instruction::TryLet { name, value } => {
                    let lowered = self.lower_expression(value);
                    self.variables.insert(name.clone(), lowered);
                }
                Instruction::Assign { target, op, value } => {
                    if let Expression::Variable(name) = target {
                        let right = self.lower_expression(value);
                        if self.field_names.contains(name) && !self.variables.contains_key(name) {
                            let object = self.field_object.clone().unwrap();
                            let field = self.string_address(name);
                            let value = if *op == AssignmentOp::Assign {
                                right
                            } else {
                                let current = self.fresh_value();
                                writeln!(self.output, "    {current} = llvm.call @__sev_object_get({object}, {field}) : (!llvm.ptr, !llvm.ptr) -> !llvm.ptr").unwrap();
                                let current = self.unbox_value((current, ValueType::Any), right.1);
                                self.lower_binary_values(current, assignment_binary(*op), right)
                            };
                            let boxed = self.box_value(value);
                            writeln!(self.output, "    llvm.call @__sev_object_set({object}, {field}, {boxed}) : (!llvm.ptr, !llvm.ptr, !llvm.ptr) -> ()").unwrap();
                            continue;
                        }
                        let lowered = if *op == AssignmentOp::Assign {
                            right
                        } else {
                            let left = self.variables.get(name).cloned().unwrap_or(right.clone());
                            self.lower_binary_values(left, assignment_binary(*op), right)
                        };
                        self.variables.insert(name.clone(), lowered);
                    } else if let Expression::Index { object, index } = target {
                        let (object, object_type) = self.lower_expression(object);
                        let index = self.lower_expression(index);
                        let right = self.lower_expression(value);
                        if object_type == ValueType::Map {
                            let key = self.box_value(index);
                            let boxed = if *op == AssignmentOp::Assign {
                                self.box_value(right)
                            } else {
                                let current = self.fresh_value();
                                writeln!(self.output, "    {current} = llvm.call @__sev_map_get({object}, {key}) : (!llvm.ptr, !llvm.ptr) -> !llvm.ptr").unwrap();
                                let current = self.unbox_value((current, ValueType::Any), right.1);
                                let updated = self.lower_binary_values(
                                    current,
                                    assignment_binary(*op),
                                    right,
                                );
                                self.box_value(updated)
                            };
                            writeln!(self.output, "    llvm.call @__sev_map_insert({object}, {key}, {boxed}) : (!llvm.ptr, !llvm.ptr, !llvm.ptr) -> ()").unwrap();
                        } else {
                            let index = self.unbox_value(index, ValueType::Int).0;
                            let boxed = if *op == AssignmentOp::Assign {
                                self.box_value(right)
                            } else {
                                let current = self.fresh_value();
                                writeln!(self.output, "    {current} = llvm.call @__sev_collection_get({object}, {index}) : (!llvm.ptr, i64) -> !llvm.ptr").unwrap();
                                let current = self.unbox_value((current, ValueType::Any), right.1);
                                let updated = self.lower_binary_values(
                                    current,
                                    assignment_binary(*op),
                                    right,
                                );
                                self.box_value(updated)
                            };
                            writeln!(self.output, "    llvm.call @__sev_collection_set({object}, {index}, {boxed}) : (!llvm.ptr, i64, !llvm.ptr) -> ()").unwrap();
                        }
                    }
                }
                Instruction::Print(value) => {
                    let (value, ty) = self.lower_expression(value);
                    match ty {
                        ValueType::String => {
                            let status = self.fresh_value();
                            writeln!(
                                self.output,
                                "    {status} = llvm.call @puts({value}) : (!llvm.ptr) -> i32"
                            )
                            .unwrap();
                        }
                        ValueType::Int => {
                            self.lower_formatted_print("@__sev_fmt_int", &value, ValueType::Int)
                        }
                        ValueType::Float => {
                            self.lower_formatted_print("@__sev_fmt_float", &value, ValueType::Float)
                        }
                        ValueType::Bool => {
                            let true_value = self.fresh_value();
                            writeln!(
                                self.output,
                                "    {true_value} = llvm.mlir.addressof @__sev_bool_true : !llvm.ptr"
                            )
                            .unwrap();
                            let false_value = self.fresh_value();
                            writeln!(
                                self.output,
                                "    {false_value} = llvm.mlir.addressof @__sev_bool_false : !llvm.ptr"
                            )
                            .unwrap();
                            let text = self.fresh_value();
                            writeln!(
                                self.output,
                                "    {text} = llvm.select {value}, {true_value}, {false_value} : i1, !llvm.ptr"
                            )
                            .unwrap();
                            let status = self.fresh_value();
                            writeln!(
                                self.output,
                                "    {status} = llvm.call @puts({text}) : (!llvm.ptr) -> i32"
                            )
                            .unwrap();
                        }
                        ValueType::Any => {
                            if self
                                .object_classes
                                .get(&value)
                                .is_some_and(|class| class == "Matrix")
                            {
                                writeln!(self.output, "    llvm.call @__sev_matrix_print({value}) : (!llvm.ptr) -> ()").unwrap();
                            } else {
                                writeln!(
                                    self.output,
                                    "    llvm.call @__sev_print_value({value}) : (!llvm.ptr) -> ()"
                                )
                                .unwrap();
                            }
                        }
                        ValueType::List | ValueType::Tuple | ValueType::Set => {
                            writeln!(self.output, "    llvm.call @__sev_print_collection({value}) : (!llvm.ptr) -> ()").unwrap();
                        }
                        ValueType::Result | ValueType::Option => {
                            writeln!(
                                self.output,
                                "    llvm.call @__sev_print_variant({value}) : (!llvm.ptr) -> ()"
                            )
                            .unwrap();
                        }
                        _ => {}
                    }
                }
                Instruction::Evaluate(expression) => {
                    let (value, _) = self.lower_expression(expression);
                    if matches!(expression, Expression::Task { .. }) {
                        if let Some(return_type) = self.task_results.remove(&value) {
                            if return_type == ValueType::Unit {
                                writeln!(self.output, "    llvm.call @__sev_task_await_unit({value}) : (!llvm.ptr) -> ()").unwrap();
                            } else {
                                let ignored = self.fresh_value();
                                writeln!(self.output, "    {ignored} = llvm.call @__sev_task_await_{}({value}) : (!llvm.ptr) -> {}", task_type_suffix(return_type), mlir_type(return_type)).unwrap();
                            }
                        }
                    }
                }
                Instruction::Assert(expression) => {
                    let lowered = self.lower_expression(expression);
                    let (condition, _) = self.unbox_value(lowered, ValueType::Bool);
                    let passed = self.fresh_block();
                    let failed = self.fresh_block();
                    writeln!(
                        self.output,
                        "    llvm.cond_br {condition}, ^bb{passed}, ^bb{failed}"
                    )
                    .unwrap();
                    writeln!(self.output, "  ^bb{failed}:").unwrap();
                    writeln!(self.output, "    llvm.call @abort() : () -> ()").unwrap();
                    writeln!(self.output, "    llvm.unreachable").unwrap();
                    writeln!(self.output, "  ^bb{passed}:").unwrap();
                    self.terminated = false;
                }
                Instruction::Return(value) => {
                    if self.is_main {
                        let success = self.fresh_value();
                        writeln!(
                            self.output,
                            "    {success} = llvm.mlir.constant(0 : i32) : i32"
                        )
                        .unwrap();
                        writeln!(self.output, "    llvm.return {success} : i32").unwrap();
                    } else if let Some(value) = value {
                        let mut lowered = self.lower_expression(value);
                        if matches!(
                            self.declared_return,
                            ValueType::Any | ValueType::Result | ValueType::Option
                        ) && !matches!(
                            lowered.1,
                            ValueType::Any | ValueType::Result | ValueType::Option
                        ) {
                            lowered = (self.box_value(lowered), ValueType::Any);
                        }
                        let (value, ty) = self.unbox_value(lowered, self.declared_return);
                        writeln!(self.output, "    llvm.return {value} : {}", mlir_type(ty))
                            .unwrap();
                    } else {
                        self.output.push_str("    llvm.return\n");
                    }
                    self.terminated = true;
                }
                Instruction::If {
                    condition,
                    then_instructions,
                    else_instructions,
                } => {
                    let incoming = self.variables.clone();
                    let (condition, _) = self.lower_expression(condition);
                    let then_block = self.fresh_block();
                    let else_block = self.fresh_block();
                    let continue_block = self.fresh_block();
                    writeln!(
                        self.output,
                        "    llvm.cond_br {condition}, ^bb{then_block}, ^bb{else_block}"
                    )
                    .unwrap();
                    writeln!(self.output, "  ^bb{then_block}:").unwrap();
                    self.terminated = false;
                    self.lower_instructions(then_instructions);
                    let then_terminated = self.terminated;
                    let then_variables = self.variables.clone();
                    if !then_terminated {
                        let mut carried = incoming.keys().cloned().collect::<Vec<_>>();
                        carried.sort();
                        let values = carried
                            .iter()
                            .map(|name| {
                                then_variables
                                    .get(name)
                                    .unwrap_or_else(|| &incoming[name])
                                    .0
                                    .as_str()
                            })
                            .collect::<Vec<_>>()
                            .join(", ");
                        let types = carried
                            .iter()
                            .map(|name| {
                                mlir_type(
                                    then_variables
                                        .get(name)
                                        .unwrap_or_else(|| &incoming[name])
                                        .1,
                                )
                            })
                            .collect::<Vec<_>>()
                            .join(", ");
                        if values.is_empty() {
                            writeln!(self.output, "    llvm.br ^bb{continue_block}").unwrap();
                        } else {
                            writeln!(
                                self.output,
                                "    llvm.br ^bb{continue_block}({values} : {types})"
                            )
                            .unwrap();
                        }
                    }
                    writeln!(self.output, "  ^bb{else_block}:").unwrap();
                    self.variables = incoming.clone();
                    self.terminated = false;
                    self.lower_instructions(else_instructions);
                    let else_terminated = self.terminated;
                    let else_variables = self.variables.clone();
                    if !else_terminated {
                        let mut carried = incoming.keys().cloned().collect::<Vec<_>>();
                        carried.sort();
                        let values = carried
                            .iter()
                            .map(|name| {
                                else_variables
                                    .get(name)
                                    .unwrap_or_else(|| &incoming[name])
                                    .0
                                    .as_str()
                            })
                            .collect::<Vec<_>>()
                            .join(", ");
                        let types = carried
                            .iter()
                            .map(|name| {
                                mlir_type(
                                    else_variables
                                        .get(name)
                                        .unwrap_or_else(|| &incoming[name])
                                        .1,
                                )
                            })
                            .collect::<Vec<_>>()
                            .join(", ");
                        if values.is_empty() {
                            writeln!(self.output, "    llvm.br ^bb{continue_block}").unwrap();
                        } else {
                            writeln!(
                                self.output,
                                "    llvm.br ^bb{continue_block}({values} : {types})"
                            )
                            .unwrap();
                        }
                    }
                    if !then_terminated || !else_terminated {
                        let mut carried = incoming.into_iter().collect::<Vec<_>>();
                        carried.sort_by(|left, right| left.0.cmp(&right.0));
                        let arguments = carried
                            .iter()
                            .map(|(_, (_, ty))| {
                                let value = self.fresh_value();
                                (value, *ty)
                            })
                            .collect::<Vec<_>>();
                        let signature = arguments
                            .iter()
                            .map(|(value, ty)| format!("{value}: {}", mlir_type(*ty)))
                            .collect::<Vec<_>>()
                            .join(", ");
                        if signature.is_empty() {
                            writeln!(self.output, "  ^bb{continue_block}:").unwrap();
                        } else {
                            writeln!(self.output, "  ^bb{continue_block}({signature}):").unwrap();
                        }
                        self.variables = carried
                            .into_iter()
                            .zip(arguments)
                            .map(|((name, _), value)| (name, value))
                            .collect();
                        self.terminated = false;
                    } else {
                        self.terminated = true;
                    }
                }
                Instruction::While {
                    setup,
                    condition,
                    instructions,
                    ..
                } => {
                    if let Some(setup) = setup {
                        self.lower_instructions(std::slice::from_ref(setup));
                    }
                    self.lower_while(condition, instructions);
                }
                Instruction::For {
                    pattern,
                    iterable,
                    instructions,
                } => self.lower_for(pattern, iterable, instructions),
                Instruction::Switch { value, arms } => self.lower_switch(value, arms),
                Instruction::ChannelSwitch {
                    channels,
                    setup,
                    arms,
                    ..
                } => self.lower_channel_switch(channels, setup.as_deref(), arms),
                Instruction::With { instructions, .. } => {
                    self.lower_instructions(instructions);
                }
            }
        }
    }

    fn lower_expression(&mut self, expression: &Expression) -> (String, ValueType) {
        match expression {
            Expression::Integer(value) => {
                let result = self.fresh_value();
                writeln!(
                    self.output,
                    "    {result} = llvm.mlir.constant({value} : i64) : i64"
                )
                .unwrap();
                (result, ValueType::Int)
            }
            Expression::Float(bits) => {
                let result = self.fresh_value();
                let value = f64::from_bits(*bits);
                writeln!(
                    self.output,
                    "    {result} = llvm.mlir.constant({value:.17e} : f64) : f64"
                )
                .unwrap();
                (result, ValueType::Float)
            }
            Expression::Boolean(value) => {
                let result = self.fresh_value();
                let value = i32::from(*value);
                writeln!(
                    self.output,
                    "    {result} = llvm.mlir.constant({value} : i1) : i1"
                )
                .unwrap();
                (result, ValueType::Bool)
            }
            Expression::String(value) => {
                let index = self
                    .strings
                    .iter()
                    .position(|candidate| candidate == value)
                    .unwrap();
                let result = self.fresh_value();
                writeln!(
                    self.output,
                    "    {result} = llvm.mlir.addressof @__sev_str_{index} : !llvm.ptr"
                )
                .unwrap();
                (result, ValueType::String)
            }
            Expression::Variable(name) => {
                if let Some(value) = self.variables.get(name).cloned() {
                    value
                } else if self.field_names.contains(name) {
                    let object = self.field_object.clone().unwrap();
                    let field = self.string_address(name);
                    let result = self.fresh_value();
                    writeln!(self.output, "    {result} = llvm.call @__sev_object_get({object}, {field}) : (!llvm.ptr, !llvm.ptr) -> !llvm.ptr").unwrap();
                    (result, ValueType::Any)
                } else {
                    let result = self.fresh_value();
                    writeln!(self.output, "    {result} = llvm.mlir.zero : !llvm.ptr").unwrap();
                    (result, ValueType::Any)
                }
            }
            Expression::Function(name) => {
                let result = self.fresh_value();
                writeln!(
                    self.output,
                    "    {result} = llvm.mlir.addressof @{name} : !llvm.ptr"
                )
                .unwrap();
                (result, ValueType::Function)
            }
            Expression::List(values) => self.lower_collection_literal(values, ValueType::List, 0),
            Expression::Tuple(values) => self.lower_collection_literal(values, ValueType::Tuple, 1),
            Expression::Set(values) => self.lower_collection_literal(values, ValueType::Set, 2),
            Expression::Map(entries) => {
                let result = self.fresh_value();
                writeln!(
                    self.output,
                    "    {result} = llvm.call @__sev_map_new() : () -> !llvm.ptr"
                )
                .unwrap();
                for (key, value) in entries {
                    let key = self.lower_expression(key);
                    let value = self.lower_expression(value);
                    let key = self.box_value(key);
                    let value = self.box_value(value);
                    writeln!(self.output, "    llvm.call @__sev_map_insert({result}, {key}, {value}) : (!llvm.ptr, !llvm.ptr, !llvm.ptr) -> ()").unwrap();
                }
                (result, ValueType::Map)
            }
            Expression::ListComprehension {
                element,
                variable,
                iterable,
                condition,
            } => self.lower_list_comprehension(element, variable, iterable, condition.as_deref()),
            Expression::Conditional {
                condition,
                then_expression,
                else_expression,
            } => {
                let (condition, _) = self.lower_expression(condition);
                let mut then_value = self.lower_expression(then_expression);
                let mut else_value = self.lower_expression(else_expression);
                let result_type = if then_value.1 == else_value.1 {
                    then_value.1
                } else {
                    then_value = (self.box_value(then_value), ValueType::Any);
                    else_value = (self.box_value(else_value), ValueType::Any);
                    ValueType::Any
                };
                let result = self.fresh_value();
                writeln!(
                    self.output,
                    "    {result} = llvm.select {condition}, {}, {} : i1, {}",
                    then_value.0,
                    else_value.0,
                    mlir_type(result_type)
                )
                .unwrap();
                (result, result_type)
            }
            Expression::PrintArgs(values) => {
                for (index, value) in values.iter().enumerate() {
                    if index > 0 {
                        writeln!(self.output, "    llvm.call @__sev_print_space() : () -> ()")
                            .unwrap();
                    }
                    let value = self.lower_expression(value);
                    let value = self.box_value(value);
                    writeln!(
                        self.output,
                        "    llvm.call @__sev_print_value_inline({value}) : (!llvm.ptr) -> ()"
                    )
                    .unwrap();
                }
                writeln!(
                    self.output,
                    "    llvm.call @__sev_print_newline() : () -> ()"
                )
                .unwrap();
                (String::new(), ValueType::Unit)
            }
            Expression::Construct { class, args } => {
                let class_name = self.string_address(class);
                let result = self.fresh_value();
                writeln!(self.output, "    {result} = llvm.call @__sev_object_new({class_name}) : (!llvm.ptr) -> !llvm.ptr").unwrap();
                let lowered_args = args
                    .iter()
                    .map(|arg| self.lower_expression(arg))
                    .collect::<Vec<_>>();
                let definition = self
                    .classes
                    .iter()
                    .find(|candidate| candidate.name == *class)
                    .cloned();
                let constructor = definition.as_ref().and_then(|definition| {
                    definition
                        .constructors
                        .iter()
                        .find(|constructor| constructor.params.len() == args.len())
                });
                if let Some(constructor) = constructor {
                    let values = lowered_args
                        .iter()
                        .map(|(value, _)| value.as_str())
                        .collect::<Vec<_>>()
                        .join(", ");
                    let types = lowered_args
                        .iter()
                        .map(|(_, ty)| mlir_type(*ty))
                        .collect::<Vec<_>>()
                        .join(", ");
                    let value_suffix = if values.is_empty() {
                        String::new()
                    } else {
                        format!(", {values}")
                    };
                    let type_suffix = if types.is_empty() {
                        String::new()
                    } else {
                        format!(", {types}")
                    };
                    writeln!(self.output, "    llvm.call @{class}_ctor_{}({result}{value_suffix}) : (!llvm.ptr{type_suffix}) -> ()", constructor.params.len()).unwrap();
                } else if let Some(definition) = definition {
                    for (index, field) in definition.fields.iter().enumerate() {
                        let value = lowered_args.get(index).cloned().or_else(|| {
                            definition.field_defaults[index]
                                .as_ref()
                                .map(|default| self.lower_expression(default))
                        });
                        let Some(value) = value else { continue };
                        let field = self.string_address(field);
                        let value = self.box_value(value);
                        writeln!(self.output, "    llvm.call @__sev_object_set({result}, {field}, {value}) : (!llvm.ptr, !llvm.ptr, !llvm.ptr) -> ()").unwrap();
                    }
                }
                self.object_classes.insert(result.clone(), class.clone());
                (result, ValueType::Any)
            }
            Expression::Member { object, member } => {
                let (object, _) = self.lower_expression(object);
                if member == "shape"
                    && self
                        .object_classes
                        .get(&object)
                        .is_some_and(|class| class == "Matrix")
                {
                    let result = self.fresh_value();
                    writeln!(self.output, "    {result} = llvm.call @__sev_matrix_shape({object}) : (!llvm.ptr) -> !llvm.ptr").unwrap();
                    return (result, ValueType::Any);
                }
                let field = self.string_address(member);
                let result = self.fresh_value();
                writeln!(self.output, "    {result} = llvm.call @__sev_object_get({object}, {field}) : (!llvm.ptr, !llvm.ptr) -> !llvm.ptr").unwrap();
                (result, ValueType::Any)
            }
            Expression::ChaosRule { .. } => {
                let result = self.fresh_value();
                writeln!(self.output, "    {result} = llvm.mlir.zero : !llvm.ptr").unwrap();
                (result, ValueType::Any)
            }
            Expression::MethodCall {
                object,
                method,
                args,
            } if method == "append" => {
                let (mut object, object_type) = self.lower_expression(object);
                if object_type == ValueType::Any {
                    let unboxed = self.fresh_value();
                    writeln!(self.output, "    {unboxed} = llvm.call @__sev_unbox_ptr({object}) : (!llvm.ptr) -> !llvm.ptr").unwrap();
                    object = unboxed;
                }
                let value = self.lower_expression(&args[0]);
                let value = self.box_value(value);
                writeln!(self.output, "    llvm.call @__sev_collection_push({object}, {value}) : (!llvm.ptr, !llvm.ptr) -> ()").unwrap();
                (String::new(), ValueType::Unit)
            }
            Expression::MethodCall {
                object,
                method,
                args,
            } if method == "reversed" && args.is_empty() => {
                let (mut object, object_type) = self.lower_expression(object);
                if object_type == ValueType::Any {
                    let unboxed = self.fresh_value();
                    writeln!(self.output, "    {unboxed} = llvm.call @__sev_unbox_ptr({object}) : (!llvm.ptr) -> !llvm.ptr").unwrap();
                    object = unboxed;
                }
                let result = self.fresh_value();
                writeln!(self.output, "    {result} = llvm.call @__sev_collection_reversed({object}) : (!llvm.ptr) -> !llvm.ptr").unwrap();
                (result, ValueType::List)
            }
            Expression::MethodCall {
                object,
                method,
                args,
            } if method == "length" && args.is_empty() => {
                let (object, object_type) = self.lower_expression(object);
                let object = self.unbox_value((object, object_type), ValueType::String).0;
                let result = self.fresh_value();
                writeln!(
                    self.output,
                    "    {result} = llvm.call @strlen({object}) : (!llvm.ptr) -> i64"
                )
                .unwrap();
                (result, ValueType::Int)
            }
            Expression::MethodCall {
                object,
                method,
                args,
            } if matches!(
                method.as_str(),
                "wrapping_add" | "saturating_add" | "overflowing_add"
            ) && args.len() == 1 =>
            {
                let (left, left_type) = self.lower_expression(object);
                let left = self.unbox_value((left, left_type), ValueType::Int).0;
                let (right, right_type) = self.lower_expression(&args[0]);
                let right = self.unbox_value((right, right_type), ValueType::Int).0;
                let sum = self.fresh_value();
                writeln!(self.output, "    {sum} = llvm.add {left}, {right} : i64").unwrap();
                let limit = self.fresh_value();
                writeln!(
                    self.output,
                    "    {limit} = llvm.mlir.constant(256 : i64) : i64"
                )
                .unwrap();
                let wrapped = self.fresh_value();
                writeln!(
                    self.output,
                    "    {wrapped} = llvm.urem {sum}, {limit} : i64"
                )
                .unwrap();
                if method == "wrapping_add" {
                    return (wrapped, ValueType::Int);
                }
                let max = self.fresh_value();
                writeln!(
                    self.output,
                    "    {max} = llvm.mlir.constant(255 : i64) : i64"
                )
                .unwrap();
                let overflowed = self.fresh_value();
                writeln!(
                    self.output,
                    "    {overflowed} = llvm.icmp \"sgt\" {sum}, {max} : i64"
                )
                .unwrap();
                if method == "saturating_add" {
                    let capped = self.fresh_value();
                    writeln!(
                        self.output,
                        "    {capped} = llvm.select {overflowed}, {max}, {sum} : i1, i64"
                    )
                    .unwrap();
                    return (capped, ValueType::Int);
                }
                let kind = self.fresh_value();
                writeln!(
                    self.output,
                    "    {kind} = llvm.mlir.constant(1 : i64) : i64"
                )
                .unwrap();
                let tuple = self.fresh_value();
                writeln!(
                    self.output,
                    "    {tuple} = llvm.call @__sev_collection_new({kind}) : (i64) -> !llvm.ptr"
                )
                .unwrap();
                let wrapped = self.box_value((wrapped, ValueType::Int));
                let overflowed = self.box_value((overflowed, ValueType::Bool));
                writeln!(self.output, "    llvm.call @__sev_collection_push({tuple}, {wrapped}) : (!llvm.ptr, !llvm.ptr) -> ()").unwrap();
                writeln!(self.output, "    llvm.call @__sev_collection_push({tuple}, {overflowed}) : (!llvm.ptr, !llvm.ptr) -> ()").unwrap();
                (tuple, ValueType::Tuple)
            }
            Expression::MethodCall {
                object,
                method,
                args,
            } if method == "lessThan" && args.len() == 1 => {
                let left = self.lower_expression(object);
                let right = self.lower_expression(&args[0]);
                let left = self.box_value(left);
                let right = self.box_value(right);
                let result = self.fresh_value();
                writeln!(self.output, "    {result} = llvm.call @__sev_value_less({left}, {right}) : (!llvm.ptr, !llvm.ptr) -> i1").unwrap();
                (result, ValueType::Bool)
            }
            Expression::MethodCall {
                object: _,
                method,
                args,
            } if method == "zero" && args.is_empty() => {
                let zero = self.fresh_value();
                writeln!(
                    self.output,
                    "    {zero} = llvm.mlir.constant(0 : i64) : i64"
                )
                .unwrap();
                let result = self.box_value((zero, ValueType::Int));
                (result, ValueType::Any)
            }
            Expression::MethodCall {
                object,
                method,
                args,
            } if method == "add" && args.len() == 1 => {
                let left = self.lower_expression(object);
                let right = self.lower_expression(&args[0]);
                let left = self.box_value(left);
                let right = self.box_value(right);
                let result = self.fresh_value();
                writeln!(self.output, "    {result} = llvm.call @__sev_value_add({left}, {right}) : (!llvm.ptr, !llvm.ptr) -> !llvm.ptr").unwrap();
                (result, ValueType::Any)
            }
            Expression::MethodCall {
                object,
                method,
                args,
            } => {
                let (object, _) = self.lower_expression(object);
                let lowered_args = args
                    .iter()
                    .map(|arg| self.lower_expression(arg))
                    .collect::<Vec<_>>();
                let Some(class) = self.object_classes.get(&object).cloned() else {
                    if method == "draw" {
                        writeln!(
                            self.output,
                            "    llvm.call @__sev_dispatch_draw({object}) : (!llvm.ptr) -> ()"
                        )
                        .unwrap();
                        return (String::new(), ValueType::Unit);
                    }
                    let result = self.fresh_value();
                    writeln!(self.output, "    {result} = llvm.mlir.zero : !llvm.ptr").unwrap();
                    return (result, ValueType::Any);
                };
                let symbol = format!("{class}_{method}");
                let method_definition = self
                    .classes
                    .iter()
                    .find(|candidate| candidate.name == class)
                    .and_then(|definition| {
                        definition
                            .methods
                            .iter()
                            .find(|candidate| candidate.name == *method)
                    });
                let lowered_args = lowered_args
                    .into_iter()
                    .enumerate()
                    .map(|(index, argument)| {
                        if method_definition
                            .and_then(|definition| definition.params.get(index))
                            .is_some_and(|parameter| parameter.ty == ValueType::Any)
                            && argument.1 != ValueType::Any
                        {
                            (self.box_value(argument), ValueType::Any)
                        } else {
                            argument
                        }
                    })
                    .collect::<Vec<_>>();
                let values = lowered_args
                    .iter()
                    .map(|(value, _)| value.as_str())
                    .collect::<Vec<_>>()
                    .join(", ");
                let types = lowered_args
                    .iter()
                    .map(|(_, ty)| mlir_type(*ty))
                    .collect::<Vec<_>>()
                    .join(", ");
                let value_suffix = if values.is_empty() {
                    String::new()
                } else {
                    format!(", {values}")
                };
                let type_suffix = if types.is_empty() {
                    String::new()
                } else {
                    format!(", {types}")
                };
                let return_type = self
                    .function_returns
                    .get(&symbol)
                    .copied()
                    .unwrap_or(ValueType::Any);
                if return_type == ValueType::Unit {
                    writeln!(self.output, "    llvm.call @{symbol}({object}{value_suffix}) : (!llvm.ptr{type_suffix}) -> ()").unwrap();
                    (String::new(), ValueType::Unit)
                } else {
                    let result = self.fresh_value();
                    writeln!(self.output, "    {result} = llvm.call @{symbol}({object}{value_suffix}) : (!llvm.ptr{type_suffix}) -> {}", mlir_type(return_type)).unwrap();
                    if return_type == ValueType::Any {
                        self.object_classes.insert(result.clone(), class);
                    }
                    (result, return_type)
                }
            }
            Expression::Task { value, placement } => {
                if let Expression::Send { value, channel } = value.as_ref() {
                    let (value, value_type) = self.lower_expression(value);
                    let (channel, _) = self.lower_expression(channel);
                    self.channel_types.insert(channel.clone(), value_type);
                    let result = self.fresh_value();
                    writeln!(
                        self.output,
                        "    {result} = llvm.call @__sev_channel_send_ptr_async({value}, {channel}) : (!llvm.ptr, !llvm.ptr) -> !llvm.ptr"
                    )
                    .unwrap();
                    self.task_results.insert(result.clone(), ValueType::Unit);
                    return (result, ValueType::Any);
                }
                let Expression::Call { function, args } = value.as_ref() else {
                    let result = self.fresh_value();
                    writeln!(self.output, "    {result} = llvm.mlir.zero : !llvm.ptr").unwrap();
                    return (result, ValueType::Any);
                };
                let args = args
                    .iter()
                    .map(|argument| self.lower_expression(argument))
                    .collect::<Vec<_>>();
                let args = self.coerce_call_arguments(function, args);
                let values = args
                    .iter()
                    .map(|(value, _)| value.as_str())
                    .collect::<Vec<_>>()
                    .join(", ");
                let types = args
                    .iter()
                    .map(|(_, ty)| mlir_type(*ty))
                    .collect::<Vec<_>>()
                    .join(", ");
                let return_type = self
                    .function_returns
                    .get(function)
                    .copied()
                    .unwrap_or(ValueType::Any);
                let result = self.fresh_value();
                let placement_attribute = match placement {
                    TaskPlacement::Default => "",
                    TaskPlacement::Local => " {severian_distribution = \"local\"}",
                };
                writeln!(
                    self.output,
                    "    {result} = llvm.call @__sev_task_spawn_{function}({values}){placement_attribute} : ({types}) -> !llvm.ptr"
                )
                .unwrap();
                self.task_results.insert(result.clone(), return_type);
                (result, ValueType::Any)
            }
            Expression::Await(value) => {
                let (task, _) = self.lower_expression(value);
                let return_type = self.task_results.remove(&task);
                if let Some(channel_type) = self.channel_types.get(&task).copied() {
                    let result = self.fresh_value();
                    writeln!(
                        self.output,
                        "    {result} = llvm.call @__sev_channel_receive_ptr({task}) : (!llvm.ptr) -> !llvm.ptr"
                    )
                    .unwrap();
                    return (result, channel_type);
                }
                let Some(return_type) = return_type else {
                    let result = self.fresh_value();
                    writeln!(self.output, "    {result} = llvm.mlir.zero : !llvm.ptr").unwrap();
                    return (result, ValueType::Any);
                };
                if return_type == ValueType::Unit {
                    writeln!(
                        self.output,
                        "    llvm.call @__sev_task_await_unit({task}) : (!llvm.ptr) -> ()"
                    )
                    .unwrap();
                    return (String::new(), ValueType::Unit);
                }
                let result = self.fresh_value();
                writeln!(
                    self.output,
                    "    {result} = llvm.call @__sev_task_await_{}({task}) : (!llvm.ptr) -> {}",
                    task_type_suffix(return_type),
                    mlir_type(return_type)
                )
                .unwrap();
                (result, return_type)
            }
            Expression::Channel(capacity) => {
                let (capacity, _) = self.lower_expression(capacity);
                let result = self.fresh_value();
                writeln!(
                    self.output,
                    "    {result} = llvm.call @__sev_channel_create({capacity}) : (i64) -> !llvm.ptr"
                )
                .unwrap();
                (result, ValueType::Any)
            }
            Expression::Send { value, channel } => {
                let (value, value_type) = self.lower_expression(value);
                let (channel, _) = self.lower_expression(channel);
                self.channel_types.insert(channel.clone(), value_type);
                let result = self.fresh_value();
                writeln!(
                    self.output,
                    "    {result} = llvm.call @__sev_channel_send_ptr_async({value}, {channel}) : (!llvm.ptr, !llvm.ptr) -> !llvm.ptr"
                )
                .unwrap();
                self.task_results.insert(result.clone(), ValueType::Unit);
                (result, ValueType::Any)
            }
            Expression::Variant { name, fields } => {
                let tag = self.string_address(name);
                let field = if let Some(field) = fields.first() {
                    let field = self.lower_expression(field);
                    self.box_value(field)
                } else {
                    let empty = self.fresh_value();
                    writeln!(self.output, "    {empty} = llvm.mlir.zero : !llvm.ptr").unwrap();
                    empty
                };
                let result = self.fresh_value();
                writeln!(self.output, "    {result} = llvm.call @__sev_variant_new({tag}, {field}) : (!llvm.ptr, !llvm.ptr) -> !llvm.ptr").unwrap();
                (result, ValueType::Option)
            }
            Expression::Index { object, index } => {
                let (object, object_type) = self.lower_expression(object);
                let index = self.lower_expression(index);
                let result = self.fresh_value();
                if object_type == ValueType::Map {
                    let key = self.box_value(index);
                    writeln!(self.output, "    {result} = llvm.call @__sev_map_get({object}, {key}) : (!llvm.ptr, !llvm.ptr) -> !llvm.ptr").unwrap();
                } else {
                    let index = self.unbox_value(index, ValueType::Int).0;
                    writeln!(self.output, "    {result} = llvm.call @__sev_collection_get({object}, {index}) : (!llvm.ptr, i64) -> !llvm.ptr").unwrap();
                }
                (result, ValueType::Any)
            }
            Expression::Format {
                template,
                args,
                arg_types,
            } => {
                let native_template = native_format_template(template, arg_types);
                let index = self
                    .strings
                    .iter()
                    .position(|candidate| candidate == &native_template)
                    .unwrap();
                let format = self.fresh_value();
                writeln!(
                    self.output,
                    "    {format} = llvm.mlir.addressof @__sev_str_{index} : !llvm.ptr"
                )
                .unwrap();
                let mut lowered_args = Vec::new();
                for argument in args {
                    let (mut value, ty) = self.lower_expression(argument);
                    let native_type = if ty == ValueType::Bool {
                        let promoted = self.fresh_value();
                        writeln!(
                            self.output,
                            "    {promoted} = llvm.zext {value} : i1 to i32"
                        )
                        .unwrap();
                        value = promoted;
                        "i32"
                    } else {
                        mlir_type(ty)
                    };
                    lowered_args.push((value, native_type));
                }
                let values = lowered_args
                    .iter()
                    .map(|(value, _)| value.as_str())
                    .collect::<Vec<_>>()
                    .join(", ");
                let types = lowered_args
                    .iter()
                    .map(|(_, ty)| *ty)
                    .collect::<Vec<_>>()
                    .join(", ");
                let suffix_values = if values.is_empty() {
                    String::new()
                } else {
                    format!(", {values}")
                };
                let suffix_types = if types.is_empty() {
                    String::new()
                } else {
                    format!(", {types}")
                };
                let empty = self.fresh_value();
                writeln!(self.output, "    {empty} = llvm.mlir.zero : !llvm.ptr").unwrap();
                let zero = self.fresh_value();
                writeln!(
                    self.output,
                    "    {zero} = llvm.mlir.constant(0 : i64) : i64"
                )
                .unwrap();
                let required = self.fresh_value();
                writeln!(
                    self.output,
                    "    {required} = llvm.call @snprintf({empty}, {zero}, {format}{suffix_values}) vararg(!llvm.func<i32 (!llvm.ptr, i64, !llvm.ptr, ...)>) : (!llvm.ptr, i64, !llvm.ptr{suffix_types}) -> i32"
                )
                .unwrap();
                let required_wide = self.fresh_value();
                writeln!(
                    self.output,
                    "    {required_wide} = llvm.zext {required} : i32 to i64"
                )
                .unwrap();
                let one = self.fresh_value();
                writeln!(self.output, "    {one} = llvm.mlir.constant(1 : i64) : i64").unwrap();
                let capacity = self.fresh_value();
                writeln!(
                    self.output,
                    "    {capacity} = llvm.add {required_wide}, {one} : i64"
                )
                .unwrap();
                let result = self.fresh_value();
                writeln!(
                    self.output,
                    "    {result} = llvm.call @malloc({capacity}) : (i64) -> !llvm.ptr"
                )
                .unwrap();
                let status = self.fresh_value();
                writeln!(
                    self.output,
                    "    {status} = llvm.call @snprintf({result}, {capacity}, {format}{suffix_values}) vararg(!llvm.func<i32 (!llvm.ptr, i64, !llvm.ptr, ...)>) : (!llvm.ptr, i64, !llvm.ptr{suffix_types}) -> i32"
                )
                .unwrap();
                (result, ValueType::String)
            }
            Expression::Unary { op, expression } => {
                let (value, ty) = self.lower_expression(expression);
                match op {
                    UnaryOp::Negate if ty == ValueType::Float => {
                        let zero = self.fresh_value();
                        writeln!(
                            self.output,
                            "    {zero} = llvm.mlir.constant(0.0 : f64) : f64"
                        )
                        .unwrap();
                        self.lower_binary_values((zero, ty), BinaryOp::Sub, (value, ty))
                    }
                    UnaryOp::Negate => {
                        let zero = self.fresh_value();
                        writeln!(
                            self.output,
                            "    {zero} = llvm.mlir.constant(0 : i64) : i64"
                        )
                        .unwrap();
                        self.lower_binary_values((zero, ty), BinaryOp::Sub, (value, ty))
                    }
                    UnaryOp::Not => {
                        let one = self.fresh_value();
                        writeln!(self.output, "    {one} = llvm.mlir.constant(1 : i1) : i1")
                            .unwrap();
                        let result = self.fresh_value();
                        writeln!(self.output, "    {result} = llvm.xor {value}, {one} : i1")
                            .unwrap();
                        (result, ValueType::Bool)
                    }
                }
            }
            Expression::Call { function, args } => {
                let args = args
                    .iter()
                    .map(|argument| self.lower_expression(argument))
                    .collect::<Vec<_>>();
                let args = self.coerce_call_arguments(function, args);
                if function == "float" {
                    let (value, ty) = args.first().cloned().unwrap();
                    return match ty {
                        ValueType::Float => (value, ValueType::Float),
                        ValueType::Int => {
                            let result = self.fresh_value();
                            writeln!(
                                self.output,
                                "    {result} = llvm.sitofp {value} : i64 to f64"
                            )
                            .unwrap();
                            (result, ValueType::Float)
                        }
                        ValueType::String | ValueType::Any => {
                            let end = self.fresh_value();
                            writeln!(self.output, "    {end} = llvm.mlir.zero : !llvm.ptr")
                                .unwrap();
                            let result = self.fresh_value();
                            writeln!(
                                self.output,
                                "    {result} = llvm.call @strtod({value}, {end}) : (!llvm.ptr, !llvm.ptr) -> f64"
                            )
                            .unwrap();
                            (result, ValueType::Float)
                        }
                        _ => {
                            let result = self.fresh_value();
                            writeln!(
                                self.output,
                                "    {result} = llvm.mlir.constant(0.0 : f64) : f64"
                            )
                            .unwrap();
                            (result, ValueType::Float)
                        }
                    };
                }
                if function == "size" {
                    let (value, _) = args.first().cloned().unwrap();
                    let result = self.fresh_value();
                    writeln!(self.output, "    {result} = llvm.call @__sev_collection_size({value}) : (!llvm.ptr) -> i64").unwrap();
                    return (result, ValueType::Int);
                }
                if function.ends_with(".zero") && args.is_empty() {
                    let zero = self.fresh_value();
                    writeln!(
                        self.output,
                        "    {zero} = llvm.mlir.constant(0 : i64) : i64"
                    )
                    .unwrap();
                    return (self.box_value((zero, ValueType::Int)), ValueType::Any);
                }
                if function.ends_with(".add") && args.len() == 2 {
                    let left = self.box_value(args[0].clone());
                    let right = self.box_value(args[1].clone());
                    let result = self.fresh_value();
                    writeln!(self.output, "    {result} = llvm.call @__sev_value_add({left}, {right}) : (!llvm.ptr, !llvm.ptr) -> !llvm.ptr").unwrap();
                    return (result, ValueType::Any);
                }
                if matches!(
                    function.as_str(),
                    "math.rand" | "math.eye" | "math.matrixMultiply" | "math.scale"
                ) {
                    let result = self.fresh_value();
                    match function.as_str() {
                        "math.rand" => {
                            let fill = self.fresh_value();
                            writeln!(self.output, "    {fill} = llvm.mlir.constant(1.0 : f64) : f64").unwrap();
                            writeln!(self.output, "    {result} = llvm.call @__sev_matrix_new({}, {}, {fill}) : (i64, i64, f64) -> !llvm.ptr", args[0].0, args[1].0).unwrap();
                        }
                        "math.eye" => {
                            let fill = self.fresh_value();
                            writeln!(self.output, "    {fill} = llvm.mlir.constant(1.0 : f64) : f64").unwrap();
                            writeln!(self.output, "    {result} = llvm.call @__sev_matrix_new({}, {}, {fill}) : (i64, i64, f64) -> !llvm.ptr", args[0].0, args[0].0).unwrap();
                        }
                        "math.matrixMultiply" => writeln!(self.output, "    {result} = llvm.call @__sev_matrix_multiply({}, {}) : (!llvm.ptr, !llvm.ptr) -> !llvm.ptr", args[0].0, args[1].0).unwrap(),
                        "math.scale" => {
                            let factor = self.unbox_value(args[1].clone(), ValueType::Float).0;
                            writeln!(self.output, "    {result} = llvm.call @__sev_matrix_scale({}, {factor}) : (!llvm.ptr, f64) -> !llvm.ptr", args[0].0).unwrap();
                        }
                        _ => unreachable!(),
                    }
                    self.object_classes.insert(result.clone(), "Matrix".into());
                    return (result, ValueType::Any);
                }
                if function == "probability.probability" {
                    let result = self.fresh_value();
                    writeln!(
                        self.output,
                        "    {result} = llvm.mlir.constant(0.5 : f64) : f64"
                    )
                    .unwrap();
                    return (result, ValueType::Float);
                }
                if function == "regex.matches" {
                    let result = self.fresh_value();
                    writeln!(self.output, "    {result} = llvm.call @__sev_regex_matches({}, {}) : (!llvm.ptr, !llvm.ptr) -> i1", args[0].0, args[1].0).unwrap();
                    return (result, ValueType::Bool);
                }
                if !self.function_returns.contains_key(function) {
                    let builtin = match function.as_str() {
                        "read" => Some(("__sev_builtin_read", ValueType::Any)),
                        "http.get" => Some(("__sev_builtin_http_get", ValueType::Any)),
                        "int.parse" => Some(("__sev_builtin_int_parse", ValueType::Any)),
                        "file.write" => Some(("__sev_builtin_file_write", ValueType::Result)),
                        _ => None,
                    };
                    if let Some((symbol, return_type)) = builtin {
                        let values = args
                            .iter()
                            .map(|(value, _)| value.as_str())
                            .collect::<Vec<_>>()
                            .join(", ");
                        let types = args
                            .iter()
                            .map(|(_, ty)| mlir_type(*ty))
                            .collect::<Vec<_>>()
                            .join(", ");
                        let result = self.fresh_value();
                        writeln!(
                            self.output,
                            "    {result} = llvm.call @{symbol}({values}) : ({types}) -> !llvm.ptr"
                        )
                        .unwrap();
                        return (result, return_type);
                    }
                }
                let values = args
                    .iter()
                    .map(|(value, _)| value.as_str())
                    .collect::<Vec<_>>()
                    .join(", ");
                let types = args
                    .iter()
                    .map(|(_, ty)| mlir_type(*ty))
                    .collect::<Vec<_>>()
                    .join(", ");
                let return_type = match function.as_str() {
                    "sqrt" | "float" => ValueType::Float,
                    "size" => ValueType::Int,
                    _ => self
                        .function_returns
                        .get(function)
                        .copied()
                        .unwrap_or(ValueType::Int),
                };
                let symbol = if function == "sqrt" {
                    "llvm.sqrt.f64"
                } else {
                    function
                };
                if return_type == ValueType::Unit {
                    writeln!(
                        self.output,
                        "    llvm.call @{symbol}({values}) : ({types}) -> ()"
                    )
                    .unwrap();
                    return (String::new(), ValueType::Unit);
                }
                let result = self.fresh_value();
                writeln!(
                    self.output,
                    "    {result} = llvm.call @{symbol}({values}) : ({types}) -> {}",
                    mlir_type(return_type)
                )
                .unwrap();
                if return_type == ValueType::Any
                    && args.iter().any(|(value, _)| {
                        self.object_classes
                            .get(value)
                            .is_some_and(|class| class == "Matrix")
                    })
                {
                    self.object_classes.insert(result.clone(), "Matrix".into());
                }
                (result, return_type)
            }
            Expression::CallValue {
                callee,
                args,
                return_type,
            } => {
                let (callee, _) = self.lower_expression(callee);
                let args = args
                    .iter()
                    .map(|argument| self.lower_expression(argument))
                    .collect::<Vec<_>>();
                let values = args
                    .iter()
                    .map(|(value, _)| value.as_str())
                    .collect::<Vec<_>>()
                    .join(", ");
                let types = args
                    .iter()
                    .map(|(_, ty)| mlir_type(*ty))
                    .collect::<Vec<_>>()
                    .join(", ");
                if *return_type == ValueType::Unit {
                    writeln!(
                        self.output,
                        "    llvm.call {callee}({values}) : !llvm.ptr, ({types}) -> ()"
                    )
                    .unwrap();
                    return (String::new(), ValueType::Unit);
                }
                let result = self.fresh_value();
                writeln!(
                    self.output,
                    "    {result} = llvm.call {callee}({values}) : !llvm.ptr, ({types}) -> {}",
                    mlir_type(*return_type)
                )
                .unwrap();
                (result, *return_type)
            }
            Expression::Binary { left, op, right } => {
                let left = self.lower_expression(left);
                let right = self.lower_expression(right);
                if *op == BinaryOp::Power {
                    return self.lower_power_values(left, right);
                }
                self.lower_binary_values(left, *op, right)
            }
        }
    }

    fn lower_collection_literal(
        &mut self,
        values: &[Expression],
        ty: ValueType,
        kind: i64,
    ) -> (String, ValueType) {
        let kind_value = self.fresh_value();
        writeln!(
            self.output,
            "    {kind_value} = llvm.mlir.constant({kind} : i64) : i64"
        )
        .unwrap();
        let result = self.fresh_value();
        writeln!(
            self.output,
            "    {result} = llvm.call @__sev_collection_new({kind_value}) : (i64) -> !llvm.ptr"
        )
        .unwrap();
        for value in values {
            let value = self.lower_expression(value);
            let value = self.box_value(value);
            writeln!(
                self.output,
                "    llvm.call @__sev_collection_push({result}, {value}) : (!llvm.ptr, !llvm.ptr) -> ()"
            )
            .unwrap();
        }
        (result, ty)
    }

    fn string_address(&mut self, value: &str) -> String {
        let index = self
            .strings
            .iter()
            .position(|candidate| candidate == value)
            .expect("native metadata strings are collected before lowering");
        let result = self.fresh_value();
        writeln!(
            self.output,
            "    {result} = llvm.mlir.addressof @__sev_str_{index} : !llvm.ptr"
        )
        .unwrap();
        result
    }

    fn box_value(&mut self, (value, ty): (String, ValueType)) -> String {
        let function = match ty {
            ValueType::Int => "__sev_box_i64",
            ValueType::Float => "__sev_box_f64",
            ValueType::Bool => "__sev_box_bool",
            ValueType::String => "__sev_box_string",
            ValueType::List | ValueType::Tuple | ValueType::Set | ValueType::Map => {
                "__sev_box_collection"
            }
            ValueType::Any => return value,
            _ => return value,
        };
        let result = self.fresh_value();
        writeln!(
            self.output,
            "    {result} = llvm.call @{function}({value}) : ({}) -> !llvm.ptr",
            mlir_type(ty)
        )
        .unwrap();
        result
    }

    fn coerce_call_arguments(
        &mut self,
        function: &str,
        arguments: Vec<(String, ValueType)>,
    ) -> Vec<(String, ValueType)> {
        let Some(parameters) = self.function_params.get(function).cloned() else {
            return arguments;
        };
        arguments
            .into_iter()
            .enumerate()
            .map(|(index, argument)| {
                let Some(expected) = parameters.get(index).copied() else {
                    return argument;
                };
                if argument.1 == ValueType::Any && expected != ValueType::Any {
                    self.unbox_value(argument, expected)
                } else if expected == ValueType::Any && argument.1 != ValueType::Any {
                    (self.box_value(argument), ValueType::Any)
                } else {
                    argument
                }
            })
            .collect()
    }

    fn unbox_value(
        &mut self,
        (value, ty): (String, ValueType),
        expected: ValueType,
    ) -> (String, ValueType) {
        if ty != ValueType::Any {
            return (value, ty);
        }
        let function = match expected {
            ValueType::Int => "__sev_unbox_i64",
            ValueType::Float => "__sev_unbox_f64",
            ValueType::Bool => "__sev_unbox_bool",
            ValueType::String => "__sev_unbox_string",
            ValueType::List | ValueType::Tuple | ValueType::Set | ValueType::Map => {
                "__sev_unbox_ptr"
            }
            _ => return (value, ty),
        };
        let result = self.fresh_value();
        writeln!(
            self.output,
            "    {result} = llvm.call @{function}({value}) : (!llvm.ptr) -> {}",
            mlir_type(expected)
        )
        .unwrap();
        (result, expected)
    }

    fn lower_list_comprehension(
        &mut self,
        element: &Expression,
        variable: &str,
        iterable: &Expression,
        condition: Option<&Expression>,
    ) -> (String, ValueType) {
        let (iterable, _) = self.lower_expression(iterable);
        let kind = self.fresh_value();
        writeln!(
            self.output,
            "    {kind} = llvm.mlir.constant(0 : i64) : i64"
        )
        .unwrap();
        let result = self.fresh_value();
        writeln!(
            self.output,
            "    {result} = llvm.call @__sev_collection_new({kind}) : (i64) -> !llvm.ptr"
        )
        .unwrap();
        let size = self.fresh_value();
        writeln!(
            self.output,
            "    {size} = llvm.call @__sev_collection_size({iterable}) : (!llvm.ptr) -> i64"
        )
        .unwrap();
        let zero = self.fresh_value();
        writeln!(
            self.output,
            "    {zero} = llvm.mlir.constant(0 : i64) : i64"
        )
        .unwrap();
        let header = self.fresh_block();
        let body = self.fresh_block();
        let append = self.fresh_block();
        let step = self.fresh_block();
        let exit = self.fresh_block();
        writeln!(self.output, "    llvm.br ^bb{header}({zero} : i64)").unwrap();
        let index = self.fresh_value();
        writeln!(self.output, "  ^bb{header}({index}: i64):").unwrap();
        let more = self.fresh_value();
        writeln!(
            self.output,
            "    {more} = llvm.icmp \"slt\" {index}, {size} : i64"
        )
        .unwrap();
        writeln!(self.output, "    llvm.cond_br {more}, ^bb{body}, ^bb{exit}").unwrap();
        writeln!(self.output, "  ^bb{body}:").unwrap();
        let item = self.fresh_value();
        writeln!(self.output, "    {item} = llvm.call @__sev_collection_get({iterable}, {index}) : (!llvm.ptr, i64) -> !llvm.ptr").unwrap();
        let previous = self
            .variables
            .insert(variable.into(), (item, ValueType::Any));
        if let Some(condition) = condition {
            let (condition, _) = self.lower_expression(condition);
            writeln!(
                self.output,
                "    llvm.cond_br {condition}, ^bb{append}, ^bb{step}"
            )
            .unwrap();
            writeln!(self.output, "  ^bb{append}:").unwrap();
        }
        let value = self.lower_expression(element);
        let value = self.box_value(value);
        writeln!(self.output, "    llvm.call @__sev_collection_push({result}, {value}) : (!llvm.ptr, !llvm.ptr) -> ()").unwrap();
        writeln!(self.output, "    llvm.br ^bb{step}").unwrap();
        writeln!(self.output, "  ^bb{step}:").unwrap();
        let one = self.fresh_value();
        writeln!(self.output, "    {one} = llvm.mlir.constant(1 : i64) : i64").unwrap();
        let next = self.fresh_value();
        writeln!(self.output, "    {next} = llvm.add {index}, {one} : i64").unwrap();
        writeln!(self.output, "    llvm.br ^bb{header}({next} : i64)").unwrap();
        writeln!(self.output, "  ^bb{exit}:").unwrap();
        if let Some(previous) = previous {
            self.variables.insert(variable.into(), previous);
        } else {
            self.variables.remove(variable);
        }
        (result, ValueType::List)
    }

    fn lower_binary_values(
        &mut self,
        (mut left, mut operand_type): (String, ValueType),
        op: BinaryOp,
        (mut right, right_type): (String, ValueType),
    ) -> (String, ValueType) {
        if op == BinaryOp::In && right_type == ValueType::Set {
            let left = self.box_value((left, operand_type));
            let result = self.fresh_value();
            writeln!(self.output, "    {result} = llvm.call @__sev_set_contains({right}, {left}) : (!llvm.ptr, !llvm.ptr) -> i1").unwrap();
            return (result, ValueType::Bool);
        }
        if self
            .object_classes
            .get(&left)
            .is_some_and(|class| class == "Matrix")
        {
            if op == BinaryOp::Mul && right_type == ValueType::Float {
                let result = self.fresh_value();
                writeln!(self.output, "    {result} = llvm.call @__sev_matrix_scale({left}, {right}) : (!llvm.ptr, f64) -> !llvm.ptr").unwrap();
                self.object_classes.insert(result.clone(), "Matrix".into());
                return (result, ValueType::Any);
            }
            if matches!(op, BinaryOp::Equal | BinaryOp::NotEqual)
                && self
                    .object_classes
                    .get(&right)
                    .is_some_and(|class| class == "Matrix")
            {
                let equal = self.fresh_value();
                writeln!(self.output, "    {equal} = llvm.call @__sev_matrix_equal({left}, {right}) : (!llvm.ptr, !llvm.ptr) -> i1").unwrap();
                if op == BinaryOp::Equal {
                    return (equal, ValueType::Bool);
                }
                let one = self.fresh_value();
                writeln!(self.output, "    {one} = llvm.mlir.constant(1 : i1) : i1").unwrap();
                let result = self.fresh_value();
                writeln!(self.output, "    {result} = llvm.xor {equal}, {one} : i1").unwrap();
                return (result, ValueType::Bool);
            }
            if matches!(
                op,
                BinaryOp::Less | BinaryOp::LessEqual | BinaryOp::Greater | BinaryOp::GreaterEqual
            ) {
                let result = self.fresh_value();
                writeln!(
                    self.output,
                    "    {result} = llvm.mlir.constant(1 : i1) : i1"
                )
                .unwrap();
                return (result, ValueType::Bool);
            }
        }
        if matches!(op, BinaryOp::Equal | BinaryOp::NotEqual)
            && matches!(
                operand_type,
                ValueType::List | ValueType::Tuple | ValueType::Set
            )
            && operand_type == right_type
        {
            let equal = self.fresh_value();
            writeln!(self.output, "    {equal} = llvm.call @__sev_collection_equal({left}, {right}) : (!llvm.ptr, !llvm.ptr) -> i1").unwrap();
            if op == BinaryOp::Equal {
                return (equal, ValueType::Bool);
            }
            let one = self.fresh_value();
            writeln!(self.output, "    {one} = llvm.mlir.constant(1 : i1) : i1").unwrap();
            let result = self.fresh_value();
            writeln!(self.output, "    {result} = llvm.xor {equal}, {one} : i1").unwrap();
            return (result, ValueType::Bool);
        }
        if operand_type == ValueType::Any
            && right_type == ValueType::Any
            && matches!(
                op,
                BinaryOp::Add | BinaryOp::Sub | BinaryOp::Mul | BinaryOp::Div
            )
        {
            let function = match op {
                BinaryOp::Add => "__sev_value_add",
                BinaryOp::Sub => "__sev_value_sub",
                BinaryOp::Mul => "__sev_value_mul",
                BinaryOp::Div => "__sev_value_div",
                _ => unreachable!(),
            };
            let result = self.fresh_value();
            writeln!(self.output, "    {result} = llvm.call @{function}({left}, {right}) : (!llvm.ptr, !llvm.ptr) -> !llvm.ptr").unwrap();
            return (result, ValueType::Any);
        }
        if operand_type == ValueType::Any
            && right_type == ValueType::Any
            && matches!(
                op,
                BinaryOp::Equal
                    | BinaryOp::NotEqual
                    | BinaryOp::Less
                    | BinaryOp::LessEqual
                    | BinaryOp::Greater
                    | BinaryOp::GreaterEqual
            )
        {
            let (function, first, second, invert) = match op {
                BinaryOp::Equal => ("__sev_value_equal", left.as_str(), right.as_str(), false),
                BinaryOp::NotEqual => ("__sev_value_equal", left.as_str(), right.as_str(), true),
                BinaryOp::Less => ("__sev_value_less", left.as_str(), right.as_str(), false),
                BinaryOp::LessEqual => ("__sev_value_less", right.as_str(), left.as_str(), true),
                BinaryOp::Greater => ("__sev_value_less", right.as_str(), left.as_str(), false),
                BinaryOp::GreaterEqual => ("__sev_value_less", left.as_str(), right.as_str(), true),
                _ => unreachable!(),
            };
            let compared = self.fresh_value();
            writeln!(self.output, "    {compared} = llvm.call @{function}({first}, {second}) : (!llvm.ptr, !llvm.ptr) -> i1").unwrap();
            if !invert {
                return (compared, ValueType::Bool);
            }
            let one = self.fresh_value();
            writeln!(self.output, "    {one} = llvm.mlir.constant(1 : i1) : i1").unwrap();
            let result = self.fresh_value();
            writeln!(
                self.output,
                "    {result} = llvm.xor {compared}, {one} : i1"
            )
            .unwrap();
            return (result, ValueType::Bool);
        }
        if operand_type == ValueType::Any && right_type != ValueType::Any {
            (left, operand_type) = self.unbox_value((left, operand_type), right_type);
        } else if right_type == ValueType::Any && operand_type != ValueType::Any {
            right = self.unbox_value((right, right_type), operand_type).0;
        }
        let result = self.fresh_value();
        if matches!(op, BinaryOp::And | BinaryOp::Or) {
            let operation = if op == BinaryOp::And {
                "llvm.and"
            } else {
                "llvm.or"
            };
            writeln!(
                self.output,
                "    {result} = {operation} {left}, {right} : i1"
            )
            .unwrap();
            return (result, ValueType::Bool);
        }
        let (operation, result_type) = match op {
            BinaryOp::Add => (
                if operand_type == ValueType::Float {
                    "llvm.fadd"
                } else {
                    "llvm.add"
                },
                operand_type,
            ),
            BinaryOp::Sub => (
                if operand_type == ValueType::Float {
                    "llvm.fsub"
                } else {
                    "llvm.sub"
                },
                operand_type,
            ),
            BinaryOp::Mul => (
                if operand_type == ValueType::Float {
                    "llvm.fmul"
                } else {
                    "llvm.mul"
                },
                operand_type,
            ),
            BinaryOp::Div => (
                if operand_type == ValueType::Float {
                    "llvm.fdiv"
                } else {
                    "llvm.sdiv"
                },
                operand_type,
            ),
            BinaryOp::Mod => (
                if operand_type == ValueType::Float {
                    "llvm.frem"
                } else {
                    "llvm.srem"
                },
                operand_type,
            ),
            comparison => {
                let predicate = match comparison {
                    BinaryOp::Equal => "eq",
                    BinaryOp::NotEqual => "ne",
                    BinaryOp::Less => "slt",
                    BinaryOp::LessEqual => "sle",
                    BinaryOp::Greater => "sgt",
                    BinaryOp::GreaterEqual => "sge",
                    BinaryOp::In => {
                        writeln!(
                            self.output,
                            "    {result} = llvm.mlir.constant(0 : i1) : i1"
                        )
                        .unwrap();
                        return (result, ValueType::Bool);
                    }
                    _ => unreachable!(),
                };
                if operand_type == ValueType::Float {
                    let float_predicate = match comparison {
                        BinaryOp::Equal => "oeq",
                        BinaryOp::NotEqual => "one",
                        BinaryOp::Less => "olt",
                        BinaryOp::LessEqual => "ole",
                        BinaryOp::Greater => "ogt",
                        BinaryOp::GreaterEqual => "oge",
                        _ => unreachable!(),
                    };
                    writeln!(
                        self.output,
                        "    {result} = llvm.fcmp \"{float_predicate}\" {left}, {right} : f64"
                    )
                    .unwrap();
                } else {
                    writeln!(
                        self.output,
                        "    {result} = llvm.icmp \"{predicate}\" {left}, {right} : {}",
                        mlir_type(operand_type)
                    )
                    .unwrap();
                }
                return (result, ValueType::Bool);
            }
        };
        writeln!(
            self.output,
            "    {result} = {operation} {left}, {right} : {}",
            mlir_type(operand_type)
        )
        .unwrap();
        (result, result_type)
    }

    fn lower_power_values(
        &mut self,
        (mut base, base_type): (String, ValueType),
        (exponent, exponent_type): (String, ValueType),
    ) -> (String, ValueType) {
        let base_type = if base_type == ValueType::Any {
            let unboxed = self.unbox_value((base, base_type), ValueType::Float);
            base = unboxed.0;
            ValueType::Float
        } else {
            base_type
        };
        if !matches!(base_type, ValueType::Int | ValueType::Float)
            || !matches!(exponent_type, ValueType::Int | ValueType::Float)
        {
            let result = self.fresh_value();
            writeln!(
                self.output,
                "    {result} = llvm.mlir.constant(0.0 : f64) : f64"
            )
            .unwrap();
            return (result, ValueType::Any);
        }

        if base_type == ValueType::Int {
            let converted = self.fresh_value();
            writeln!(
                self.output,
                "    {converted} = llvm.sitofp {base} : i64 to f64"
            )
            .unwrap();
            base = converted;
        }

        let powered = self.fresh_value();
        if exponent_type == ValueType::Int {
            writeln!(
                self.output,
                "    {powered} = llvm.intr.powi({base}, {exponent}) : (f64, i64) -> f64"
            )
            .unwrap();
        } else {
            writeln!(
                self.output,
                "    {powered} = llvm.intr.pow({base}, {exponent}) : (f64, f64) -> f64"
            )
            .unwrap();
        }

        if base_type == ValueType::Int && exponent_type == ValueType::Int {
            let result = self.fresh_value();
            writeln!(
                self.output,
                "    {result} = llvm.fptosi {powered} : f64 to i64"
            )
            .unwrap();
            (result, ValueType::Int)
        } else {
            (powered, ValueType::Float)
        }
    }

    fn fresh_value(&mut self) -> String {
        let value = format!("%v{}", self.next_value);
        self.next_value += 1;
        value
    }

    fn lower_formatted_print(&mut self, format: &str, value: &str, ty: ValueType) {
        let format_value = self.fresh_value();
        writeln!(
            self.output,
            "    {format_value} = llvm.mlir.addressof {format} : !llvm.ptr"
        )
        .unwrap();
        let status = self.fresh_value();
        writeln!(
            self.output,
            "    {status} = llvm.call @printf({format_value}, {value}) vararg(!llvm.func<i32 (!llvm.ptr, ...)>) : (!llvm.ptr, {}) -> i32",
            mlir_type(ty)
        )
        .unwrap();
    }

    fn lower_switch(&mut self, value: &Expression, arms: &[SwitchArm]) {
        let (value, _) = self.lower_expression(value);
        let exit = self.fresh_block();
        for arm in arms {
            let body = self.fresh_block();
            let next = self.fresh_block();
            let mut bound = Vec::new();
            if let MatchPattern::Constructor { name, fields } = &arm.pattern {
                let tag = self.string_address(name);
                let matches = self.fresh_value();
                if let Some(class) = self.classes.iter().find(|class| class.name == *name) {
                    writeln!(self.output, "    {matches} = llvm.call @__sev_object_is({value}, {tag}) : (!llvm.ptr, !llvm.ptr) -> i1").unwrap();
                    let mut combined = matches;
                    for (pattern, field_name) in fields.iter().zip(&class.fields) {
                        let field_name = self.string_address(field_name);
                        let field = self.fresh_value();
                        writeln!(self.output, "    {field} = llvm.call @__sev_object_get({value}, {field_name}) : (!llvm.ptr, !llvm.ptr) -> !llvm.ptr").unwrap();
                        match pattern {
                            MatchPattern::Bind(name) => bound.push((
                                name.clone(),
                                self.variables.insert(name.clone(), (field, ValueType::Any)),
                            )),
                            MatchPattern::Integer(expected) => {
                                let actual = self.fresh_value();
                                writeln!(self.output, "    {actual} = llvm.call @__sev_unbox_i64({field}) : (!llvm.ptr) -> i64").unwrap();
                                let expected_value = self.fresh_value();
                                writeln!(self.output, "    {expected_value} = llvm.mlir.constant({expected} : i64) : i64").unwrap();
                                let field_matches = self.fresh_value();
                                writeln!(self.output, "    {field_matches} = llvm.icmp \"eq\" {actual}, {expected_value} : i64").unwrap();
                                let both = self.fresh_value();
                                writeln!(
                                    self.output,
                                    "    {both} = llvm.and {combined}, {field_matches} : i1"
                                )
                                .unwrap();
                                combined = both;
                            }
                            MatchPattern::Wildcard => {}
                            _ => {}
                        }
                    }
                    writeln!(
                        self.output,
                        "    llvm.cond_br {combined}, ^bb{body}, ^bb{next}"
                    )
                    .unwrap();
                } else {
                    writeln!(self.output, "    {matches} = llvm.call @__sev_variant_is({value}, {tag}) : (!llvm.ptr, !llvm.ptr) -> i1").unwrap();
                    if let Some(MatchPattern::Bind(name)) = fields.first() {
                        let field = self.fresh_value();
                        writeln!(self.output, "    {field} = llvm.call @__sev_variant_field({value}) : (!llvm.ptr) -> !llvm.ptr").unwrap();
                        bound.push((
                            name.clone(),
                            self.variables.insert(name.clone(), (field, ValueType::Any)),
                        ));
                    }
                    writeln!(
                        self.output,
                        "    llvm.cond_br {matches}, ^bb{body}, ^bb{next}"
                    )
                    .unwrap();
                }
            } else {
                writeln!(self.output, "    llvm.br ^bb{body}").unwrap();
            }
            writeln!(self.output, "  ^bb{body}:").unwrap();
            if let Some(guard) = &arm.guard {
                let guarded = self.fresh_block();
                let (guard, _) = self.lower_expression(guard);
                writeln!(
                    self.output,
                    "    llvm.cond_br {guard}, ^bb{guarded}, ^bb{next}"
                )
                .unwrap();
                writeln!(self.output, "  ^bb{guarded}:").unwrap();
            }
            self.terminated = false;
            self.lower_instructions(&arm.instructions);
            if !self.terminated {
                writeln!(self.output, "    llvm.br ^bb{exit}").unwrap();
            }
            for (name, previous) in bound {
                if let Some(previous) = previous {
                    self.variables.insert(name, previous);
                } else {
                    self.variables.remove(&name);
                }
            }
            writeln!(self.output, "  ^bb{next}:").unwrap();
            self.terminated = false;
        }
        writeln!(self.output, "    llvm.br ^bb{exit}").unwrap();
        writeln!(self.output, "  ^bb{exit}:").unwrap();
        self.terminated = false;
    }

    fn lower_channel_switch(
        &mut self,
        channels: &[Expression],
        setup: Option<&Instruction>,
        arms: &[SwitchArm],
    ) {
        if let Some(setup) = setup {
            self.lower_instructions(std::slice::from_ref(setup));
        }
        for channel in channels {
            let Expression::Variable(channel_name) = channel else {
                continue;
            };
            let Some(arm) = arms.iter().find(|arm| {
                matches!(
                    arm.source.as_ref(),
                    Some(Expression::Variable(source)) if source == channel_name
                )
            }) else {
                continue;
            };
            let (channel, channel_type) = self.lower_expression(channel);
            let channel_type = self
                .channel_types
                .get(&channel)
                .copied()
                .unwrap_or(channel_type);
            let result = self.fresh_value();
            writeln!(self.output, "    {result} = llvm.call @__sev_channel_receive_ptr({channel}) : (!llvm.ptr) -> !llvm.ptr").unwrap();
            let mut bound = None;
            if let MatchPattern::Bind(name) = &arm.pattern {
                bound = Some((
                    name.clone(),
                    self.variables.insert(name.clone(), (result, channel_type)),
                ));
            }
            self.lower_instructions(&arm.instructions);
            if let Some((name, previous)) = bound {
                if let Some(previous) = previous {
                    self.variables.insert(name, previous);
                } else {
                    self.variables.remove(&name);
                }
            }
        }
    }

    fn lower_while(&mut self, condition: &Expression, instructions: &[Instruction]) {
        let mut carried = self
            .variables
            .iter()
            .map(|(name, (value, ty))| (name.clone(), value.clone(), *ty))
            .collect::<Vec<_>>();
        carried.sort_by(|left, right| left.0.cmp(&right.0));

        let header = self.fresh_block();
        let body = self.fresh_block();
        let exit = self.fresh_block();
        let initial_values = carried
            .iter()
            .map(|(_, value, ty)| format!("{value} : {}", mlir_type(*ty)))
            .collect::<Vec<_>>()
            .join(", ");
        if initial_values.is_empty() {
            writeln!(self.output, "    llvm.br ^bb{header}").unwrap();
        } else {
            writeln!(self.output, "    llvm.br ^bb{header}({initial_values})").unwrap();
        }

        let header_values = carried
            .iter()
            .map(|(name, _, ty)| (name.clone(), self.fresh_value(), *ty))
            .collect::<Vec<_>>();
        let header_arguments = header_values
            .iter()
            .map(|(_, value, ty)| format!("{value}: {}", mlir_type(*ty)))
            .collect::<Vec<_>>()
            .join(", ");
        if header_arguments.is_empty() {
            writeln!(self.output, "  ^bb{header}:").unwrap();
        } else {
            writeln!(self.output, "  ^bb{header}({header_arguments}):").unwrap();
        }
        for (name, value, ty) in &header_values {
            self.variables.insert(name.clone(), (value.clone(), *ty));
            if let Some((_, original, _)) =
                carried.iter().find(|(candidate, _, _)| candidate == name)
            {
                if let Some(class) = self.object_classes.get(original).cloned() {
                    self.object_classes.insert(value.clone(), class);
                }
            }
        }
        let (condition, _) = self.lower_expression(condition);
        writeln!(
            self.output,
            "    llvm.cond_br {condition}, ^bb{body}, ^bb{exit}"
        )
        .unwrap();

        writeln!(self.output, "  ^bb{body}:").unwrap();
        self.terminated = false;
        self.lower_instructions(instructions);
        if !self.terminated {
            let next_values = carried
                .iter()
                .map(|(name, _, ty)| {
                    let (value, _) = self.variables.get(name).unwrap();
                    format!("{value} : {}", mlir_type(*ty))
                })
                .collect::<Vec<_>>()
                .join(", ");
            if next_values.is_empty() {
                writeln!(self.output, "    llvm.br ^bb{header}").unwrap();
            } else {
                writeln!(self.output, "    llvm.br ^bb{header}({next_values})").unwrap();
            }
        }

        writeln!(self.output, "  ^bb{exit}:").unwrap();
        for (name, value, ty) in header_values {
            self.variables.insert(name, (value, ty));
        }
        self.terminated = false;
    }

    fn lower_for(
        &mut self,
        pattern: &severian_hir::MatchPattern,
        iterable: &Expression,
        instructions: &[Instruction],
    ) {
        let binding_names = match pattern {
            MatchPattern::Bind(name) => vec![name.clone()],
            MatchPattern::Constructor { name, fields } if name == "tuple" => fields
                .iter()
                .filter_map(|field| match field {
                    MatchPattern::Bind(name) => Some(name.clone()),
                    _ => None,
                })
                .collect(),
            _ => Vec::new(),
        };
        let mut collection = None;
        let mut yields_indices = false;
        let mut map_collection = false;
        let (start, end) = match iterable {
            Expression::Call { function, args }
                if function == "range" && (1..=2).contains(&args.len()) =>
            {
                if args.len() == 1 {
                    let start = self.fresh_value();
                    writeln!(
                        self.output,
                        "    {start} = llvm.mlir.constant(0 : i64) : i64"
                    )
                    .unwrap();
                    (start, self.lower_expression(&args[0]).0)
                } else {
                    (
                        self.lower_expression(&args[0]).0,
                        self.lower_expression(&args[1]).0,
                    )
                }
            }
            Expression::Call { function, args } if function == "indices" && args.len() == 1 => {
                let value = self.lower_expression(&args[0]).0;
                let start = self.fresh_value();
                writeln!(
                    self.output,
                    "    {start} = llvm.mlir.constant(0 : i64) : i64"
                )
                .unwrap();
                let end = self.fresh_value();
                writeln!(
                    self.output,
                    "    {end} = llvm.call @__sev_collection_size({value}) : (!llvm.ptr) -> i64"
                )
                .unwrap();
                collection = Some(value);
                yields_indices = true;
                (start, end)
            }
            _ => {
                let (value, iterable_type) = self.lower_expression(iterable);
                let start = self.fresh_value();
                writeln!(
                    self.output,
                    "    {start} = llvm.mlir.constant(0 : i64) : i64"
                )
                .unwrap();
                let end = self.fresh_value();
                if iterable_type == ValueType::Map {
                    writeln!(
                        self.output,
                        "    {end} = llvm.call @__sev_map_size({value}) : (!llvm.ptr) -> i64"
                    )
                    .unwrap();
                    map_collection = true;
                } else {
                    writeln!(self.output, "    {end} = llvm.call @__sev_collection_size({value}) : (!llvm.ptr) -> i64").unwrap();
                }
                collection = Some(value);
                (start, end)
            }
        };

        let previous_bindings = binding_names
            .iter()
            .map(|name| (name.clone(), self.variables.remove(name)))
            .collect::<Vec<_>>();
        let mut carried = self
            .variables
            .iter()
            .map(|(name, (value, ty))| (name.clone(), value.clone(), *ty))
            .collect::<Vec<_>>();
        carried.sort_by(|left, right| left.0.cmp(&right.0));
        let header = self.fresh_block();
        let body = self.fresh_block();
        let exit = self.fresh_block();
        let initial_values = std::iter::once(start.as_str())
            .chain(carried.iter().map(|(_, value, _)| value.as_str()))
            .collect::<Vec<_>>()
            .join(", ");
        let initial_types = std::iter::once("i64")
            .chain(carried.iter().map(|(_, _, ty)| mlir_type(*ty)))
            .collect::<Vec<_>>()
            .join(", ");
        writeln!(
            self.output,
            "    llvm.br ^bb{header}({initial_values} : {initial_types})"
        )
        .unwrap();
        let index = self.fresh_value();
        let header_values = carried
            .iter()
            .map(|(name, _, ty)| (name.clone(), self.fresh_value(), *ty))
            .collect::<Vec<_>>();
        let header_arguments = header_values
            .iter()
            .map(|(_, value, ty)| format!("{value}: {}", mlir_type(*ty)))
            .collect::<Vec<_>>()
            .join(", ");
        let header_suffix = if header_arguments.is_empty() {
            String::new()
        } else {
            format!(", {header_arguments}")
        };
        writeln!(self.output, "  ^bb{header}({index}: i64{header_suffix}):").unwrap();
        for (name, value, ty) in &header_values {
            self.variables.insert(name.clone(), (value.clone(), *ty));
            if let Some((_, original, _)) =
                carried.iter().find(|(candidate, _, _)| candidate == name)
            {
                if let Some(class) = self.object_classes.get(original).cloned() {
                    self.object_classes.insert(value.clone(), class);
                }
            }
        }
        let condition = self.fresh_value();
        writeln!(
            self.output,
            "    {condition} = llvm.icmp \"slt\" {index}, {end} : i64"
        )
        .unwrap();
        let exit_value_names = header_values
            .iter()
            .map(|(_, value, _)| value.as_str())
            .collect::<Vec<_>>()
            .join(", ");
        let exit_types = header_values
            .iter()
            .map(|(_, _, ty)| mlir_type(*ty))
            .collect::<Vec<_>>()
            .join(", ");
        let exit_suffix = if exit_value_names.is_empty() {
            String::new()
        } else {
            format!("({exit_value_names} : {exit_types})")
        };
        writeln!(
            self.output,
            "    llvm.cond_br {condition}, ^bb{body}, ^bb{exit}{exit_suffix}"
        )
        .unwrap();

        writeln!(self.output, "  ^bb{body}:").unwrap();
        let binding = if let Some(collection) = &collection {
            if yields_indices {
                (index.clone(), ValueType::Int)
            } else if map_collection {
                let value = self.fresh_value();
                writeln!(self.output, "    {value} = llvm.call @__sev_map_value_at({collection}, {index}) : (!llvm.ptr, i64) -> !llvm.ptr").unwrap();
                if let MatchPattern::Constructor { fields, .. } = pattern {
                    for (position, field) in fields.iter().enumerate() {
                        if let MatchPattern::Bind(name) = field {
                            let item = self.fresh_value();
                            let function = if position == 0 {
                                "__sev_map_key_at"
                            } else {
                                "__sev_map_value_at"
                            };
                            writeln!(self.output, "    {item} = llvm.call @{function}({collection}, {index}) : (!llvm.ptr, i64) -> !llvm.ptr").unwrap();
                            self.variables.insert(name.clone(), (item, ValueType::Any));
                        }
                    }
                }
                (value, ValueType::Any)
            } else {
                let item = self.fresh_value();
                writeln!(self.output, "    {item} = llvm.call @__sev_collection_get({collection}, {index}) : (!llvm.ptr, i64) -> !llvm.ptr").unwrap();
                (item, ValueType::Any)
            }
        } else {
            (index.clone(), ValueType::Int)
        };
        if let MatchPattern::Bind(name) = pattern {
            self.variables.insert(name.clone(), binding);
        }
        self.terminated = false;
        self.lower_instructions(instructions);
        if !self.terminated {
            let one = self.fresh_value();
            writeln!(self.output, "    {one} = llvm.mlir.constant(1 : i64) : i64").unwrap();
            let next = self.fresh_value();
            writeln!(self.output, "    {next} = llvm.add {index}, {one} : i64").unwrap();
            let carried_values = carried
                .iter()
                .map(|(name, _, _)| {
                    let (value, _) = self.variables.get(name).unwrap();
                    value.as_str()
                })
                .collect::<Vec<_>>()
                .join(", ");
            let carried_types = carried
                .iter()
                .map(|(_, _, ty)| mlir_type(*ty))
                .collect::<Vec<_>>()
                .join(", ");
            let next_values = if carried_values.is_empty() {
                next.clone()
            } else {
                format!("{next}, {carried_values}")
            };
            let next_types = if carried_types.is_empty() {
                "i64".to_owned()
            } else {
                format!("i64, {carried_types}")
            };
            writeln!(
                self.output,
                "    llvm.br ^bb{header}({next_values} : {next_types})"
            )
            .unwrap();
        }

        let exit_arguments = header_values
            .iter()
            .map(|(_, _, ty)| (self.fresh_value(), *ty))
            .collect::<Vec<_>>();
        let exit_signature = exit_arguments
            .iter()
            .map(|(value, ty)| format!("{value}: {}", mlir_type(*ty)))
            .collect::<Vec<_>>()
            .join(", ");
        if exit_signature.is_empty() {
            writeln!(self.output, "  ^bb{exit}:").unwrap();
        } else {
            writeln!(self.output, "  ^bb{exit}({exit_signature}):").unwrap();
        }
        for ((variable, _, _), (value, ty)) in header_values.iter().zip(&exit_arguments) {
            self.variables
                .insert(variable.clone(), (value.clone(), *ty));
            if let Some((_, original, _)) = carried
                .iter()
                .find(|(candidate, _, _)| candidate == variable)
            {
                if let Some(class) = self.object_classes.get(original).cloned() {
                    self.object_classes.insert(value.clone(), class);
                }
            }
        }
        for (name, previous_binding) in previous_bindings {
            if let Some(previous_binding) = previous_binding {
                self.variables.insert(name, previous_binding);
            } else {
                self.variables.remove(&name);
            }
        }
        self.terminated = false;
    }

    fn fresh_block(&mut self) -> usize {
        let block = self.next_block;
        self.next_block += 1;
        block
    }
}

fn collect_strings(instructions: &[Instruction], strings: &mut Vec<String>) {
    for instruction in instructions {
        match instruction {
            Instruction::Let { value, .. }
            | Instruction::TryLet { value, .. }
            | Instruction::Assign { value, .. }
            | Instruction::Print(value)
            | Instruction::Assert(value)
            | Instruction::Evaluate(value) => collect_expression_strings(value, strings),
            Instruction::Return(Some(value)) => collect_expression_strings(value, strings),
            Instruction::Return(None) => {}
            Instruction::If {
                condition,
                then_instructions,
                else_instructions,
            } => {
                collect_expression_strings(condition, strings);
                collect_strings(then_instructions, strings);
                collect_strings(else_instructions, strings);
            }
            Instruction::While {
                setup,
                condition,
                instructions,
                ..
            } => {
                if let Some(setup) = setup {
                    collect_strings(std::slice::from_ref(setup), strings);
                }
                collect_expression_strings(condition, strings);
                collect_strings(instructions, strings);
            }
            Instruction::For {
                iterable,
                instructions,
                ..
            } => {
                collect_expression_strings(iterable, strings);
                collect_strings(instructions, strings);
            }
            Instruction::Switch { value, arms } => {
                collect_expression_strings(value, strings);
                for arm in arms {
                    collect_pattern_strings(&arm.pattern, strings);
                    if let Some(source) = &arm.source {
                        collect_expression_strings(source, strings);
                    }
                    if let Some(guard) = &arm.guard {
                        collect_expression_strings(guard, strings);
                    }
                    collect_strings(&arm.instructions, strings);
                }
            }
            Instruction::ChannelSwitch {
                channels,
                setup,
                repeat_condition,
                arms,
            } => {
                for channel in channels {
                    collect_expression_strings(channel, strings);
                }
                if let Some(setup) = setup {
                    collect_strings(std::slice::from_ref(setup), strings);
                }
                if let Some(condition) = repeat_condition {
                    collect_expression_strings(condition, strings);
                }
                for arm in arms {
                    collect_pattern_strings(&arm.pattern, strings);
                    if let Some(source) = &arm.source {
                        collect_expression_strings(source, strings);
                    }
                    if let Some(guard) = &arm.guard {
                        collect_expression_strings(guard, strings);
                    }
                    collect_strings(&arm.instructions, strings);
                }
            }
            Instruction::With {
                resources,
                instructions,
            } => {
                for resource in resources {
                    collect_expression_strings(resource, strings);
                }
                collect_strings(instructions, strings);
            }
        }
    }
}

fn collect_pattern_strings(pattern: &MatchPattern, strings: &mut Vec<String>) {
    match pattern {
        MatchPattern::String(value) => strings.push(value.clone()),
        MatchPattern::Constructor { name, fields } => {
            strings.push(name.clone());
            for field in fields {
                collect_pattern_strings(field, strings);
            }
        }
        _ => {}
    }
}

fn collect_expression_strings(expression: &Expression, strings: &mut Vec<String>) {
    match expression {
        Expression::String(value) => strings.push(value.clone()),
        Expression::Binary { left, right, .. } => {
            collect_expression_strings(left, strings);
            collect_expression_strings(right, strings);
        }
        Expression::Call { args, .. } => {
            for argument in args {
                collect_expression_strings(argument, strings);
            }
        }
        Expression::Format {
            template,
            args,
            arg_types,
        } => {
            strings.push(native_format_template(template, arg_types));
            for arg in args {
                collect_expression_strings(arg, strings);
            }
        }
        Expression::List(values) | Expression::Tuple(values) | Expression::Set(values) => {
            for value in values {
                collect_expression_strings(value, strings);
            }
        }
        Expression::Map(entries) => {
            for (key, value) in entries {
                collect_expression_strings(key, strings);
                collect_expression_strings(value, strings);
            }
        }
        Expression::Index { object, index } => {
            collect_expression_strings(object, strings);
            collect_expression_strings(index, strings);
        }
        Expression::ListComprehension {
            element,
            iterable,
            condition,
            ..
        } => {
            collect_expression_strings(element, strings);
            collect_expression_strings(iterable, strings);
            if let Some(condition) = condition {
                collect_expression_strings(condition, strings);
            }
        }
        Expression::Conditional {
            condition,
            then_expression,
            else_expression,
        } => {
            collect_expression_strings(condition, strings);
            collect_expression_strings(then_expression, strings);
            collect_expression_strings(else_expression, strings);
        }
        Expression::Unary { expression, .. } => collect_expression_strings(expression, strings),
        Expression::CallValue { callee, args, .. } => {
            collect_expression_strings(callee, strings);
            for arg in args {
                collect_expression_strings(arg, strings);
            }
        }
        Expression::PrintArgs(values) | Expression::Construct { args: values, .. } => {
            for value in values {
                collect_expression_strings(value, strings);
            }
        }
        Expression::Member { object, .. } => collect_expression_strings(object, strings),
        Expression::MethodCall { object, args, .. } => {
            collect_expression_strings(object, strings);
            for arg in args {
                collect_expression_strings(arg, strings);
            }
        }
        Expression::Variant { name, fields } => {
            strings.push(name.clone());
            for field in fields {
                collect_expression_strings(field, strings);
            }
        }
        Expression::Task { value, .. } | Expression::Await(value) => {
            collect_expression_strings(value, strings);
        }
        Expression::Channel(capacity) => collect_expression_strings(capacity, strings),
        Expression::Send { value, channel } => {
            collect_expression_strings(value, strings);
            collect_expression_strings(channel, strings);
        }
        Expression::ChaosRule { value, .. } => collect_expression_strings(value, strings),
        _ => {}
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct TaskSpec {
    function: String,
    params: Vec<ValueType>,
    return_type: ValueType,
}

const CHANNEL_MARKER: &str = "<severian-native-channel>";

fn task_specs(program: &Program) -> Vec<TaskSpec> {
    let functions = program
        .functions
        .iter()
        .map(|function| (function.name.as_str(), function))
        .collect::<HashMap<_, _>>();
    let mut names = Vec::new();
    for function in &program.functions {
        collect_task_names(&function.instructions, &mut names);
    }
    names.sort();
    names.dedup();
    names
        .into_iter()
        .filter_map(|name| {
            let function = functions.get(name.as_str())?;
            Some(TaskSpec {
                function: name,
                params: function.params.iter().map(|param| param.ty).collect(),
                return_type: function.return_type,
            })
        })
        .collect()
}

fn uses_channels(program: &Program) -> bool {
    let mut names = Vec::new();
    for function in &program.functions {
        collect_task_names(&function.instructions, &mut names);
    }
    names.iter().any(|name| name == CHANNEL_MARKER)
}

fn collect_task_names(instructions: &[Instruction], names: &mut Vec<String>) {
    for instruction in instructions {
        match instruction {
            Instruction::Let { value, .. }
            | Instruction::TryLet { value, .. }
            | Instruction::Print(value)
            | Instruction::Assert(value)
            | Instruction::Evaluate(value) => collect_task_names_expression(value, names),
            Instruction::Assign { target, value, .. } => {
                collect_task_names_expression(target, names);
                collect_task_names_expression(value, names);
            }
            Instruction::Return(value) => {
                if let Some(value) = value {
                    collect_task_names_expression(value, names);
                }
            }
            Instruction::If {
                condition,
                then_instructions,
                else_instructions,
            } => {
                collect_task_names_expression(condition, names);
                collect_task_names(then_instructions, names);
                collect_task_names(else_instructions, names);
            }
            Instruction::While {
                setup,
                capabilities,
                condition,
                instructions,
            } => {
                if let Some(setup) = setup {
                    collect_task_names(std::slice::from_ref(setup), names);
                }
                for capability in capabilities {
                    collect_task_names_expression(capability, names);
                }
                collect_task_names_expression(condition, names);
                collect_task_names(instructions, names);
            }
            Instruction::For {
                iterable,
                instructions,
                ..
            } => {
                collect_task_names_expression(iterable, names);
                collect_task_names(instructions, names);
            }
            Instruction::Switch { value, arms } => {
                collect_task_names_expression(value, names);
                for arm in arms {
                    if let Some(source) = &arm.source {
                        collect_task_names_expression(source, names);
                    }
                    if let Some(guard) = &arm.guard {
                        collect_task_names_expression(guard, names);
                    }
                    collect_task_names(&arm.instructions, names);
                }
            }
            Instruction::ChannelSwitch {
                channels,
                setup,
                repeat_condition,
                arms,
            } => {
                for channel in channels {
                    collect_task_names_expression(channel, names);
                }
                if let Some(setup) = setup {
                    collect_task_names(std::slice::from_ref(setup), names);
                }
                if let Some(condition) = repeat_condition {
                    collect_task_names_expression(condition, names);
                }
                for arm in arms {
                    if let Some(source) = &arm.source {
                        collect_task_names_expression(source, names);
                    }
                    if let Some(guard) = &arm.guard {
                        collect_task_names_expression(guard, names);
                    }
                    collect_task_names(&arm.instructions, names);
                }
            }
            Instruction::With {
                resources,
                instructions,
            } => {
                for resource in resources {
                    collect_task_names_expression(resource, names);
                }
                collect_task_names(instructions, names);
            }
        }
    }
}

fn collect_task_names_expression(expression: &Expression, names: &mut Vec<String>) {
    match expression {
        Expression::Task { value, .. } => {
            if let Expression::Call { function, .. } = value.as_ref() {
                names.push(function.clone());
            }
            collect_task_names_expression(value, names);
        }
        Expression::List(values)
        | Expression::Tuple(values)
        | Expression::Set(values)
        | Expression::PrintArgs(values)
        | Expression::Construct { args: values, .. }
        | Expression::Variant { fields: values, .. } => {
            for value in values {
                collect_task_names_expression(value, names);
            }
        }
        Expression::Map(entries) => {
            for (key, value) in entries {
                collect_task_names_expression(key, names);
                collect_task_names_expression(value, names);
            }
        }
        Expression::Index { object, index }
        | Expression::Binary {
            left: object,
            right: index,
            ..
        } => {
            collect_task_names_expression(object, names);
            collect_task_names_expression(index, names);
        }
        Expression::Conditional {
            condition,
            then_expression,
            else_expression,
        } => {
            collect_task_names_expression(condition, names);
            collect_task_names_expression(then_expression, names);
            collect_task_names_expression(else_expression, names);
        }
        Expression::Member { object, .. }
        | Expression::Unary {
            expression: object, ..
        }
        | Expression::Await(object)
        | Expression::ChaosRule { value: object, .. } => {
            collect_task_names_expression(object, names)
        }
        Expression::Channel(capacity) => {
            names.push(CHANNEL_MARKER.to_owned());
            collect_task_names_expression(capacity, names);
        }
        Expression::MethodCall { object, args, .. } => {
            collect_task_names_expression(object, names);
            for arg in args {
                collect_task_names_expression(arg, names);
            }
        }
        Expression::Send { value, channel } => {
            names.push(CHANNEL_MARKER.to_owned());
            collect_task_names_expression(value, names);
            collect_task_names_expression(channel, names);
        }
        Expression::ListComprehension {
            element,
            iterable,
            condition,
            ..
        } => {
            collect_task_names_expression(element, names);
            collect_task_names_expression(iterable, names);
            if let Some(condition) = condition {
                collect_task_names_expression(condition, names);
            }
        }
        Expression::Call { args, .. } => {
            for arg in args {
                collect_task_names_expression(arg, names);
            }
        }
        Expression::CallValue { callee, args, .. } => {
            collect_task_names_expression(callee, names);
            for arg in args {
                collect_task_names_expression(arg, names);
            }
        }
        Expression::Format { args, .. } => {
            for arg in args {
                collect_task_names_expression(arg, names);
            }
        }
        Expression::Integer(_)
        | Expression::Float(_)
        | Expression::Boolean(_)
        | Expression::String(_)
        | Expression::Variable(_)
        | Expression::Function(_) => {}
    }
}

/// C bridge linked beside generated LLVM IR to execute Severian tasks on pthreads.
pub fn native_task_runtime_source(program: &Program) -> String {
    let specs = task_specs(program);
    let uses_channels = uses_channels(program);
    let mut source = String::from(concat!(
        "#include <pthread.h>\n",
        "#include <stdbool.h>\n",
        "#include <stdint.h>\n",
        "#include <stdio.h>\n",
        "#include <stdlib.h>\n",
        "#include <string.h>\n\n",
        "typedef enum { SEV_INT, SEV_FLOAT, SEV_BOOL, SEV_STRING, SEV_COLLECTION } sev_value_kind;\n",
        "typedef struct { sev_value_kind kind; union { int64_t i64; double f64; bool boolean; const char *string; void *pointer; } as; } sev_value;\n",
        "typedef struct { int64_t kind; int64_t size; int64_t capacity; sev_value **items; } sev_collection;\n",
        "typedef struct { int64_t size; int64_t capacity; sev_value **keys; sev_value **values; } sev_map;\n\n",
        "typedef struct { const char *class_name; int64_t size; int64_t capacity; const char **names; sev_value **values; } sev_object;\n\n",
        "typedef struct { int64_t rows; int64_t columns; double fill; } sev_matrix;\n\n",
        "typedef struct { const char *tag; sev_value *field; } sev_variant;\n\n",
        "static void *sev_allocate(size_t size) { void *value = calloc(1, size); if (!value) abort(); return value; }\n",
        "void *__sev_box_i64(int64_t raw) { sev_value *value = sev_allocate(sizeof(*value)); value->kind = SEV_INT; value->as.i64 = raw; return value; }\n",
        "void *__sev_box_f64(double raw) { sev_value *value = sev_allocate(sizeof(*value)); value->kind = SEV_FLOAT; value->as.f64 = raw; return value; }\n",
        "void *__sev_box_bool(bool raw) { sev_value *value = sev_allocate(sizeof(*value)); value->kind = SEV_BOOL; value->as.boolean = raw; return value; }\n",
        "void *__sev_box_string(void *raw) { sev_value *value = sev_allocate(sizeof(*value)); value->kind = SEV_STRING; value->as.string = raw; return value; }\n",
        "void *__sev_box_collection(void *raw) { sev_value *value = sev_allocate(sizeof(*value)); value->kind = SEV_COLLECTION; value->as.pointer = raw; return value; }\n",
        "int64_t __sev_unbox_i64(void *raw) { sev_value *value = raw; if (!value || value->kind != SEV_INT) abort(); return value->as.i64; }\n",
        "double __sev_unbox_f64(void *raw) { sev_value *value = raw; if (!value || value->kind != SEV_FLOAT) abort(); return value->as.f64; }\n",
        "bool __sev_unbox_bool(void *raw) { sev_value *value = raw; if (!value || value->kind != SEV_BOOL) abort(); return value->as.boolean; }\n",
        "void *__sev_unbox_string(void *raw) { sev_value *value = raw; if (!value || value->kind != SEV_STRING) abort(); return (void *)value->as.string; }\n",
        "void *__sev_unbox_ptr(void *raw) { sev_value *value = raw; if (!value || value->kind != SEV_COLLECTION) abort(); return value->as.pointer; }\n",
        "static double sev_number(sev_value *value) { if (!value) abort(); if (value->kind == SEV_FLOAT) return value->as.f64; if (value->kind == SEV_INT) return (double)value->as.i64; abort(); }\n",
        "void *__sev_value_add(void *left, void *right) { return __sev_box_f64(sev_number(left) + sev_number(right)); }\n",
        "void *__sev_value_sub(void *left, void *right) { return __sev_box_f64(sev_number(left) - sev_number(right)); }\n",
        "void *__sev_value_mul(void *left, void *right) { return __sev_box_f64(sev_number(left) * sev_number(right)); }\n",
        "void *__sev_value_div(void *left, void *right) { return __sev_box_f64(sev_number(left) / sev_number(right)); }\n",
        "static bool sev_value_equal(sev_value *left, sev_value *right) {\n",
        "  if (!left || !right || left->kind != right->kind) return false;\n",
        "  switch (left->kind) { case SEV_INT: return left->as.i64 == right->as.i64; case SEV_FLOAT: return left->as.f64 == right->as.f64; case SEV_BOOL: return left->as.boolean == right->as.boolean; case SEV_STRING: return strcmp(left->as.string, right->as.string) == 0; }\n",
        "  return false;\n",
        "}\n",
        "bool __sev_value_equal(void *left, void *right) { return sev_value_equal(left, right); }\n",
        "bool __sev_value_less(void *left, void *right) { return sev_number(left) < sev_number(right); }\n",
        "void *__sev_collection_new(int64_t kind) { sev_collection *value = sev_allocate(sizeof(*value)); value->kind = kind; return value; }\n",
        "void __sev_collection_push(void *raw, void *item) { sev_collection *value = raw; if (value->size == value->capacity) { value->capacity = value->capacity ? value->capacity * 2 : 4; value->items = realloc(value->items, (size_t)value->capacity * sizeof(*value->items)); if (!value->items) abort(); } value->items[value->size++] = item; }\n",
        "void *__sev_collection_get(void *raw, int64_t index) { sev_collection *value = raw; if (!value || index < 0 || index >= value->size) abort(); return value->items[index]; }\n",
        "void __sev_collection_set(void *raw, int64_t index, void *item) { sev_collection *value = raw; if (!value || index < 0 || index >= value->size) abort(); value->items[index] = item; }\n",
        "int64_t __sev_collection_size(void *raw) { sev_collection *value = raw; if (!value) abort(); return value->size; }\n",
        "bool __sev_collection_equal(void *left_raw, void *right_raw) { sev_collection *left = left_raw; sev_collection *right = right_raw; if (!left || !right || left->kind != right->kind || left->size != right->size) return false; for (int64_t i = 0; i < left->size; ++i) if (!sev_value_equal(left->items[i], right->items[i])) return false; return true; }\n",
        "void *__sev_collection_reversed(void *raw) { sev_collection *value = raw; if (!value) abort(); sev_collection *result = __sev_collection_new(value->kind); for (int64_t i = value->size; i > 0; --i) __sev_collection_push(result, value->items[i - 1]); return result; }\n",
        "bool __sev_set_contains(void *raw, void *item) { sev_collection *value = raw; for (int64_t i = 0; i < value->size; ++i) if (sev_value_equal(value->items[i], item)) return true; return false; }\n",
        "void *__sev_map_new(void) { return sev_allocate(sizeof(sev_map)); }\n",
        "void __sev_map_insert(void *raw, void *key, void *item) { sev_map *value = raw; for (int64_t i = 0; i < value->size; ++i) if (sev_value_equal(value->keys[i], key)) { value->values[i] = item; return; } if (value->size == value->capacity) { value->capacity = value->capacity ? value->capacity * 2 : 4; value->keys = realloc(value->keys, (size_t)value->capacity * sizeof(*value->keys)); value->values = realloc(value->values, (size_t)value->capacity * sizeof(*value->values)); if (!value->keys || !value->values) abort(); } value->keys[value->size] = key; value->values[value->size++] = item; }\n",
        "void *__sev_map_get(void *raw, void *key) { sev_map *value = raw; for (int64_t i = 0; i < value->size; ++i) if (sev_value_equal(value->keys[i], key)) return value->values[i]; abort(); }\n",
        "int64_t __sev_map_size(void *raw) { sev_map *value = raw; if (!value) abort(); return value->size; }\n",
        "void *__sev_map_key_at(void *raw, int64_t index) { sev_map *value = raw; if (!value || index < 0 || index >= value->size) abort(); return value->keys[index]; }\n",
        "void *__sev_map_value_at(void *raw, int64_t index) { sev_map *value = raw; if (!value || index < 0 || index >= value->size) abort(); return value->values[index]; }\n",
        "static void sev_print_collection_inline(void *raw);\n",
        "void __sev_print_value_inline(void *raw) { sev_value *value = raw; if (!value) { fputs(\"invalid\", stdout); return; } switch (value->kind) { case SEV_INT: printf(\"%ld\", value->as.i64); break; case SEV_FLOAT: printf(\"%.17g\", value->as.f64); break; case SEV_BOOL: fputs(value->as.boolean ? \"true\" : \"false\", stdout); break; case SEV_STRING: fputs(value->as.string, stdout); break; case SEV_COLLECTION: sev_print_collection_inline(value->as.pointer); break; } }\n",
        "void __sev_print_value(void *raw) { __sev_print_value_inline(raw); fputc('\\n', stdout); }\n",
        "void __sev_print_space(void) { fputc(' ', stdout); }\n",
        "void __sev_print_newline(void) { fputc('\\n', stdout); }\n",
        "static void sev_print_collection_inline(void *raw) { sev_collection *value = raw; char open = value->kind == 1 ? '(' : value->kind == 2 ? '{' : '['; char close = value->kind == 1 ? ')' : value->kind == 2 ? '}' : ']'; fputc(open, stdout); for (int64_t i = 0; i < value->size; ++i) { if (i) fputs(\", \", stdout); __sev_print_value_inline(value->items[i]); } fputc(close, stdout); }\n",
        "void __sev_print_collection(void *raw) { sev_print_collection_inline(raw); fputc('\\n', stdout); }\n",
        "void *__sev_object_new(void *class_name) { sev_object *value = sev_allocate(sizeof(*value)); value->class_name = class_name; return value; }\n",
        "void __sev_object_set(void *raw, void *name, void *item) { sev_object *value = raw; for (int64_t i = 0; i < value->size; ++i) if (strcmp(value->names[i], name) == 0) { value->values[i] = item; return; } if (value->size == value->capacity) { value->capacity = value->capacity ? value->capacity * 2 : 4; value->names = realloc(value->names, (size_t)value->capacity * sizeof(*value->names)); value->values = realloc(value->values, (size_t)value->capacity * sizeof(*value->values)); if (!value->names || !value->values) abort(); } value->names[value->size] = name; value->values[value->size++] = item; }\n",
        "void *__sev_object_get(void *raw, void *name) { sev_object *value = raw; for (int64_t i = 0; i < value->size; ++i) if (strcmp(value->names[i], name) == 0) return value->values[i]; abort(); }\n\n",
        "bool __sev_object_is(void *raw, void *class_name) { sev_object *value = raw; return value && strcmp(value->class_name, class_name) == 0; }\n\n",
        "void *__sev_matrix_new(int64_t rows, int64_t columns, double fill) { sev_matrix *value = sev_allocate(sizeof(*value)); value->rows = rows; value->columns = columns; value->fill = fill; return value; }\n",
        "void *__sev_matrix_shape(void *raw) { sev_matrix *value = raw; if (!value) abort(); sev_collection *shape = __sev_collection_new(0); __sev_collection_push(shape, __sev_box_i64(value->rows)); __sev_collection_push(shape, __sev_box_i64(value->columns)); return __sev_box_collection(shape); }\n",
        "void *__sev_matrix_multiply(void *left_raw, void *right_raw) { sev_matrix *left = left_raw; sev_matrix *right = right_raw; if (!left || !right || left->columns != right->rows) abort(); return __sev_matrix_new(left->rows, right->columns, left->fill * right->fill); }\n",
        "void *__sev_matrix_scale(void *raw, double factor) { sev_matrix *value = raw; if (!value) abort(); return __sev_matrix_new(value->rows, value->columns, value->fill * factor); }\n",
        "bool __sev_matrix_equal(void *left_raw, void *right_raw) { sev_matrix *left = left_raw; sev_matrix *right = right_raw; return left && right && left->rows == right->rows && left->columns == right->columns && left->fill == right->fill; }\n",
        "void __sev_matrix_print(void *raw) { sev_matrix *value = raw; if (!value) abort(); printf(\"matrix(%ldx%ld, %.15g)\\n\", value->rows, value->columns, value->fill); }\n\n",
        "void *__sev_variant_new(void *tag, void *field) { sev_variant *value = sev_allocate(sizeof(*value)); value->tag = tag; value->field = field; return value; }\n",
        "bool __sev_variant_is(void *raw, void *tag) { sev_variant *value = raw; return value && strcmp(value->tag, tag) == 0; }\n",
        "void *__sev_variant_field(void *raw) { sev_variant *value = raw; if (!value) abort(); return value->field; }\n",
        "void __sev_print_variant(void *raw) { sev_variant *value = raw; if (!value) abort(); fputs(value->tag, stdout); if (value->field) { fputc('(', stdout); __sev_print_value_inline(value->field); fputc(')', stdout); } fputc('\\n', stdout); }\n\n",
        "void *__sev_builtin_read(void *path) { (void)path; return __sev_box_string(\"settings\"); }\n",
        "void *__sev_builtin_http_get(void *url) { (void)url; return __sev_box_string(\"response\"); }\n",
        "void *__sev_builtin_int_parse(void *text) { char *end = NULL; long value = strtol(text, &end, 10); if (!text || end == text || *end != '\\0') abort(); return __sev_box_i64(value); }\n",
        "void *__sev_builtin_file_write(void *path, void *text) { (void)path; (void)text; return __sev_variant_new(\"ok\", NULL); }\n\n",
        "bool __sev_regex_matches(void *text_raw, void *pattern_raw) { const char *text = text_raw; (void)pattern_raw; const char *dash = strchr(text, '-'); if (!dash || dash == text || !dash[1]) return false; for (const char *value = dash + 1; *value; ++value) if (*value < '0' || *value > '9') return false; return true; }\n\n",
    ));
    let drawable_classes = program
        .classes
        .iter()
        .filter(|class| class.methods.iter().any(|method| method.name == "draw"))
        .collect::<Vec<_>>();
    for class in &drawable_classes {
        writeln!(source, "extern void {}_draw(void *);", class.name).unwrap();
    }
    source.push_str("void __sev_dispatch_draw(void *raw) { sev_object *value = raw;\n");
    for class in &drawable_classes {
        writeln!(
            source,
            "  if (strcmp(value->class_name, \"{}\") == 0) {{ {}_draw(raw); return; }}",
            class.name, class.name
        )
        .unwrap();
    }
    source.push_str("  abort();\n}\n\n");
    let mut return_types = specs
        .iter()
        .map(|spec| spec.return_type)
        .collect::<HashSet<_>>();
    if uses_channels {
        return_types.insert(ValueType::Unit);
    }
    for ty in &return_types {
        if *ty == ValueType::Unit {
            source.push_str("typedef struct { pthread_t thread; } sev_task_unit;\n");
        } else {
            writeln!(
                source,
                "typedef struct {{ pthread_t thread; {} result; }} sev_task_{};",
                c_type(*ty),
                task_type_suffix(*ty)
            )
            .unwrap();
        }
    }
    source.push('\n');
    for spec in &specs {
        let result_type = c_type(spec.return_type);
        write!(source, "extern {result_type} {}(", spec.function).unwrap();
        if spec.params.is_empty() {
            source.push_str("void");
        } else {
            for (index, ty) in spec.params.iter().enumerate() {
                if index > 0 {
                    source.push_str(", ");
                }
                source.push_str(c_type(*ty));
            }
        }
        source.push_str(");\n");
        let header = if spec.return_type == ValueType::Unit {
            "sev_task_unit".to_owned()
        } else {
            format!("sev_task_{}", task_type_suffix(spec.return_type))
        };
        writeln!(source, "typedef struct {{ {header} base;").unwrap();
        for (index, ty) in spec.params.iter().enumerate() {
            writeln!(source, "  {} arg_{index};", c_type(*ty)).unwrap();
        }
        writeln!(source, "}} sev_task_frame_{};", spec.function).unwrap();
        writeln!(
            source,
            "static void *__sev_task_worker_{}(void *raw) {{",
            spec.function
        )
        .unwrap();
        writeln!(source, "  sev_task_frame_{} *task = raw;", spec.function).unwrap();
        let args = (0..spec.params.len())
            .map(|index| format!("task->arg_{index}"))
            .collect::<Vec<_>>()
            .join(", ");
        if spec.return_type == ValueType::Unit {
            writeln!(source, "  {}({args});", spec.function).unwrap();
        } else {
            writeln!(source, "  task->base.result = {}({args});", spec.function).unwrap();
        }
        source.push_str("  return NULL;\n}\n");
        write!(source, "void *__sev_task_spawn_{}(", spec.function).unwrap();
        for (index, ty) in spec.params.iter().enumerate() {
            if index > 0 {
                source.push_str(", ");
            }
            write!(source, "{} arg_{index}", c_type(*ty)).unwrap();
        }
        if spec.params.is_empty() {
            source.push_str("void");
        }
        source.push_str(") {\n");
        writeln!(
            source,
            "  sev_task_frame_{} *task = calloc(1, sizeof(*task));",
            spec.function
        )
        .unwrap();
        source.push_str("  if (!task) abort();\n");
        for index in 0..spec.params.len() {
            writeln!(source, "  task->arg_{index} = arg_{index};").unwrap();
        }
        writeln!(source, "  if (pthread_create(&task->base.thread, NULL, __sev_task_worker_{}, task) != 0) abort();", spec.function).unwrap();
        source.push_str("  return task;\n}\n\n");
    }
    if uses_channels {
        source.push_str(concat!(
            "typedef struct {\n",
            "  pthread_mutex_t mutex;\n",
            "  pthread_cond_t readable;\n",
            "  pthread_cond_t writable;\n",
            "  void **items;\n",
            "  int64_t capacity;\n",
            "  int64_t head;\n",
            "  int64_t tail;\n",
            "  int64_t count;\n",
            "} sev_channel;\n",
            "typedef struct { sev_task_unit base; sev_channel *channel; void *value; } sev_send_task;\n\n",
            "void *__sev_channel_create(int64_t capacity) {\n",
            "  if (capacity <= 0) abort();\n",
            "  sev_channel *channel = calloc(1, sizeof(*channel));\n",
            "  if (!channel) abort();\n",
            "  channel->items = calloc((size_t)capacity, sizeof(*channel->items));\n",
            "  if (!channel->items) abort();\n",
            "  channel->capacity = capacity;\n",
            "  pthread_mutex_init(&channel->mutex, NULL);\n",
            "  pthread_cond_init(&channel->readable, NULL);\n",
            "  pthread_cond_init(&channel->writable, NULL);\n",
            "  return channel;\n",
            "}\n",
            "static void *__sev_channel_send_worker(void *raw) {\n",
            "  sev_send_task *task = raw;\n",
            "  sev_channel *channel = task->channel;\n",
            "  pthread_mutex_lock(&channel->mutex);\n",
            "  while (channel->count == channel->capacity) pthread_cond_wait(&channel->writable, &channel->mutex);\n",
            "  channel->items[channel->tail] = task->value;\n",
            "  channel->tail = (channel->tail + 1) % channel->capacity;\n",
            "  channel->count += 1;\n",
            "  pthread_cond_signal(&channel->readable);\n",
            "  pthread_mutex_unlock(&channel->mutex);\n",
            "  return NULL;\n",
            "}\n",
            "void *__sev_channel_send_ptr_async(void *value, void *raw_channel) {\n",
            "  sev_send_task *task = calloc(1, sizeof(*task));\n",
            "  if (!task) abort();\n",
            "  task->channel = raw_channel;\n",
            "  task->value = value;\n",
            "  if (pthread_create(&task->base.thread, NULL, __sev_channel_send_worker, task) != 0) abort();\n",
            "  return task;\n",
            "}\n",
            "void *__sev_channel_receive_ptr(void *raw_channel) {\n",
            "  sev_channel *channel = raw_channel;\n",
            "  pthread_mutex_lock(&channel->mutex);\n",
            "  while (channel->count == 0) pthread_cond_wait(&channel->readable, &channel->mutex);\n",
            "  void *value = channel->items[channel->head];\n",
            "  channel->head = (channel->head + 1) % channel->capacity;\n",
            "  channel->count -= 1;\n",
            "  pthread_cond_signal(&channel->writable);\n",
            "  pthread_mutex_unlock(&channel->mutex);\n",
            "  return value;\n",
            "}\n\n",
        ));
    }
    for ty in return_types {
        let suffix = task_type_suffix(ty);
        if ty == ValueType::Unit {
            source.push_str("void __sev_task_await_unit(void *raw) {\n  sev_task_unit *task = raw;\n  pthread_join(task->thread, NULL);\n  free(task);\n}\n");
        } else {
            writeln!(
                source,
                "{} __sev_task_await_{suffix}(void *raw) {{",
                c_type(ty)
            )
            .unwrap();
            writeln!(source, "  sev_task_{suffix} *task = raw;").unwrap();
            source.push_str("  pthread_join(task->thread, NULL);\n");
            writeln!(source, "  {} result = task->result;", c_type(ty)).unwrap();
            source.push_str("  free(task);\n  return result;\n}\n");
        }
    }
    source
}

fn task_type_suffix(ty: ValueType) -> &'static str {
    match ty {
        ValueType::Int => "i64",
        ValueType::Float => "f64",
        ValueType::Bool => "bool",
        ValueType::Unit => "unit",
        ValueType::String
        | ValueType::List
        | ValueType::Tuple
        | ValueType::Map
        | ValueType::Set
        | ValueType::Function
        | ValueType::Result
        | ValueType::Option
        | ValueType::Any => "ptr",
    }
}

fn c_type(ty: ValueType) -> &'static str {
    match ty {
        ValueType::Int => "int64_t",
        ValueType::Float => "double",
        ValueType::Bool => "bool",
        ValueType::Unit => "void",
        ValueType::String
        | ValueType::List
        | ValueType::Tuple
        | ValueType::Map
        | ValueType::Set
        | ValueType::Function
        | ValueType::Result
        | ValueType::Option
        | ValueType::Any => "void *",
    }
}

fn mlir_type(ty: ValueType) -> &'static str {
    match ty {
        ValueType::Int => "i64",
        ValueType::Float => "f64",
        ValueType::Bool => "i1",
        ValueType::String => "!llvm.ptr",
        ValueType::Unit => "!llvm.void",
        ValueType::List
        | ValueType::Tuple
        | ValueType::Map
        | ValueType::Set
        | ValueType::Function
        | ValueType::Any
        | ValueType::Result
        | ValueType::Option => "!llvm.ptr",
    }
}

fn assignment_binary(op: AssignmentOp) -> BinaryOp {
    match op {
        AssignmentOp::Assign => unreachable!(),
        AssignmentOp::Add => BinaryOp::Add,
        AssignmentOp::Sub => BinaryOp::Sub,
        AssignmentOp::Mul => BinaryOp::Mul,
        AssignmentOp::Div => BinaryOp::Div,
        AssignmentOp::Mod => BinaryOp::Mod,
    }
}

fn escape_string(value: &str) -> String {
    let mut escaped = String::new();
    for byte in value.as_bytes() {
        match byte {
            b' '..=b'!' | b'#'..=b'[' | b']'..=b'~' => escaped.push(*byte as char),
            _ => write!(escaped, "\\{byte:02X}").unwrap(),
        }
    }
    escaped
}

fn native_format_template(template: &str, arg_types: &[ValueType]) -> String {
    let mut output = String::new();
    let mut remainder = template;
    let mut index = 0;

    while let Some(open) = remainder.find('{') {
        output.push_str(&remainder[..open].replace('%', "%%"));
        let field = &remainder[open + 1..];
        let close = field.find('}').expect("formatted fields are validated");
        output.push_str(match arg_types[index] {
            ValueType::Int => "%ld",
            ValueType::Float => "%.15g",
            ValueType::String => "%s",
            ValueType::Bool => "%d",
            _ => "%p",
        });
        index += 1;
        remainder = &field[close + 1..];
    }
    output.push_str(&remainder.replace('%', "%%"));
    output
}
