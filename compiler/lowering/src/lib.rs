#![forbid(unsafe_code)]

use severian_hir::{
    AssignmentOp, BinaryOp, Expression, Function, Instruction, Program, UnaryOp, ValueType,
};
use severian_mlir::Module;
use std::collections::HashMap;
use std::fmt::Write;

pub fn lower(program: &Program) -> Module {
    let mut strings = Vec::new();
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
        "  llvm.func @llvm.sqrt.f64(f64) -> f64\n\n",
    ));

    let function_returns = program
        .functions
        .iter()
        .map(|function| (function.name.clone(), function.return_type))
        .collect::<HashMap<_, _>>();
    for function in &program.functions {
        lower_function(
            function,
            &program.globals,
            &strings,
            &function_returns,
            &mut output,
        );
    }
    output.push_str("}\n");
    Module::new(output)
}

fn lower_function(
    function: &Function,
    globals: &[severian_hir::Global],
    strings: &[String],
    function_returns: &HashMap<String, ValueType>,
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
        }
    }
    context.output.push_str("  }\n");
}

struct LowerContext<'a> {
    output: &'a mut String,
    strings: &'a [String],
    function_returns: &'a HashMap<String, ValueType>,
    variables: HashMap<String, (String, ValueType)>,
    next_value: usize,
    next_block: usize,
    terminated: bool,
    is_main: bool,
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
                        let lowered = if *op == AssignmentOp::Assign {
                            right
                        } else {
                            let left = self.variables.get(name).cloned().unwrap_or(right.clone());
                            self.lower_binary_values(left, assignment_binary(*op), right)
                        };
                        self.variables.insert(name.clone(), lowered);
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
                        _ => {}
                    }
                }
                Instruction::Evaluate(expression) => {
                    self.lower_expression(expression);
                }
                Instruction::Assert(_) => {}
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
                        let (value, ty) = self.lower_expression(value);
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
                    if !then_terminated {
                        writeln!(self.output, "    llvm.br ^bb{continue_block}").unwrap();
                    }
                    writeln!(self.output, "  ^bb{else_block}:").unwrap();
                    self.terminated = false;
                    self.lower_instructions(else_instructions);
                    let else_terminated = self.terminated;
                    if !else_terminated {
                        writeln!(self.output, "    llvm.br ^bb{continue_block}").unwrap();
                    }
                    if !then_terminated || !else_terminated {
                        writeln!(self.output, "  ^bb{continue_block}:").unwrap();
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
                Instruction::Switch { .. } => {}
                Instruction::ChannelSwitch { .. } => {}
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
            Expression::Variable(name) => self.variables.get(name).cloned().unwrap(),
            Expression::Function(_)
            | Expression::List(_)
            | Expression::Tuple(_)
            | Expression::Map(_)
            | Expression::Set(_)
            | Expression::ListComprehension { .. }
            | Expression::PrintArgs(_)
            | Expression::Construct { .. }
            | Expression::Member { .. }
            | Expression::MethodCall { .. }
            | Expression::Task(_)
            | Expression::Await(_)
            | Expression::Channel(_)
            | Expression::Send { .. }
            | Expression::ChaosRule { .. } => {
                let result = self.fresh_value();
                writeln!(self.output, "    {result} = llvm.mlir.zero : !llvm.ptr").unwrap();
                (result, ValueType::Any)
            }
            Expression::Variant { .. } => {
                let result = self.fresh_value();
                writeln!(self.output, "    {result} = llvm.mlir.zero : !llvm.ptr").unwrap();
                (result, ValueType::Any)
            }
            Expression::Index { .. } => {
                let result = self.fresh_value();
                writeln!(
                    self.output,
                    "    {result} = llvm.mlir.constant(0 : i64) : i64"
                )
                .unwrap();
                (result, ValueType::Any)
            }
            Expression::Format(value) => {
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
                (result, return_type)
            }
            Expression::CallValue { .. } => {
                let result = self.fresh_value();
                writeln!(
                    self.output,
                    "    {result} = llvm.mlir.constant(0 : i64) : i64"
                )
                .unwrap();
                (result, ValueType::Any)
            }
            Expression::Binary { left, op, right } => {
                let left = self.lower_expression(left);
                let right = self.lower_expression(right);
                self.lower_binary_values(left, *op, right)
            }
        }
    }

    fn lower_binary_values(
        &mut self,
        (left, operand_type): (String, ValueType),
        op: BinaryOp,
        (right, _): (String, ValueType),
    ) -> (String, ValueType) {
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
                writeln!(
                    self.output,
                    "    {result} = llvm.icmp \"{predicate}\" {left}, {right} : {}",
                    mlir_type(operand_type)
                )
                .unwrap();
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
        let severian_hir::MatchPattern::Bind(name) = pattern else {
            return;
        };
        let Expression::Call { function, args } = iterable else {
            return;
        };
        if function != "range" || !(1..=2).contains(&args.len()) {
            return;
        }

        let (start, end) = if args.len() == 1 {
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
        };

        let previous_binding = self.variables.remove(name);
        let header = self.fresh_block();
        let body = self.fresh_block();
        let exit = self.fresh_block();
        writeln!(self.output, "    llvm.br ^bb{header}({start} : i64)").unwrap();

        let index = self.fresh_value();
        writeln!(self.output, "  ^bb{header}({index}: i64):").unwrap();
        let condition = self.fresh_value();
        writeln!(
            self.output,
            "    {condition} = llvm.icmp \"slt\" {index}, {end} : i64"
        )
        .unwrap();
        writeln!(
            self.output,
            "    llvm.cond_br {condition}, ^bb{body}, ^bb{exit}"
        )
        .unwrap();

        writeln!(self.output, "  ^bb{body}:").unwrap();
        self.variables
            .insert(name.clone(), (index.clone(), ValueType::Int));
        self.terminated = false;
        self.lower_instructions(instructions);
        if !self.terminated {
            let one = self.fresh_value();
            writeln!(self.output, "    {one} = llvm.mlir.constant(1 : i64) : i64").unwrap();
            let next = self.fresh_value();
            writeln!(self.output, "    {next} = llvm.add {index}, {one} : i64").unwrap();
            writeln!(self.output, "    llvm.br ^bb{header}({next} : i64)").unwrap();
        }

        writeln!(self.output, "  ^bb{exit}:").unwrap();
        if let Some(previous_binding) = previous_binding {
            self.variables.insert(name.clone(), previous_binding);
        } else {
            self.variables.remove(name);
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
        Expression::Format(value) => strings.push(value.clone()),
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
        Expression::Unary { expression, .. } => collect_expression_strings(expression, strings),
        Expression::CallValue { callee, args } => {
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
        Expression::Variant { fields, .. } => {
            for field in fields {
                collect_expression_strings(field, strings);
            }
        }
        Expression::Task(value) | Expression::Await(value) => {
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
