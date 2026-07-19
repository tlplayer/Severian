#![forbid(unsafe_code)]

use severian_hir::{BinaryOp, Expression, Function, Instruction, Program, ValueType};
use severian_mlir::Module;
use std::collections::HashMap;
use std::fmt::Write;

pub fn lower(program: &Program) -> Module {
    let mut strings = Vec::new();
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
    if !strings.is_empty() {
        output.push_str("  llvm.func @puts(!llvm.ptr) -> i32\n\n");
    }

    for function in &program.functions {
        lower_function(function, &strings, &mut output);
    }
    output.push_str("}\n");
    Module::new(output)
}

fn lower_function(function: &Function, strings: &[String], output: &mut String) {
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
                Instruction::Print(value) => {
                    let (value, ty) = self.lower_expression(value);
                    if ty == ValueType::String {
                        let status = self.fresh_value();
                        writeln!(
                            self.output,
                            "    {status} = llvm.call @puts({value}) : (!llvm.ptr) -> i32"
                        )
                        .unwrap();
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
            Expression::Call { function, args } => {
                let args = args
                    .iter()
                    .map(|argument| self.lower_expression(argument))
                    .collect::<Vec<_>>();
                let result = self.fresh_value();
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
                writeln!(
                    self.output,
                    "    {result} = llvm.call @{function}({values}) : ({types}) -> i64"
                )
                .unwrap();
                (result, ValueType::Int)
            }
            Expression::Binary { left, op, right } => {
                let (left, operand_type) = self.lower_expression(left);
                let (right, _) = self.lower_expression(right);
                let result = self.fresh_value();
                let (operation, result_type) = match op {
                    BinaryOp::Add => ("llvm.add", ValueType::Int),
                    BinaryOp::Sub => ("llvm.sub", ValueType::Int),
                    BinaryOp::Mul => ("llvm.mul", ValueType::Int),
                    BinaryOp::Div => ("llvm.sdiv", ValueType::Int),
                    BinaryOp::Mod => ("llvm.srem", ValueType::Int),
                    comparison => {
                        let predicate = match comparison {
                            BinaryOp::Equal => "eq",
                            BinaryOp::NotEqual => "ne",
                            BinaryOp::Less => "slt",
                            BinaryOp::LessEqual => "sle",
                            BinaryOp::Greater => "sgt",
                            BinaryOp::GreaterEqual => "sge",
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
                    "    {result} = {operation} {left}, {right} : i64"
                )
                .unwrap();
                (result, result_type)
            }
        }
    }

    fn fresh_value(&mut self) -> String {
        let value = format!("%v{}", self.next_value);
        self.next_value += 1;
        value
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
        _ => {}
    }
}

fn mlir_type(ty: ValueType) -> &'static str {
    match ty {
        ValueType::Int => "i64",
        ValueType::Bool => "i1",
        ValueType::String => "!llvm.ptr",
        ValueType::Unit => "!llvm.void",
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
