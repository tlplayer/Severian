use crate::{Diagnostic, DiagnosticBag, Severity, SourceRange};
use severian_hir::{Expression, Function, Instruction, OwnershipOp, Program};
use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LintLevel {
    Allow,
    Warn,
    Deny,
}

#[derive(Debug, Clone, Default)]
pub struct LintConfig {
    levels: BTreeMap<String, LintLevel>,
}

impl LintConfig {
    pub fn set(&mut self, lint: impl Into<String>, level: LintLevel) {
        self.levels.insert(lint.into(), level);
    }

    pub fn level(&self, lint: &str) -> LintLevel {
        self.levels.get(lint).copied().unwrap_or(LintLevel::Warn)
    }
}

pub fn run(program: &Program, config: &LintConfig) -> DiagnosticBag {
    let mut bag = DiagnosticBag::default();

    for function in &program.functions {
        lint_function(function, config, &mut bag);
    }
    for class in &program.classes {
        for function in class.methods.iter().chain(&class.constructors) {
            lint_function(function, config, &mut bag);
        }
    }

    bag
}

pub fn run_with_source(
    program: &Program,
    config: &LintConfig,
    path: &Path,
    source: &str,
) -> DiagnosticBag {
    let mut bag = run(program, config);
    for diagnostic in bag.diagnostics_mut() {
        let needle = match diagnostic.code.0.as_str() {
            "W001" => diagnostic.message.split('`').nth(1).map(str::to_owned),
            "W002" => Some("async".to_owned()),
            _ => None,
        };
        let Some(needle) = needle else {
            continue;
        };
        let start = if diagnostic.code.0 == "W001" {
            source.find(&needle)
        } else {
            source.find("async")
        };
        let Some(start) = start else {
            continue;
        };
        let (start_line, start_column) = line_column(source, start);
        let (end_line, end_column) = line_column(source, start + needle.len());
        diagnostic.source = Some(SourceRange {
            file: path.to_path_buf(),
            start_byte: start,
            end_byte: start + needle.len(),
            start_line: Some(start_line),
            start_column: Some(start_column),
            end_line: Some(end_line),
            end_column: Some(end_column),
        });
    }
    bag
}

fn line_column(source: &str, byte: usize) -> (u32, u32) {
    let byte = byte.min(source.len());
    let prefix = source.get(..byte).unwrap_or("");
    let line = prefix
        .bytes()
        .filter(|character| *character == b'\n')
        .count() as u32
        + 1;
    let line_start = prefix.rfind('\n').map_or(0, |index| index + 1);
    let column = source.get(line_start..byte).unwrap_or("").chars().count() as u32 + 1;
    (line, column)
}

fn lint_function(function: &Function, config: &LintConfig, bag: &mut DiagnosticBag) {
    let mut defined = BTreeSet::new();
    let mut used = BTreeSet::new();

    for parameter in &function.params {
        defined.insert(parameter.name.clone());
    }

    collect_bindings_and_uses(&function.instructions, &mut defined, &mut used, config, bag);

    for name in defined.difference(&used) {
        if !name.starts_with('_') {
            emit(
                bag,
                config,
                "unused-binding",
                format!("binding `{name}` is never read in `{}`", function.name),
                Some("prefix intentionally-unused bindings with `_`, or remove them"),
            );
        }
    }
}

fn collect_bindings_and_uses(
    instructions: &[Instruction],
    defined: &mut BTreeSet<String>,
    used: &mut BTreeSet<String>,
    config: &LintConfig,
    bag: &mut DiagnosticBag,
) {
    for instruction in instructions {
        match instruction {
            Instruction::Let { name, value } | Instruction::TryLet { name, value } => {
                defined.insert(name.clone());
                inspect_expression(value, used, config, bag, false);
            }

            Instruction::Assign { target, value, .. } => {
                inspect_expression(target, used, config, bag, false);
                inspect_expression(value, used, config, bag, false);
            }

            Instruction::Evaluate(expression) => {
                inspect_expression(expression, used, config, bag, true);
            }

            Instruction::Print(expression) | Instruction::Assert(expression) => {
                inspect_expression(expression, used, config, bag, false);
            }

            Instruction::Return(expression) => {
                if let Some(expression) = expression {
                    inspect_expression(expression, used, config, bag, false);
                }
            }

            Instruction::If {
                condition,
                then_instructions,
                else_instructions,
            } => {
                inspect_expression(condition, used, config, bag, false);
                collect_bindings_and_uses(then_instructions, defined, used, config, bag);
                collect_bindings_and_uses(else_instructions, defined, used, config, bag);
            }

            Instruction::While {
                setup,
                capabilities,
                condition,
                instructions,
            } => {
                if let Some(setup) = setup {
                    collect_bindings_and_uses(
                        std::slice::from_ref(setup.as_ref()),
                        defined,
                        used,
                        config,
                        bag,
                    );
                }
                for capability in capabilities {
                    inspect_expression(capability, used, config, bag, false);
                }
                inspect_expression(condition, used, config, bag, false);
                collect_bindings_and_uses(instructions, defined, used, config, bag);
            }

            Instruction::For {
                setup,
                iterable,
                instructions,
                ..
            } => {
                if let Some(setup) = setup {
                    collect_bindings_and_uses(
                        std::slice::from_ref(setup.as_ref()),
                        defined,
                        used,
                        config,
                        bag,
                    );
                }
                inspect_expression(iterable, used, config, bag, false);
                collect_bindings_and_uses(instructions, defined, used, config, bag);
            }

            Instruction::Switch { value, arms } => {
                inspect_expression(value, used, config, bag, false);
                for arm in arms {
                    if let Some(source) = &arm.source {
                        inspect_expression(source, used, config, bag, false);
                    }
                    if let Some(guard) = &arm.guard {
                        inspect_expression(guard, used, config, bag, false);
                    }
                    collect_bindings_and_uses(&arm.instructions, defined, used, config, bag);
                }
            }

            Instruction::ChannelSwitch {
                channels,
                setup,
                repeat_condition,
                arms,
            } => {
                for channel in channels {
                    inspect_expression(channel, used, config, bag, false);
                }
                if let Some(setup) = setup {
                    collect_bindings_and_uses(
                        std::slice::from_ref(setup.as_ref()),
                        defined,
                        used,
                        config,
                        bag,
                    );
                }
                if let Some(condition) = repeat_condition {
                    inspect_expression(condition, used, config, bag, false);
                }
                for arm in arms {
                    collect_bindings_and_uses(&arm.instructions, defined, used, config, bag);
                }
            }

            Instruction::With {
                resources,
                instructions,
                ..
            } => {
                for resource in resources {
                    inspect_expression(resource, used, config, bag, false);
                }
                collect_bindings_and_uses(instructions, defined, used, config, bag);
            }

            Instruction::Break | Instruction::Continue => {}
        }
    }
}

fn inspect_expression(
    expression: &Expression,
    used: &mut BTreeSet<String>,
    config: &LintConfig,
    bag: &mut DiagnosticBag,
    discarded: bool,
) {
    match expression {
        Expression::Typed { expression, .. } => {
            inspect_expression(expression, used, config, bag, discarded)
        }
        Expression::Variable(name) => {
            used.insert(name.clone());
        }

        Expression::Task { value, .. } if discarded => {
            emit(
                bag,
                config,
                "discarded-task",
                "task is created and immediately discarded",
                Some("bind the task and await it, or document intentional detached execution"),
            );
            inspect_expression(value, used, config, bag, false);
        }

        Expression::Send { value, channel } if discarded => {
            emit(
                bag,
                config,
                "discarded-send",
                "asynchronous channel send result is discarded",
                Some("await the send task when delivery ordering or failure matters"),
            );
            inspect_expression(value, used, config, bag, false);
            inspect_expression(channel, used, config, bag, false);
        }

        Expression::Ownership {
            op: OwnershipOp::Clone,
            value,
        } => {
            if matches!(
                value.as_ref(),
                Expression::Integer(_) | Expression::Float(_) | Expression::Boolean(_)
            ) {
                emit(
                    bag,
                    config,
                    "unnecessary-clone",
                    "clone of a scalar value is unnecessary",
                    Some("use the scalar directly"),
                );
            }
            inspect_expression(value, used, config, bag, false);
        }

        Expression::List(values)
        | Expression::Tuple(values)
        | Expression::Set(values)
        | Expression::PrintArgs(values)
        | Expression::Construct { args: values, .. }
        | Expression::Variant { fields: values, .. } => {
            for value in values {
                inspect_expression(value, used, config, bag, false);
            }
        }

        Expression::Map(entries) => {
            for (key, value) in entries {
                inspect_expression(key, used, config, bag, false);
                inspect_expression(value, used, config, bag, false);
            }
        }

        Expression::Index { object, index } => {
            inspect_expression(object, used, config, bag, false);
            inspect_expression(index, used, config, bag, false);
        }

        Expression::Slice {
            object,
            start,
            end,
            step,
        } => {
            inspect_expression(object, used, config, bag, false);
            for bound in [start, end, step].into_iter().flatten() {
                inspect_expression(bound, used, config, bag, false);
            }
        }

        Expression::Lambda { body, .. }
        | Expression::Ownership { value: body, .. }
        | Expression::Task { value: body, .. }
        | Expression::Await(body)
        | Expression::Channel(body)
        | Expression::ChaosRule { value: body, .. }
        | Expression::FusedPipeline { input: body, .. }
        | Expression::Unary {
            expression: body, ..
        }
        | Expression::Member { object: body, .. } => {
            inspect_expression(body, used, config, bag, false)
        }

        Expression::MethodCall { object, args, .. } => {
            inspect_expression(object, used, config, bag, false);
            for argument in args {
                inspect_expression(argument, used, config, bag, false);
            }
        }

        Expression::Send { value, channel } => {
            inspect_expression(value, used, config, bag, false);
            inspect_expression(channel, used, config, bag, false);
        }

        Expression::Conditional {
            condition,
            then_expression,
            else_expression,
        } => {
            inspect_expression(condition, used, config, bag, false);
            inspect_expression(then_expression, used, config, bag, false);
            inspect_expression(else_expression, used, config, bag, false);
        }

        Expression::Binary { left, right, .. } => {
            inspect_expression(left, used, config, bag, false);
            inspect_expression(right, used, config, bag, false);
        }

        Expression::Call { args, .. } => {
            for argument in args {
                inspect_expression(argument, used, config, bag, false);
            }
        }

        Expression::CallValue { callee, args, .. } => {
            inspect_expression(callee, used, config, bag, false);
            for argument in args {
                inspect_expression(argument, used, config, bag, false);
            }
        }

        Expression::ListComprehension { element, clauses }
        | Expression::SetComprehension { element, clauses } => {
            inspect_expression(element, used, config, bag, false);
            for clause in clauses {
                inspect_expression(&clause.iterable, used, config, bag, false);
                if let Some(condition) = &clause.condition {
                    inspect_expression(condition, used, config, bag, false);
                }
            }
        }

        Expression::MapComprehension {
            key,
            value,
            clauses,
        } => {
            inspect_expression(key, used, config, bag, false);
            inspect_expression(value, used, config, bag, false);
            for clause in clauses {
                inspect_expression(&clause.iterable, used, config, bag, false);
                if let Some(condition) = &clause.condition {
                    inspect_expression(condition, used, config, bag, false);
                }
            }
        }

        Expression::Format { args, .. } => {
            for argument in args {
                inspect_expression(argument, used, config, bag, false);
            }
        }

        Expression::Integer(_)
        | Expression::Float(_)
        | Expression::Boolean(_)
        | Expression::String(_)
        | Expression::Function(_) => {}
    }
}

fn emit(
    bag: &mut DiagnosticBag,
    config: &LintConfig,
    lint: &str,
    message: impl Into<String>,
    help: Option<&str>,
) {
    let level = config.level(lint);
    if level == LintLevel::Allow {
        return;
    }

    let code = match lint {
        "unused-binding" => "W001",
        "discarded-task" => "W002",
        other => return emit_named(bag, level, format!("lint::{other}"), message, help),
    };
    emit_named(bag, level, code.to_owned(), message, help);
}

fn emit_named(
    bag: &mut DiagnosticBag,
    level: LintLevel,
    code: String,
    message: impl Into<String>,
    help: Option<&str>,
) {
    let mut diagnostic = Diagnostic::warning(code.as_str(), message);
    diagnostic.severity = match level {
        LintLevel::Allow => Severity::Allow,
        LintLevel::Warn => Severity::Warning,
        LintLevel::Deny => Severity::Error,
    };
    if let Some(help) = help {
        diagnostic.help = Some(help.to_owned());
    }
    bag.push(diagnostic);
}
