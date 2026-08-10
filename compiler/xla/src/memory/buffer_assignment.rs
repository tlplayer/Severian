use super::buffer_reuse::apply_reuse;
use severian_hir::{Expression, Function, Instruction};
use std::collections::HashMap;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct BufferSlotId(pub usize);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ValueLifetime {
    pub definition: usize,
    pub last_use: usize,
}

impl ValueLifetime {
    pub fn overlaps(self, other: Self) -> bool {
        self.definition <= other.last_use && other.definition <= self.last_use
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BufferAssignment {
    pub value: String,
    pub buffer: BufferSlotId,
    pub lifetime: ValueLifetime,
    pub bytes: Option<usize>,
}

#[derive(Debug, Clone, Default)]
pub struct BufferPlan {
    pub assignments: Vec<BufferAssignment>,
    pub peak_buffers: usize,
    pub unique_buffers: usize,
}

pub fn assign_function_buffers(function: &Function) -> BufferPlan {
    let lifetimes = collect_lifetimes(function);
    let mut assignments = lifetimes
        .into_iter()
        .enumerate()
        .map(|(index, (value, lifetime))| BufferAssignment {
            value,
            buffer: BufferSlotId(index),
            lifetime,
            bytes: None,
        })
        .collect::<Vec<_>>();

    assignments.sort_by_key(|assignment| assignment.lifetime.definition);

    let decisions = apply_reuse(&assignments);
    for decision in decisions {
        if let Some(assignment) = assignments.get_mut(decision.assignment_index) {
            assignment.buffer = decision.reuse_buffer;
        }
    }

    let unique_buffers = assignments
        .iter()
        .map(|assignment| assignment.buffer)
        .collect::<std::collections::HashSet<_>>()
        .len();

    let peak_buffers = peak_live_buffers(&assignments);

    BufferPlan {
        assignments,
        peak_buffers,
        unique_buffers,
    }
}

fn collect_lifetimes(function: &Function) -> Vec<(String, ValueLifetime)> {
    let mut point = 0usize;
    let mut definitions = HashMap::<String, usize>::new();
    let mut last_uses = HashMap::<String, usize>::new();

    for parameter in &function.params {
        definitions.insert(parameter.name.clone(), point);
        last_uses.insert(parameter.name.clone(), point);
        point += 1;
    }

    walk_instructions(
        &function.instructions,
        &mut point,
        &mut definitions,
        &mut last_uses,
    );

    let mut result = definitions
        .into_iter()
        .map(|(name, definition)| {
            let last_use = last_uses.get(&name).copied().unwrap_or(definition);
            (
                name,
                ValueLifetime {
                    definition,
                    last_use,
                },
            )
        })
        .collect::<Vec<_>>();

    result.sort_by_key(|(_, lifetime)| lifetime.definition);
    result
}

fn walk_instructions(
    instructions: &[Instruction],
    point: &mut usize,
    definitions: &mut HashMap<String, usize>,
    last_uses: &mut HashMap<String, usize>,
) {
    for instruction in instructions {
        let current = *point;
        *point += 1;

        match instruction {
            Instruction::Let { name, value } | Instruction::TryLet { name, value } => {
                collect_expression_uses(value, current, last_uses);
                definitions.entry(name.clone()).or_insert(current);
                last_uses.entry(name.clone()).or_insert(current);
            }

            Instruction::Assign { target, value, .. } => {
                collect_expression_uses(target, current, last_uses);
                collect_expression_uses(value, current, last_uses);
            }

            Instruction::Print(value)
            | Instruction::Assert(value)
            | Instruction::Evaluate(value) => {
                collect_expression_uses(value, current, last_uses);
            }

            Instruction::Return(value) => {
                if let Some(value) = value {
                    collect_expression_uses(value, current, last_uses);
                }
            }

            Instruction::If {
                condition,
                then_instructions,
                else_instructions,
            } => {
                collect_expression_uses(condition, current, last_uses);
                walk_instructions(then_instructions, point, definitions, last_uses);
                walk_instructions(else_instructions, point, definitions, last_uses);
            }

            Instruction::While {
                setup,
                capabilities,
                condition,
                instructions,
            } => {
                if let Some(setup) = setup {
                    walk_instructions(
                        std::slice::from_ref(setup.as_ref()),
                        point,
                        definitions,
                        last_uses,
                    );
                }
                for capability in capabilities {
                    collect_expression_uses(capability, current, last_uses);
                }
                collect_expression_uses(condition, current, last_uses);
                walk_instructions(instructions, point, definitions, last_uses);
            }

            Instruction::For {
                setup,
                iterable,
                instructions,
                ..
            } => {
                if let Some(setup) = setup {
                    walk_instructions(
                        std::slice::from_ref(setup.as_ref()),
                        point,
                        definitions,
                        last_uses,
                    );
                }
                collect_expression_uses(iterable, current, last_uses);
                walk_instructions(instructions, point, definitions, last_uses);
            }

            Instruction::Switch { value, arms } => {
                collect_expression_uses(value, current, last_uses);
                for arm in arms {
                    if let Some(source) = &arm.source {
                        collect_expression_uses(source, current, last_uses);
                    }
                    if let Some(guard) = &arm.guard {
                        collect_expression_uses(guard, current, last_uses);
                    }
                    walk_instructions(&arm.instructions, point, definitions, last_uses);
                }
            }

            Instruction::ChannelSwitch {
                channels,
                setup,
                repeat_condition,
                arms,
            } => {
                for channel in channels {
                    collect_expression_uses(channel, current, last_uses);
                }
                if let Some(setup) = setup {
                    walk_instructions(
                        std::slice::from_ref(setup.as_ref()),
                        point,
                        definitions,
                        last_uses,
                    );
                }
                if let Some(condition) = repeat_condition {
                    collect_expression_uses(condition, current, last_uses);
                }
                for arm in arms {
                    if let Some(source) = &arm.source {
                        collect_expression_uses(source, current, last_uses);
                    }
                    if let Some(guard) = &arm.guard {
                        collect_expression_uses(guard, current, last_uses);
                    }
                    walk_instructions(&arm.instructions, point, definitions, last_uses);
                }
            }

            Instruction::With {
                resources,
                instructions,
                ..
            } => {
                for resource in resources {
                    collect_expression_uses(resource, current, last_uses);
                }
                walk_instructions(instructions, point, definitions, last_uses);
            }

            Instruction::Break | Instruction::Continue => {}
        }
    }
}

fn collect_expression_uses(
    expression: &Expression,
    point: usize,
    last_uses: &mut HashMap<String, usize>,
) {
    match expression {
        Expression::Variable(name) => {
            last_uses.insert(name.clone(), point);
        }

        Expression::Unary { expression, .. }
        | Expression::Ownership {
            value: expression, ..
        }
        | Expression::Member {
            object: expression, ..
        }
        | Expression::Await(expression)
        | Expression::Channel(expression)
        | Expression::Task {
            value: expression, ..
        }
        | Expression::ChaosRule {
            value: expression, ..
        }
        | Expression::FusedPipeline {
            input: expression, ..
        } => collect_expression_uses(expression, point, last_uses),

        Expression::Binary { left, right, .. } => {
            collect_expression_uses(left, point, last_uses);
            collect_expression_uses(right, point, last_uses);
        }

        Expression::List(values)
        | Expression::Tuple(values)
        | Expression::Set(values)
        | Expression::PrintArgs(values)
        | Expression::Construct { args: values, .. }
        | Expression::Variant { fields: values, .. }
        | Expression::Format { args: values, .. }
        | Expression::Call { args: values, .. } => {
            for value in values {
                collect_expression_uses(value, point, last_uses);
            }
        }

        Expression::Map(entries) => {
            for (key, value) in entries {
                collect_expression_uses(key, point, last_uses);
                collect_expression_uses(value, point, last_uses);
            }
        }

        Expression::Index { object, index } => {
            collect_expression_uses(object, point, last_uses);
            collect_expression_uses(index, point, last_uses);
        }

        Expression::Slice {
            object,
            start,
            end,
            step,
        } => {
            collect_expression_uses(object, point, last_uses);
            for bound in [start, end, step].into_iter().flatten() {
                collect_expression_uses(bound, point, last_uses);
            }
        }

        Expression::MethodCall { object, args, .. } => {
            collect_expression_uses(object, point, last_uses);
            for arg in args {
                collect_expression_uses(arg, point, last_uses);
            }
        }

        Expression::Send { value, channel } => {
            collect_expression_uses(value, point, last_uses);
            collect_expression_uses(channel, point, last_uses);
        }

        Expression::Conditional {
            condition,
            then_expression,
            else_expression,
        } => {
            collect_expression_uses(condition, point, last_uses);
            collect_expression_uses(then_expression, point, last_uses);
            collect_expression_uses(else_expression, point, last_uses);
        }

        Expression::CallValue { callee, args, .. } => {
            collect_expression_uses(callee, point, last_uses);
            for arg in args {
                collect_expression_uses(arg, point, last_uses);
            }
        }

        Expression::Lambda { body, .. } => {
            collect_expression_uses(body, point, last_uses);
        }

        Expression::ListComprehension { element, clauses }
        | Expression::SetComprehension { element, clauses } => {
            for clause in clauses {
                collect_expression_uses(&clause.iterable, point, last_uses);
                if let Some(condition) = &clause.condition {
                    collect_expression_uses(condition, point, last_uses);
                }
            }
            collect_expression_uses(element, point, last_uses);
        }

        Expression::MapComprehension {
            key,
            value,
            clauses,
        } => {
            for clause in clauses {
                collect_expression_uses(&clause.iterable, point, last_uses);
                if let Some(condition) = &clause.condition {
                    collect_expression_uses(condition, point, last_uses);
                }
            }
            collect_expression_uses(key, point, last_uses);
            collect_expression_uses(value, point, last_uses);
        }

        Expression::Integer(_)
        | Expression::Float(_)
        | Expression::Boolean(_)
        | Expression::String(_)
        | Expression::Function(_) => {}
    }
}

fn peak_live_buffers(assignments: &[BufferAssignment]) -> usize {
    let max_point = assignments
        .iter()
        .map(|assignment| assignment.lifetime.last_use)
        .max()
        .unwrap_or(0);

    (0..=max_point)
        .map(|point| {
            assignments
                .iter()
                .filter(|assignment| {
                    assignment.lifetime.definition <= point
                        && point <= assignment.lifetime.last_use
                })
                .map(|assignment| assignment.buffer)
                .collect::<std::collections::HashSet<_>>()
                .len()
        })
        .max()
        .unwrap_or(0)
}
