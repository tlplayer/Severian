use severian_hir::{Expression, Function, Instruction};
use std::collections::HashMap;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MultiOutputFusionCandidate {
    pub producer_index: usize,
    pub producer_binding: String,
    pub consumer_indices: Vec<usize>,
}

pub fn find_multi_output_candidates(function: &Function) -> Vec<MultiOutputFusionCandidate> {
    let mut definitions = HashMap::<String, usize>::new();
    let mut uses = HashMap::<String, Vec<usize>>::new();

    for (index, instruction) in function.instructions.iter().enumerate() {
        if let Instruction::Let { name, .. } | Instruction::TryLet { name, .. } = instruction {
            definitions.insert(name.clone(), index);
        }

        collect_top_level_uses(instruction, &mut |name| {
            uses.entry(name.to_string()).or_default().push(index);
        });
    }

    let mut result = Vec::new();

    for (binding, producer_index) in definitions {
        let Some(consumers) = uses.get(&binding) else {
            continue;
        };

        let mut unique = consumers.clone();
        unique.sort_unstable();
        unique.dedup();

        if unique.len() >= 2 {
            result.push(MultiOutputFusionCandidate {
                producer_index,
                producer_binding: binding,
                consumer_indices: unique,
            });
        }
    }

    result
}

fn collect_top_level_uses(instruction: &Instruction, visitor: &mut impl FnMut(&str)) {
    match instruction {
        Instruction::Let { value, .. }
        | Instruction::TryLet { value, .. }
        | Instruction::Print(value)
        | Instruction::Assert(value)
        | Instruction::Evaluate(value) => collect_expression_uses(value, visitor),

        Instruction::Assign { target, value, .. } => {
            collect_expression_uses(target, visitor);
            collect_expression_uses(value, visitor);
        }

        Instruction::Return(value) => {
            if let Some(value) = value {
                collect_expression_uses(value, visitor);
            }
        }

        // Nested control flow is intentionally not attributed to the outer
        // instruction index as a fusion consumer in this first planner.
        _ => {}
    }
}

fn collect_expression_uses(expression: &Expression, visitor: &mut impl FnMut(&str)) {
    match expression {
        Expression::Variable(name) => visitor(name),

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
        } => collect_expression_uses(expression, visitor),

        Expression::Binary { left, right, .. } => {
            collect_expression_uses(left, visitor);
            collect_expression_uses(right, visitor);
        }

        Expression::Call { args, .. }
        | Expression::Format { args, .. }
        | Expression::Construct { args, .. }
        | Expression::PrintArgs(args)
        | Expression::Variant { fields: args, .. }
        | Expression::List(args)
        | Expression::Tuple(args)
        | Expression::Set(args) => {
            for arg in args {
                collect_expression_uses(arg, visitor);
            }
        }

        Expression::MethodCall { object, args, .. } => {
            collect_expression_uses(object, visitor);
            for arg in args {
                collect_expression_uses(arg, visitor);
            }
        }

        Expression::CallValue { callee, args, .. } => {
            collect_expression_uses(callee, visitor);
            for arg in args {
                collect_expression_uses(arg, visitor);
            }
        }

        Expression::Map(entries) => {
            for (key, value) in entries {
                collect_expression_uses(key, visitor);
                collect_expression_uses(value, visitor);
            }
        }

        Expression::Index { object, index } => {
            collect_expression_uses(object, visitor);
            collect_expression_uses(index, visitor);
        }

        Expression::Slice {
            object,
            start,
            end,
            step,
        } => {
            collect_expression_uses(object, visitor);
            for bound in [start, end, step].into_iter().flatten() {
                collect_expression_uses(bound, visitor);
            }
        }

        Expression::Send { value, channel } => {
            collect_expression_uses(value, visitor);
            collect_expression_uses(channel, visitor);
        }

        Expression::Conditional {
            condition,
            then_expression,
            else_expression,
        } => {
            collect_expression_uses(condition, visitor);
            collect_expression_uses(then_expression, visitor);
            collect_expression_uses(else_expression, visitor);
        }

        Expression::Lambda { body, .. } => collect_expression_uses(body, visitor),

        Expression::ListComprehension { element, clauses }
        | Expression::SetComprehension { element, clauses } => {
            for clause in clauses {
                collect_expression_uses(&clause.iterable, visitor);
                if let Some(condition) = &clause.condition {
                    collect_expression_uses(condition, visitor);
                }
            }
            collect_expression_uses(element, visitor);
        }

        Expression::MapComprehension {
            key,
            value,
            clauses,
        } => {
            for clause in clauses {
                collect_expression_uses(&clause.iterable, visitor);
                if let Some(condition) = &clause.condition {
                    collect_expression_uses(condition, visitor);
                }
            }
            collect_expression_uses(key, visitor);
            collect_expression_uses(value, visitor);
        }

        Expression::Integer(_)
        | Expression::Float(_)
        | Expression::Boolean(_)
        | Expression::String(_)
        | Expression::Function(_) => {}
    }
}
