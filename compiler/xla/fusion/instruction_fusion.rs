use crate::{Pass, PassError};
use severian_hir::{Expression, Function, Instruction, Program};
use std::collections::HashMap;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FusionKind {
    ProducerConsumer,
    ElementwiseChain,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FusionCandidate {
    pub producer_index: usize,
    pub consumer_index: usize,
    pub producer_binding: String,
    pub kind: FusionKind,
}

#[derive(Debug, Clone, Copy)]
pub struct InstructionFusion {
    pub max_distance: usize,
}

impl Default for InstructionFusion {
    fn default() -> Self {
        Self { max_distance: 8 }
    }
}

impl Pass for InstructionFusion {
    fn name(&self) -> &'static str {
        "xla-instruction-fusion-analysis"
    }

    fn run(&self, program: &mut Program) -> Result<(), PassError> {
        // HIR has no generic fused-op carrier. Discovery is intentionally kept
        // as analysis; StableHLO/MLIR lowering can consume these candidates.
        for function in &program.functions {
            let _ = find_instruction_fusion_candidates_with_distance(
                function,
                self.max_distance,
            );
        }
        Ok(())
    }
}

pub fn find_instruction_fusion_candidates(function: &Function) -> Vec<FusionCandidate> {
    find_instruction_fusion_candidates_with_distance(function, usize::MAX)
}

fn find_instruction_fusion_candidates_with_distance(
    function: &Function,
    max_distance: usize,
) -> Vec<FusionCandidate> {
    let mut definitions: HashMap<String, usize> = HashMap::new();
    let mut uses: HashMap<String, Vec<usize>> = HashMap::new();

    for (index, instruction) in function.instructions.iter().enumerate() {
        if let Some((name, _)) = binding_definition(instruction) {
            definitions.insert(name.to_string(), index);
        }

        collect_instruction_uses(instruction, &mut |name| {
            uses.entry(name.to_string()).or_default().push(index);
        });
    }

    let mut candidates = Vec::new();

    for (binding, producer_index) in definitions {
        let Some(binding_uses) = uses.get(&binding) else {
            continue;
        };

        if binding_uses.len() != 1 {
            continue;
        }

        let consumer_index = binding_uses[0];
        if consumer_index <= producer_index
            || consumer_index.saturating_sub(producer_index) > max_distance
        {
            continue;
        }

        let Some((_, producer_expression)) =
            binding_definition(&function.instructions[producer_index])
        else {
            continue;
        };

        if !is_fusible_expression(producer_expression) {
            continue;
        }

        let kind = if is_elementwise_expression(producer_expression) {
            FusionKind::ElementwiseChain
        } else {
            FusionKind::ProducerConsumer
        };

        candidates.push(FusionCandidate {
            producer_index,
            consumer_index,
            producer_binding: binding,
            kind,
        });
    }

    candidates
}

fn binding_definition(instruction: &Instruction) -> Option<(&str, &Expression)> {
    match instruction {
        Instruction::Let { name, value } | Instruction::TryLet { name, value } => {
            Some((name, value))
        }
        _ => None,
    }
}

fn is_fusible_expression(expression: &Expression) -> bool {
    matches!(
        expression,
        Expression::Unary { .. }
            | Expression::Binary { .. }
            | Expression::Index { .. }
            | Expression::Slice { .. }
            | Expression::Call { .. }
            | Expression::MethodCall { .. }
            | Expression::FusedPipeline { .. }
    )
}

fn is_elementwise_expression(expression: &Expression) -> bool {
    matches!(
        expression,
        Expression::Unary { .. } | Expression::Binary { .. } | Expression::FusedPipeline { .. }
    )
}

fn collect_instruction_uses(instruction: &Instruction, visitor: &mut impl FnMut(&str)) {
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

        Instruction::If {
            condition,
            then_instructions,
            else_instructions,
        } => {
            collect_expression_uses(condition, visitor);
            for instruction in then_instructions.iter().chain(else_instructions) {
                collect_instruction_uses(instruction, visitor);
            }
        }

        Instruction::While {
            setup,
            capabilities,
            condition,
            instructions,
        } => {
            if let Some(setup) = setup {
                collect_instruction_uses(setup, visitor);
            }
            for capability in capabilities {
                collect_expression_uses(capability, visitor);
            }
            collect_expression_uses(condition, visitor);
            for instruction in instructions {
                collect_instruction_uses(instruction, visitor);
            }
        }

        Instruction::For {
            setup,
            iterable,
            instructions,
            ..
        } => {
            if let Some(setup) = setup {
                collect_instruction_uses(setup, visitor);
            }
            collect_expression_uses(iterable, visitor);
            for instruction in instructions {
                collect_instruction_uses(instruction, visitor);
            }
        }

        Instruction::Switch { value, arms } => {
            collect_expression_uses(value, visitor);
            for arm in arms {
                if let Some(source) = &arm.source {
                    collect_expression_uses(source, visitor);
                }
                if let Some(guard) = &arm.guard {
                    collect_expression_uses(guard, visitor);
                }
                for instruction in &arm.instructions {
                    collect_instruction_uses(instruction, visitor);
                }
            }
        }

        Instruction::ChannelSwitch {
            channels,
            setup,
            repeat_condition,
            arms,
        } => {
            for channel in channels {
                collect_expression_uses(channel, visitor);
            }
            if let Some(setup) = setup {
                collect_instruction_uses(setup, visitor);
            }
            if let Some(condition) = repeat_condition {
                collect_expression_uses(condition, visitor);
            }
            for arm in arms {
                if let Some(source) = &arm.source {
                    collect_expression_uses(source, visitor);
                }
                if let Some(guard) = &arm.guard {
                    collect_expression_uses(guard, visitor);
                }
                for instruction in &arm.instructions {
                    collect_instruction_uses(instruction, visitor);
                }
            }
        }

        Instruction::With {
            resources,
            instructions,
            ..
        } => {
            for resource in resources {
                collect_expression_uses(resource, visitor);
            }
            for instruction in instructions {
                collect_instruction_uses(instruction, visitor);
            }
        }

        Instruction::Break | Instruction::Continue => {}
    }
}

fn collect_expression_uses(expression: &Expression, visitor: &mut impl FnMut(&str)) {
    match expression {
        Expression::Variable(name) => visitor(name),

        Expression::Lambda { body, .. }
        | Expression::Ownership { value: body, .. }
        | Expression::Member { object: body, .. }
        | Expression::Await(body)
        | Expression::Channel(body)
        | Expression::Task { value: body, .. }
        | Expression::ChaosRule { value: body, .. }
        | Expression::FusedPipeline { input: body, .. }
        | Expression::Unary {
            expression: body, ..
        } => collect_expression_uses(body, visitor),

        Expression::List(values)
        | Expression::Tuple(values)
        | Expression::Set(values)
        | Expression::PrintArgs(values)
        | Expression::Construct { args: values, .. }
        | Expression::Variant { fields: values, .. } => {
            for value in values {
                collect_expression_uses(value, visitor);
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

        Expression::Format { args, .. } | Expression::Call { args, .. } => {
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

        Expression::Send { value, channel } => {
            collect_expression_uses(value, visitor);
            collect_expression_uses(channel, visitor);
        }

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

        Expression::Conditional {
            condition,
            then_expression,
            else_expression,
        } => {
            collect_expression_uses(condition, visitor);
            collect_expression_uses(then_expression, visitor);
            collect_expression_uses(else_expression, visitor);
        }

        Expression::Binary { left, right, .. } => {
            collect_expression_uses(left, visitor);
            collect_expression_uses(right, visitor);
        }

        Expression::CallValue { callee, args, .. } => {
            collect_expression_uses(callee, visitor);
            for arg in args {
                collect_expression_uses(arg, visitor);
            }
        }

        Expression::Integer(_)
        | Expression::Float(_)
        | Expression::Boolean(_)
        | Expression::String(_)
        | Expression::Function(_) => {}
    }
}
