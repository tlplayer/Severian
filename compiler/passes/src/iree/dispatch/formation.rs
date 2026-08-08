use super::fusion::DispatchFusion;
use severian_hir::{Expression, Function, Instruction};
use std::collections::{BTreeSet, HashMap};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct DispatchId(pub usize);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DispatchRoot {
    TensorCall,
    FusedPipeline,
    Comprehension,
    Elementwise,
    ReductionLike,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DispatchRegion {
    pub id: DispatchId,
    pub instruction_indices: Vec<usize>,
    pub root: DispatchRoot,
    pub inputs: Vec<String>,
    pub outputs: Vec<String>,
    pub fused_into: Option<DispatchId>,
}

#[derive(Debug, Clone, Default)]
pub struct DispatchPlan {
    pub regions: Vec<DispatchRegion>,
    pub fusions: Vec<DispatchFusion>,
}

impl DispatchPlan {
    pub fn region(&self, id: DispatchId) -> Option<&DispatchRegion> {
        self.regions.iter().find(|region| region.id == id)
    }

    pub fn mark_fused(&mut self, fusion: &DispatchFusion) {
        for producer in &fusion.producers {
            if let Some(region) = self.regions.iter_mut().find(|region| region.id == *producer) {
                region.fused_into = Some(fusion.consumer);
            }
        }
    }

    pub fn active_regions(&self) -> impl Iterator<Item = &DispatchRegion> {
        self.regions.iter().filter(|region| region.fused_into.is_none())
    }
}

pub fn analyze_dispatches(function: &Function) -> DispatchPlan {
    let mut definitions = HashMap::<String, usize>::new();

    for (index, instruction) in function.instructions.iter().enumerate() {
        if let Some(name) = defined_binding(instruction) {
            definitions.insert(name.to_string(), index);
        }
    }

    let mut regions = Vec::new();

    for (index, instruction) in function.instructions.iter().enumerate() {
        let Some((root, expression, output)) = dispatchable_instruction(instruction) else {
            continue;
        };

        let mut inputs = BTreeSet::new();
        collect_expression_uses(expression, &mut |name| {
            // Treat values defined before this instruction, plus parameters or
            // globals with no local definition, as dispatch inputs.
            match definitions.get(name) {
                Some(definition) if *definition == index => {}
                _ => {
                    inputs.insert(name.to_string());
                }
            }
        });

        let outputs = output
            .map(|name| vec![name.to_string()])
            .unwrap_or_default();

        regions.push(DispatchRegion {
            id: DispatchId(regions.len()),
            instruction_indices: vec![index],
            root,
            inputs: inputs.into_iter().collect(),
            outputs,
            fused_into: None,
        });
    }

    DispatchPlan {
        regions,
        fusions: Vec::new(),
    }
}

fn dispatchable_instruction(
    instruction: &Instruction,
) -> Option<(DispatchRoot, &Expression, Option<&str>)> {
    match instruction {
        Instruction::Let { name, value } | Instruction::TryLet { name, value } => {
            classify_expression(value).map(|root| (root, value, Some(name.as_str())))
        }
        Instruction::Assign { value, .. } => {
            classify_expression(value).map(|root| (root, value, None))
        }
        Instruction::Evaluate(value) => classify_expression(value).map(|root| (root, value, None)),
        _ => None,
    }
}

fn classify_expression(expression: &Expression) -> Option<DispatchRoot> {
    match expression {
        Expression::FusedPipeline { .. } => Some(DispatchRoot::FusedPipeline),

        Expression::ListComprehension { .. }
        | Expression::SetComprehension { .. }
        | Expression::MapComprehension { .. } => Some(DispatchRoot::Comprehension),

        Expression::Binary { .. } | Expression::Unary { .. } => Some(DispatchRoot::Elementwise),

        Expression::Call { function, .. } => {
            if reduction_like(function) {
                Some(DispatchRoot::ReductionLike)
            } else if tensor_like(function) {
                Some(DispatchRoot::TensorCall)
            } else {
                None
            }
        }

        Expression::MethodCall { method, .. } => {
            if reduction_like(method) {
                Some(DispatchRoot::ReductionLike)
            } else if tensor_like(method) {
                Some(DispatchRoot::TensorCall)
            } else {
                None
            }
        }

        _ => None,
    }
}

fn tensor_like(name: &str) -> bool {
    const TOKENS: &[&str] = &[
        "matmul",
        "dot",
        "conv",
        "tensor",
        "linear",
        "relu",
        "gelu",
        "softmax",
        "transpose",
        "reshape",
        "broadcast",
        "attention",
    ];

    let lowered = name.to_ascii_lowercase();
    TOKENS.iter().any(|token| lowered.contains(token))
}

fn reduction_like(name: &str) -> bool {
    const TOKENS: &[&str] = &[
        "sum",
        "mean",
        "max",
        "min",
        "reduce",
        "argmax",
        "argmin",
        "softmax",
    ];

    let lowered = name.to_ascii_lowercase();
    TOKENS.iter().any(|token| lowered.contains(token))
}

fn defined_binding(instruction: &Instruction) -> Option<&str> {
    match instruction {
        Instruction::Let { name, .. } | Instruction::TryLet { name, .. } => Some(name),
        _ => None,
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

        Expression::List(values)
        | Expression::Tuple(values)
        | Expression::Set(values)
        | Expression::PrintArgs(values)
        | Expression::Construct { args: values, .. }
        | Expression::Variant { fields: values, .. }
        | Expression::Format { args: values, .. }
        | Expression::Call { args: values, .. } => {
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

        Expression::Conditional {
            condition,
            then_expression,
            else_expression,
        } => {
            collect_expression_uses(condition, visitor);
            collect_expression_uses(then_expression, visitor);
            collect_expression_uses(else_expression, visitor);
        }

        Expression::CallValue { callee, args, .. } => {
            collect_expression_uses(callee, visitor);
            for arg in args {
                collect_expression_uses(arg, visitor);
            }
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
