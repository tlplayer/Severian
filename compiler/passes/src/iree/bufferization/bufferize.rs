use crate::iree::dispatch::DispatchPlan;
use severian_hir::{Expression, Function, Instruction, OwnershipOp};
use std::collections::{HashMap, HashSet};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AllocationKind {
    Borrowed,
    Stack,
    Heap,
    DispatchLocal,
    Alias,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BufferizationStrategy {
    InPlace,
    OutOfPlace,
    AliasInput,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BufferizedValue {
    pub name: String,
    pub allocation: AllocationKind,
    pub strategy: BufferizationStrategy,
    pub aliases: Option<String>,
    pub escapes_dispatch: bool,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct BufferizationPlan {
    pub values: Vec<BufferizedValue>,
    pub by_name: HashMap<String, usize>,
}

impl BufferizationPlan {
    pub fn value(&self, name: &str) -> Option<&BufferizedValue> {
        self.by_name
            .get(name)
            .and_then(|index| self.values.get(*index))
    }

    fn push(&mut self, value: BufferizedValue) {
        let index = self.values.len();
        self.by_name.insert(value.name.clone(), index);
        self.values.push(value);
    }
}

pub fn analyze_bufferization(
    function: &Function,
    dispatch: &DispatchPlan,
) -> BufferizationPlan {
    let dispatch_outputs = dispatch
        .regions
        .iter()
        .flat_map(|region| region.outputs.iter().cloned())
        .collect::<HashSet<_>>();

    let returned = returned_bindings(function);
    let mut plan = BufferizationPlan::default();

    for parameter in &function.params {
        plan.push(BufferizedValue {
            name: parameter.name.clone(),
            allocation: AllocationKind::Borrowed,
            strategy: BufferizationStrategy::AliasInput,
            aliases: None,
            escapes_dispatch: true,
        });
    }

    for instruction in &function.instructions {
        let Some((name, value)) = definition(instruction) else {
            continue;
        };

        let escapes_dispatch =
            returned.contains(name) || dispatch_outputs.contains(name);

        let (allocation, strategy, aliases) =
            classify_value(name, value, escapes_dispatch);

        plan.push(BufferizedValue {
            name: name.to_string(),
            allocation,
            strategy,
            aliases,
            escapes_dispatch,
        });
    }

    plan
}

fn classify_value(
    _name: &str,
    value: &Expression,
    escapes_dispatch: bool,
) -> (AllocationKind, BufferizationStrategy, Option<String>) {
    match value {
        Expression::Ownership {
            op: OwnershipOp::View,
            value,
        }
        | Expression::Ownership {
            op: OwnershipOp::Borrow,
            value,
        }
        | Expression::Ownership {
            op: OwnershipOp::AddressOf,
            value,
        } => {
            if let Expression::Variable(source) = value.as_ref() {
                (
                    AllocationKind::Alias,
                    BufferizationStrategy::AliasInput,
                    Some(source.clone()),
                )
            } else {
                (
                    AllocationKind::Alias,
                    BufferizationStrategy::AliasInput,
                    None,
                )
            }
        }

        Expression::Ownership {
            op: OwnershipOp::Move,
            value,
        } => {
            if let Expression::Variable(source) = value.as_ref() {
                (
                    AllocationKind::Alias,
                    BufferizationStrategy::InPlace,
                    Some(source.clone()),
                )
            } else {
                allocation_for_escape(escapes_dispatch)
            }
        }

        Expression::Ownership {
            op: OwnershipOp::Clone,
            ..
        } => (
            if escapes_dispatch {
                AllocationKind::Heap
            } else {
                AllocationKind::DispatchLocal
            },
            BufferizationStrategy::OutOfPlace,
            None,
        ),

        Expression::Variable(source) => (
            AllocationKind::Alias,
            BufferizationStrategy::AliasInput,
            Some(source.clone()),
        ),

        Expression::Binary { .. }
        | Expression::Unary { .. }
        | Expression::FusedPipeline { .. }
        | Expression::Call { .. }
        | Expression::MethodCall { .. }
        | Expression::ListComprehension { .. }
        | Expression::SetComprehension { .. }
        | Expression::MapComprehension { .. } => allocation_for_escape(escapes_dispatch),

        _ => (
            AllocationKind::Stack,
            BufferizationStrategy::OutOfPlace,
            None,
        ),
    }
}

fn allocation_for_escape(
    escapes_dispatch: bool,
) -> (AllocationKind, BufferizationStrategy, Option<String>) {
    if escapes_dispatch {
        (
            AllocationKind::Heap,
            BufferizationStrategy::OutOfPlace,
            None,
        )
    } else {
        (
            AllocationKind::DispatchLocal,
            BufferizationStrategy::InPlace,
            None,
        )
    }
}

fn definition(instruction: &Instruction) -> Option<(&str, &Expression)> {
    match instruction {
        Instruction::Let { name, value } | Instruction::TryLet { name, value } => {
            Some((name, value))
        }
        _ => None,
    }
}

fn returned_bindings(function: &Function) -> HashSet<String> {
    let mut result = HashSet::new();
    collect_returns(&function.instructions, &mut result);
    result
}

fn collect_returns(instructions: &[Instruction], result: &mut HashSet<String>) {
    for instruction in instructions {
        match instruction {
            Instruction::Return(Some(Expression::Variable(name))) => {
                result.insert(name.clone());
            }

            Instruction::If {
                then_instructions,
                else_instructions,
                ..
            } => {
                collect_returns(then_instructions, result);
                collect_returns(else_instructions, result);
            }

            Instruction::While { instructions, .. }
            | Instruction::For { instructions, .. }
            | Instruction::With { instructions, .. } => {
                collect_returns(instructions, result);
            }

            Instruction::Switch { arms, .. }
            | Instruction::ChannelSwitch { arms, .. } => {
                for arm in arms {
                    collect_returns(&arm.instructions, result);
                }
            }

            _ => {}
        }
    }
}
