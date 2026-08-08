use crate::{Pass, PassError};
use severian_hir::{Expression, Function, Instruction, Program};

/// Performs conservative structured-loop cleanup before lowering.
///
/// This pass is intentionally HIR-level. Tiling, vectorization, LICM, induction
/// variables, and affine transforms belong in later MLIR-oriented passes.
#[derive(Debug, Default, Clone, Copy)]
pub struct LoopSimplification;

impl Pass for LoopSimplification {
    fn name(&self) -> &'static str {
        "loop-simplification"
    }

    fn run(&self, program: &mut Program) -> Result<(), PassError> {
        for function in &mut program.functions {
            simplify_function(function);
        }

        for class in &mut program.classes {
            for function in class.methods.iter_mut().chain(&mut class.constructors) {
                simplify_function(function);
            }
        }

        Ok(())
    }
}

pub fn simplify_function(function: &mut Function) {
    simplify_block(&mut function.instructions);

    for test in &mut function.tests {
        simplify_block(&mut test.instructions);
    }
}

fn simplify_block(instructions: &mut Vec<Instruction>) {
    let old = std::mem::take(instructions);
    let mut result = Vec::with_capacity(old.len());

    for mut instruction in old {
        simplify_nested(&mut instruction);

        match instruction {
            Instruction::While {
                setup,
                capabilities,
                condition: Expression::Boolean(false),
                instructions: _,
            } => {
                if let Some(setup) = setup {
                    result.push(*setup);
                }

                // Capabilities are expressions and may carry calls/effects.
                for capability in capabilities {
                    result.push(Instruction::Evaluate(capability));
                }
            }
            Instruction::For {
                setup,
                pattern: _,
                iterable,
                instructions: _,
            } if is_empty_literal(&iterable) => {
                if let Some(setup) = setup {
                    result.push(*setup);
                }
                // Empty literal iterable itself has no effect.
            }
            Instruction::While {
                setup,
                capabilities,
                condition,
                instructions,
            } if instructions.is_empty() && capabilities.is_empty() => {
                if let Some(setup) = setup {
                    result.push(*setup);
                }

                // Preserve the condition evaluation. Do not keep an empty
                // potentially-infinite loop at HIR level.
                result.push(Instruction::Evaluate(condition));
            }
            Instruction::For {
                setup,
                pattern: _,
                iterable,
                instructions,
            } if instructions.is_empty() => {
                if let Some(setup) = setup {
                    result.push(*setup);
                }

                // Iterating may evaluate an arbitrary expression. Preserve that
                // evaluation even though the loop body is empty.
                result.push(Instruction::Evaluate(iterable));
            }
            instruction => result.push(instruction),
        }
    }

    *instructions = result;
}

fn simplify_nested(instruction: &mut Instruction) {
    match instruction {
        Instruction::While {
            setup,
            instructions,
            ..
        }
        | Instruction::For {
            setup,
            instructions,
            ..
        } => {
            if let Some(setup) = setup {
                simplify_nested(setup);
            }
            simplify_block(instructions);
            remove_redundant_continue(instructions);
        }
        Instruction::If {
            then_instructions,
            else_instructions,
            ..
        } => {
            simplify_block(then_instructions);
            simplify_block(else_instructions);
        }
        Instruction::Switch { arms, .. } | Instruction::ChannelSwitch { arms, .. } => {
            for arm in arms {
                simplify_block(&mut arm.instructions);
            }
        }
        Instruction::With { instructions, .. } => simplify_block(instructions),
        Instruction::Let { .. }
        | Instruction::TryLet { .. }
        | Instruction::Assign { .. }
        | Instruction::Print(_)
        | Instruction::Assert(_)
        | Instruction::Return(_)
        | Instruction::Break
        | Instruction::Continue
        | Instruction::Evaluate(_) => {}
    }
}

fn remove_redundant_continue(instructions: &mut Vec<Instruction>) {
    if matches!(instructions.last(), Some(Instruction::Continue)) {
        instructions.pop();
    }
}

fn is_empty_literal(expression: &Expression) -> bool {
    matches!(
        expression,
        Expression::List(values)
            | Expression::Tuple(values)
            | Expression::Set(values)
            if values.is_empty()
    ) || matches!(expression, Expression::Map(entries) if entries.is_empty())
}

