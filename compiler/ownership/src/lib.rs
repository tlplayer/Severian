#![forbid(unsafe_code)]

use severian_hir::{Instruction, Program};

/// Checks the ownership invariants represented by the current HIR.
///
/// The hello-world slice contains only owned string literals passed to `print`,
/// so traversal is sufficient today. Later slices will add bindings, moves,
/// views, and borrows here without changing the driver pipeline.
pub fn check(program: &Program) -> Result<(), OwnershipError> {
    for function in &program.functions {
        for instruction in &function.instructions {
            match instruction {
                Instruction::Print(_) => {}
            }
        }
    }
    Ok(())
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OwnershipError {
    pub message: String,
}

impl std::fmt::Display for OwnershipError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(&self.message)
    }
}

impl std::error::Error for OwnershipError {}
