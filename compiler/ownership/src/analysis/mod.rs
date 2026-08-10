//! Supplemental ownership analyses.
//!
//! These analyses are intentionally separate from the primary ownership checker.
//! They provide reusable compiler facts for diagnostics, optimization, lowering,
//! and later interprocedural passes.

pub mod alias;
pub mod borrow_check;
pub mod escape;
pub mod liveness;
pub mod move_check;

use std::collections::HashMap;

#[derive(Debug, Clone)]
pub struct OwnershipAnalysis {
    pub aliases: HashMap<String, alias::AliasAnalysis>,
    pub borrows: HashMap<String, borrow_check::BorrowReport>,
    pub escapes: escape::ProgramEscapeAnalysis,
    pub liveness: HashMap<String, liveness::FunctionLiveness>,
    pub moves: HashMap<String, move_check::MoveReport>,
}

pub fn analyze(program: &severian_hir::Program) -> OwnershipAnalysis {
    OwnershipAnalysis {
        aliases: alias::analyze(program),
        borrows: borrow_check::check(program),
        escapes: escape::analyze(program),
        liveness: liveness::analyze(program),
        moves: move_check::check(program),
    }
}
