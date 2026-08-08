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