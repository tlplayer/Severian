mod constant_fold;
mod simplify;

pub use constant_fold::{fold_constant_binary, fold_constant_unary};
pub use simplify::{simplify_expression, AlgebraicSimplification};
