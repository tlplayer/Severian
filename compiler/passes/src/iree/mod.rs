//! IREE-inspired planning passes for Severian.
//!
//! These passes keep dispatch formation, tiling, vectorization, and
//! bufferization decisions separate from the final MLIR lowering. The HIR
//! remains language-oriented while lowering can consume the generated plans.

pub mod bufferization;
pub mod dispatch;
pub mod tiling;
pub mod vectorization;

use severian_hir::{Function, Program};
use std::collections::HashMap;

#[derive(Debug, Clone, Default)]
pub struct FunctionIreePlan {
    pub dispatch: dispatch::DispatchPlan,
    pub tiling: tiling::TilingPlan,
    pub vectorization: vectorization::VectorizationPlan,
    pub bufferization: bufferization::BufferizationPlan,
}

#[derive(Debug, Clone, Default)]
pub struct IreePlan {
    pub functions: HashMap<String, FunctionIreePlan>,
}

impl IreePlan {
    pub fn analyze(program: &Program) -> Self {
        let mut result = Self::default();

        for function in &program.functions {
            result
                .functions
                .insert(function.name.clone(), analyze_function(function));
        }

        for class in &program.classes {
            for function in class.methods.iter().chain(&class.constructors) {
                result.functions.insert(
                    format!("{}::{}", class.name, function.name),
                    analyze_function(function),
                );
            }
        }

        result
    }
}

pub fn analyze_function(function: &Function) -> FunctionIreePlan {
    let dispatch = dispatch::analyze_function(function);
    let tiling = tiling::analyze_function(function, &dispatch);
    let vectorization = vectorization::analyze_function(function, &tiling);
    let bufferization = bufferization::analyze_function(function, &dispatch);

    FunctionIreePlan {
        dispatch,
        tiling,
        vectorization,
        bufferization,
    }
}
