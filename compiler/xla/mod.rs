//! XLA-inspired optimization and planning infrastructure for Severian.
//!
//! These modules operate on Severian HIR today and are intended to feed the
//! StableHLO/MLIR lowering path later. Algebraic simplification mutates HIR;
//! fusion, layout, memory, and scheduling expose conservative plans that can be
//! consumed by lowerings without forcing XLA-specific state into the HIR.

pub mod algebraic;
pub mod fusion;
pub mod layout;
pub mod memory;
pub mod scheduling;

use severian_hir::Program;
use std::collections::HashMap;

#[derive(Debug, Clone, Default)]
pub struct FunctionOptimizationPlan {
    pub fusion: fusion::FusionPlan,
    pub layout: layout::LayoutPlan,
    pub memory: memory::BufferPlan,
    pub schedule: scheduling::Schedule,
}

#[derive(Debug, Clone, Default)]
pub struct XlaOptimizationPlan {
    pub functions: HashMap<String, FunctionOptimizationPlan>,
}

impl XlaOptimizationPlan {
    pub fn analyze(program: &Program) -> Self {
        let mut plan = Self::default();

        for function in &program.functions {
            plan.functions.insert(
                function.name.clone(),
                FunctionOptimizationPlan {
                    fusion: fusion::analyze_function(function),
                    layout: layout::assign_function_layouts(function),
                    memory: memory::assign_function_buffers(function),
                    schedule: scheduling::schedule_function(function),
                },
            );
        }

        for class in &program.classes {
            for function in class.methods.iter().chain(&class.constructors) {
                let name = format!("{}::{}", class.name, function.name);
                plan.functions.insert(
                    name,
                    FunctionOptimizationPlan {
                        fusion: fusion::analyze_function(function),
                        layout: layout::assign_function_layouts(function),
                        memory: memory::assign_function_buffers(function),
                        schedule: scheduling::schedule_function(function),
                    },
                );
            }
        }

        plan
    }
}
