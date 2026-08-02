#![forbid(unsafe_code)]

use severian_hir::{Expression, Program};
use severian_package::{FusionAlias, FusionRule};
use std::collections::HashMap;
use std::fmt;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PassError {
    pub pass: &'static str,
    pub message: String,
}

impl fmt::Display for PassError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{} pass failed: {}", self.pass, self.message)
    }
}

impl std::error::Error for PassError {}

pub trait Pass {
    fn name(&self) -> &'static str;
    fn run(&self, program: &mut Program) -> Result<(), PassError>;
}

#[derive(Default)]
pub struct PassManager {
    passes: Vec<Box<dyn Pass>>,
}

impl PassManager {
    pub fn add(&mut self, pass: impl Pass + 'static) {
        self.passes.push(Box::new(pass));
    }

    pub fn run(&self, program: &mut Program) -> Result<(), PassError> {
        for pass in &self.passes {
            pass.run(program).map_err(|mut error| {
                error.pass = pass.name();
                error
            })?;
        }
        Ok(())
    }
}

pub fn standard_pipeline(
    rules: impl IntoIterator<Item = FusionRule>,
    aliases: impl IntoIterator<Item = FusionAlias>,
) -> PassManager {
    let mut pipeline = PassManager::default();
    let fusion = ElementwiseFusion::new(rules, aliases);
    if !fusion.rules.is_empty() || !fusion.configuration_errors.is_empty() {
        pipeline.add(fusion);
    }
    pipeline
}

struct ElementwiseFusion {
    rules: HashMap<String, FusionRule>,
    configuration_errors: Vec<String>,
}

impl ElementwiseFusion {
    fn new(
        rules: impl IntoIterator<Item = FusionRule>,
        aliases: impl IntoIterator<Item = FusionAlias>,
    ) -> Self {
        let mut rules_by_name = HashMap::new();
        let mut configuration_errors = Vec::new();
        for rule in rules {
            if let Some(previous) = rules_by_name.insert(rule.function.clone(), rule.clone()) {
                if previous != rule {
                    configuration_errors.push(format!(
                        "conflicting fusion contracts for `{}`",
                        rule.function
                    ));
                }
            }
        }
        for alias in aliases {
            if let Some(target) = rules_by_name.get(&alias.target).cloned() {
                rules_by_name.insert(
                    alias.function.clone(),
                    FusionRule {
                        function: alias.function,
                        ..target
                    },
                );
            } else {
                configuration_errors.push(format!(
                    "fusion alias `{}` targets unknown operation `{}`",
                    alias.function, alias.target
                ));
            }
        }
        Self {
            rules: rules_by_name,
            configuration_errors,
        }
    }

    fn rewrite(&self, expression: &mut Expression) {
        let Expression::Call { function, args } = expression else {
            return;
        };
        let Some(outer) = self.rules.get(function) else {
            return;
        };
        if args.len() != 1 {
            return;
        }

        let replacement = match &args[0] {
            Expression::FusedPipeline {
                input,
                runtime_symbol,
                operations,
                packing_bits,
            } if runtime_symbol == &outer.runtime_symbol
                && *packing_bits == outer.packing_bits
                && operations.len() < outer.max_chain =>
            {
                let mut operations = operations.clone();
                operations.push(outer.opcode);
                Some((input.as_ref().clone(), operations))
            }
            Expression::Call {
                function: inner,
                args: inner_args,
            } if inner_args.len() == 1 => self.rules.get(inner).and_then(|inner| {
                (inner.runtime_symbol == outer.runtime_symbol
                    && inner.packing_bits == outer.packing_bits
                    && outer.max_chain >= 2)
                    .then(|| (inner_args[0].clone(), vec![inner.opcode, outer.opcode]))
            }),
            _ => None,
        };

        if let Some((input, operations)) = replacement {
            *expression = Expression::FusedPipeline {
                input: Box::new(input),
                runtime_symbol: outer.runtime_symbol.clone(),
                operations,
                packing_bits: outer.packing_bits,
            };
        }
    }
}

impl Pass for ElementwiseFusion {
    fn name(&self) -> &'static str {
        "elementwise-fusion"
    }

    fn run(&self, program: &mut Program) -> Result<(), PassError> {
        if !self.configuration_errors.is_empty() {
            return Err(PassError {
                pass: self.name(),
                message: self.configuration_errors.join("; "),
            });
        }
        program.visit_expressions_mut(&mut |expression| self.rewrite(expression));
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use severian_hir::{Function, ValueType};

    #[test]
    fn fusion_is_driven_by_package_rules_not_function_names() {
        let mut program = Program {
            globals: Vec::new(),
            classes: Vec::new(),
            functions: vec![Function {
                name: "forward".into(),
                native_symbol: None,
                decorators: Vec::new(),
                contract: None,
                params: Vec::new(),
                return_type: ValueType::List,
                instructions: vec![severian_hir::Instruction::Return(Some(Expression::Call {
                    function: "custom.curve".into(),
                    args: vec![Expression::Call {
                        function: "custom.clip".into(),
                        args: vec![Expression::Variable("X".into())],
                    }],
                }))],
                tests: Vec::new(),
            }],
        };
        let rules = [
            FusionRule {
                function: "custom.clip".into(),
                runtime_symbol: "__custom_pipeline".into(),
                opcode: 7,
                packing_bits: 4,
                max_chain: 8,
            },
            FusionRule {
                function: "custom.curve".into(),
                runtime_symbol: "__custom_pipeline".into(),
                opcode: 9,
                packing_bits: 4,
                max_chain: 8,
            },
        ];

        standard_pipeline(rules, []).run(&mut program).unwrap();

        let severian_hir::Instruction::Return(Some(Expression::FusedPipeline {
            runtime_symbol,
            operations,
            ..
        })) = &program.functions[0].instructions[0]
        else {
            panic!("expected a fused package pipeline");
        };
        assert_eq!(runtime_symbol, "__custom_pipeline");
        assert_eq!(operations, &[7, 9]);
    }

    #[test]
    fn rejects_an_alias_to_an_unknown_package_operation() {
        let mut program = Program {
            globals: Vec::new(),
            classes: Vec::new(),
            functions: Vec::new(),
        };
        let error = standard_pipeline(
            [],
            [FusionAlias {
                function: "models.Relu".into(),
                target: "tensor.missing".into(),
            }],
        )
        .run(&mut program)
        .unwrap_err();

        assert!(error.message.contains("tensor.missing"));
    }
}
