#![forbid(unsafe_code)]

mod error;
mod model;
mod planner;
mod registry;

pub use error::CompileError;
pub use model::{
    CompileContext, CompilePlan, CompileRegion, EffectSet, PlanSegment, StandardRegion,
};
pub use planner::plan;
pub use registry::{CompileHandler, CompilerRegistry};

#[cfg(test)]
mod tests {
    use super::*;
    use severian_mir::{Module, Operation, Value, ValueId};
    use severian_mlir::{LoweredType, MlirArtifact};
    use severian_target::TargetSpec;
    use severian_universal::{
        BinaryOperator, CompilerId, IntegerWidth, LiteralValue, PrimitiveCategory,
        PrimitiveRepresentation, TypeContext, TypeContextBuilder, TypeId, UnaryOperator,
    };
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Arc;

    fn type_builder() -> (TypeContextBuilder, TypeId, TypeId, CompilerId) {
        let mut types = TypeContextBuilder::new();
        let standard = types.register_declaration("test.i32", "i32").unwrap();
        types
            .define_primitive(
                standard,
                PrimitiveCategory::Integer,
                PrimitiveRepresentation::Integer {
                    bits: IntegerWidth::Fixed(32),
                    signed: true,
                },
                false,
            )
            .unwrap();
        let special = types
            .register_declaration("test.ir_value", "TestIR")
            .unwrap();
        types
            .define_primitive(
                special,
                PrimitiveCategory::Integer,
                PrimitiveRepresentation::Integer {
                    bits: IntegerWidth::Fixed(32),
                    signed: true,
                },
                false,
            )
            .unwrap();
        let compiler_declaration = types
            .register_declaration("test.compiler", "TestCompiler")
            .unwrap();
        let compiler = types.compiler_id(compiler_declaration).unwrap();
        types.set_compile_route(special, compiler).unwrap();
        (types, standard, special, compiler)
    }

    fn types() -> (TypeContext, TypeId, TypeId, CompilerId) {
        let (types, standard, special, compiler) = type_builder();
        (types.build(), standard, special, compiler)
    }

    fn value(id: u32, type_id: TypeId) -> Value {
        Value {
            id: ValueId(id),
            type_id,
        }
    }

    struct Handler {
        calls: Arc<AtomicUsize>,
        invalid: bool,
    }

    impl CompileHandler for Handler {
        fn compile(
            &self,
            region: &CompileRegion,
            _: &CompileContext<'_>,
        ) -> Result<MlirArtifact, CompileError> {
            self.calls.fetch_add(1, Ordering::SeqCst);
            let parameters = (0..region.inputs.len())
                .map(|index| format!("%arg{index}: i32"))
                .collect::<Vec<_>>()
                .join(", ");
            let result_type = match region.outputs.len() {
                0 => String::new(),
                1 => " -> i32".into(),
                count => format!(" -> ({})", vec!["i32"; count].join(", ")),
            };
            let constants = (0..region.outputs.len())
                .map(|index| format!("    %result{index} = arith.constant 0 : i32"))
                .collect::<Vec<_>>()
                .join("\n");
            let return_values = (0..region.outputs.len())
                .map(|index| format!("%result{index}"))
                .collect::<Vec<_>>()
                .join(", ");
            let return_types = vec!["i32"; region.outputs.len()].join(", ");
            let terminator = if region.outputs.is_empty() {
                "    return".into()
            } else {
                format!("    return {return_values} : {return_types}")
            };
            Ok(MlirArtifact {
                module: if self.invalid {
                    "module { func.func @entry() {".into()
                } else {
                    format!(
                        "module {{\n  func.func @entry({parameters}){result_type} {{\n{constants}\n{terminator}\n  }}\n}}"
                    )
                },
                inputs: vec![
                    LoweredType::Integer {
                        bits: 32,
                        signed: true,
                    };
                    region.inputs.len()
                ],
                outputs: vec![
                    LoweredType::Integer {
                        bits: 32,
                        signed: true,
                    };
                    region.outputs.len()
                ],
            })
        }
    }

    #[test]
    fn standard_operations_never_invoke_a_handler() {
        let (types, standard, _, compiler) = types();
        let module = Module {
            values: vec![value(0, standard)],
            operations: vec![Operation::Constant {
                value: LiteralValue::Integer("1".into()),
                result: ValueId(0),
            }],
            bindings: vec![],
        };
        let plan = plan(&module, &types).unwrap();
        assert!(matches!(plan.segments[0], PlanSegment::Standard(_)));
        let calls = Arc::new(AtomicUsize::new(0));
        let mut registry = CompilerRegistry::new();
        registry
            .register(
                compiler,
                Handler {
                    calls: calls.clone(),
                    invalid: false,
                },
            )
            .unwrap();
        registry
            .compile(
                &plan,
                &CompileContext {
                    types: &types,
                    target: &TargetSpec::new("x86_64-unknown-linux"),
                },
            )
            .unwrap();
        assert_eq!(calls.load(Ordering::SeqCst), 0);
    }

    #[test]
    fn custom_operations_bypass_generic_operation_lowering() {
        let (types, _, special, _) = types();
        let module = Module {
            values: vec![value(0, special)],
            operations: vec![Operation::Constant {
                value: LiteralValue::Integer("1".into()),
                result: ValueId(0),
            }],
            bindings: vec![],
        };
        let resumed = plan(&module, &types).unwrap().resumed_mir();
        assert!(matches!(
            resumed.operations[0],
            Operation::CompiledRegionCall { .. }
        ));
    }

    #[test]
    fn mixed_program_produces_verified_artifacts_and_a_typed_resume_call() {
        let (types, standard, special, compiler) = types();
        let module = Module {
            values: vec![
                value(0, standard),
                value(1, special),
                value(2, standard),
                value(3, standard),
            ],
            operations: vec![
                Operation::Constant {
                    value: LiteralValue::Integer("2".into()),
                    result: ValueId(0),
                },
                Operation::Constant {
                    value: LiteralValue::Integer("1".into()),
                    result: ValueId(1),
                },
                Operation::Unary {
                    operator: UnaryOperator::Positive,
                    operand: ValueId(1),
                    result: ValueId(2),
                },
                Operation::Binary {
                    operator: BinaryOperator::Add,
                    left: ValueId(2),
                    right: ValueId(0),
                    result: ValueId(3),
                },
            ],
            bindings: vec![],
        };
        let target = TargetSpec::new("x86_64-unknown-linux");
        let plan = plan(&module, &types).unwrap();
        assert_eq!(plan.segments.len(), 3);
        let mut registry = CompilerRegistry::new();
        registry
            .register(
                compiler,
                Handler {
                    calls: Arc::new(AtomicUsize::new(0)),
                    invalid: false,
                },
            )
            .unwrap();
        let artifacts = registry
            .compile(
                &plan,
                &CompileContext {
                    types: &types,
                    target: &target,
                },
            )
            .unwrap();
        assert_eq!(artifacts.len(), 1);
        assert!(plan
            .resumed_mir()
            .operations
            .iter()
            .any(|operation| matches!(operation, Operation::CompiledRegionCall { .. })));
    }

    #[test]
    fn missing_handler_is_a_deterministic_error() {
        let (types, _, special, compiler) = types();
        let module = Module {
            values: vec![value(0, special)],
            operations: vec![Operation::Constant {
                value: LiteralValue::Integer("1".into()),
                result: ValueId(0),
            }],
            bindings: vec![],
        };
        let target = TargetSpec::new("x86_64-unknown-linux");
        let plan = plan(&module, &types).unwrap();
        assert_eq!(
            CompilerRegistry::new()
                .compile(
                    &plan,
                    &CompileContext {
                        types: &types,
                        target: &target,
                    },
                )
                .unwrap_err(),
            CompileError::MissingHandler(compiler)
        );
    }

    #[test]
    fn conflicting_compile_types_are_a_deterministic_error() {
        let (mut types, standard, special, _) = type_builder();
        let other = types.register_declaration("test.other", "OtherIR").unwrap();
        let other_compiler_declaration = types
            .register_declaration("test.other_compiler", "OtherCompiler")
            .unwrap();
        let other_compiler = types.compiler_id(other_compiler_declaration).unwrap();
        types.set_compile_route(other, other_compiler).unwrap();
        let types = types.build();
        let conflict = Module {
            values: vec![value(0, special), value(1, other), value(2, standard)],
            operations: vec![Operation::Binary {
                operator: BinaryOperator::Add,
                left: ValueId(0),
                right: ValueId(1),
                result: ValueId(2),
            }],
            bindings: vec![],
        };
        assert!(matches!(
            plan(&conflict, &types),
            Err(CompileError::ConflictingCompilers { .. })
        ));
    }

    #[test]
    fn invalid_handler_mlir_is_rejected() {
        let (types, _, special, compiler) = types();
        let module = Module {
            values: vec![value(0, special)],
            operations: vec![Operation::Constant {
                value: LiteralValue::Integer("1".into()),
                result: ValueId(0),
            }],
            bindings: vec![],
        };
        let plan = plan(&module, &types).unwrap();
        let target = TargetSpec::new("x86_64-unknown-linux");
        let mut registry = CompilerRegistry::new();
        registry
            .register(
                compiler,
                Handler {
                    calls: Arc::new(AtomicUsize::new(0)),
                    invalid: true,
                },
            )
            .unwrap();
        assert!(matches!(
            registry.compile(
                &plan,
                &CompileContext {
                    types: &types,
                    target: &target,
                },
            ),
            Err(CompileError::InvalidArtifact(_))
        ));
    }

    #[test]
    fn duplicate_handlers_are_rejected() {
        let (_, _, _, compiler) = types();
        let mut registry = CompilerRegistry::new();
        registry
            .register(
                compiler,
                Handler {
                    calls: Arc::new(AtomicUsize::new(0)),
                    invalid: false,
                },
            )
            .unwrap();
        assert!(matches!(
            registry.register(
                compiler,
                Handler {
                    calls: Arc::new(AtomicUsize::new(0)),
                    invalid: false,
                },
            ),
            Err(CompileError::DuplicateHandler(found)) if found == compiler
        ));
    }

    #[test]
    fn compiler_identity_is_stable_under_declaration_reordering() {
        let compiler_id = |unrelated_first: bool| {
            let mut types = TypeContextBuilder::new();
            if unrelated_first {
                types
                    .register_declaration("test.unrelated", "Unrelated")
                    .unwrap();
            }
            let compiler = types
                .register_declaration("test.compiler", "TestCompiler")
                .unwrap();
            if !unrelated_first {
                types
                    .register_declaration("test.unrelated", "Unrelated")
                    .unwrap();
            }
            types.compiler_id(compiler).unwrap()
        };
        assert_eq!(compiler_id(false), compiler_id(true));
    }
}
