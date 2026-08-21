#![forbid(unsafe_code)]

pub mod config;
mod pipeline;

pub use pipeline::{check_file, compile_file, compile_source, CompileError, Compiler};

#[cfg(test)]
mod tests {
    use super::*;
    use severian_compile::{CompileContext, CompileHandler, CompileRegion};
    use severian_mir::{Module, Operation, Value, ValueId};
    use severian_mlir::{LoweredType, MlirArtifact};
    use severian_source::SourceFile;
    use severian_target::TargetSpec;
    use severian_universal::{
        BinaryOperator, CompilerId, IntegerWidth, LiteralValue, PrimitiveCategory,
        PrimitiveRepresentation, TypeContextBuilder, UniversalContext,
    };
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Arc;

    struct TestHandler {
        calls: Option<Arc<AtomicUsize>>,
    }

    impl CompileHandler for TestHandler {
        fn compile(
            &self,
            region: &CompileRegion,
            _: &CompileContext<'_>,
        ) -> Result<MlirArtifact, severian_compile::CompileError> {
            if let Some(calls) = &self.calls {
                calls.fetch_add(1, Ordering::SeqCst);
            }
            assert!(region.inputs.is_empty());
            assert_eq!(region.outputs.len(), 1);
            Ok(MlirArtifact {
                module: "module { func.func @handler_entry() -> i32 { %0 = arith.constant 7 : i32 return %0 : i32 } }".into(),
                inputs: vec![],
                outputs: vec![LoweredType::Integer {
                    bits: 32,
                    signed: true,
                }],
            })
        }
    }

    fn compile_type_context() -> (
        UniversalContext,
        severian_universal::TypeId,
        severian_universal::TypeId,
        CompilerId,
    ) {
        let mut types = TypeContextBuilder::new();
        let i32_type = types.register_declaration("test.i32", "i32").unwrap();
        types
            .define_primitive(
                i32_type,
                PrimitiveCategory::Integer,
                PrimitiveRepresentation::Integer {
                    bits: IntegerWidth::Fixed(32),
                    signed: true,
                },
                true,
            )
            .unwrap();
        let string_type = types.register_declaration("test.string", "string").unwrap();
        types
            .define_primitive(
                string_type,
                PrimitiveCategory::Text,
                PrimitiveRepresentation::String,
                true,
            )
            .unwrap();
        let unit_type = types.register_declaration("test.unit", "unit").unwrap();
        types
            .define_primitive(
                unit_type,
                PrimitiveCategory::Unit,
                PrimitiveRepresentation::Unit,
                true,
            )
            .unwrap();
        let arguments_type = types.register_declaration("test.args", "args").unwrap();
        types
            .define_primitive(
                arguments_type,
                PrimitiveCategory::Arguments,
                PrimitiveRepresentation::Arguments,
                false,
            )
            .unwrap();
        let custom = types
            .register_declaration("test.custom", "CustomValue")
            .unwrap();
        types
            .define_primitive(
                custom,
                PrimitiveCategory::Integer,
                PrimitiveRepresentation::Integer {
                    bits: IntegerWidth::Fixed(32),
                    signed: true,
                },
                false,
            )
            .unwrap();
        let compiler = types
            .register_declaration("test.compiler", "TestCompiler")
            .unwrap();
        let compiler = types.compiler_id(compiler).unwrap();
        types.set_compile_route(custom, compiler).unwrap();
        (
            UniversalContext::new(types.build()),
            i32_type,
            custom,
            compiler,
        )
    }

    #[test]
    fn compiles_symmetric_i32_additions_to_a_runnable_executable() {
        let source =
            SourceFile::virtual_source("addition.sev", "x: i32 = 10\na = x + 1\nb = 1 + x\n");
        let output = std::env::temp_dir().join(format!(
            "severian-universal-pipeline-{}",
            std::process::id()
        ));
        let artifact = compile_source(&source, &output).unwrap();
        assert!(artifact.path.exists());
        assert!(std::process::Command::new(&artifact.path)
            .status()
            .unwrap()
            .success());
        std::fs::remove_file(output).unwrap();
    }

    #[test]
    fn ordinary_compilation_validates_external_boundaries() {
        let source = SourceFile::virtual_source(
            "invalid-ffi.sev",
            "@c\ndef invalid(value: nullable[i32]) -> i32\nx: i32 = 1\n",
        );
        let output =
            std::env::temp_dir().join(format!("severian-invalid-ffi-{}", std::process::id()));
        let error = compile_source(&source, &output).unwrap_err();
        assert!(error
            .to_string()
            .contains("only pointer representations may be nullable"));
    }

    #[test]
    fn driver_orchestrates_standard_lowering_and_custom_artifact_composition() {
        let (context, i32_type, custom, compiler_id) = compile_type_context();
        let module = Module {
            values: vec![
                Value {
                    id: ValueId(0),
                    type_id: i32_type,
                },
                Value {
                    id: ValueId(1),
                    type_id: custom,
                },
                Value {
                    id: ValueId(2),
                    type_id: i32_type,
                },
                Value {
                    id: ValueId(3),
                    type_id: i32_type,
                },
            ],
            initializer: severian_mir::Block {
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
                        operator: severian_universal::UnaryOperator::Positive,
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
            },
            bindings: vec![],
            ..Module::default()
        };
        let mut compiler = Compiler::with_context(context, TargetSpec::new("x86_64-unknown-linux"));
        compiler
            .register_compile_handler(compiler_id, TestHandler { calls: None })
            .unwrap();

        let mlir = compiler.compile_mir_to_mlir(&module).unwrap();
        assert!(mlir.contains("call @__sev_artifact_0"), "{mlir}");
        assert!(mlir.contains("func.func @__sev_artifact_0"));
    }

    #[test]
    fn source_compilation_routes_compile_types_through_the_mlir_toolchain() {
        let (context, _, _, compiler_id) = compile_type_context();
        let mut compiler = Compiler::with_context(context, TargetSpec::new("x86_64-unknown-linux"));
        let calls = Arc::new(AtomicUsize::new(0));
        compiler
            .register_compile_handler(
                compiler_id,
                TestHandler {
                    calls: Some(calls.clone()),
                },
            )
            .unwrap();
        let source = SourceFile::virtual_source("custom.sev", "value: CustomValue = 1\n");
        let output = std::env::temp_dir().join(format!(
            "severian-compile-type-source-{}",
            std::process::id()
        ));

        let artifact = compiler.compile_source(&source, &output).unwrap();
        assert_eq!(calls.load(Ordering::SeqCst), 1);
        assert!(artifact.path.exists());
        assert!(std::process::Command::new(&artifact.path)
            .status()
            .unwrap()
            .success());
        std::fs::remove_file(output).unwrap();
    }

    #[test]
    fn mixed_compile_type_programs_resume_through_ordinary_print() {
        let (context, _, _, compiler_id) = compile_type_context();
        let mut compiler = Compiler::with_context(context, TargetSpec::new("x86_64-unknown-linux"));
        compiler
            .register_compile_handler(compiler_id, TestHandler { calls: None })
            .unwrap();
        let source = SourceFile::virtual_source(
            "custom-print.sev",
            "value: CustomValue = 1\ndef main():\n    print(\"resumed\")\n",
        );
        let output = std::env::temp_dir().join(format!(
            "severian-compile-type-print-{}",
            std::process::id()
        ));

        let artifact = compiler.compile_source(&source, &output).unwrap();
        let result = std::process::Command::new(&artifact.path).output().unwrap();
        assert!(result.status.success());
        assert_eq!(result.stdout, b"resumed\n");
        std::fs::remove_file(output).unwrap();
    }

    #[test]
    fn core_character_literals_and_numeric_separators_compile_end_to_end() {
        let source = SourceFile::virtual_source(
            "primitive-literals.sev",
            "letter: char = '\u{03bb}'\nlarge: i64 = 1_000_000\n",
        );
        let output = std::env::temp_dir().join(format!(
            "severian-primitive-literals-{}",
            std::process::id()
        ));

        let artifact = compile_source(&source, &output).unwrap();
        assert!(std::process::Command::new(&artifact.path)
            .status()
            .unwrap()
            .success());
        std::fs::remove_file(output).unwrap();
    }

    #[test]
    fn global_initialization_runs_before_root_main() {
        let source = SourceFile::virtual_source(
            "entry.sev",
            "print(\"initializing\")\nseed := 7\ndef main():\n    observed := seed\n    print(\"running\")\n",
        );
        let output =
            std::env::temp_dir().join(format!("severian-entry-contract-{}", std::process::id()));
        compile_source(&source, &output).unwrap();
        let result = std::process::Command::new(&output).output().unwrap();
        assert!(result.status.success());
        assert_eq!(
            String::from_utf8(result.stdout).unwrap(),
            "initializing\nrunning\n"
        );
        std::fs::remove_file(output).unwrap();
    }
}
