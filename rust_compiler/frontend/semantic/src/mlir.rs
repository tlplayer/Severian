use super::*;
use severian_ast::DecoratorValue;
use severian_source::Span;

impl Analyzer<'_> {
    pub(super) fn memory_buffer_length(&mut self, value: Expression, span: Span) -> Expression {
        let index = self.types.resolve_name("index").expect("bootstrap defines index");
        let axis = Expression {
            id: self.next_id(),
            type_id: index,
            kind: ExpressionKind::Literal(LiteralValue::Integer("0".into())),
            span,
        };
        let dimension = self.memory_operation("memref", "dim", vec![value, axis], index, span);
        let integer = self.types.resolve_name("int").expect("bootstrap defines int");
        self.memory_operation("arith", "index_cast", vec![dimension], integer, span)
    }

    fn memory_operation(
        &mut self, dialect: &str, operation: &str, arguments: Vec<Expression>,
        result: TypeId, span: Span,
    ) -> Expression {
        let attributes = severian_universal::Attrs::from([(
            severian_universal::MLIR_OPERATION_NAME_ATTRIBUTE,
            severian_universal::AttrValue::String(format!("{dialect}.{operation}")),
        )]);
        Expression {
            id: self.next_id(), type_id: result, span,
            kind: ExpressionKind::Call {
                callee: severian_hir::Callee::Intrinsic {
                    operation: severian_universal::OpId::named(dialect, operation), attributes,
                },
                arguments, evaluation_order: Vec::new(),
            },
        }
    }

    /// A typed MLIR declaration is an implementation, not an unresolved
    /// foreign function. Give it a body so ordinary package/object emission
    /// retains the operation under the declaration's stable function identity.
    pub(super) fn lower_mlir_boundary(
        &mut self,
        source: &severian_ast::FunctionDeclaration,
        function: &mut FunctionDeclaration,
    ) -> Result<bool, Diagnostic> {
        let Some(decorator) = source.decorators.iter().find(|value| value.name == "mlir") else {
            return Ok(false);
        };
        let invalid = |message: &str| Diagnostic::new("E000212", message, Some(decorator.span));
        if source.body.is_some() {
            return Err(invalid(
                "MLIR boundary declarations cannot have source bodies",
            ));
        }
        let mut operation = None;
        let mut attributes = Vec::new();
        let mut names = BTreeSet::new();
        let mut callee = None;
        for argument in &decorator.arguments {
            let DecoratorValue::String(value) = &argument.value else {
                return Err(invalid("MLIR boundary arguments require string literals"));
            };
            let name = argument.name.as_deref().unwrap_or("");
            if !names.insert(name) {
                return Err(invalid("duplicate MLIR boundary argument"));
            }
            match name {
                "" => operation = Some(value.as_str()),
                "lowering" if value == "convert-vector-to-llvm" => {}
                "libraries" if value == "mlir_c_runner_utils" => {}
                "operand_segments" => {
                    let segments = value.split(',')
                        .map(|part| part.parse::<i32>().ok().filter(|size| *size >= 0))
                        .collect::<Option<Vec<_>>>()
                        .ok_or_else(|| invalid("MLIR operand segments require nonnegative i32 sizes"))?;
                    if segments.iter().map(|size| *size as u64).sum::<u64>()
                        != function.parameters.len() as u64
                    {
                        return Err(invalid("MLIR operand segments must cover every parameter"));
                    }
                    attributes.push(format!("operandSegmentSizes = array<i32: {}>",
                        segments.iter().map(ToString::to_string).collect::<Vec<_>>().join(", ")));
                }
                "lowering" | "libraries" => {
                    return Err(invalid("unsupported MLIR boundary lowering requirement"));
                }
                "callee" => callee = Some(value.clone()),
                _ => attributes.push(format!(
                    "{name} = {}",
                    if value.starts_with('#') {
                        value.clone()
                    } else {
                        format!("{value:?}")
                    }
                )),
            }
        }
        let operation =
            operation.ok_or_else(|| invalid("MLIR boundary requires an operation name"))?;
        let Some((dialect, mnemonic)) = operation.split_once('.') else {
            return Err(invalid("MLIR operation must have a dialect and mnemonic"));
        };
        if dialect.is_empty() || mnemonic.is_empty() {
            return Err(invalid("MLIR operation must have a dialect and mnemonic"));
        }
        if matches!(
            operation,
            "cf.br" | "cf.cond_br" | "cf.switch" | "func.return" | "llvm.unreachable"
        ) {
            return Err(invalid(
                "CFG terminators cannot be used as callable MLIR boundaries",
            ));
        }
        if operation == "func.call" {
            let symbol =
                callee.ok_or_else(|| invalid("MLIR func.call requires a callee symbol"))?;
            if !attributes.is_empty() {
                return Err(invalid("unsupported attributes on MLIR library call"));
            }
            // This emits a func declaration and call in MLIR. Library
            // composition supplies the symbol's IR implementation before
            // lowering to an object; no C source or foreign decorator is used.
            function.call_type = CallType::Mlir(severian_hir::SymbolId(symbol));
            return Ok(true);
        }
        let arguments = function
            .parameters
            .iter()
            .map(|parameter| Expression {
                id: self.next_id(),
                type_id: parameter.contract.ty,
                kind: ExpressionKind::Binding(parameter.binding),
                span: source.span,
            })
            .collect::<Vec<_>>();
        let expression = {
            if callee.is_some() {
                return Err(invalid(
                    "callee is only supported on MLIR func.call boundaries",
                ));
            }
            let mut metadata = severian_universal::Attrs::new();
            metadata.insert(
                severian_universal::MLIR_OPERATION_NAME_ATTRIBUTE,
                severian_universal::AttrValue::String(operation.to_owned()),
            );
            if !attributes.is_empty() {
                metadata.insert(
                    severian_universal::MLIR_OPERATION_PARAMETERS_ATTRIBUTE,
                    severian_universal::AttrValue::String(format!(
                        "<{{{}}}>",
                        attributes.join(", ")
                    )),
                );
            }
            Expression {
                id: self.next_id(),
                type_id: function.result.ty,
                kind: ExpressionKind::Call {
                    callee: severian_hir::Callee::Intrinsic {
                        operation: severian_universal::OpId::named(dialect, mnemonic),
                        attributes: metadata,
                    },
                    arguments,
                    evaluation_order: Vec::new(),
                },
                span: source.span,
            }
        };
        function.body = Some(Block {
            statements: if self.types.resolve_name("unit") == Some(function.result.ty) {
                vec![Statement::Expression(expression), Statement::Return(None)]
            } else {
                vec![Statement::Return(Some(expression))]
            },
        });
        Ok(true)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn memory_allocation_preserves_valid_operand_segments_and_rejects_bad_sizes() {
        let context = severian_bootstrap::load().unwrap();
        for (segments, valid) in [
            ("1,0", true), ("0,1", true), ("2,0", false),
            ("-1,2", false), ("1,,0", false), ("2147483648", false),
        ] {
            let source = severian_source::SourceFile::virtual_source(
                "allocation.sev",
                &format!("@mlir(\"memref.alloc\", operand_segments=\"{segments}\")\ndef allocate(count: index) -> array[u8]\n"),
            );
            let ast = severian_parser::parse(&severian_lexer::scan(&source).unwrap()).unwrap();
            let result = crate::analyze(&ast, &context.types);
            if !valid {
                assert!(result.unwrap_err().message.contains("operand segments"));
                continue;
            }
            let program = result.unwrap();
            let body = program.modules[0].functions[0].body.as_ref().unwrap();
            let Statement::Return(Some(Expression { kind: ExpressionKind::Call {
                callee: severian_hir::Callee::Intrinsic { attributes, .. }, ..
            }, .. })) = &body.statements[0] else { panic!("expected MLIR operation") };
            let expected = format!("<{{operandSegmentSizes = array<i32: {}>}}>", segments.replace(',', ", "));
            assert_eq!(attributes.get(&severian_universal::MLIR_OPERATION_PARAMETERS_ATTRIBUTE),
                Some(&severian_universal::AttrValue::String(expected)));
        }
    }
}
