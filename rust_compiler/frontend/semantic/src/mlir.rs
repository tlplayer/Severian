use super::*;
use severian_ast::DecoratorValue;

impl Analyzer<'_> {
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
                "lowering" | "libraries" | "operand_segments" => {
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
