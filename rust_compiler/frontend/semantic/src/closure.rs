use super::*;
use severian_source::Span;

impl Analyzer<'_> {
    pub(super) fn closure_value(
        &mut self,
        ast: &AstExpression,
        ty: TypeId,
    ) -> Result<Option<Expression>, Diagnostic> {
        let signature = self.function_types[&ty].clone();
        let (parameters, body, captures, direct) = match &ast.kind {
            AstExpressionKind::Lambda { parameters, body } => {
                let mut mentioned = BTreeSet::new();
                collect_expression_names(body, &mut mentioned);
                for parameter in parameters {
                    mentioned.remove(parameter);
                }
                let mut captures = Vec::new();
                for name in mentioned {
                    if let Some(value) = self.value_substitutions.get(&name).cloned() {
                        captures.push((name, value));
                    } else if let Some((binding, _, ty)) = self.names.get(&name).copied() {
                        let value = Expression {
                            id: self.next_id(),
                            type_id: ty,
                            kind: ExpressionKind::Binding(binding),
                            span: ast.span,
                        };
                        captures.push((name, value));
                    }
                }
                (
                    parameters.clone(),
                    Some(body.as_ref().clone()),
                    captures,
                    None,
                )
            }
            AstExpressionKind::Name(name)
                if self.functions.contains_key(name) && !self.names.contains_key(name) =>
            {
                let callable = self.resolve_callable_value(ast, &signature)?;
                let CallableValue::Direct(function) = callable.value else {
                    return Ok(None);
                };
                (Vec::new(), None, Vec::new(), Some(function))
            }
            AstExpressionKind::Name(name) => {
                let callable = self
                    .names
                    .get(name)
                    .and_then(|(_, variable, _)| self.callable_bindings.get(variable))
                    .cloned();
                let Some(CallableValue::Lambda {
                    parameters,
                    body,
                    closure,
                    closure_type,
                    captures,
                }) = callable
                else {
                    return Ok(None);
                };
                let owner = Expression {
                    id: self.next_id(),
                    type_id: closure_type,
                    kind: ExpressionKind::Binding(closure),
                    span: ast.span,
                };
                let captures = captures
                    .into_iter()
                    .enumerate()
                    .map(|(index, (name, ty))| {
                        (
                            name,
                            Expression {
                                id: self.next_id(),
                                type_id: ty,
                                kind: ExpressionKind::Field {
                                    object: Box::new(owner.clone()),
                                    index: index as u32,
                                },
                                span: ast.span,
                            },
                        )
                    })
                    .collect();
                (parameters, Some(body), captures, None)
            }
            _ => return Ok(None),
        };
        if direct.is_none() && parameters.len() != signature.parameters.len() {
            return Err(Diagnostic::new(
                "E000206",
                "lambda parameter count does not match the callable type",
                Some(ast.span),
            ));
        }
        let span = ast.span;
        let any = self.ensure_any_type();
        let capture_types = captures
            .iter()
            .map(|(_, value)| value.type_id)
            .collect::<Vec<_>>();
        let environment_type = self.instantiate_tuple_type(&capture_types);
        let environment = Expression {
            id: self.next_id(),
            type_id: environment_type,
            span,
            kind: ExpressionKind::Aggregate {
                class: environment_type,
                fields: captures.iter().map(|(_, value)| value.clone()).collect(),
            },
        };
        let environment = self.box_any_value(environment, span)?;
        let symbol = format!(
            "__sev_closure_{}_{}_{}_{}_{}",
            span.source,
            span.start,
            ty.0,
            environment_type.0,
            direct.map(|id| id.0).unwrap_or(0)
        );
        let mut helper_parameters = vec![any];
        helper_parameters.extend_from_slice(&signature.parameters);
        let definition =
            self.ensure_runtime_function(&symbol, &helper_parameters, signature.result);
        let helper = self
            .runtime_functions
            .iter()
            .find(|function| function.definition == definition)
            .unwrap();
        if helper.body.is_none() {
            let bindings = helper
                .parameters
                .iter()
                .map(|parameter| parameter.binding)
                .collect::<Vec<_>>();
            let values = helper_parameters
                .iter()
                .zip(&bindings)
                .map(|(ty, binding)| Expression {
                    id: self.next_id(),
                    type_id: *ty,
                    span,
                    kind: ExpressionKind::Binding(*binding),
                })
                .collect::<Vec<_>>();
            let result = if let Some(function) = direct {
                Expression {
                    id: self.next_id(),
                    type_id: signature.result,
                    span,
                    kind: ExpressionKind::Call {
                        callee: severian_hir::Callee::Direct {
                            instance: Some(function),
                            function: self.function_definitions[&function],
                            substitution: self.function_substitutions[&function].clone(),
                        },
                        arguments: values[1..].to_vec(),
                        evaluation_order: Vec::new(),
                    },
                }
            } else {
                let previous_names = std::mem::take(&mut self.names);
                let previous_values = std::mem::take(&mut self.value_substitutions);
                let storage = self.erased_record_read(values[0].clone(), environment_type, span);
                for (index, (name, value)) in captures.iter().enumerate() {
                    let field = Expression {
                        id: self.next_id(),
                        type_id: value.type_id,
                        span,
                        kind: ExpressionKind::Field {
                            object: Box::new(storage.clone()),
                            index: index as u32,
                        },
                    };
                    self.value_substitutions.insert(name.clone(), field);
                }
                for (name, value) in parameters.into_iter().zip(values[1..].iter().cloned()) {
                    self.value_substitutions.insert(name, value);
                }
                let lowered = self.expression(body.as_ref().unwrap(), Some(signature.result));
                self.names = previous_names;
                self.value_substitutions = previous_values;
                lowered?
            };
            let helper = self
                .runtime_functions
                .iter_mut()
                .find(|function| function.definition == definition)
                .unwrap();
            helper.call_type = CallType::Severian;
            for parameter in &mut helper.parameters {
                parameter
                    .contract
                    .modifiers
                    .push(severian_hir::BoundaryModifier {
                        name: "callable_value".into(),
                    });
            }
            helper.body = Some(Block {
                statements: if signature.result == self.types.resolve_name("unit").unwrap() {
                    vec![Statement::Expression(result), Statement::Return(None)]
                } else {
                    vec![Statement::Return(Some(result))]
                },
            });
        }
        let pointer = self.instantiate_pointer_type(self.types.resolve_name("u8").unwrap());
        let code = Expression {
            id: self.next_id(),
            type_id: pointer,
            span,
            kind: ExpressionKind::Function(definition),
        };
        Ok(Some(Expression {
            id: self.next_id(),
            type_id: ty,
            span,
            kind: ExpressionKind::Aggregate {
                class: ty,
                fields: vec![code, environment],
            },
        }))
    }

    pub(super) fn closure_call(
        &mut self,
        callee: &AstExpression,
        arguments: &[severian_ast::CallArgument],
        expected: Option<TypeId>,
        span: Span,
    ) -> Result<Option<Expression>, Diagnostic> {
        // Declaration names use ordinary overload resolution. Values and fields
        // carry a callable signature and an environment through normal storage.
        if callable_path(callee).is_some_and(|name| {
            self.functions.contains_key(&name)
                && !self.names.contains_key(&name)
                && !self.value_substitutions.contains_key(&name)
        }) {
            return Ok(None);
        }
        let Ok(value) = self.expression(callee, None) else {
            return Ok(None);
        };
        let Some(signature) = self.function_types.get(&value.type_id).cloned() else {
            return Ok(None);
        };
        if arguments.len() != signature.parameters.len()
            || arguments.iter().any(|argument| argument.name.is_some())
        {
            return Err(Diagnostic::new(
                "E000206",
                "callable received the wrong arguments",
                Some(span),
            ));
        }
        if expected
            .is_some_and(|expected| !self.accepts_expression_type(signature.result, expected))
        {
            return Err(Diagnostic::new(
                "E000204",
                "callable result does not satisfy the expected type",
                Some(span),
            ));
        }
        let callable_type = value.type_id;
        let mut values = vec![value];
        for (argument, ty) in arguments.iter().zip(&signature.parameters) {
            values.push(self.expression(&argument.value, Some(*ty))?);
        }
        let mut parameter_types = vec![callable_type];
        parameter_types.extend_from_slice(&signature.parameters);
        let symbol = format!("__sev_invoke_{}", callable_type.0);
        let definition = self.ensure_runtime_function(&symbol, &parameter_types, signature.result);
        let helper = self
            .runtime_functions
            .iter()
            .find(|function| function.definition == definition)
            .unwrap();
        if helper.body.is_none() {
            let bindings = helper
                .parameters
                .iter()
                .map(|parameter| parameter.binding)
                .collect::<Vec<_>>();
            let values = parameter_types
                .iter()
                .zip(bindings)
                .map(|(ty, binding)| Expression {
                    id: self.next_id(),
                    type_id: *ty,
                    span,
                    kind: ExpressionKind::Binding(binding),
                })
                .collect::<Vec<_>>();
            let pointer = self.instantiate_pointer_type(self.types.resolve_name("u8").unwrap());
            let code = Expression {
                id: self.next_id(),
                type_id: pointer,
                span,
                kind: ExpressionKind::Field {
                    object: Box::new(values[0].clone()),
                    index: 0,
                },
            };
            let environment = Expression {
                id: self.next_id(),
                type_id: self.ensure_any_type(),
                span,
                kind: ExpressionKind::Field {
                    object: Box::new(values[0].clone()),
                    index: 1,
                },
            };
            let mut arguments = vec![environment];
            arguments.extend_from_slice(&values[1..]);
            let result = Expression {
                id: self.next_id(),
                type_id: signature.result,
                span,
                kind: ExpressionKind::Call {
                    callee: severian_hir::Callee::FunctionValue(Box::new(code)),
                    arguments,
                    evaluation_order: Vec::new(),
                },
            };
            let helper = self
                .runtime_functions
                .iter_mut()
                .find(|function| function.definition == definition)
                .unwrap();
            helper.call_type = CallType::Severian;
            helper.body = Some(Block {
                statements: if signature.result == self.types.resolve_name("unit").unwrap() {
                    vec![Statement::Expression(result), Statement::Return(None)]
                } else {
                    vec![Statement::Return(Some(result))]
                },
            });
        }
        Ok(Some(self.runtime_call(
            &symbol,
            &parameter_types,
            signature.result,
            values,
            span,
        )))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parenthesized_optional_callback_keeps_function_context() {
        let source = severian_source::SourceFile::virtual_source("optional_callback.sev", "class Callback:\n    check: (() -> int) | None = None\ndef make() -> Callback:\n    return Callback(check=lambda: 7)\n");
        let ast = severian_parser::parse(&severian_lexer::scan(&source).unwrap()).unwrap();
        let context = severian_bootstrap::load().unwrap();
        let program = analyze(&ast, &context.types).unwrap();
        severian_mir::build(&program).unwrap();
    }

    #[test]
    fn callbacks_survive_returns_fields_and_captured_values() {
        let source = severian_source::SourceFile::virtual_source("closure.sev", "class Callback:\n    invoke: (int) -> int\ndef plus_one(value: int) -> int:\n    return value + 1\ndef make(offset: int) -> Callback:\n    return Callback(lambda value: value + offset)\ndef apply(callback: (int) -> int, value: int) -> int:\n    answer = callback(value)\n    return answer\ndef main() -> int:\n    saved = make(5)\n    assert(saved.invoke(7) == 12)\n    assert(apply(plus_one, 8) == 9)\n    return 0\n");
        let ast = severian_parser::parse(&severian_lexer::scan(&source).unwrap()).unwrap();
        let context = severian_bootstrap::load().unwrap();
        let program = analyze(&ast, &context.types).unwrap();
        severian_mir::build(&program).unwrap();
    }
}
