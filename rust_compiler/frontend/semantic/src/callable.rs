use super::*;

impl Analyzer<'_> {
    pub(super) fn lower_callable_body(
        &mut self,
        ast_function: &severian_ast::FunctionDeclaration,
        function: &mut FunctionDeclaration,
        bindings: &mut Vec<Binding>,
        globals: &BTreeMap<String, (BindingId, severian_hir::VariableId, TypeId)>,
        global_values: &BTreeMap<String, Expression>,
        aliases: BTreeMap<String, TypeId>,
    ) -> Result<(), Diagnostic> {
        let Some(ast_body) = &ast_function.body else {
            return Ok(());
        };
        self.names = globals.clone();
        self.active_function_name = Some(ast_function.name.clone());
        self.declarations.clear();
        self.active_type_aliases = aliases;
        for (index, name) in ast_function.type_parameters.iter().enumerate() {
            if let Some(ty) = function
                .substitution
                .get(severian_universal::GenericParamId(index as u32))
            {
                self.active_type_aliases.insert(name.clone(), ty);
            }
        }
        self.value_substitutions = global_values.clone();
        self.loop_depth = 0;
        self.mocks.clear();
        self.callable_substitutions.clear();
        self.active_operator_namespaces = operator_namespaces(&ast_function.decorators);
        for parameter in &function.parameters {
            let type_id = parameter.contract.ty;
            if !self.declarations.insert(parameter.name.clone()) {
                return Err(Diagnostic::new(
                    "E000203",
                    format!("parameter `{}` is declared more than once", parameter.name),
                    Some(ast_function.span),
                ));
            }
            let variable = severian_hir::VariableId(parameter.binding.0);
            self.mutable_variables.insert(variable);
            self.names.insert(
                parameter.name.clone(),
                (parameter.binding, variable, type_id),
            );
        }
        let result_type = function.result.ty;
        let (mut body, hooks) = self.lower_function_hooks(ast_function, bindings, result_type)?;
        for contract in ast_function
            .contracts
            .iter()
            .filter(|contract| !contract.deferred)
        {
            body.statements.push(self.contract_assertion(contract)?);
        }
        body.statements
            .extend(self.block(ast_body, bindings, result_type)?.statements);
        let deferred = ast_function
            .contracts
            .iter()
            .filter(|contract| contract.deferred)
            .map(|contract| self.contract_assertion(contract))
            .collect::<Result<Vec<_>, _>>()?;
        if !deferred.is_empty() {
            insert_before_returns(&mut body, &deferred);
            if block_flow(ast_body) == ControlFlow::FallsThrough {
                body.statements.extend(deferred);
            }
        }
        if !hooks.is_empty() {
            insert_hook_exits(&mut body, &hooks);
            if block_flow(ast_body) == ControlFlow::FallsThrough {
                for hook in hooks.iter().rev() {
                    if let Some((field, duration)) = &hook.duration {
                        body.statements.push(Statement::FieldSet {
                            binding: hook.context,
                            field: *field,
                            value: duration.clone(),
                        });
                    }
                    body.statements
                        .extend(hook.without_phase.statements.iter().cloned());
                }
            }
        }
        let success_type = self
            .fallible_types
            .get(&result_type)
            .map_or(result_type, |fallible| fallible.success);
        let allows_fallthrough = success_type
            == self
                .types
                .resolve_name("unit")
                .expect("bootstrap defines unit")
            || self
                .types
                .definition(success_type)
                .is_some_and(|definition| definition.name == "None");
        let falls_through = block_flow(ast_body) == ControlFlow::FallsThrough;
        if !allows_fallthrough && falls_through {
            return Err(Diagnostic::new(
                "E000209",
                "not every path in this function returns its declared result",
                Some(ast_function.span),
            ));
        }
        if allows_fallthrough
            && falls_through
            && self
                .types
                .definition(result_type)
                .is_some_and(|definition| definition.name == "None")
        {
            body.statements.push(Statement::Return(Some(
                self.default_expression(result_type, ast_function.span)?,
            )));
        }
        if allows_fallthrough && falls_through && self.fallible_types.contains_key(&result_type) {
            body.statements.push(self.statement(
                &AstStatement::Return {
                    value: None,
                    span: ast_function.span,
                },
                bindings,
                result_type,
            )?);
        }
        if let Some(fallible) = self.fallible_types.get(&result_type).copied() {
            let catch_binding = self.new_binding_id();
            let error = Expression {
                id: self.next_id(),
                type_id: fallible.error,
                kind: ExpressionKind::Binding(catch_binding),
                span: ast_function.span,
            };
            let core_error = self
                .types
                .resolve_name("Error")
                .expect("bootstrap defines Error");
            let error = if fallible.error == core_error {
                let string = self
                    .types
                    .resolve_name("string")
                    .expect("bootstrap defines string");
                let frame = Expression {
                    id: self.next_id(),
                    type_id: string,
                    kind: ExpressionKind::Literal(LiteralValue::String(ast_function.name.clone())),
                    span: ast_function.span,
                };
                self.runtime_call(
                    "__sev_error_propagate",
                    &[fallible.error, string],
                    fallible.error,
                    vec![error, frame],
                    ast_function.span,
                )
            } else {
                error
            };
            let propagated =
                self.fallible_error_expression(result_type, fallible, error, ast_function.span)?;
            body = Block {
                statements: vec![Statement::Try {
                    body,
                    catch_binding,
                    catch_type: fallible.error,
                    catch_body: Block {
                        statements: vec![Statement::Return(Some(propagated))],
                    },
                    span: ast_function.span,
                }],
            };
        }
        let effects = function
            .parameters
            .iter()
            .map(|parameter| self.inferred_parameter_effect(&body, parameter.binding))
            .collect();
        self.parameter_effects.insert(function.id, effects);
        function.body = Some(body);
        Ok(())
    }

    pub(super) fn receiver_field_expression(
        &mut self,
        name: &str,
        expected: Option<TypeId>,
        span: severian_source::Span,
    ) -> Result<Option<Expression>, Diagnostic> {
        if !self.declarations.contains(name) {
            if let Some((binding, owner)) = self.active_receiver.clone() {
                if let Some((index, field)) = owner
                    .fields
                    .iter()
                    .enumerate()
                    .find(|(_, field)| field.name == name)
                {
                    if expected.is_some_and(|expected| !self.types.assignable(field.ty, expected)) {
                        return Err(semantic_error(
                            "field does not satisfy the expected type".into(),
                            span,
                        ));
                    }
                    let receiver = Expression {
                        id: self.next_id(),
                        type_id: owner.ty,
                        kind: ExpressionKind::Binding(binding),
                        span: span,
                    };
                    return Ok(Some(Expression {
                        id: self.next_id(),
                        type_id: field.ty,
                        kind: ExpressionKind::Field {
                            object: Box::new(receiver),
                            index: index as u32,
                        },
                        span: span,
                    }));
                }
            }
        }

        Ok(None)
    }

    pub(super) fn receiver_sibling_call(
        &mut self,
        callee: &AstExpression,
        arguments: &[severian_ast::CallArgument],
        expected: Option<TypeId>,
        span: severian_source::Span,
    ) -> Result<Option<Expression>, Diagnostic> {
        if let AstExpressionKind::Name(name) = &callee.kind {
            if !self.declarations.contains(name) {
                if let Some((binding, owner)) = self.active_receiver.clone() {
                    if let Some(method) = owner
                        .methods
                        .iter()
                        .find(|method| method.name == *name)
                        .cloned()
                    {
                        let receiver = Expression {
                            id: self.next_id(),
                            type_id: owner.ty,
                            kind: ExpressionKind::Binding(binding),
                            span: callee.span,
                        };
                        return self
                            .lower_method_callable(
                                &owner, &method, receiver, arguments, expected, span,
                            )
                            .map(Some);
                    }
                }
            }
        }
        Ok(None)
    }

    pub(super) fn lower_method_callable(
        &mut self,
        owner: &ClassInstance,
        method: &severian_ast::FunctionDeclaration,
        receiver: Expression,
        arguments: &[severian_ast::CallArgument],
        expected: Option<TypeId>,
        span: severian_source::Span,
    ) -> Result<Expression, Diagnostic> {
        if method.body.is_none() {
            return Err(Diagnostic::new(
                "E000211",
                format!("method `{}` has no implementation", method.name),
                Some(method.span),
            ));
        }
        let mut aliases = self
            .classes
            .get(&owner.name)
            .map(|class| {
                class
                    .type_parameters
                    .iter()
                    .cloned()
                    .zip(owner.arguments.iter().copied())
                    .collect::<BTreeMap<_, _>>()
            })
            .unwrap_or_default();
        aliases.insert("Self".into(), owner.ty);
        let parameters = method
            .parameters
            .iter()
            .map(|parameter| {
                let element = self.resolve_instantiated_type(&parameter.annotation, &aliases)?;
                Ok(SignatureParameter {
                    name: parameter.name.clone(),
                    type_id: if parameter.variadic {
                        self.instantiate_list_type(element)
                    } else {
                        element
                    },
                    variadic_element: parameter.variadic.then_some(element),
                    default: parameter.default.clone(),
                })
            })
            .collect::<Result<Vec<_>, Diagnostic>>()?;
        let result = self.resolve_instantiated_type(&method.result, &aliases)?;
        let exposed_result = self
            .fallible_types
            .get(&result)
            .map_or(result, |fallible| fallible.success);
        if expected.is_some_and(|expected| !self.types.assignable(exposed_result, expected)) {
            return Err(semantic_error(
                "method result does not satisfy the expected type".into(),
                span,
            ));
        }
        let signature = FunctionSignature { parameters, result };
        let Some((resolved, _, evaluation_order)) =
            self.resolve_signature_arguments(&signature, arguments, span)?
        else {
            return Err(Diagnostic::new(
                "E000206",
                format!("method `{}` does not accept these arguments", method.name),
                Some(span),
            ));
        };
        let key = (owner.ty, method.name.clone(), method.span.start as usize);
        let id = if let Some(id) = self.method_instances.get(&key) {
            *id
        } else {
            let definition = synthetic_extension_definition(
                &format!("{}.{}", owner.name, method.name),
                method.span,
                &[owner.ty],
            );
            let id = FunctionId(definition.declaration.0);
            let mut hir_parameters = vec![FunctionParameter {
                binding: self.new_binding_id(),
                name: "self".into(),
                contract: universal_boundary(owner.ty),
            }];
            for parameter in &signature.parameters {
                hir_parameters.push(FunctionParameter {
                    binding: self.new_binding_id(),
                    name: parameter.name.clone(),
                    contract: universal_boundary(parameter.type_id),
                });
            }
            let function = FunctionDeclaration {
                id,
                definition,
                substitution: severian_universal::Substitution::default(),
                name: format!("{}.{}", owner.name, method.name),
                generic_parameters: Vec::new(),
                type_parameters: Vec::new(),
                parameters: hir_parameters,
                result: universal_boundary(result),
                compile_route: severian_universal::CompileRoute::Standard,
                call_type: CallType::Severian,
                body: None,
            };
            self.method_instances.insert(key, id);
            self.function_definitions.insert(id, definition);
            self.function_substitutions
                .insert(id, function.substitution.clone());
            self.parameter_effects
                .insert(id, vec![ParameterEffect::Shared; function.parameters.len()]);
            self.pending_methods
                .push((method.clone(), function, owner.clone(), aliases));
            id
        };
        let arguments = self.apply_parameter_effects(
            id,
            std::iter::once(receiver).chain(resolved).collect(),
            span,
        );
        let call = Expression {
            id: self.next_id(),
            type_id: result,
            kind: ExpressionKind::Call {
                evaluation_order: std::iter::once(0)
                    .chain(evaluation_order.into_iter().map(|index| index + 1))
                    .collect(),
                callee: severian_hir::Callee::Direct {
                    instance: Some(id),
                    function: self.function_definitions[&id],
                    substitution: self.function_substitutions[&id].clone(),
                },
                arguments,
            },
            span,
        };
        if self.preserve_error_depth == 0 {
            if let Some(fallible) = self.fallible_types.get(&result).copied() {
                return Ok(self.unwrap_fallible_expression(call, fallible, span));
            }
        }
        Ok(call)
    }

    pub(super) fn finish_callable_effects(&mut self, module: &mut Module) {
        // Signatures are reserved before bodies, so recursive and forward calls
        // initially carry shared effects. Complete the finite effect lattice
        // before exposing argument borrows or the reference calling convention.
        loop {
            let mut changed = false;
            for function in &module.functions {
                if let Some(body) = &function.body {
                    let effects = function
                        .parameters
                        .iter()
                        .map(|parameter| self.inferred_parameter_effect(body, parameter.binding))
                        .collect::<Vec<_>>();
                    if self.parameter_effects.get(&function.id) != Some(&effects) {
                        self.parameter_effects.insert(function.id, effects);
                        changed = true;
                    }
                }
            }
            if !changed {
                break;
            }
        }
        let mut refresh = |expression: &mut Expression| {
            let ExpressionKind::Call {
                callee:
                    severian_hir::Callee::Direct {
                        instance: Some(instance),
                        ..
                    },
                arguments,
                ..
            } = &mut expression.kind
            else {
                return;
            };
            let Some(effects) = self.parameter_effects.get(instance) else {
                return;
            };
            for (argument, effect) in arguments.iter_mut().zip(effects) {
                if let ExpressionKind::Borrow { exclusive, .. } = &mut argument.kind {
                    if *effect == ParameterEffect::Exclusive {
                        *exclusive = true;
                    }
                    if *effect == ParameterEffect::Move {
                        let ExpressionKind::Borrow { operand, .. } = std::mem::replace(
                            &mut argument.kind,
                            ExpressionKind::Literal(LiteralValue::None),
                        ) else {
                            unreachable!();
                        };
                        argument.kind = ExpressionKind::Move(operand);
                    }
                }
            }
        };
        for binding in &mut module.bindings {
            visit_expression(&mut binding.value, &mut refresh);
        }
        visit_block(&mut module.initializer, &mut refresh);
        for function in &mut module.functions {
            if let Some(body) = &mut function.body {
                visit_block(body, &mut refresh);
            }
        }
    }
}

fn visit_expression(expression: &mut Expression, visit: &mut impl FnMut(&mut Expression)) {
    match &mut expression.kind {
        ExpressionKind::Variant { fields, .. } | ExpressionKind::Aggregate { fields, .. } => {
            for field in fields {
                visit_expression(field, visit);
            }
        }
        ExpressionKind::Call { arguments, .. } => {
            for argument in arguments {
                visit_expression(argument, visit);
            }
        }
        ExpressionKind::Field { object, .. } => visit_expression(object, visit),
        ExpressionKind::Async { expression, .. } => visit_expression(expression, visit),
        ExpressionKind::AsyncFieldUpdate { value, .. } => visit_expression(value, visit),
        ExpressionKind::Await(operand)
        | ExpressionKind::Throw(operand)
        | ExpressionKind::Move(operand)
        | ExpressionKind::Convert { operand, .. }
        | ExpressionKind::Borrow { operand, .. }
        | ExpressionKind::Unary { operand, .. } => visit_expression(operand, visit),
        ExpressionKind::Fallback {
            condition,
            value,
            fallback,
        } => {
            visit_expression(condition, visit);
            visit_expression(value, visit);
            visit_expression(fallback, visit);
        }
        ExpressionKind::Binary { left, right, .. } => {
            visit_expression(left, visit);
            visit_expression(right, visit);
        }
        ExpressionKind::Literal(_)
        | ExpressionKind::Binding(_)
        | ExpressionKind::Function(_)
        | ExpressionKind::AddressOf(_) => {}
    }
    visit(expression);
}

fn visit_block(block: &mut Block, visit: &mut impl FnMut(&mut Expression)) {
    for statement in &mut block.statements {
        match statement {
            Statement::Sequence(block)
            | Statement::Placement { body: block, .. }
            | Statement::ExpectThrow { body: block, .. } => visit_block(block, visit),
            Statement::FieldSet { value, .. }
            | Statement::FieldUpdate { value, .. }
            | Statement::Expression(value)
            | Statement::Return(Some(value)) => visit_expression(value, visit),
            Statement::Assert {
                condition, message, ..
            } => {
                visit_expression(condition, visit);
                if let Some(message) = message {
                    visit_expression(message, visit);
                }
            }
            Statement::Try {
                body, catch_body, ..
            } => {
                visit_block(body, visit);
                visit_block(catch_body, visit);
            }
            Statement::If {
                condition,
                then_block,
                else_block,
            } => {
                visit_expression(condition, visit);
                visit_block(then_block, visit);
                visit_block(else_block, visit);
            }
            Statement::While {
                condition, body, ..
            } => {
                visit_expression(condition, visit);
                visit_block(body, visit);
            }
            Statement::Match { subject, arms } => {
                visit_expression(subject, visit);
                for arm in arms {
                    visit_block(&mut arm.body, visit);
                }
            }
            Statement::Binding(_)
            | Statement::Return(None)
            | Statement::Break { .. }
            | Statement::Continue { .. } => {}
        }
    }
}
