use super::*;

impl Analyzer<'_> {
    pub(super) fn lower_constructor_call(
        &mut self,
        owner: &ClassInstance,
        arguments: &[severian_ast::CallArgument],
        span: severian_source::Span,
    ) -> Result<Expression, Diagnostic> {
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
        let mut candidates = Vec::new();
        let mut inference_error = None;
        for source in &owner.constructors {
            let mut constructor = source.clone();
            if !constructor.type_parameters.is_empty() {
                // Inference follows parameter order, including named arguments.
                // Signature resolution below remains responsible for validating
                // duplicate names, defaults, arity, and conversion ranks.
                let mut actuals = Vec::new();
                let mut complete = true;
                for (index, parameter) in constructor.parameters.iter().enumerate() {
                    let argument = arguments
                        .iter()
                        .find(|arg| arg.name.as_deref() == Some(&parameter.name))
                        .or_else(|| arguments.get(index).filter(|arg| arg.name.is_none()));
                    let expression = argument
                        .map(|arg| &arg.value)
                        .or(parameter.default.as_ref());
                    let Some(expression) = expression else {
                        complete = false;
                        break;
                    };
                    match self.expression(expression, None) {
                        Ok(value) => {
                            actuals.push(self.constructor_argument_type_name(value.type_id))
                        }
                        Err(error) => {
                            inference_error = Some(error);
                            complete = false;
                            break;
                        }
                    }
                }
                if !complete {
                    continue;
                }
                match package::generic::specialize_constructor(&constructor, &actuals) {
                    Ok(specialized) => constructor = specialized,
                    Err(error) => {
                        inference_error = Some(error);
                        continue;
                    }
                }
            }
            let parameters = constructor
                .parameters
                .iter()
                .map(|parameter| {
                    let element =
                        self.resolve_instantiated_type(&parameter.annotation, &aliases)?;
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
                .collect::<Result<Vec<_>, Diagnostic>>();
            let parameters = match parameters {
                Ok(parameters) => parameters,
                Err(error) => {
                    inference_error = Some(error);
                    continue;
                }
            };
            let signature = FunctionSignature {
                parameters,
                result: owner.ty,
            };
            if let Some((values, ranks, order)) =
                self.resolve_signature_arguments(&signature, arguments, span)?
            {
                candidates.push((
                    constructor,
                    signature,
                    values,
                    ranks,
                    order,
                    source.type_parameters.is_empty(),
                ));
            }
        }
        let best = candidates
            .iter()
            .enumerate()
            .filter(|(index, candidate)| {
                !candidates.iter().enumerate().any(|(other_index, other)| {
                    other_index != *index
                        && (dominates(&other.3, &candidate.3)
                            || (other.3 == candidate.3 && other.5 && !candidate.5))
                })
            })
            .map(|(index, _)| index)
            .collect::<Vec<_>>();
        let selected = match best.as_slice() {
            [index] => *index,
            [] => {
                return Err(inference_error.unwrap_or_else(|| {
                    Diagnostic::new(
                        "E000221",
                        format!(
                            "constructor `{}` has no overload accepting these arguments",
                            owner.name
                        ),
                        Some(span),
                    )
                }))
            }
            _ => {
                return Err(Diagnostic::new(
                    "E000221",
                    format!("ambiguous constructor call for `{}`", owner.name),
                    Some(span),
                ))
            }
        };
        let (mut constructor, signature, values, _, evaluation_order, _) =
            candidates.swap_remove(selected);
        if constructor.body.is_none() {
            return Err(Diagnostic::new(
                "E000211",
                format!("constructor `{}` has no implementation", owner.name),
                Some(constructor.span),
            ));
        }
        let declared_result = self.resolve_instantiated_type(&constructor.result, &aliases)?;
        let result_type = if self.types.resolve_name("unit") == Some(declared_result) {
            owner.ty
        } else {
            declared_result
        };
        let success_type = self
            .fallible_types
            .get(&result_type)
            .map_or(result_type, |result| result.success);
        if success_type != owner.ty {
            return Err(Diagnostic::new(
                "E000221",
                "a field-initializing constructor must return its class or a fallible result of its class",
                Some(constructor.result.span),
            ));
        }
        let mut identity_types = vec![owner.ty];
        identity_types.extend(
            signature
                .parameters
                .iter()
                .map(|parameter| parameter.type_id),
        );
        let definition = synthetic_extension_definition(
            &format!("{}.constructor", owner.name),
            constructor.span,
            &identity_types,
        );
        let id = FunctionId(definition.declaration.0);
        if !self.function_definitions.contains_key(&id) {
            constructor.type_parameters.clear();
            let parameters = signature
                .parameters
                .iter()
                .map(|parameter| FunctionParameter {
                    binding: self.new_binding_id(),
                    name: parameter.name.clone(),
                    contract: universal_boundary(parameter.type_id),
                })
                .collect::<Vec<_>>();
            let function = FunctionDeclaration {
                id,
                definition,
                substitution: severian_universal::Substitution::default(),
                name: format!("{}.constructor", owner.name),
                generic_parameters: Vec::new(),
                type_parameters: Vec::new(),
                parameters,
                result: universal_boundary(result_type),
                compile_route: severian_universal::CompileRoute::Standard,
                call_type: CallType::Severian,
                body: None,
            };
            self.function_definitions.insert(id, definition);
            self.function_substitutions
                .insert(id, function.substitution.clone());
            self.parameter_effects
                .insert(id, vec![ParameterEffect::Shared; function.parameters.len()]);
            self.pending_methods
                .push((constructor, function, owner.clone(), aliases, true));
        }
        let arguments = self.apply_parameter_effects(id, values, span);
        let call = Expression {
            id: self.next_id(),
            type_id: result_type,
            span,
            kind: ExpressionKind::Call {
                callee: severian_hir::Callee::Direct {
                    instance: Some(id),
                    function: definition,
                    substitution: severian_universal::Substitution::default(),
                },
                arguments,
                evaluation_order,
            },
        };
        Ok(
            if let Some(fallible) = self.fallible_types.get(&result_type).copied() {
                self.unwrap_fallible_expression(call, fallible, span)
            } else {
                call
            },
        )
    }

    pub(super) fn begin_constructor(
        &mut self,
        owner: &ClassInstance,
        bindings: &mut Vec<Binding>,
        span: severian_source::Span,
    ) -> Result<(BindingId, Block), Diagnostic> {
        if self.declarations.contains("self") {
            return Err(Diagnostic::new(
                "E000203",
                "constructor parameters cannot bind `self`",
                Some(span),
            ));
        }
        let receiver = self.new_binding_id();
        let variable = severian_hir::VariableId(receiver.0);
        // An empty aggregate reserves storage without evaluating defaults or
        // constructing temporary owned values. finish_constructor proves every
        // field read and every successful exit initialized before this reaches MIR.
        bindings.push(Binding {
            id: receiver,
            variable,
            type_id: owner.ty,
            value: Expression {
                id: self.next_id(),
                type_id: owner.ty,
                kind: ExpressionKind::Aggregate {
                    class: owner.ty,
                    fields: Vec::new(),
                },
                span,
            },
            mutable: true,
            preserve_error: false,
            span,
        });
        self.names
            .insert("self".into(), (receiver, variable, owner.ty));
        self.declarations.insert("self".into());
        self.mutable_variables.insert(variable);
        self.active_receiver = Some((receiver, owner.clone()));
        let mut body = Block {
            statements: vec![Statement::Binding(receiver)],
        };
        for (index, field) in owner.source_fields.iter().enumerate() {
            if let Some(default) = &field.default {
                let value = self.class_field_default(owner.ty, default, owner.fields[index].ty)?;
                body.statements.push(Statement::FieldSet {
                    binding: receiver,
                    field: index as u32,
                    value,
                });
            }
        }
        Ok((receiver, body))
    }

    pub(super) fn finish_constructor(
        &mut self,
        owner: &ClassInstance,
        receiver: BindingId,
        body: &mut Block,
        bindings: &mut Vec<Binding>,
        result_type: TypeId,
        span: severian_source::Span,
    ) -> Result<(), Diagnostic> {
        let checker = Initialization {
            owner,
            receiver,
            bindings,
            span,
        };
        checker.block(body, BTreeSet::new())?;
        let mut fields = Vec::new();
        for (index, field) in owner.fields.iter().enumerate() {
            let value = Expression {
                id: self.next_id(),
                type_id: field.ty,
                span,
                kind: ExpressionKind::Field {
                    object: Box::new(Expression {
                        id: self.next_id(),
                        type_id: owner.ty,
                        kind: ExpressionKind::Binding(receiver),
                        span,
                    }),
                    index: index as u32,
                },
            };
            fields.push(self.validate_field_value(owner, index, value, span)?);
        }
        let result = Expression {
            id: self.next_id(),
            type_id: owner.ty,
            kind: ExpressionKind::Aggregate {
                class: owner.ty,
                fields,
            },
            span,
        };
        let result = if let Some(fallible) = self.fallible_types.get(&result_type).copied() {
            self.fallible_success_expression(result_type, fallible, result, span)?
        } else {
            result
        };
        constructor_returns(body, &result, &mut |value| {
            self.check_constructor_result(owner, value, bindings, span)
        })?;
        body.statements.push(Statement::Return(Some(result)));
        Ok(())
    }

    fn check_constructor_result(
        &mut self,
        owner: &ClassInstance,
        value: &mut Expression,
        bindings: &mut Vec<Binding>,
        span: severian_source::Span,
    ) -> Result<Option<BindingId>, Diagnostic> {
        if owner
            .source_fields
            .iter()
            .all(|field| field.constraints.is_empty())
        {
            return Ok(None);
        }
        let returned = if value.type_id == owner.ty {
            value
        } else if let ExpressionKind::Variant {
            variant: 1, fields, ..
        } = &mut value.kind
        {
            let Some(returned) = fields.first_mut().filter(|field| field.type_id == owner.ty)
            else {
                return Ok(None);
            };
            returned
        } else {
            // An error return has no constructed value to validate.
            return Ok(None);
        };
        // Evaluate an explicit result once before checking its fields. This
        // also covers factory returns and keeps checks on `return self`.
        let binding = self.new_binding_id();
        bindings.push(Binding {
            id: binding,
            variable: severian_hir::VariableId(binding.0),
            type_id: owner.ty,
            value: returned.clone(),
            mutable: false,
            preserve_error: false,
            span,
        });
        let object = Expression {
            id: self.next_id(),
            type_id: owner.ty,
            span,
            kind: ExpressionKind::Binding(binding),
        };
        let previous = self.value_substitutions.clone();
        let mut fields = Vec::new();
        for (index, field) in owner.fields.iter().enumerate() {
            let field_value = Expression {
                id: self.next_id(),
                type_id: field.ty,
                span,
                kind: ExpressionKind::Field {
                    object: Box::new(object.clone()),
                    index: index as u32,
                },
            };
            self.value_substitutions
                .insert(field.name.clone(), field_value.clone());
            fields.push(field_value);
        }
        let checked = fields
            .into_iter()
            .enumerate()
            .map(|(index, field)| self.validate_field_value(owner, index, field, span))
            .collect::<Result<Vec<_>, _>>();
        self.value_substitutions = previous;
        returned.id = self.next_id();
        returned.kind = ExpressionKind::Aggregate {
            class: owner.ty,
            fields: checked?,
        };
        Ok(Some(binding))
    }
}

struct Initialization<'a> {
    owner: &'a ClassInstance,
    receiver: BindingId,
    bindings: &'a [Binding],
    span: severian_source::Span,
}

impl Initialization<'_> {
    fn require(
        &self,
        initialized: &BTreeSet<u32>,
        field: u32,
        span: severian_source::Span,
    ) -> Result<(), Diagnostic> {
        if !initialized.contains(&field) {
            return Err(Diagnostic::new(
                "E000221",
                format!(
                    "constructor `{}` uses or returns uninitialized field `{}`",
                    self.owner.name, self.owner.fields[field as usize].name
                ),
                Some(span),
            ));
        }
        Ok(())
    }
    fn complete(&self, initialized: &BTreeSet<u32>) -> Result<(), Diagnostic> {
        for field in 0..self.owner.fields.len() {
            self.require(initialized, field as u32, self.span)?;
        }
        Ok(())
    }
    fn expression(
        &self,
        value: &Expression,
        initialized: &BTreeSet<u32>,
    ) -> Result<(), Diagnostic> {
        match &value.kind {
            ExpressionKind::Binding(binding) | ExpressionKind::AddressOf(binding)
                if *binding == self.receiver =>
            {
                self.complete(initialized)?
            }
            ExpressionKind::Field { object, index } if matches!(object.kind, ExpressionKind::Binding(binding) if binding == self.receiver) => {
                self.require(initialized, *index, value.span)?
            }
            ExpressionKind::Aggregate { fields, .. } | ExpressionKind::Variant { fields, .. } => {
                for field in fields {
                    self.expression(field, initialized)?;
                }
            }
            ExpressionKind::Call { arguments, .. } => {
                for argument in arguments {
                    self.expression(argument, initialized)?;
                }
            }
            ExpressionKind::Field { object, .. } => self.expression(object, initialized)?,
            ExpressionKind::Unary { operand, .. }
            | ExpressionKind::Convert { operand, .. }
            | ExpressionKind::Borrow { operand, .. }
            | ExpressionKind::Move(operand)
            | ExpressionKind::Throw(operand)
            | ExpressionKind::Await(operand) => self.expression(operand, initialized)?,
            ExpressionKind::Async { expression, .. } => self.expression(expression, initialized)?,
            ExpressionKind::AsyncFieldUpdate {
                binding,
                field,
                value,
                ..
            } => {
                if *binding == self.receiver {
                    self.require(initialized, *field, value.span)?;
                }
                self.expression(value, initialized)?;
            }
            ExpressionKind::Binary { left, right, .. } => {
                self.expression(left, initialized)?;
                self.expression(right, initialized)?;
            }
            ExpressionKind::Fallback {
                condition,
                value,
                fallback,
            } => {
                self.expression(condition, initialized)?;
                self.expression(value, initialized)?;
                self.expression(fallback, initialized)?;
            }
            _ => {}
        }
        Ok(())
    }
    // None means this path cannot fall through. Branch joins intersect only
    // continuing paths; loops never promise that their body executes once.
    fn walk(
        &self,
        body: &Block,
        mut initialized: BTreeSet<u32>,
    ) -> Result<Option<BTreeSet<u32>>, Diagnostic> {
        for statement in &body.statements {
            match statement {
                Statement::Binding(id) if *id == self.receiver => {}
                Statement::Binding(id) => {
                    let binding = self
                        .bindings
                        .iter()
                        .find(|binding| binding.id == *id)
                        .ok_or_else(|| {
                            Diagnostic::new(
                                "E000221",
                                "constructor binding metadata is missing",
                                Some(self.span),
                            )
                        })?;
                    self.expression(&binding.value, &initialized)?;
                }
                Statement::FieldSet {
                    binding,
                    field,
                    value,
                } => {
                    self.expression(value, &initialized)?;
                    if *binding == self.receiver {
                        initialized.insert(*field);
                    }
                }
                Statement::FieldUpdate {
                    binding,
                    field,
                    value,
                    ..
                } => {
                    if *binding == self.receiver {
                        self.require(&initialized, *field, value.span)?;
                    }
                    self.expression(value, &initialized)?;
                }
                Statement::Expression(value) | Statement::Destroy(value) => {
                    self.expression(value, &initialized)?;
                    if matches!(value.kind, ExpressionKind::Throw(_)) {
                        return Ok(None);
                    }
                }
                Statement::Assert {
                    condition, message, ..
                } => {
                    self.expression(condition, &initialized)?;
                    if let Some(message) = message {
                        self.expression(message, &initialized)?;
                    }
                }
                Statement::Return(value) => {
                    if let Some(value) = value {
                        self.expression(value, &initialized)?;
                    } else {
                        self.complete(&initialized)?;
                    }
                    return Ok(None);
                }
                Statement::Sequence(body) | Statement::Placement { body, .. } => {
                    match self.walk(body, initialized)? {
                        Some(next) => initialized = next,
                        None => return Ok(None),
                    }
                }
                Statement::If {
                    condition,
                    then_block,
                    else_block,
                } => {
                    self.expression(condition, &initialized)?;
                    match merge(
                        self.walk(then_block, initialized.clone())?,
                        self.walk(else_block, initialized.clone())?,
                    ) {
                        Some(next) => initialized = next,
                        None => return Ok(None),
                    }
                }
                Statement::While {
                    condition, body, ..
                } => {
                    self.expression(condition, &initialized)?;
                    self.walk(body, initialized.clone())?;
                }
                Statement::Try {
                    body, catch_body, ..
                } => {
                    match merge(
                        self.walk(body, initialized.clone())?,
                        self.walk(catch_body, initialized.clone())?,
                    ) {
                        Some(next) => initialized = next,
                        None => return Ok(None),
                    }
                }
                Statement::Match { subject, arms } => {
                    self.expression(subject, &initialized)?;
                    let mut joined = None;
                    for arm in arms {
                        joined = merge(joined, self.walk(&arm.body, initialized.clone())?);
                    }
                    match joined {
                        Some(next) => initialized = next,
                        None => return Ok(None),
                    }
                }
                Statement::ExpectThrow { body, .. } => {
                    self.walk(body, initialized.clone())?;
                }
                Statement::Break { .. } | Statement::Continue { .. } => return Ok(None),
            }
        }
        Ok(Some(initialized))
    }
    fn block(&self, body: &Block, initialized: BTreeSet<u32>) -> Result<(), Diagnostic> {
        if let Some(initialized) = self.walk(body, initialized)? {
            self.complete(&initialized)?;
        }
        Ok(())
    }
}

fn merge(left: Option<BTreeSet<u32>>, right: Option<BTreeSet<u32>>) -> Option<BTreeSet<u32>> {
    match (left, right) {
        (Some(left), Some(right)) => Some(left.intersection(&right).copied().collect()),
        (left, right) => left.or(right),
    }
}

fn constructor_returns(
    body: &mut Block,
    result: &Expression,
    check: &mut impl FnMut(&mut Expression) -> Result<Option<BindingId>, Diagnostic>,
) -> Result<(), Diagnostic> {
    for statement in &mut body.statements {
        match statement {
            Statement::Return(value) if value.is_none() => *value = Some(result.clone()),
            Statement::Return(Some(value)) => {
                if let Some(binding) = check(value)? {
                    *statement = Statement::Sequence(Block {
                        statements: vec![
                            Statement::Binding(binding),
                            Statement::Return(Some(value.clone())),
                        ],
                    });
                }
            }
            Statement::Sequence(body)
            | Statement::Placement { body, .. }
            | Statement::While { body, .. }
            | Statement::ExpectThrow { body, .. } => constructor_returns(body, result, check)?,
            Statement::If {
                then_block,
                else_block,
                ..
            } => {
                constructor_returns(then_block, result, check)?;
                constructor_returns(else_block, result, check)?;
            }
            Statement::Try {
                body, catch_body, ..
            } => {
                constructor_returns(body, result, check)?;
                constructor_returns(catch_body, result, check)?;
            }
            Statement::Match { arms, .. } => {
                for arm in arms {
                    constructor_returns(&mut arm.body, result, check)?;
                }
            }
            _ => {}
        }
    }
    Ok(())
}
