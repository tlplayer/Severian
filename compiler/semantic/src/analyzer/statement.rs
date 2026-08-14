use super::*;

pub(super) fn lower_block(
    block: &Block,
    scope: &mut HashMap<String, Binding>,
    return_type: ValueType,
    signatures: &HashMap<String, Signature>,
    aliases: &HashMap<String, String>,
) -> Result<Vec<Instruction>, SemanticError> {
    let mut instructions = Vec::new();
    for statement in &block.statements {
        match statement {
            Stmt::Function(function) => {
                if !function.generic_params.is_empty() {
                    return Err(error(
                        function.span,
                        "nested functions do not support generic parameters",
                    ));
                }
                if function
                    .params
                    .iter()
                    .any(|parameter| parameter.default.is_some())
                {
                    return Err(error(
                        function.span,
                        "nested function parameters do not support default values",
                    ));
                }
                if !function.decorators.is_empty() || !function.tests.is_empty() {
                    return Err(error(
                        function.span,
                        "nested functions do not support decorators or attached tests",
                    ));
                }

                let mut function_scope = scope.clone();
                let mut params = Vec::new();
                for parameter in &function.params {
                    let ty = parameter
                        .ty
                        .as_ref()
                        .map(lower_type)
                        .transpose()?
                        .unwrap_or(ValueType::Any);
                    function_scope.insert(
                        parameter.name.name.clone(),
                        Binding {
                            reference: source_binding(&parameter.name),
                            ty,
                            class: parameter.ty.as_ref().and_then(class_type_name),
                            function_return: function_return_type(parameter.ty.as_ref()),
                            collection_len: None,
                            mutable: false,
                            field: false,
                            integer_max: None,
                            known_integer: None,
                            any_origin: declared_any_origin(parameter.ty.as_ref(), ty),
                        },
                    );
                    params.push(Parameter {
                        name: function_scope[&parameter.name.name].reference.clone(),
                        ty,
                        default: None,
                        receiver: parameter
                            .ty
                            .as_ref()
                            .and_then(|ty| declared_receiver_type(ty, aliases)),
                    });
                }
                let return_type = function
                    .return_type
                    .as_ref()
                    .map(lower_type)
                    .transpose()?
                    .unwrap_or(ValueType::Unit);
                let mut body = lower_block(
                    &function.body,
                    &mut function_scope,
                    return_type,
                    signatures,
                    aliases,
                )?;
                if return_type != ValueType::Unit && !always_returns(&body) {
                    return Err(error(
                        function.span,
                        format!("function `{}` must return a value", function.name.name),
                    ));
                }
                let contract = lower_function_contract(
                    function.contract.as_ref(),
                    &function_scope,
                    signatures,
                    aliases,
                )?;
                enforce_function_contract(&mut body, contract.as_ref());

                let binding = Binding {
                    reference: source_binding(&function.name),
                    ty: ValueType::Function,
                    class: None,
                    function_return: Some(return_type),
                    collection_len: None,
                    mutable: false,
                    field: false,
                    integer_max: None,
                    known_integer: None,
                    any_origin: None,
                };
                if scope
                    .insert(function.name.name.clone(), binding.clone())
                    .is_some()
                {
                    return Err(error(
                        function.name.span,
                        format!("duplicate binding `{}`", function.name.name),
                    ));
                }
                instructions.push(Instruction::Let {
                    name: binding.reference,
                    value: Expression::Typed {
                        id: HirId::from_source_range(function.span.start, function.span.end),
                        ty: ValueType::Function,
                        any_origin: None,
                        expression: Box::new(Expression::Closure {
                            params,
                            body,
                            return_type,
                        }),
                    },
                });
            }
            Stmt::Let(binding) => {
                let source = binding.value.as_ref().ok_or_else(|| {
                    error(
                        binding.span,
                        format!(
                            "E000205: binding `{}` requires an initializer",
                            binding.name.name
                        ),
                    )
                })?;
                if binding.ty.as_ref().is_some_and(|ty| {
                    matches!(ty, Type::Named(path) if path.segments.first().is_some_and(|segment| segment.name == "u8"))
                }) && constant_integer(source).is_some_and(|value| !(0..=u8::MAX as i64).contains(&value))
                {
                    return Err(error(
                        source.span(),
                        "E000501: Checked integer arithmetic cannot produce a value outside the destination type.",
                    ));
                }
                if checked_integer_overflow(source, scope) {
                    return Err(error(
                        source.span(),
                        "E000501: Checked integer arithmetic cannot produce a value outside the destination type.",
                    ));
                }
                let (value, inferred) = lower_expression(source, scope, signatures, aliases)?;
                let propagates = inferred == ValueType::Result;
                let declared = binding.ty.as_ref().map(lower_type).transpose()?;
                if let Some(declared) = declared.filter(|_| !propagates) {
                    compatible(binding.span, inferred, declared)?;
                }
                // `=` and `:=` are the propagation boundary for a Result. The
                // binding receives the success payload; an explicit annotation
                // therefore describes that payload rather than the Result
                // carrier itself.
                let ty = if propagates {
                    declared.unwrap_or(ValueType::Any)
                } else {
                    declared.unwrap_or(inferred)
                };
                let any_origin = declared_any_origin(binding.ty.as_ref(), ty)
                    .or_else(|| value.any_origin())
                    .or_else(|| {
                        matches!(ty, ValueType::Any | ValueType::TensorAny)
                            .then_some(AnyOrigin::InferenceFallback)
                    });
                let integer_max = binding
                    .ty
                    .as_ref()
                    .filter(|ty| named_type_is(ty, "u8"))
                    .map(|_| u8::MAX as i64);
                let known_integer = constant_integer(source);
                let receiver = propagates
                    .then(|| file_read_receiver_type(source, aliases))
                    .flatten();
                let class = binding
                    .ty
                    .as_ref()
                    .and_then(class_type_name)
                    .or_else(|| receiver.as_ref().map(|receiver| receiver.name.clone()))
                    .or_else(|| expression_class(source, scope, aliases));
                if scope
                    .get(&binding.name.name)
                    .is_some_and(|existing| existing.field || existing.mutable)
                {
                    let existing = scope[&binding.name.name].reference.clone();
                    if propagates {
                        instructions.push(Instruction::TryLet {
                            name: existing,
                            value,
                            receiver,
                        });
                    } else {
                        instructions.push(Instruction::Assign {
                            target: Expression::Variable(existing),
                            op: AssignmentOp::Assign,
                            value,
                        });
                    }
                    continue;
                }
                if scope
                    .insert(
                        binding.name.name.clone(),
                        Binding {
                            reference: source_binding(&binding.name),
                            ty,
                            class,
                            function_return: None,
                            collection_len: binding.value.as_ref().and_then(collection_length),
                            mutable: binding.kind == LetKind::Changeable,
                            field: false,
                            integer_max,
                            known_integer,
                            any_origin,
                        },
                    )
                    .is_some()
                {
                    return Err(error(
                        binding.name.span,
                        format!("duplicate binding `{}`", binding.name.name),
                    ));
                }
                if propagates {
                    instructions.push(Instruction::TryLet {
                        name: scope[&binding.name.name].reference.clone(),
                        value,
                        receiver,
                    });
                } else {
                    instructions.push(Instruction::Let {
                        name: scope[&binding.name.name].reference.clone(),
                        value,
                    });
                }
            }
            Stmt::DestructureLet(binding) => {
                let (value, value_type) =
                    lower_expression(&binding.value, scope, signatures, aliases)?;
                let temporary = format!("__destructure_{}", binding.span.start);
                let temporary = BindingRef::synthetic(temporary);
                if value_type == ValueType::Result {
                    instructions.push(Instruction::TryLet {
                        name: temporary.clone(),
                        value,
                        receiver: None,
                    });
                } else {
                    instructions.push(Instruction::Let {
                        name: temporary.clone(),
                        value,
                    });
                }
                for (index, name) in binding.names.iter().enumerate() {
                    let reference = source_binding(name);
                    scope.insert(
                        name.name.clone(),
                        Binding {
                            reference: reference.clone(),
                            ty: ValueType::Any,
                            class: None,
                            function_return: None,
                            collection_len: None,
                            mutable: false,
                            field: false,
                            integer_max: None,
                            known_integer: None,
                            any_origin: Some(AnyOrigin::LostTypeInformation),
                        },
                    );
                    instructions.push(Instruction::Let {
                        name: reference,
                        value: Expression::Index {
                            object: Box::new(Expression::Variable(temporary.clone())),
                            index: Box::new(Expression::Integer(index as i64)),
                        },
                    });
                }
            }
            Stmt::Assign(assignment) => {
                let (target, target_type) =
                    lower_expression(&assignment.target, scope, signatures, aliases)?;
                if let Expr::Identifier(name) = &assignment.target {
                    if !scope.get(&name.name).is_some_and(|binding| binding.mutable) {
                        return Err(error(
                            name.span,
                            format!("binding `{}` is not changeable", name.name),
                        ));
                    }
                } else if let Expr::Member(member) = &assignment.target {
                    if let Expr::Identifier(object) = member.object.as_ref() {
                        if !scope
                            .get(&object.name)
                            .is_some_and(|binding| binding.mutable || binding.field)
                        {
                            return Err(error(
                                object.span,
                                format!(
                                    "object `{}` is not changeable; bind it with `:=` before assigning a field",
                                    object.name
                                ),
                            ));
                        }
                    }
                } else if !matches!(assignment.target, Expr::Index(_)) {
                    return Err(error(
                        assignment.target.span(),
                        "assignment target is not mutable",
                    ));
                }
                let (mut value, mut value_type) =
                    lower_expression(&assignment.value, scope, signatures, aliases)?;
                if value_type == ValueType::Result {
                    let temporary = format!("__assignment_{}", assignment.span.start);
                    let temporary = BindingRef::synthetic(temporary);
                    instructions.push(Instruction::TryLet {
                        name: temporary.clone(),
                        value,
                        receiver: None,
                    });
                    value = Expression::Variable(temporary);
                    value_type = ValueType::Any;
                }
                if target_type != ValueType::Any && value_type != ValueType::Any {
                    compatible(assignment.span, value_type, target_type)?;
                }
                instructions.push(Instruction::Assign {
                    target,
                    op: match assignment.op {
                        AstAssignOp::Assign => AssignmentOp::Assign,
                        AstAssignOp::AddAssign => AssignmentOp::Add,
                        AstAssignOp::SubAssign => AssignmentOp::Sub,
                        AstAssignOp::MulAssign => AssignmentOp::Mul,
                        AstAssignOp::DivAssign => AssignmentOp::Div,
                        AstAssignOp::ModAssign => AssignmentOp::Mod,
                    },
                    value,
                });
            }
            Stmt::TryBind(binding) => {
                let (value, inferred) =
                    lower_expression(&binding.value, scope, signatures, aliases)?;
                if !matches!(inferred, ValueType::Result | ValueType::Any) {
                    return Err(error(
                        binding.value.span(),
                        "`?=` safely captures a Result and requires a fallible expression",
                    ));
                }
                if scope
                    .insert(
                        binding.name.name.clone(),
                        Binding {
                            reference: source_binding(&binding.name),
                            ty: ValueType::Result,
                            class: None,
                            function_return: None,
                            collection_len: None,
                            mutable: false,
                            field: false,
                            integer_max: None,
                            known_integer: None,
                            any_origin: None,
                        },
                    )
                    .is_some()
                {
                    return Err(error(
                        binding.name.span,
                        format!("duplicate binding `{}`", binding.name.name),
                    ));
                }
                instructions.push(Instruction::Let {
                    name: scope[&binding.name.name].reference.clone(),
                    value,
                });
            }
            Stmt::Expr(expression) => {
                let (expression, expression_type) =
                    lower_expression(expression, scope, signatures, aliases)?;
                if let Expression::MethodCall { object, method, .. } = expression.kind() {
                    if collection_shape_mutating_method(method) {
                        if let Expression::Variable(name) = object.kind() {
                            if let Some(binding) = scope.get_mut(&name.name) {
                                binding.collection_len = None;
                            }
                        }
                    }
                }
                if expression_type == ValueType::Result {
                    return Err(error(
                        statement.span(),
                        "E000801: A recoverable error must be propagated, handled, or explicitly discarded with a reason.",
                    ));
                }
                match expression.kind() {
                    Expression::Call { target, args } if target.name == "print" => {
                        let mut args = args.clone();
                        let value = if args.len() == 1 {
                            args.remove(0)
                        } else {
                            Expression::Typed {
                                id: HirId::from_source_range(
                                    statement.span().start,
                                    statement.span().end,
                                ),
                                ty: ValueType::Tuple,
                                any_origin: None,
                                expression: Box::new(Expression::PrintArgs(args)),
                            }
                        };
                        instructions.push(Instruction::Print(value));
                    }
                    _ => instructions.push(Instruction::Evaluate(expression)),
                }
            }
            Stmt::Assert(assertion) => {
                let (condition, ty) =
                    lower_expression(&assertion.condition, scope, signatures, aliases)?;
                compatible(assertion.condition.span(), ty, ValueType::Bool)?;
                instructions.push(Instruction::Assert(condition));
            }
            Stmt::Return(statement) => {
                let value = statement
                    .value
                    .as_ref()
                    .map(|value| lower_expression(value, scope, signatures, aliases))
                    .transpose()?;
                let actual = value.as_ref().map_or(ValueType::Unit, |(_, ty)| *ty);
                let value = value.map(|(value, _)| value);
                let value = match (return_type, actual, value) {
                    (ValueType::Result, ValueType::Result, value) => value,
                    (ValueType::Result, ValueType::Unit, None) => Some(Expression::Variant {
                        type_id: Some(TypeDefinitionId::from_name("Result")),
                        variant_id: VariantId::from_name("ok"),
                        name: "ok".into(),
                        fields: Vec::new(),
                    }),
                    (ValueType::Result, _, Some(value)) => Some(Expression::Variant {
                        type_id: Some(TypeDefinitionId::from_name("Result")),
                        variant_id: VariantId::from_name("ok"),
                        name: "ok".into(),
                        fields: vec![value],
                    }),
                    (ValueType::Option, ValueType::Option, value) => value,
                    (_, _, value) => {
                        compatible(statement.span, actual, return_type)?;
                        value
                    }
                };
                instructions.push(Instruction::Return(value));
            }
            Stmt::If(statement) => {
                let (condition, ty) =
                    lower_expression(&statement.condition, scope, signatures, aliases)?;
                compatible(statement.condition.span(), ty, ValueType::Bool)?;
                let mut then_scope = scope.clone();
                let then_instructions = lower_block(
                    &statement.then_block,
                    &mut then_scope,
                    return_type,
                    signatures,
                    aliases,
                )?;
                let else_instructions = match &statement.else_branch {
                    None => Vec::new(),
                    Some(ElseBranch::Block(block)) => {
                        let mut else_scope = scope.clone();
                        lower_block(block, &mut else_scope, return_type, signatures, aliases)?
                    }
                    Some(ElseBranch::If(branch)) => {
                        let mut else_scope = scope.clone();
                        lower_block(
                            &Block {
                                span: branch.span,
                                statements: vec![Stmt::If((**branch).clone())],
                            },
                            &mut else_scope,
                            return_type,
                            signatures,
                            aliases,
                        )?
                    }
                };
                instructions.push(Instruction::If {
                    condition,
                    then_instructions,
                    else_instructions,
                });
            }
            Stmt::While(statement) => {
                let setup = statement
                    .setup
                    .as_ref()
                    .map(|setup| {
                        let lowered = lower_block(
                            &Block {
                                span: setup.span(),
                                statements: vec![(**setup).clone()],
                            },
                            scope,
                            return_type,
                            signatures,
                            aliases,
                        )?;
                        Ok(Box::new(lowered.into_iter().next().unwrap()))
                    })
                    .transpose()?;
                let (condition, ty) =
                    lower_expression(&statement.condition, scope, signatures, aliases)?;
                compatible(statement.condition.span(), ty, ValueType::Bool)?;
                let capabilities = statement
                    .capabilities
                    .iter()
                    .map(|capability| {
                        lower_expression(capability, scope, signatures, aliases)
                            .map(|(capability, _)| capability)
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                let mut body_scope = scope.clone();
                let body = lower_block(
                    &statement.body,
                    &mut body_scope,
                    return_type,
                    signatures,
                    aliases,
                )?;
                propagate_unknown_collection_shapes(scope, &body_scope);
                instructions.push(Instruction::While {
                    setup,
                    capabilities,
                    condition,
                    instructions: body,
                });
            }
            Stmt::For(statement) => {
                let setup = statement
                    .setup
                    .as_ref()
                    .map(|setup| {
                        let lowered = lower_block(
                            &Block {
                                span: setup.span(),
                                statements: vec![(**setup).clone()],
                            },
                            scope,
                            return_type,
                            signatures,
                            aliases,
                        )?;
                        Ok(Box::new(lowered.into_iter().next().unwrap()))
                    })
                    .transpose()?;
                if inclusive_collection_range(&statement.iterable) {
                    return Err(error(
                        statement.iterable.span(),
                        "E000402: An inclusive range ending at a collection's element count includes one invalid index.",
                    ));
                }
                let (iterable, _) =
                    lower_expression(&statement.iterable, scope, signatures, aliases)?;
                let mut body_scope = scope.clone();
                let pattern = lower_pattern(&statement.pattern, &mut body_scope, aliases)?;
                let body = lower_block(
                    &statement.body,
                    &mut body_scope,
                    return_type,
                    signatures,
                    aliases,
                )?;
                propagate_unknown_collection_shapes(scope, &body_scope);
                instructions.push(Instruction::For {
                    setup,
                    pattern,
                    iterable,
                    instructions: body,
                });
            }
            Stmt::Switch(statement) => {
                validate_exhaustive_enum_switch(statement, scope, aliases)?;
                let file_receiver = (statement.values.len() == 1)
                    .then(|| file_read_receiver_type(&statement.values[0], aliases))
                    .flatten();
                let setup = statement
                    .setup
                    .as_ref()
                    .map(|setup| {
                        let lowered = lower_block(
                            &Block {
                                span: setup.span(),
                                statements: vec![(**setup).clone()],
                            },
                            scope,
                            return_type,
                            signatures,
                            aliases,
                        )?;
                        Ok(Box::new(lowered.into_iter().next().unwrap()))
                    })
                    .transpose()?;
                let repeat_condition = statement
                    .repeat_condition
                    .as_ref()
                    .map(|condition| {
                        let (condition, ty) =
                            lower_expression(condition, scope, signatures, aliases)?;
                        compatible(statement.span, ty, ValueType::Bool)?;
                        Ok(condition)
                    })
                    .transpose()?;
                let mut arms = Vec::new();
                for arm in &statement.arms {
                    let mut arm_scope = scope.clone();
                    let pattern = lower_pattern(&arm.pattern, &mut arm_scope, aliases)?;
                    if let Some(receiver) = &file_receiver {
                        refine_success_pattern_bindings(&pattern, &receiver.name, &mut arm_scope);
                    }
                    let receivers = file_receiver
                        .as_ref()
                        .map(|receiver| success_pattern_receivers(&pattern, receiver))
                        .unwrap_or_default();
                    let source = arm
                        .source
                        .as_ref()
                        .map(|source| {
                            lower_expression(source, &arm_scope, signatures, aliases)
                                .map(|(source, _)| source)
                        })
                        .transpose()?;
                    let guard = arm
                        .guard
                        .as_ref()
                        .map(|guard| {
                            lower_expression(guard, &arm_scope, signatures, aliases)
                                .map(|(guard, _)| guard)
                        })
                        .transpose()?;
                    let arm_instructions =
                        lower_block(&arm.body, &mut arm_scope, return_type, signatures, aliases)?;
                    arms.push(HirSwitchArm {
                        source,
                        pattern,
                        guard,
                        instructions: arm_instructions,
                        receivers,
                    });
                }
                if statement.values.len() == 1
                    && statement.repeat_condition.is_none()
                    && statement.setup.is_none()
                    && statement.arms.iter().all(|arm| arm.source.is_none())
                {
                    let value =
                        lower_expression(&statement.values[0], scope, signatures, aliases)?.0;
                    instructions.push(Instruction::Switch { value, arms });
                } else {
                    let channels = statement
                        .values
                        .iter()
                        .map(|channel| {
                            lower_expression(channel, scope, signatures, aliases)
                                .map(|(channel, _)| channel)
                        })
                        .collect::<Result<Vec<_>, _>>()?;
                    instructions.push(Instruction::ChannelSwitch {
                        channels,
                        setup,
                        repeat_condition,
                        arms,
                    });
                }
            }
            Stmt::Unsafe(block) => {
                instructions.extend(lower_block(
                    &block.body,
                    scope,
                    return_type,
                    signatures,
                    aliases,
                )?);
            }
            Stmt::With(block) => {
                let mut resources = Vec::new();
                let mut placement = TaskPlacement::Default;
                for resource in &block.resources {
                    if let Expr::Identifier(identifier) = resource {
                        match identifier.name.as_str() {
                            "gpu" | "simd" => {
                                if !aliases.values().any(|module| module == "parallel") {
                                    return Err(error(
                                        identifier.span,
                                        format!(
                                            "execution placement `{}` requires `import parallel`",
                                            identifier.name
                                        ),
                                    ));
                                }
                                let requested = if identifier.name == "gpu" {
                                    TaskPlacement::Gpu
                                } else {
                                    TaskPlacement::Simd
                                };
                                if placement != TaskPlacement::Default {
                                    return Err(error(
                                        identifier.span,
                                        "an execution region accepts only one placement",
                                    ));
                                }
                                placement = requested;
                                continue;
                            }
                            "self" | "runtime" | "local" | "simt" => continue,
                            _ => {}
                        }
                    }
                    resources.push(lower_expression(resource, scope, signatures, aliases)?.0);
                }
                let mut with_scope = scope.clone();
                let body = lower_block(
                    &block.body,
                    &mut with_scope,
                    return_type,
                    signatures,
                    aliases,
                )?;
                instructions.push(Instruction::With {
                    placement,
                    resources,
                    instructions: body,
                });
            }
            Stmt::Break(_) => instructions.push(Instruction::Break),
            Stmt::Continue(_) => instructions.push(Instruction::Continue),
        }
    }
    Ok(instructions)
}
