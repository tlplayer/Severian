use super::*;

pub(super) fn lower_expression(
    expression: &Expr,
    scope: &HashMap<String, Binding>,
    signatures: &HashMap<String, Signature>,
    aliases: &HashMap<String, String>,
) -> Result<(Expression, ValueType), SemanticError> {
    let (lowered, ty) = lower_expression_kind(expression, scope, signatures, aliases)?;
    let span = expression.span();
    let any_origin = expression_any_origin(expression, &lowered, ty, scope);
    Ok((
        Expression::Typed {
            id: HirId::from_source_range(span.start, span.end),
            ty,
            any_origin,
            expression: Box::new(lowered),
        },
        ty,
    ))
}

fn expression_any_origin(
    source: &Expr,
    lowered: &Expression,
    ty: ValueType,
    scope: &HashMap<String, Binding>,
) -> Option<AnyOrigin> {
    if !matches!(ty, ValueType::Any | ValueType::TensorAny) {
        return None;
    }
    if let Expr::Identifier(identifier) = source {
        return scope
            .get(&identifier.name)
            .and_then(|binding| binding.any_origin)
            .or_else(|| {
                (identifier.name == "invalid")
                    .then_some(AnyOrigin::Explicit)
                    .or(Some(AnyOrigin::UnresolvedType))
            });
    }
    let origin = match lowered {
        Expression::Call { target, args } => combine_any_origins(
            target
                .signature
                .as_ref()
                .and_then(|signature| signature.return_any_origin)
                .into_iter()
                .chain(args.iter().filter_map(Expression::any_origin)),
        ),
        Expression::CallValue { callee, args, .. } => combine_any_origins(
            callee
                .any_origin()
                .into_iter()
                .chain(args.iter().filter_map(Expression::any_origin)),
        ),
        Expression::Binary { left, right, .. } => combine_any_origins(
            [left.any_origin(), right.any_origin()]
                .into_iter()
                .flatten(),
        ),
        Expression::Unary { expression, .. }
        | Expression::Await(expression)
        | Expression::Channel(expression)
        | Expression::Ownership {
            value: expression, ..
        }
        | Expression::Task {
            value: expression, ..
        }
        | Expression::FusedPipeline {
            input: expression, ..
        } => expression.any_origin(),
        Expression::Conditional {
            then_expression,
            else_expression,
            ..
        } => combine_any_origins(
            [then_expression.any_origin(), else_expression.any_origin()]
                .into_iter()
                .flatten(),
        ),
        Expression::Index { object, .. }
        | Expression::Slice { object, .. }
        | Expression::Member { object, .. } => object.any_origin(),
        Expression::MethodCall { object, args, .. } => combine_any_origins(
            object
                .any_origin()
                .into_iter()
                .chain(args.iter().filter_map(Expression::any_origin)),
        ),
        Expression::Construct { .. }
        | Expression::ConstructFields { .. }
        | Expression::ObjectUpdate { .. }
        | Expression::Variant { .. } => Some(AnyOrigin::Explicit),
        Expression::ChaosRule { value, .. } => value.any_origin(),
        _ => None,
    };
    origin.or(Some(AnyOrigin::LostTypeInformation))
}

fn combine_any_origins(origins: impl IntoIterator<Item = AnyOrigin>) -> Option<AnyOrigin> {
    origins.into_iter().max_by_key(|origin| match origin {
        AnyOrigin::Explicit => 0,
        AnyOrigin::InferenceFallback => 1,
        AnyOrigin::UnresolvedType => 2,
        AnyOrigin::UnresolvedGeneric => 3,
        AnyOrigin::LostTypeInformation => 4,
    })
}

pub(super) fn lower_expression_kind(
    expression: &Expr,
    scope: &HashMap<String, Binding>,
    signatures: &HashMap<String, Signature>,
    aliases: &HashMap<String, String>,
) -> Result<(Expression, ValueType), SemanticError> {
    match expression {
        Expr::Literal(Literal::Integer { value, .. }) => {
            Ok((Expression::Integer(*value), ValueType::Int))
        }
        Expr::Literal(Literal::Float { value, .. }) => {
            Ok((Expression::Float(value.to_bits()), ValueType::Float))
        }
        Expr::Literal(Literal::Boolean { value, .. }) => {
            Ok((Expression::Boolean(*value), ValueType::Bool))
        }
        Expr::Literal(Literal::String { value, .. }) => {
            Ok((Expression::String(value.clone()), ValueType::String))
        }
        Expr::Identifier(identifier) => {
            if let Some(binding) = scope.get(&identifier.name) {
                Ok((Expression::Variable(binding.reference.clone()), binding.ty))
            } else if signatures.contains_key(&identifier.name) {
                Ok((
                    Expression::Function(signatures[&identifier.name].target.clone()),
                    ValueType::Function,
                ))
            } else if identifier.name == "invalid" {
                Ok((
                    Expression::Variant {
                        type_id: None,
                        variant_id: VariantId::from_name("invalid"),
                        name: "invalid".into(),
                        fields: Vec::new(),
                    },
                    ValueType::Any,
                ))
            } else if identifier.name == "absent" {
                Ok((
                    Expression::Variant {
                        type_id: Some(TypeDefinitionId::from_name("Option")),
                        variant_id: VariantId::from_name("absent"),
                        name: "absent".into(),
                        fields: Vec::new(),
                    },
                    ValueType::Option,
                ))
            } else if identifier.name == "None" {
                Ok((
                    Expression::Variant {
                        type_id: Some(TypeDefinitionId::from_name("Option")),
                        variant_id: VariantId::from_name("None"),
                        name: "None".into(),
                        fields: Vec::new(),
                    },
                    ValueType::Option,
                ))
            } else if identifier
                .name
                .as_bytes()
                .first()
                .is_some_and(u8::is_ascii_uppercase)
            {
                Ok((
                    Expression::Variant {
                        type_id: None,
                        variant_id: VariantId::from_name(&identifier.name),
                        name: identifier.name.clone(),
                        fields: Vec::new(),
                    },
                    ValueType::Any,
                ))
            } else {
                Err(error(
                    identifier.span,
                    format!("unknown binding `{}`", identifier.name),
                ))
            }
        }
        Expr::Unary(unary) => {
            let (expression, ty) = lower_expression(&unary.expr, scope, signatures, aliases)?;
            let (op, expected, result) = match unary.op {
                AstUnaryOp::Negate => (UnaryOp::Negate, ty, ty),
                AstUnaryOp::Not => (UnaryOp::Not, ValueType::Bool, ValueType::Bool),
            };
            compatible(unary.span, ty, expected)?;
            Ok((
                Expression::Unary {
                    op,
                    expression: Box::new(expression),
                },
                result,
            ))
        }
        Expr::Binary(binary) => {
            let (left, left_type) = lower_expression(&binary.left, scope, signatures, aliases)?;
            let (right, right_type) = lower_expression(&binary.right, scope, signatures, aliases)?;
            if left_type == ValueType::Result || right_type == ValueType::Result {
                return Err(error(
                    binary.span,
                    "E000801: a recoverable Result cannot be used as an operator operand; bind it to propagate success or handle it with switch",
                ));
            }
            let (op, result_type) = match binary.op {
                AstBinaryOp::Add
                    if left_type == ValueType::List && right_type == ValueType::List =>
                {
                    (BinaryOp::Add, ValueType::List)
                }
                AstBinaryOp::Add => (
                    BinaryOp::Add,
                    merge_numeric(left_type, right_type, binary.span)?,
                ),
                AstBinaryOp::Sub => (
                    BinaryOp::Sub,
                    merge_numeric(left_type, right_type, binary.span)?,
                ),
                AstBinaryOp::Mul
                    if matches!(
                        (left_type, right_type),
                        (ValueType::List, ValueType::Int) | (ValueType::Int, ValueType::List)
                    ) =>
                {
                    (BinaryOp::Mul, ValueType::List)
                }
                AstBinaryOp::Mul => (
                    BinaryOp::Mul,
                    merge_numeric(left_type, right_type, binary.span)?,
                ),
                AstBinaryOp::Div | AstBinaryOp::Mod
                    if constant_integer(&binary.right) == Some(0) =>
                {
                    return Err(error(
                        binary.right.span(),
                        "E000502: division by zero is known at compile time",
                    ));
                }
                AstBinaryOp::Div => (
                    BinaryOp::Div,
                    merge_numeric(left_type, right_type, binary.span)?,
                ),
                AstBinaryOp::Mod => (
                    BinaryOp::Mod,
                    merge_numeric(left_type, right_type, binary.span)?,
                ),
                AstBinaryOp::Power => {
                    let result = power_type(left_type, right_type, binary.span)?;
                    (BinaryOp::Power, result)
                }
                AstBinaryOp::MatMul => {
                    if let Some(candidates) = aliases.get("__semantic.operator_candidates.@") {
                        let candidates = candidates
                            .split(',')
                            .filter(|candidate| !candidate.is_empty())
                            .collect::<Vec<_>>();
                        if aliases.get("__semantic.operator.@").is_none() && candidates.len() > 1 {
                            return Err(error(
                                binary.span,
                                format!(
                                    "E000210: ambiguous operator `@`; provided by {}; select a lowering context with the semantic decorator",
                                    candidates.join(", ")
                                ),
                            ));
                        }
                        if candidates.is_empty() {
                            return Err(error(
                                binary.span,
                                "operator `@` has no valid provider in the active semantic context",
                            ));
                        }
                    } else {
                        let package = aliases
                            .get("__symbol.@")
                            .or_else(|| aliases.get("__symbol.X"))
                            .map(String::as_str);
                        if package != Some("tensor") {
                            return Err(error(
                                binary.span,
                                "operator `@` requires a tensor semantic decorator with an explicit provider",
                            ));
                        }
                    }
                    let result_type =
                        tensor::infer_matmul_operator(left_type, right_type, binary.span)?;
                    let target = signatures
                        .get("tensor.ranked_matmul")
                        .map(|signature| signature.target.clone())
                        .unwrap_or_else(|| {
                            CallTarget::native(
                                "tensor.ranked_matmul",
                                severian_hir::TENSOR_MATMUL_NATIVE_SYMBOL,
                            )
                        });
                    return Ok((
                        Expression::Call {
                            target,
                            args: vec![left, right],
                        },
                        result_type,
                    ));
                }
                AstBinaryOp::BitOr | AstBinaryOp::BitXor | AstBinaryOp::BitAnd => {
                    let (symbol, op) = match binary.op {
                        AstBinaryOp::BitOr => ("|", BinaryOp::BitOr),
                        AstBinaryOp::BitXor => ("^", BinaryOp::BitXor),
                        AstBinaryOp::BitAnd => ("&", BinaryOp::BitAnd),
                        _ => unreachable!(),
                    };
                    let selected = aliases
                        .get(&format!("__symbol.{symbol}"))
                        .map(String::as_str);
                    if let Some(package) = selected {
                        if package != "bits" {
                            return Err(error(
                                binary.span,
                                format!(
                                    "operator `{symbol}` is selected from `@{package}`, which does not apply to integer bit operations"
                                ),
                            ));
                        }
                    } else if aliases.contains_key("__capability.bits") {
                        return Err(error(
                            binary.span,
                            format!(
                                "operator `{symbol}` is not enabled by this function's `@bits(...)` decorator"
                            ),
                        ));
                    }
                    if left_type != ValueType::Int || right_type != ValueType::Int {
                        return Err(error(
                            binary.span,
                            format!(
                                "operator `{symbol}` requires two integer operands in the `bits` algebra"
                            ),
                        ));
                    }
                    (op, ValueType::Int)
                }
                AstBinaryOp::Equal => (BinaryOp::Equal, ValueType::Bool),
                AstBinaryOp::NotEqual => (BinaryOp::NotEqual, ValueType::Bool),
                AstBinaryOp::Less => (BinaryOp::Less, ValueType::Bool),
                AstBinaryOp::LessEqual => (BinaryOp::LessEqual, ValueType::Bool),
                AstBinaryOp::Greater => (BinaryOp::Greater, ValueType::Bool),
                AstBinaryOp::GreaterEqual => (BinaryOp::GreaterEqual, ValueType::Bool),
                AstBinaryOp::And => (BinaryOp::And, ValueType::Bool),
                AstBinaryOp::Or => (BinaryOp::Or, ValueType::Bool),
                AstBinaryOp::In => (BinaryOp::In, ValueType::Bool),
            };
            Ok((
                Expression::Binary {
                    left: Box::new(left),
                    op,
                    right: Box::new(right),
                },
                result_type,
            ))
        }
        Expr::Call(call) => lower_call(call, scope, signatures, aliases),
        Expr::List(collection) => {
            lower_collection(&collection.elements, scope, signatures, aliases)
                .map(|elements| (Expression::List(elements), ValueType::List))
        }
        Expr::Tuple(collection) => {
            lower_collection(&collection.elements, scope, signatures, aliases)
                .map(|elements| (Expression::Tuple(elements), ValueType::Tuple))
        }
        Expr::Set(collection) => lower_collection(&collection.elements, scope, signatures, aliases)
            .map(|elements| (Expression::Set(elements), ValueType::Set)),
        Expr::Map(map) => {
            let entries = map
                .entries
                .iter()
                .map(|entry| {
                    Ok((
                        lower_expression(&entry.key, scope, signatures, aliases)?.0,
                        lower_expression(&entry.value, scope, signatures, aliases)?.0,
                    ))
                })
                .collect::<Result<Vec<_>, SemanticError>>()?;
            Ok((Expression::Map(entries), ValueType::Map))
        }
        Expr::Index(index) => {
            if let Expr::Literal(Literal::Integer {
                value: element_index,
                span,
            }) = index.index.as_ref()
            {
                let length = match index.object.as_ref() {
                    Expr::List(collection) => Some(collection.elements.len()),
                    Expr::Identifier(identifier) => scope
                        .get(&identifier.name)
                        .and_then(|binding| binding.collection_len),
                    _ => None,
                };
                if length
                    .is_some_and(|length| *element_index < 0 || *element_index as usize >= length)
                {
                    return Err(error(
                        *span,
                        format!(
                            "E000401: An index known to be outside a fixed-length collection is rejected at compile time. (index {element_index}, length {})",
                            length.unwrap()
                        ),
                    ));
                }
            }
            let (object, object_type) =
                lower_expression(&index.object, scope, signatures, aliases)?;
            let index_value = lower_expression(&index.index, scope, signatures, aliases)?.0;
            Ok((
                Expression::Index {
                    object: Box::new(object),
                    index: Box::new(index_value),
                },
                if object_type == ValueType::String {
                    ValueType::String
                } else {
                    ValueType::Any
                },
            ))
        }
        Expr::Slice(slice) => {
            let (object, object_type) =
                lower_expression(&slice.object, scope, signatures, aliases)?;
            let lower_bound = |bound: &Option<Box<Expr>>| {
                bound
                    .as_ref()
                    .map(|bound| {
                        let span = bound.span();
                        let (bound, ty) = lower_expression(bound, scope, signatures, aliases)?;
                        compatible(span, ty, ValueType::Int)?;
                        Ok(Box::new(bound))
                    })
                    .transpose()
            };
            Ok((
                Expression::Slice {
                    object: Box::new(object),
                    start: lower_bound(&slice.start)?,
                    end: lower_bound(&slice.end)?,
                    step: lower_bound(&slice.step)?,
                },
                object_type,
            ))
        }
        Expr::Member(member) => {
            let object_class = expression_class(&member.object, scope, aliases);
            let object = lower_expression(&member.object, scope, signatures, aliases)?.0;
            let field_type = if let Some(class) = object_class {
                let fields = aliases
                    .get(&format!("__class_fields.{class}"))
                    .map(|fields| fields.split(',').collect::<Vec<_>>())
                    .unwrap_or_default();
                if !fields.contains(&member.member.name.as_str()) {
                    return Err(error(
                        member.member.span,
                        format!("class `{class}` has no field `{}`", member.member.name),
                    ));
                }
                aliases
                    .get(&format!(
                        "__class_field_type.{class}.{}",
                        member.member.name
                    ))
                    .and_then(|value| decode_field_type(value))
                    .unwrap_or(ValueType::Any)
            } else {
                ValueType::Any
            };
            Ok((
                Expression::Member {
                    object: Box::new(object),
                    member: member.member.name.clone(),
                },
                field_type,
            ))
        }
        Expr::ListComprehension(comprehension) => {
            let mut inner_scope = scope.clone();
            let clauses = lower_comprehension_clauses(
                &comprehension.clauses,
                &mut inner_scope,
                signatures,
                aliases,
            )?;
            let element =
                lower_expression(&comprehension.element, &inner_scope, signatures, aliases)?.0;
            Ok((
                Expression::ListComprehension {
                    element: Box::new(element),
                    clauses,
                },
                ValueType::List,
            ))
        }
        Expr::SetComprehension(comprehension) => {
            let mut inner_scope = scope.clone();
            let clauses = lower_comprehension_clauses(
                &comprehension.clauses,
                &mut inner_scope,
                signatures,
                aliases,
            )?;
            let element =
                lower_expression(&comprehension.element, &inner_scope, signatures, aliases)?.0;
            Ok((
                Expression::SetComprehension {
                    element: Box::new(element),
                    clauses,
                },
                ValueType::Set,
            ))
        }
        Expr::MapComprehension(comprehension) => {
            let mut inner_scope = scope.clone();
            let clauses = lower_comprehension_clauses(
                &comprehension.clauses,
                &mut inner_scope,
                signatures,
                aliases,
            )?;
            let key = lower_expression(&comprehension.key, &inner_scope, signatures, aliases)?.0;
            let value =
                lower_expression(&comprehension.value, &inner_scope, signatures, aliases)?.0;
            Ok((
                Expression::MapComprehension {
                    key: Box::new(key),
                    value: Box::new(value),
                    clauses,
                },
                ValueType::Map,
            ))
        }
        Expr::If(conditional) => {
            let (condition, condition_type) =
                lower_expression(&conditional.condition, scope, signatures, aliases)?;
            compatible(
                conditional.condition.span(),
                condition_type,
                ValueType::Bool,
            )?;
            let (then_expression, then_type) =
                lower_expression(&conditional.then_expr, scope, signatures, aliases)?;
            let (else_expression, else_type) =
                lower_expression(&conditional.else_expr, scope, signatures, aliases)?;
            let result_type = if then_type == else_type {
                then_type
            } else if then_type == ValueType::Any || else_type == ValueType::Any {
                ValueType::Any
            } else {
                return Err(error(
                    conditional.span,
                    format!(
                        "conditional branches have incompatible types `{then_type:?}` and `{else_type:?}`"
                    ),
                ));
            };
            Ok((
                Expression::Conditional {
                    condition: Box::new(condition),
                    then_expression: Box::new(then_expression),
                    else_expression: Box::new(else_expression),
                },
                result_type,
            ))
        }
        Expr::Async(task) => {
            if let Expr::Call(call) = task.value.as_ref() {
                if let Expr::Member(member) = call.callee.as_ref() {
                    if let Expr::Identifier(object) = member.object.as_ref() {
                        if scope
                            .get(&object.name)
                            .is_some_and(|binding| binding.mutable)
                            && !task.captures.iter().any(|capture| capture.name == "lock")
                        {
                            return Err(error(
                                task.span,
                                "E000601: Mutable method calls across an async boundary require transferring the value's `lock` capability.",
                            ));
                        }
                    }
                }
            }
            let placement = match task.placement {
                severian_ast::TaskPlacement::Default => TaskPlacement::Default,
                severian_ast::TaskPlacement::Local => {
                    if !aliases.values().any(|module| module == "distributed") {
                        return Err(error(
                            task.span,
                            "task placement `local` requires `import distributed`",
                        ));
                    }
                    TaskPlacement::Local
                }
                severian_ast::TaskPlacement::Gpu
                | severian_ast::TaskPlacement::Simd
                | severian_ast::TaskPlacement::Simt => {
                    if !aliases.values().any(|module| module == "parallel") {
                        return Err(error(
                            task.span,
                            format!(
                                "task placement `{}` requires `import parallel`",
                                match task.placement {
                                    severian_ast::TaskPlacement::Gpu => "gpu",
                                    severian_ast::TaskPlacement::Simd => "simd",
                                    severian_ast::TaskPlacement::Simt => "simt",
                                    _ => unreachable!(),
                                }
                            ),
                        ));
                    }
                    match task.placement {
                        severian_ast::TaskPlacement::Gpu => TaskPlacement::Gpu,
                        severian_ast::TaskPlacement::Simd => TaskPlacement::Simd,
                        severian_ast::TaskPlacement::Simt => TaskPlacement::Simt,
                        _ => unreachable!(),
                    }
                }
            };
            let (value, _) = lower_expression(&task.value, scope, signatures, aliases)?;
            Ok((
                Expression::Task {
                    value: Box::new(value),
                    placement,
                },
                ValueType::Any,
            ))
        }
        Expr::Await(task) => {
            let (value, task_type) = lower_expression(&task.value, scope, signatures, aliases)?;
            let awaited_type = match value.kind() {
                Expression::Task { value, .. } => value.ty().unwrap_or(ValueType::Any),
                _ if task_type == ValueType::Channel => ValueType::Any,
                _ => task_type,
            };
            Ok((Expression::Await(Box::new(value)), awaited_type))
        }
        Expr::Channel(channel) => {
            let capacity = lower_expression(&channel.capacity, scope, signatures, aliases)?.0;
            Ok((Expression::Channel(Box::new(capacity)), ValueType::Channel))
        }
        Expr::Send(send) => {
            let value = lower_expression(&send.value, scope, signatures, aliases)?.0;
            let channel = lower_expression(&send.channel, scope, signatures, aliases)?.0;
            Ok((
                Expression::Send {
                    value: Box::new(value),
                    channel: Box::new(channel),
                },
                ValueType::Unit,
            ))
        }
        Expr::Ownership(ownership) => {
            let (value, ty) = lower_expression(&ownership.value, scope, signatures, aliases)?;
            let op = match ownership.op {
                AstOwnershipOp::View => OwnershipOp::View,
                AstOwnershipOp::Borrow => OwnershipOp::Borrow,
                AstOwnershipOp::Clone => OwnershipOp::Clone,
                AstOwnershipOp::Move => OwnershipOp::Move,
                AstOwnershipOp::AddressOf => OwnershipOp::AddressOf,
            };
            Ok((
                Expression::Ownership {
                    op,
                    value: Box::new(value),
                },
                ty,
            ))
        }
        Expr::Lambda(lambda) => {
            let severian_ast::LambdaBody::Expr(body) = &lambda.body else {
                return Err(error(
                    lambda.span,
                    "lambda blocks require an expression body",
                ));
            };
            let mut lambda_scope = scope.clone();
            let mut params = Vec::new();
            for parameter in &lambda.params {
                let ty = parameter
                    .ty
                    .as_ref()
                    .map(lower_type)
                    .transpose()?
                    .unwrap_or(ValueType::Any);
                lambda_scope.insert(
                    parameter.name.name.clone(),
                    Binding {
                        reference: source_binding(&parameter.name),
                        ty,
                        class: None,
                        function_return: None,
                        collection_len: None,
                        mutable: false,
                        field: false,
                        integer_max: None,
                        known_integer: None,
                        any_origin: declared_any_origin(parameter.ty.as_ref(), ty),
                    },
                );
                params.push(lambda_scope[&parameter.name.name].reference.clone());
            }
            let body = lower_expression(body, &lambda_scope, signatures, aliases)?.0;
            Ok((
                Expression::Lambda {
                    params,
                    body: Box::new(body),
                },
                ValueType::Function,
            ))
        }
        Expr::ChaosRule(rule) => {
            let (function, return_type) =
                lower_expression(&rule.function, scope, signatures, aliases)?;
            let Expression::Function(function) = function.into_kind() else {
                return Err(error(
                    rule.function.span(),
                    "chaos injection target must be a function",
                ));
            };
            let (value, value_type) = lower_expression(&rule.value, scope, signatures, aliases)?;
            if rule.action == severian_ast::ChaosAction::Return {
                let declared_return = signatures
                    .get(&function.name)
                    .map_or(return_type, |signature| signature.returns.resolved(aliases));
                compatible(rule.value.span(), value_type, declared_return)?;
            }
            Ok((
                Expression::ChaosRule {
                    function,
                    action: match rule.action {
                        severian_ast::ChaosAction::Return => HirChaosAction::Return,
                        severian_ast::ChaosAction::Throw => HirChaosAction::Throw,
                    },
                    value: Box::new(value),
                },
                ValueType::Any,
            ))
        }
        _ => Err(error(
            expression.span(),
            "expression is not supported in this compiler slice yet",
        )),
    }
}

pub(super) fn add_test_bindings(
    scope: &mut HashMap<String, Binding>,
    modes: &[severian_ast::TestMode],
) {
    scope.insert(
        "chaos".into(),
        Binding {
            reference: BindingRef::synthetic("chaos"),
            ty: ValueType::List,
            class: None,
            function_return: None,
            collection_len: None,
            mutable: false,
            field: false,
            integer_max: None,
            known_integer: None,
            any_origin: None,
        },
    );
    if modes.contains(&severian_ast::TestMode::Integration) {
        for name in ["stdout", "stderr"] {
            scope.insert(
                name.into(),
                Binding {
                    reference: BindingRef::synthetic(name),
                    ty: ValueType::String,
                    class: None,
                    function_return: None,
                    collection_len: None,
                    mutable: false,
                    field: false,
                    integer_max: None,
                    known_integer: None,
                    any_origin: None,
                },
            );
        }
    }
    if modes.contains(&severian_ast::TestMode::Profile) {
        for name in ["time", "memory", "allocations"] {
            scope.insert(
                name.into(),
                Binding {
                    reference: BindingRef::synthetic(name),
                    ty: ValueType::Int,
                    class: None,
                    function_return: None,
                    collection_len: None,
                    mutable: false,
                    field: false,
                    integer_max: None,
                    known_integer: None,
                    any_origin: None,
                },
            );
        }
    }
}
