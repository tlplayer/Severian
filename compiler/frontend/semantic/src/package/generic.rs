use super::*;
use std::collections::BTreeSet;

pub(super) type Substitution = BTreeMap<String, String>;
/// Concrete generic instances and the first source call that requested each
/// one. Keeping the origin beside the substitution lets later constraint
/// validation report the call site instead of only the declaration.
pub(super) type Specializations = BTreeMap<DefId, BTreeMap<Substitution, severian_source::Span>>;

#[derive(Debug, Clone, PartialEq, Eq)]
struct InferenceConflict {
    parameter: String,
    known: String,
    inferred: String,
}

pub(super) fn validate_generic_bodies(
    module_graph: &ModuleGraph,
    index: &ProgramIndex,
    types: &severian_universal::TypeContext,
) -> Result<(), Diagnostic> {
    for module in &module_graph.modules {
        for function in module.ast.items.iter().filter_map(|item| match item {
            Item::Function(function) if !function.type_parameters.is_empty() => Some(function),
            _ => None,
        }) {
            let Some(body) = &function.body else {
                continue;
            };
            let parameters = function
                .type_parameters
                .iter()
                .cloned()
                .collect::<BTreeSet<_>>();
            let mut names = BTreeMap::new();
            for parameter in &function.parameters {
                if let Some(name) = parameter.annotation.simple_name() {
                    if parameters.contains(name) {
                        names.insert(parameter.name.clone(), name.to_owned());
                    }
                }
            }
            validate_generic_statements(body, &mut names, function, index, types)?;
        }
    }
    Ok(())
}

fn validate_generic_statements(
    statements: &[severian_ast::Statement],
    names: &mut BTreeMap<String, String>,
    function: &severian_ast::FunctionDeclaration,
    index: &ProgramIndex,
    types: &severian_universal::TypeContext,
) -> Result<(), Diagnostic> {
    for statement in statements {
        match statement {
            severian_ast::Statement::Binding(binding) => {
                let inferred =
                    validate_generic_expression(&binding.value, names, function, index, types)?;
                if let Some(parameter) = binding
                    .annotation
                    .as_ref()
                    .and_then(TypeAnnotation::simple_name)
                    .filter(|name| function.type_parameters.iter().any(|known| known == name))
                    .map(str::to_owned)
                    .or(inferred)
                {
                    names.insert(binding.name.clone(), parameter);
                }
            }
            severian_ast::Statement::Destructure {
                names: bound,
                value,
                ..
            } => {
                validate_generic_expression(value, names, function, index, types)?;
                for name in bound {
                    names.remove(name);
                }
            }
            severian_ast::Statement::FieldAssignment { object, value, .. } => {
                validate_generic_expression(object, names, function, index, types)?;
                validate_generic_expression(value, names, function, index, types)?;
            }
            severian_ast::Statement::IndexAssignment {
                object,
                index: offset,
                value,
                ..
            } => {
                validate_generic_expression(object, names, function, index, types)?;
                validate_generic_expression(offset, names, function, index, types)?;
                validate_generic_expression(value, names, function, index, types)?;
            }
            severian_ast::Statement::Expression(expression)
            | severian_ast::Statement::Defer { expression, .. }
            | severian_ast::Statement::Return {
                value: Some(expression),
                ..
            } => {
                validate_generic_expression(expression, names, function, index, types)?;
            }
            severian_ast::Statement::Return { value: None, .. }
            | severian_ast::Statement::Break { .. }
            | severian_ast::Statement::Continue { .. } => {}
            severian_ast::Statement::Assert {
                condition, message, ..
            } => {
                validate_generic_expression(condition, names, function, index, types)?;
                if let Some(message) = message {
                    validate_generic_expression(message, names, function, index, types)?;
                }
            }
            severian_ast::Statement::Unsafe { body, .. }
            | severian_ast::Statement::Placement { body, .. } => {
                validate_generic_statements(body, &mut names.clone(), function, index, types)?;
            }
            severian_ast::Statement::Try {
                body,
                catch_binding,
                catch_body,
                ..
            } => {
                validate_generic_statements(body, &mut names.clone(), function, index, types)?;
                let mut catch_names = names.clone();
                catch_names.remove(catch_binding);
                validate_generic_statements(catch_body, &mut catch_names, function, index, types)?;
            }
            severian_ast::Statement::FallibleElse {
                value,
                error_binding,
                body,
                ..
            } => {
                validate_generic_expression(value, names, function, index, types)?;
                let mut handler_names = names.clone();
                handler_names.remove(error_binding);
                validate_generic_statements(body, &mut handler_names, function, index, types)?;
            }
            severian_ast::Statement::If {
                condition,
                then_block,
                else_block,
                ..
            } => {
                validate_generic_expression(condition, names, function, index, types)?;
                validate_generic_statements(
                    then_block,
                    &mut names.clone(),
                    function,
                    index,
                    types,
                )?;
                validate_generic_statements(
                    else_block,
                    &mut names.clone(),
                    function,
                    index,
                    types,
                )?;
            }
            severian_ast::Statement::While {
                condition,
                initializer,
                guards,
                body,
                ..
            } => {
                let mut loop_names = names.clone();
                if let Some(initializer) = initializer {
                    validate_generic_expression(
                        &initializer.value,
                        &loop_names,
                        function,
                        index,
                        types,
                    )?;
                }
                validate_generic_expression(condition, &loop_names, function, index, types)?;
                validate_generic_statements(body, &mut loop_names, function, index, types)?;
                for guard in guards {
                    validate_generic_expression(
                        &guard.condition,
                        &loop_names,
                        function,
                        index,
                        types,
                    )?;
                }
            }
            severian_ast::Statement::For {
                iterable,
                initializer,
                body,
                ..
            } => {
                validate_generic_expression(iterable, names, function, index, types)?;
                if let Some(initializer) = initializer {
                    validate_generic_expression(&initializer.value, names, function, index, types)?;
                }
                validate_generic_statements(body, &mut names.clone(), function, index, types)?;
            }
            severian_ast::Statement::Match { subject, cases, .. } => {
                validate_generic_expression(subject, names, function, index, types)?;
                for case in cases {
                    validate_generic_statements(
                        &case.body,
                        &mut names.clone(),
                        function,
                        index,
                        types,
                    )?;
                }
            }
            severian_ast::Statement::Select {
                limit,
                cases,
                error_body,
                ..
            } => {
                validate_generic_expression(limit, names, function, index, types)?;
                for case in cases {
                    validate_generic_expression(&case.channel, names, function, index, types)?;
                    let mut case_names = names.clone();
                    case_names.remove(&case.binding);
                    validate_generic_statements(
                        &case.body,
                        &mut case_names,
                        function,
                        index,
                        types,
                    )?;
                }
                validate_generic_statements(
                    error_body,
                    &mut names.clone(),
                    function,
                    index,
                    types,
                )?;
            }
        }
    }
    Ok(())
}

fn validate_generic_expression(
    expression: &severian_ast::Expression,
    names: &BTreeMap<String, String>,
    function: &severian_ast::FunctionDeclaration,
    index: &ProgramIndex,
    types: &severian_universal::TypeContext,
) -> Result<Option<String>, Diagnostic> {
    use severian_ast::ExpressionKind as Expression;
    match &expression.kind {
        Expression::Name(name) => Ok(names.get(name).cloned()),
        Expression::Literal(_) => Ok(None),
        Expression::List(values) | Expression::Set(values) | Expression::Tuple(values) => {
            for value in values {
                validate_generic_expression(value, names, function, index, types)?;
            }
            Ok(None)
        }
        Expression::Map(entries) => {
            for entry in entries {
                validate_generic_expression(&entry.key, names, function, index, types)?;
                validate_generic_expression(&entry.value, names, function, index, types)?;
            }
            Ok(None)
        }
        Expression::ListComprehension { value, clauses }
        | Expression::SetComprehension { value, clauses } => {
            for clause in clauses {
                validate_generic_expression(&clause.iterable, names, function, index, types)?;
                if let Some(condition) = &clause.condition {
                    validate_generic_expression(condition, names, function, index, types)?;
                }
            }
            validate_generic_expression(value, names, function, index, types)?;
            Ok(None)
        }
        Expression::MapComprehension {
            key,
            value,
            clauses,
        } => {
            for clause in clauses {
                validate_generic_expression(&clause.iterable, names, function, index, types)?;
                if let Some(condition) = &clause.condition {
                    validate_generic_expression(condition, names, function, index, types)?;
                }
            }
            validate_generic_expression(key, names, function, index, types)?;
            validate_generic_expression(value, names, function, index, types)?;
            Ok(None)
        }
        Expression::Mock { cases, fallback } => {
            for case in cases {
                validate_generic_expression(&case.call, names, function, index, types)?;
                validate_generic_expression(&case.result, names, function, index, types)?;
            }
            validate_generic_expression(fallback, names, function, index, types)?;
            Ok(None)
        }
        Expression::Lambda { parameters, body } => {
            let mut lambda_names = names.clone();
            for parameter in parameters {
                lambda_names.remove(parameter);
            }
            validate_generic_expression(body, &lambda_names, function, index, types)?;
            Ok(None)
        }
        Expression::Member { object, .. } => {
            validate_generic_expression(object, names, function, index, types)
        }
        Expression::Index {
            object,
            index: offset,
        } => {
            let object = validate_generic_expression(object, names, function, index, types)?;
            validate_generic_expression(offset, names, function, index, types)?;
            Ok(object)
        }
        Expression::Slice {
            object,
            start,
            end,
            step,
            ..
        } => {
            let object = validate_generic_expression(object, names, function, index, types)?;
            for bound in [start, end, step].into_iter().flatten() {
                validate_generic_expression(bound, names, function, index, types)?;
            }
            Ok(object)
        }
        Expression::TypeApplication { callee, .. } => {
            validate_generic_expression(callee, names, function, index, types)
        }
        Expression::Call { callee, arguments } => {
            validate_generic_expression(callee, names, function, index, types)?;
            for argument in arguments {
                validate_generic_expression(&argument.value, names, function, index, types)?;
            }
            Ok(None)
        }
        Expression::Async { expression, .. } | Expression::Await { expression } => {
            validate_generic_expression(expression, names, function, index, types)
        }
        Expression::Conditional {
            value,
            condition,
            fallback,
        } => {
            let value = validate_generic_expression(value, names, function, index, types)?;
            validate_generic_expression(condition, names, function, index, types)?;
            let fallback = validate_generic_expression(fallback, names, function, index, types)?;
            Ok(value.or(fallback))
        }
        Expression::Fallback { value, fallback } => {
            let value = validate_generic_expression(value, names, function, index, types)?;
            let fallback = validate_generic_expression(fallback, names, function, index, types)?;
            Ok(value.or(fallback))
        }
        Expression::Throw { error } => {
            validate_generic_expression(error, names, function, index, types)?;
            Ok(None)
        }
        Expression::Unary { operator, operand } => {
            let parameter = validate_generic_expression(operand, names, function, index, types)?;
            if let Some(parameter) = &parameter {
                let operator = match operator {
                    severian_ast::UnaryOperator::Positive => {
                        severian_universal::UnaryOperator::Positive
                    }
                    severian_ast::UnaryOperator::Negative => {
                        severian_universal::UnaryOperator::Negative
                    }
                    severian_ast::UnaryOperator::Not => severian_universal::UnaryOperator::Not,
                    severian_ast::UnaryOperator::Borrow
                    | severian_ast::UnaryOperator::BorrowMut
                    | severian_ast::UnaryOperator::AddressOf
                    | severian_ast::UnaryOperator::Copy
                    | severian_ast::UnaryOperator::Move => return Ok(parameter.clone().into()),
                };
                if !parameter_allows_unary(parameter, operator, function, index, types) {
                    return Err(missing_operator_constraint(
                        function,
                        parameter,
                        format!("{operator:?}"),
                        expression.span,
                    ));
                }
            }
            Ok(parameter)
        }
        Expression::Binary {
            operator,
            left,
            right,
        } => {
            let left = validate_generic_expression(left, names, function, index, types)?;
            let right = validate_generic_expression(right, names, function, index, types)?;
            if *operator == severian_ast::BinaryOperator::Pipe {
                return Ok(left.or(right));
            }
            let universal = ast_binary(*operator);
            for parameter in [&left, &right].into_iter().flatten() {
                if !parameter_allows_binary(parameter, universal, function, index, types) {
                    return Err(missing_operator_constraint(
                        function,
                        parameter,
                        format!("{operator:?}"),
                        expression.span,
                    ));
                }
            }
            Ok(left.or(right))
        }
    }
}

fn missing_operator_constraint(
    function: &severian_ast::FunctionDeclaration,
    parameter: &str,
    operator: String,
    span: severian_source::Span,
) -> Diagnostic {
    Diagnostic::new(
        "E000219",
        format!(
            "generic body `{}` uses operator {operator} on `{parameter}` without a capability that guarantees it",
            function.name
        ),
        Some(span),
    )
}

fn parameter_bounds<'a>(
    parameter: &'a str,
    function: &'a severian_ast::FunctionDeclaration,
) -> impl Iterator<Item = &'a TypeAnnotation> {
    function
        .constraints
        .iter()
        .filter_map(move |constraint| match constraint {
            severian_ast::GenericConstraint::Parameter {
                parameter: known,
                bound,
                ..
            } if known == parameter => Some(bound),
            _ => None,
        })
}

fn parameter_allows_binary(
    parameter: &str,
    operator: severian_universal::BinaryOperator,
    function: &severian_ast::FunctionDeclaration,
    index: &ProgramIndex,
    types: &severian_universal::TypeContext,
) -> bool {
    parameter_bounds(parameter, function).any(|bound| {
        let Some((name, _)) = bound.named_parts() else {
            return false;
        };
        types
            .resolve_name(name)
            .is_some_and(|trait_id| types.trait_supports_binary(trait_id, operator))
            || index.definitions.values().any(|definition| {
                definition.name == name
                    && matches!(&definition.kind, DefKind::Trait(declaration) if trait_supports_source_binary(declaration, operator))
            })
    })
}

fn parameter_allows_unary(
    parameter: &str,
    operator: severian_universal::UnaryOperator,
    function: &severian_ast::FunctionDeclaration,
    index: &ProgramIndex,
    types: &severian_universal::TypeContext,
) -> bool {
    parameter_bounds(parameter, function).any(|bound| {
        let Some((name, _)) = bound.named_parts() else {
            return false;
        };
        types
            .resolve_name(name)
            .is_some_and(|trait_id| types.trait_supports_unary(trait_id, operator))
            || index.definitions.values().any(|definition| {
                definition.name == name
                    && matches!(&definition.kind, DefKind::Trait(declaration) if declaration.operators.iter().any(|known| ast_unary_syntax(known.operator) == Some(operator)))
            })
    })
}

fn ast_binary(operator: severian_ast::BinaryOperator) -> severian_universal::BinaryOperator {
    use severian_ast::BinaryOperator as Ast;
    use severian_universal::BinaryOperator as Universal;
    match operator {
        Ast::Pipe => Universal::BitwiseOr,
        Ast::BitwiseAnd => Universal::BitwiseAnd,
        Ast::BitwiseXor => Universal::BitwiseXor,
        Ast::Add => Universal::Add,
        Ast::Subtract => Universal::Subtract,
        Ast::Multiply => Universal::Multiply,
        Ast::Divide => Universal::Divide,
        Ast::Remainder => Universal::Remainder,
        Ast::Power => Universal::Power,
        Ast::Equal | Ast::Identity => Universal::Equal,
        Ast::NotEqual => Universal::NotEqual,
        Ast::Less => Universal::Less,
        Ast::LessEqual => Universal::LessEqual,
        Ast::Greater => Universal::Greater,
        Ast::GreaterEqual => Universal::GreaterEqual,
        Ast::Contains => Universal::Contains,
        Ast::And => Universal::And,
        Ast::Or => Universal::Or,
    }
}

fn ast_binary_syntax(
    operator: severian_ast::OperatorSyntax,
) -> Option<severian_universal::BinaryOperator> {
    use severian_ast::OperatorSyntax as Ast;
    Some(ast_binary(match operator {
        Ast::Pipe => severian_ast::BinaryOperator::Pipe,
        Ast::BitwiseAnd => severian_ast::BinaryOperator::BitwiseAnd,
        Ast::BitwiseXor => severian_ast::BinaryOperator::BitwiseXor,
        Ast::Plus => severian_ast::BinaryOperator::Add,
        Ast::Minus => severian_ast::BinaryOperator::Subtract,
        Ast::Multiply => severian_ast::BinaryOperator::Multiply,
        Ast::Divide => severian_ast::BinaryOperator::Divide,
        Ast::Remainder => severian_ast::BinaryOperator::Remainder,
        Ast::Power => severian_ast::BinaryOperator::Power,
        Ast::Equal => severian_ast::BinaryOperator::Equal,
        Ast::NotEqual => severian_ast::BinaryOperator::NotEqual,
        Ast::Less => severian_ast::BinaryOperator::Less,
        Ast::LessEqual => severian_ast::BinaryOperator::LessEqual,
        Ast::Greater => severian_ast::BinaryOperator::Greater,
        Ast::GreaterEqual => severian_ast::BinaryOperator::GreaterEqual,
        Ast::Contains => severian_ast::BinaryOperator::Contains,
        Ast::And => severian_ast::BinaryOperator::And,
        Ast::Or => severian_ast::BinaryOperator::Or,
        Ast::Not => return None,
    }))
}

fn trait_supports_source_binary(
    declaration: &TraitDecl,
    operator: severian_universal::BinaryOperator,
) -> bool {
    declaration
        .operators
        .iter()
        .any(|known| ast_binary_syntax(known.operator) == Some(operator))
        || declaration
            .methods
            .iter()
            .any(|method| method_binary_syntax(&method.name) == Some(operator))
}

fn method_binary_syntax(name: &str) -> Option<severian_universal::BinaryOperator> {
    use severian_universal::BinaryOperator as Operator;
    match name {
        "add" => Some(Operator::Add),
        "subtract" => Some(Operator::Subtract),
        "multiply" => Some(Operator::Multiply),
        "divide" => Some(Operator::Divide),
        "remainder" => Some(Operator::Remainder),
        "power" => Some(Operator::Power),
        "equal" => Some(Operator::Equal),
        "not_equal" => Some(Operator::NotEqual),
        "less_than" => Some(Operator::Less),
        "less_equal" => Some(Operator::LessEqual),
        "greater_than" => Some(Operator::Greater),
        "greater_equal" => Some(Operator::GreaterEqual),
        "contains" => Some(Operator::Contains),
        _ => None,
    }
}

fn ast_unary_syntax(
    operator: severian_ast::OperatorSyntax,
) -> Option<severian_universal::UnaryOperator> {
    match operator {
        severian_ast::OperatorSyntax::Plus => Some(severian_universal::UnaryOperator::Positive),
        severian_ast::OperatorSyntax::Minus => Some(severian_universal::UnaryOperator::Negative),
        severian_ast::OperatorSyntax::Not => Some(severian_universal::UnaryOperator::Not),
        _ => None,
    }
}

pub(super) fn collect_generic_specializations(
    module_graph: &ModuleGraph,
    index: &ProgramIndex,
    types: &severian_universal::TypeContext,
) -> Result<Specializations, Diagnostic> {
    let mut specializations = BTreeMap::new();
    // A specialization can make calls inside another generic body concrete,
    // so walk to a fixed point. This is declaration-driven and independent of
    // module initialization order.
    loop {
        let before = specialization_count(&specializations);
        for module in &module_graph.modules {
            let mut globals = BTreeMap::new();
            for item in &module.ast.items {
                match item {
                    Item::Binding(binding) => {
                        let expected = binding.annotation.as_ref().and_then(simple_type_name);
                        visit_expression_for_specializations(
                            module.id,
                            &binding.value,
                            expected.as_deref(),
                            &globals,
                            index,
                            &mut specializations,
                        )?;
                        if let Some(ty) = expected.or_else(|| {
                            expression_type_name(module.id, &binding.value, &globals, index)
                        }) {
                            globals.insert(binding.name.clone(), ty);
                        }
                    }
                    Item::Expression(expression) => visit_expression_for_specializations(
                        module.id,
                        expression,
                        None,
                        &globals,
                        index,
                        &mut specializations,
                    )?,
                    Item::Function(function) => {
                        let id = function_def_id(module.package, module.id, &module.ast, function);
                        let substitution = if function.type_parameters.is_empty() {
                            Some(Substitution::new())
                        } else {
                            specializations
                                .get(&id)
                                .and_then(|instances| instances.keys().next().cloned())
                        };
                        let Some(substitution) = substitution else {
                            continue;
                        };
                        let mut names = globals.clone();
                        for parameter in &function.parameters {
                            if let Some(name) =
                                specialized_type_name(&parameter.annotation, &substitution)
                            {
                                names.insert(parameter.name.clone(), name);
                            }
                        }
                        let result = specialized_type_name(&function.result, &substitution);
                        if let Some(body) = &function.body {
                            visit_statements_for_specializations(
                                module.id,
                                body,
                                result.as_deref(),
                                &mut names,
                                index,
                                &mut specializations,
                            )?;
                        }
                    }
                    Item::Test(test) => {
                        let mut names = globals.clone();
                        visit_statements_for_specializations(
                            module.id,
                            &test.body,
                            None,
                            &mut names,
                            index,
                            &mut specializations,
                        )?;
                    }
                    _ => {}
                }
            }
        }
        if specialization_count(&specializations) == before {
            break;
        }
    }
    validate_specializations(module_graph, index, types, &specializations)?;
    Ok(specializations)
}

fn validate_specializations(
    module_graph: &ModuleGraph,
    index: &ProgramIndex,
    types: &severian_universal::TypeContext,
    specializations: &Specializations,
) -> Result<(), Diagnostic> {
    for (definition, instances) in specializations {
        let DefKind::Function(function) = &index.definitions[definition].kind else {
            continue;
        };
        for (substitution, origin) in instances {
            for constraint in &function.constraints {
                let severian_ast::GenericConstraint::Parameter {
                    parameter,
                    bound,
                    span,
                } = constraint
                else {
                    return Err(Diagnostic::new(
                        "E000218",
                        "compile-time value predicates are parsed but are not supported by the initial generic solver",
                        Some(constraint_span(constraint)),
                    ));
                };
                let Some(actual_name) = substitution.get(parameter) else {
                    continue;
                };
                let Some((bound_name, _)) = bound.named_parts() else {
                    return Err(Diagnostic::new(
                        "E000217",
                        "a generic capability constraint must name a trait",
                        Some(bound.span),
                    ));
                };
                let satisfied = types
                    .resolve_name(actual_name)
                    .is_some_and(|actual| satisfies_bound(actual, bound_name, index, types))
                    || source_class_satisfies_bound(actual_name, bound_name, module_graph, index);
                if satisfied {
                    continue;
                }
                let function_name = &index.definitions[definition].name;
                let specialization = format_substitution(substitution);
                return Err(Diagnostic::new(
                    "E000217",
                    format!(
                        "cannot specialize `{function_name}[{specialization}]`: `{actual_name}` does not satisfy `{bound_name}`"
                    ),
                    Some(*origin),
                )
                .with_label(*origin, "specialization requested here")
                .with_note(format!(
                    "type parameter `{parameter}` was inferred as `{actual_name}`"
                ))
                .with_additional([Diagnostic::new(
                    "E000217",
                    format!("`{parameter}` must satisfy `{bound_name}`"),
                    Some(*span),
                )
                .with_label(*span, "constraint declared here")]));
            }
        }
    }
    Ok(())
}

fn source_class_satisfies_bound(
    actual_name: &str,
    bound_name: &str,
    module_graph: &ModuleGraph,
    index: &ProgramIndex,
) -> bool {
    module_graph.modules.iter().any(|module| {
        module.ast.items.iter().any(|item| {
            let Item::Class(class) = item else {
                return false;
            };
            class.name == actual_name
                && class.traits.iter().any(|implemented| {
                    implemented.simple_name().is_some_and(|name| {
                        source_trait_extends(name, bound_name, index, &mut BTreeSet::new())
                    })
                })
        })
    })
}

fn source_trait_extends(
    trait_name: &str,
    bound_name: &str,
    index: &ProgramIndex,
    visiting: &mut BTreeSet<String>,
) -> bool {
    if trait_name == bound_name {
        return true;
    }
    if !visiting.insert(trait_name.to_owned()) {
        return false;
    }
    index.definitions.values().any(|definition| {
        definition.name == trait_name
            && matches!(&definition.kind, DefKind::Trait(declaration) if declaration.bases.iter().any(|base| {
                base.simple_name().is_some_and(|base| {
                    source_trait_extends(base, bound_name, index, visiting)
                })
            }))
    })
}

fn constraint_span(constraint: &severian_ast::GenericConstraint) -> severian_source::Span {
    match constraint {
        severian_ast::GenericConstraint::Parameter { span, .. } => *span,
        severian_ast::GenericConstraint::Predicate(expression) => expression.span,
    }
}

fn satisfies_bound(
    actual: severian_universal::TypeId,
    bound_name: &str,
    index: &ProgramIndex,
    types: &severian_universal::TypeContext,
) -> bool {
    if let Some(bound) = types.resolve_name(bound_name) {
        return types.implements(actual, bound);
    }
    index.definitions.values().any(|definition| {
        definition.name == bound_name
            && matches!(&definition.kind, DefKind::Trait(trait_decl) if trait_is_structurally_satisfied(actual, trait_decl, index, types, &mut BTreeSet::new()))
    })
}

fn trait_is_structurally_satisfied(
    actual: severian_universal::TypeId,
    declaration: &TraitDecl,
    index: &ProgramIndex,
    types: &severian_universal::TypeContext,
    visiting: &mut BTreeSet<String>,
) -> bool {
    for base in &declaration.bases {
        let Some((name, _)) = base.named_parts() else {
            return false;
        };
        if !visiting.insert(name.to_owned()) {
            continue;
        }
        if !satisfies_bound(actual, name, index, types) {
            return false;
        }
    }
    let operators_satisfied = declaration.operators.iter().all(|operator| {
        use severian_ast::OperatorSyntax as Syntax;
        use severian_universal::{BinaryOperator as Binary, UnaryOperator as Unary};
        match (operator.operator, operator.parameters.is_empty()) {
            (Syntax::Plus, true) => types.supports_unary(Unary::Positive, actual),
            (Syntax::Minus, true) => types.supports_unary(Unary::Negative, actual),
            (Syntax::Not, _) => types.supports_unary(Unary::Not, actual),
            (syntax, _) => {
                let operator = match syntax {
                    Syntax::Pipe => Binary::BitwiseOr,
                    Syntax::BitwiseAnd => Binary::BitwiseAnd,
                    Syntax::BitwiseXor => Binary::BitwiseXor,
                    Syntax::Plus => Binary::Add,
                    Syntax::Minus => Binary::Subtract,
                    Syntax::Multiply => Binary::Multiply,
                    Syntax::Divide => Binary::Divide,
                    Syntax::Remainder => Binary::Remainder,
                    Syntax::Power => Binary::Power,
                    Syntax::Equal => Binary::Equal,
                    Syntax::NotEqual => Binary::NotEqual,
                    Syntax::Less => Binary::Less,
                    Syntax::LessEqual => Binary::LessEqual,
                    Syntax::Greater => Binary::Greater,
                    Syntax::GreaterEqual => Binary::GreaterEqual,
                    Syntax::Contains => Binary::Contains,
                    Syntax::And => Binary::And,
                    Syntax::Or => Binary::Or,
                    Syntax::Not => return false,
                };
                types.supports_binary(operator, actual)
            }
        }
    });
    operators_satisfied
        && declaration.methods.iter().all(|method| {
            method_binary_syntax(&method.name)
                .is_some_and(|operator| types.supports_binary(operator, actual))
                || (method.name == "zero"
                    && method.parameters.is_empty()
                    && types.supports_binary(severian_universal::BinaryOperator::Add, actual))
                || (method.name == "hash"
                    && method.parameters.is_empty()
                    && types.primitive(actual).is_some())
        })
}

fn specialization_count(specializations: &Specializations) -> usize {
    specializations.values().map(BTreeMap::len).sum()
}

fn format_substitution(substitution: &Substitution) -> String {
    substitution
        .iter()
        .map(|(parameter, actual)| format!("{parameter}={actual}"))
        .collect::<Vec<_>>()
        .join(", ")
}

fn visit_statements_for_specializations(
    module: ModuleId,
    statements: &[severian_ast::Statement],
    result: Option<&str>,
    names: &mut BTreeMap<String, String>,
    index: &ProgramIndex,
    specializations: &mut Specializations,
) -> Result<(), Diagnostic> {
    for statement in statements {
        match statement {
            severian_ast::Statement::Binding(binding) => {
                let expected = binding.annotation.as_ref().and_then(simple_type_name);
                visit_expression_for_specializations(
                    module,
                    &binding.value,
                    expected.as_deref(),
                    names,
                    index,
                    specializations,
                )?;
                if let Some(ty) =
                    expected.or_else(|| expression_type_name(module, &binding.value, names, index))
                {
                    names.insert(binding.name.clone(), ty);
                }
            }
            severian_ast::Statement::Destructure {
                names: bound,
                value,
                ..
            } => {
                visit_expression_for_specializations(
                    module,
                    value,
                    None,
                    names,
                    index,
                    specializations,
                )?;
                for name in bound {
                    names.remove(name);
                }
            }
            severian_ast::Statement::FieldAssignment { object, value, .. } => {
                visit_expression_for_specializations(
                    module,
                    object,
                    None,
                    names,
                    index,
                    specializations,
                )?;
                visit_expression_for_specializations(
                    module,
                    value,
                    None,
                    names,
                    index,
                    specializations,
                )?;
            }
            severian_ast::Statement::IndexAssignment {
                object,
                index: offset,
                value,
                ..
            } => {
                for expression in [object, offset, value] {
                    visit_expression_for_specializations(
                        module,
                        expression,
                        None,
                        names,
                        index,
                        specializations,
                    )?;
                }
            }
            severian_ast::Statement::Expression(expression)
            | severian_ast::Statement::Defer { expression, .. } => {
                visit_expression_for_specializations(
                    module,
                    expression,
                    None,
                    names,
                    index,
                    specializations,
                )?;
            }
            severian_ast::Statement::Return {
                value: Some(value), ..
            } => {
                visit_expression_for_specializations(
                    module,
                    value,
                    result,
                    names,
                    index,
                    specializations,
                )?;
            }
            severian_ast::Statement::Return { value: None, .. }
            | severian_ast::Statement::Break { .. }
            | severian_ast::Statement::Continue { .. } => {}
            severian_ast::Statement::Assert {
                condition, message, ..
            } => {
                visit_expression_for_specializations(
                    module,
                    condition,
                    Some("bool"),
                    names,
                    index,
                    specializations,
                )?;
                if let Some(message) = message {
                    visit_expression_for_specializations(
                        module,
                        message,
                        Some("string"),
                        names,
                        index,
                        specializations,
                    )?;
                }
            }
            severian_ast::Statement::Unsafe { body, .. }
            | severian_ast::Statement::Placement { body, .. } => {
                visit_statements_for_specializations(
                    module,
                    body,
                    result,
                    &mut names.clone(),
                    index,
                    specializations,
                )?;
            }
            severian_ast::Statement::Try {
                body, catch_body, ..
            } => {
                visit_statements_for_specializations(
                    module,
                    body,
                    result,
                    &mut names.clone(),
                    index,
                    specializations,
                )?;
                visit_statements_for_specializations(
                    module,
                    catch_body,
                    result,
                    &mut names.clone(),
                    index,
                    specializations,
                )?;
            }
            severian_ast::Statement::FallibleElse { value, body, .. } => {
                visit_expression_for_specializations(
                    module,
                    value,
                    None,
                    names,
                    index,
                    specializations,
                )?;
                visit_statements_for_specializations(
                    module,
                    body,
                    result,
                    &mut names.clone(),
                    index,
                    specializations,
                )?;
            }
            severian_ast::Statement::If {
                condition,
                then_block,
                else_block,
                ..
            } => {
                visit_expression_for_specializations(
                    module,
                    condition,
                    Some("bool"),
                    names,
                    index,
                    specializations,
                )?;
                visit_statements_for_specializations(
                    module,
                    then_block,
                    result,
                    &mut names.clone(),
                    index,
                    specializations,
                )?;
                visit_statements_for_specializations(
                    module,
                    else_block,
                    result,
                    &mut names.clone(),
                    index,
                    specializations,
                )?;
            }
            severian_ast::Statement::While {
                condition,
                initializer,
                guards,
                body,
                ..
            } => {
                if let Some(initializer) = initializer {
                    visit_expression_for_specializations(
                        module,
                        &initializer.value,
                        None,
                        names,
                        index,
                        specializations,
                    )?;
                }
                visit_expression_for_specializations(
                    module,
                    condition,
                    Some("bool"),
                    names,
                    index,
                    specializations,
                )?;
                for guard in guards {
                    visit_expression_for_specializations(
                        module,
                        &guard.condition,
                        Some("bool"),
                        names,
                        index,
                        specializations,
                    )?;
                }
                visit_statements_for_specializations(
                    module,
                    body,
                    result,
                    &mut names.clone(),
                    index,
                    specializations,
                )?;
            }
            severian_ast::Statement::For {
                iterable,
                initializer,
                body,
                ..
            } => {
                visit_expression_for_specializations(
                    module,
                    iterable,
                    None,
                    names,
                    index,
                    specializations,
                )?;
                if let Some(initializer) = initializer {
                    visit_expression_for_specializations(
                        module,
                        &initializer.value,
                        None,
                        names,
                        index,
                        specializations,
                    )?;
                }
                visit_statements_for_specializations(
                    module,
                    body,
                    result,
                    &mut names.clone(),
                    index,
                    specializations,
                )?;
            }
            severian_ast::Statement::Match { subject, cases, .. } => {
                visit_expression_for_specializations(
                    module,
                    subject,
                    None,
                    names,
                    index,
                    specializations,
                )?;
                for case in cases {
                    let mut case_names = names.clone();
                    if let (Some(binding), Some(annotation)) = (&case.binding, &case.annotation) {
                        if let Some(ty) = simple_type_name(annotation) {
                            case_names.insert(binding.clone(), ty);
                        }
                    }
                    visit_statements_for_specializations(
                        module,
                        &case.body,
                        result,
                        &mut case_names,
                        index,
                        specializations,
                    )?;
                }
            }
            severian_ast::Statement::Select {
                limit,
                cases,
                error_body,
                ..
            } => {
                visit_expression_for_specializations(
                    module,
                    limit,
                    None,
                    names,
                    index,
                    specializations,
                )?;
                for case in cases {
                    visit_expression_for_specializations(
                        module,
                        &case.channel,
                        None,
                        names,
                        index,
                        specializations,
                    )?;
                    visit_statements_for_specializations(
                        module,
                        &case.body,
                        result,
                        &mut names.clone(),
                        index,
                        specializations,
                    )?;
                }
                visit_statements_for_specializations(
                    module,
                    error_body,
                    result,
                    &mut names.clone(),
                    index,
                    specializations,
                )?;
            }
        }
    }
    Ok(())
}

fn visit_expression_for_specializations(
    module: ModuleId,
    expression: &severian_ast::Expression,
    expected: Option<&str>,
    names: &BTreeMap<String, String>,
    index: &ProgramIndex,
    specializations: &mut Specializations,
) -> Result<(), Diagnostic> {
    match &expression.kind {
        severian_ast::ExpressionKind::Call { callee, arguments } => {
            if let Some(path) = ast_callable_path(callee) {
                let definitions = resolve_path(module, &path, index);
                let unambiguous = definitions.len() == 1;
                for definition in definitions {
                    let DefKind::Function(signature) = &index.definitions[&definition].kind else {
                        continue;
                    };
                    let variadic = signature
                        .parameter_variadics
                        .last()
                        .copied()
                        .unwrap_or(false);
                    let fixed = signature.parameters.len() - usize::from(variadic);
                    if signature.type_parameters.is_empty()
                        || arguments.len() < fixed
                        || (!variadic && arguments.len() != fixed)
                    {
                        continue;
                    }
                    let mut substitution = Substitution::new();
                    let mut conflict = None;
                    if !variadic {
                        if let Some(expected) = expected {
                            conflict = infer_substitution(
                                &signature.result,
                                expected,
                                &signature.type_parameters,
                                &mut substitution,
                            )
                            .err();
                        }
                    }
                    for (parameter, argument) in signature.parameters[..fixed].iter().zip(arguments)
                    {
                        if conflict.is_some() {
                            break;
                        }
                        if let Some(actual) =
                            expression_type_name(module, &argument.value, names, index)
                        {
                            conflict = infer_substitution(
                                parameter,
                                &actual,
                                &signature.type_parameters,
                                &mut substitution,
                            )
                            .err();
                        }
                    }
                    if variadic && conflict.is_none() {
                        let parameter = &signature.parameters[fixed];
                        for argument in &arguments[fixed..] {
                            if let Some(mut actual) =
                                expression_type_name(module, &argument.value, names, index)
                            {
                                if argument.spread {
                                    actual = actual
                                        .strip_prefix("list[")
                                        .and_then(|actual| actual.strip_suffix(']'))
                                        .unwrap_or(&actual)
                                        .to_owned();
                                }
                                if let Some(name) = parameter.simple_name().filter(|name| {
                                    signature
                                        .type_parameters
                                        .iter()
                                        .any(|parameter| parameter == name)
                                }) {
                                    match substitution.get(name) {
                                        Some(known) if known != &actual && known != "Any" => {
                                            substitution.insert(name.to_owned(), "Any".into());
                                        }
                                        None => {
                                            substitution.insert(name.to_owned(), actual);
                                        }
                                        _ => {}
                                    }
                                } else {
                                    conflict = infer_substitution(
                                        parameter,
                                        &actual,
                                        &signature.type_parameters,
                                        &mut substitution,
                                    )
                                    .err();
                                    if conflict.is_some() {
                                        break;
                                    }
                                }
                            }
                        }
                    }
                    if variadic && conflict.is_none() {
                        if let Some(expected) = expected {
                            conflict = infer_substitution(
                                &signature.result,
                                expected,
                                &signature.type_parameters,
                                &mut substitution,
                            )
                            .err();
                        }
                    }
                    if let Some(conflict) = conflict {
                        if unambiguous {
                            return Err(inference_conflict_diagnostic(
                                &index.definitions[&definition].name,
                                &conflict,
                                expression.span,
                            ));
                        }
                        continue;
                    }
                    if signature
                        .type_parameters
                        .iter()
                        .all(|parameter| substitution.contains_key(parameter))
                    {
                        specializations
                            .entry(definition)
                            .or_default()
                            .entry(substitution)
                            .or_insert(expression.span);
                    }
                }
            }
            for argument in arguments {
                visit_expression_for_specializations(
                    module,
                    &argument.value,
                    None,
                    names,
                    index,
                    specializations,
                )?;
            }
        }
        severian_ast::ExpressionKind::Async { expression, .. }
        | severian_ast::ExpressionKind::Await { expression } => {
            visit_expression_for_specializations(
                module,
                expression,
                expected,
                names,
                index,
                specializations,
            )?;
        }
        severian_ast::ExpressionKind::Conditional {
            value,
            condition,
            fallback,
        } => {
            visit_expression_for_specializations(
                module,
                value,
                expected,
                names,
                index,
                specializations,
            )?;
            visit_expression_for_specializations(
                module,
                condition,
                None,
                names,
                index,
                specializations,
            )?;
            visit_expression_for_specializations(
                module,
                fallback,
                expected,
                names,
                index,
                specializations,
            )?;
        }
        severian_ast::ExpressionKind::Fallback { value, fallback } => {
            visit_expression_for_specializations(
                module,
                value,
                expected,
                names,
                index,
                specializations,
            )?;
            visit_expression_for_specializations(
                module,
                fallback,
                expected,
                names,
                index,
                specializations,
            )?;
        }
        severian_ast::ExpressionKind::Throw { error } => {
            visit_expression_for_specializations(
                module,
                error,
                None,
                names,
                index,
                specializations,
            )?;
        }
        severian_ast::ExpressionKind::Member { object, .. } => {
            visit_expression_for_specializations(
                module,
                object,
                None,
                names,
                index,
                specializations,
            )?;
        }
        severian_ast::ExpressionKind::Index {
            object,
            index: offset,
        } => {
            visit_expression_for_specializations(
                module,
                object,
                expected,
                names,
                index,
                specializations,
            )?;
            visit_expression_for_specializations(
                module,
                offset,
                None,
                names,
                index,
                specializations,
            )?;
        }
        severian_ast::ExpressionKind::Slice {
            object,
            start,
            end,
            step,
            ..
        } => {
            visit_expression_for_specializations(
                module,
                object,
                expected,
                names,
                index,
                specializations,
            )?;
            for bound in [start, end, step].into_iter().flatten() {
                visit_expression_for_specializations(
                    module,
                    bound,
                    None,
                    names,
                    index,
                    specializations,
                )?;
            }
        }
        severian_ast::ExpressionKind::TypeApplication { callee, .. } => {
            visit_expression_for_specializations(
                module,
                callee,
                expected,
                names,
                index,
                specializations,
            )?;
        }
        severian_ast::ExpressionKind::Unary { operator, operand } => {
            let operand_expected = match operator {
                severian_ast::UnaryOperator::Not => Some("bool"),
                _ => expected,
            };
            visit_expression_for_specializations(
                module,
                operand,
                operand_expected,
                names,
                index,
                specializations,
            )?;
        }
        severian_ast::ExpressionKind::Binary {
            operator,
            left,
            right,
        } => {
            use severian_ast::BinaryOperator as Binary;
            // A comparison's result is boolean, but its operands are not.
            // Forwarding the result expectation into either operand can poison
            // generic result inference (for example, inferring `V = bool` from
            // `generic_call() == 42`). Logical operators, conversely, do
            // require boolean operands. Value-producing operators retain the
            // surrounding expected type.
            let operand_expected = match operator {
                Binary::Equal
                | Binary::Identity
                | Binary::NotEqual
                | Binary::Less
                | Binary::LessEqual
                | Binary::Greater
                | Binary::GreaterEqual
                | Binary::Contains => None,
                Binary::And | Binary::Or => Some("bool"),
                _ => expected,
            };
            visit_expression_for_specializations(
                module,
                left,
                operand_expected,
                names,
                index,
                specializations,
            )?;
            visit_expression_for_specializations(
                module,
                right,
                operand_expected,
                names,
                index,
                specializations,
            )?;
        }
        severian_ast::ExpressionKind::List(values)
        | severian_ast::ExpressionKind::Set(values)
        | severian_ast::ExpressionKind::Tuple(values) => {
            for value in values {
                visit_expression_for_specializations(
                    module,
                    value,
                    None,
                    names,
                    index,
                    specializations,
                )?;
            }
        }
        severian_ast::ExpressionKind::Map(entries) => {
            for entry in entries {
                visit_expression_for_specializations(
                    module,
                    &entry.key,
                    None,
                    names,
                    index,
                    specializations,
                )?;
                visit_expression_for_specializations(
                    module,
                    &entry.value,
                    None,
                    names,
                    index,
                    specializations,
                )?;
            }
        }
        severian_ast::ExpressionKind::ListComprehension { value, clauses }
        | severian_ast::ExpressionKind::SetComprehension { value, clauses } => {
            visit_expression_for_specializations(
                module,
                value,
                None,
                names,
                index,
                specializations,
            )?;
            for clause in clauses {
                visit_expression_for_specializations(
                    module,
                    &clause.iterable,
                    None,
                    names,
                    index,
                    specializations,
                )?;
                if let Some(condition) = &clause.condition {
                    visit_expression_for_specializations(
                        module,
                        condition,
                        Some("bool"),
                        names,
                        index,
                        specializations,
                    )?;
                }
            }
        }
        severian_ast::ExpressionKind::MapComprehension {
            key,
            value,
            clauses,
        } => {
            for expression in [key.as_ref(), value.as_ref()] {
                visit_expression_for_specializations(
                    module,
                    expression,
                    None,
                    names,
                    index,
                    specializations,
                )?;
            }
            for clause in clauses {
                visit_expression_for_specializations(
                    module,
                    &clause.iterable,
                    None,
                    names,
                    index,
                    specializations,
                )?;
                if let Some(condition) = &clause.condition {
                    visit_expression_for_specializations(
                        module,
                        condition,
                        Some("bool"),
                        names,
                        index,
                        specializations,
                    )?;
                }
            }
        }
        severian_ast::ExpressionKind::Mock { cases, fallback } => {
            for case in cases {
                visit_expression_for_specializations(
                    module,
                    &case.call,
                    None,
                    names,
                    index,
                    specializations,
                )?;
                visit_expression_for_specializations(
                    module,
                    &case.result,
                    None,
                    names,
                    index,
                    specializations,
                )?;
            }
            visit_expression_for_specializations(
                module,
                fallback,
                None,
                names,
                index,
                specializations,
            )?;
        }
        severian_ast::ExpressionKind::Lambda { body, .. } => {
            visit_expression_for_specializations(
                module,
                body,
                expected,
                names,
                index,
                specializations,
            )?;
        }
        severian_ast::ExpressionKind::Literal(_) | severian_ast::ExpressionKind::Name(_) => {}
    }
    Ok(())
}

fn resolve_path(module: ModuleId, path: &str, index: &ProgramIndex) -> Vec<DefId> {
    let mut parts = path.split('.');
    let Some(first) = parts.next() else {
        return Vec::new();
    };
    let Some(mut resolution) = index.modules[&module].scope.bindings.get(first) else {
        return Vec::new();
    };
    for part in parts {
        let Resolution::Module(target) = resolution else {
            return Vec::new();
        };
        let Some(next) = index
            .exports
            .get(target)
            .and_then(|exports| exports.get(part))
        else {
            return Vec::new();
        };
        resolution = next;
    }
    if let Resolution::Module(target) = resolution {
        // A same-named export is the package's default callable surface, so
        // generic specialization must see the same `tensor(...)` shorthand
        // that semantic call resolution accepts.
        if let Some(default) = index
            .exports
            .get(target)
            .and_then(|exports| exports.get(first))
        {
            return resolution_definitions(default);
        }
    }
    resolution_definitions(resolution)
}

fn ast_callable_path(expression: &severian_ast::Expression) -> Option<String> {
    match &expression.kind {
        severian_ast::ExpressionKind::Name(name) => Some(name.clone()),
        severian_ast::ExpressionKind::Member { object, name } => {
            Some(format!("{}.{}", ast_callable_path(object)?, name))
        }
        _ => None,
    }
}

fn infer_substitution(
    pattern: &TypeAnnotation,
    actual: &str,
    parameters: &[String],
    substitution: &mut Substitution,
) -> Result<(), InferenceConflict> {
    let Some((name, arguments)) = pattern.named_parts() else {
        return Ok(());
    };
    if arguments.is_empty() && parameters.iter().any(|parameter| parameter == name) {
        if let Some(known) = substitution.get(name) {
            if known == actual || default_numeric_matches(actual, known) {
                return Ok(());
            }
            if default_numeric_matches(known, actual) {
                substitution.insert(name.to_owned(), actual.to_owned());
                return Ok(());
            }
            return Err(InferenceConflict {
                parameter: name.to_owned(),
                known: known.clone(),
                inferred: actual.to_owned(),
            });
        } else {
            substitution.insert(name.to_owned(), actual.to_owned());
        }
        return Ok(());
    }
    let Some((actual_name, actual_arguments)) = type_application_parts(actual) else {
        return Ok(());
    };
    if !same_type_constructor(name, actual_name) || arguments.len() != actual_arguments.len() {
        return Ok(());
    }
    for (pattern, actual) in arguments.iter().zip(actual_arguments) {
        infer_substitution(pattern, actual, parameters, substitution)?;
    }
    Ok(())
}

fn same_type_constructor(left: &str, right: &str) -> bool {
    left == right
        || left
            .rsplit('.')
            .next()
            .is_some_and(|left| right.rsplit('.').next().is_some_and(|right| left == right))
}

fn default_numeric_matches(default: &str, concrete: &str) -> bool {
    match default {
        "int" => matches!(
            concrete,
            "i8" | "i16"
                | "i32"
                | "i64"
                | "i128"
                | "isize"
                | "u8"
                | "u16"
                | "u32"
                | "u64"
                | "u128"
                | "usize"
        ),
        "float" => matches!(
            concrete,
            "f8e4m3fn" | "f8e5m2" | "f16" | "bf16" | "f32" | "f64" | "f128"
        ),
        _ => false,
    }
}

fn inference_conflict_diagnostic(
    function: &str,
    conflict: &InferenceConflict,
    span: severian_source::Span,
) -> Diagnostic {
    Diagnostic::new(
        "E000217",
        format!(
            "conflicting inferences for `{}` while specializing `{function}`: `{}` and `{}`",
            conflict.parameter, conflict.known, conflict.inferred
        ),
        Some(span),
    )
    .with_label(span, "generic call has incompatible type evidence")
    .with_note(format!(
        "`{}` was first inferred as `{}` and later as `{}`",
        conflict.parameter, conflict.known, conflict.inferred
    ))
    .with_help("make the generic result or argument types agree")
}

fn type_application_parts(value: &str) -> Option<(&str, Vec<&str>)> {
    let Some(open) = value.find('[') else {
        return Some((value.trim(), Vec::new()));
    };
    if !value.ends_with(']') {
        return None;
    }
    let name = value[..open].trim();
    let contents = &value[open + 1..value.len() - 1];
    let mut depth = 0usize;
    let mut start = 0usize;
    let mut arguments = Vec::new();
    for (offset, character) in contents.char_indices() {
        match character {
            '[' => depth += 1,
            ']' => depth = depth.checked_sub(1)?,
            ',' if depth == 0 => {
                arguments.push(contents[start..offset].trim());
                start = offset + character.len_utf8();
            }
            _ => {}
        }
    }
    if depth != 0 {
        return None;
    }
    arguments.push(contents[start..].trim());
    Some((name, arguments))
}

fn expression_type_name(
    module: ModuleId,
    expression: &severian_ast::Expression,
    names: &BTreeMap<String, String>,
    index: &ProgramIndex,
) -> Option<String> {
    match &expression.kind {
        severian_ast::ExpressionKind::Name(name) => names.get(name).cloned(),
        severian_ast::ExpressionKind::Literal(literal) => Some(
            match literal {
                severian_ast::Literal::Integer(_) => "int",
                severian_ast::Literal::Float(_) => "float",
                severian_ast::Literal::Measured { suffix, .. } => {
                    return crate::measured_type_name(suffix).map(str::to_owned)
                }
                severian_ast::Literal::Boolean(_) => "bool",
                severian_ast::Literal::Character(_) => "char",
                severian_ast::Literal::String(_) => "string",
                severian_ast::Literal::Bytes(_) => "bytes",
                severian_ast::Literal::None => "none",
                severian_ast::Literal::Unit => "unit",
            }
            .to_owned(),
        ),
        severian_ast::ExpressionKind::Call { callee, arguments } => {
            let path = ast_callable_path(callee)?;
            resolve_path(module, &path, index)
                .into_iter()
                .find_map(|definition| match &index.definitions[&definition].kind {
                    DefKind::Function(function) if function.type_parameters.is_empty() => {
                        type_annotation_name(&function.result)
                    }
                    DefKind::Function(function) => {
                        let mut substitution = Substitution::new();
                        for (parameter, argument) in function.parameters.iter().zip(arguments) {
                            let actual =
                                expression_type_name(module, &argument.value, names, index)?;
                            infer_substitution(
                                parameter,
                                &actual,
                                &function.type_parameters,
                                &mut substitution,
                            )
                            .ok()?;
                        }
                        function
                            .type_parameters
                            .iter()
                            .all(|parameter| substitution.contains_key(parameter))
                            .then(|| {
                                type_annotation_name(&specialize_annotation(
                                    &function.result,
                                    &substitution,
                                ))
                            })
                            .flatten()
                    }
                    DefKind::Type => Some(index.definitions[&definition].name.clone()),
                    _ => None,
                })
        }
        severian_ast::ExpressionKind::List(values) => {
            let element = values
                .first()
                .and_then(|value| expression_type_name(module, value, names, index))?;
            Some(format!("list[{element}]"))
        }
        severian_ast::ExpressionKind::Tuple(values) => {
            let elements = values
                .iter()
                .map(|value| expression_type_name(module, value, names, index))
                .collect::<Option<Vec<_>>>()?;
            Some(format!("tuple[{}]", elements.join(", ")))
        }
        severian_ast::ExpressionKind::Map(entries) => {
            let first = entries.first()?;
            let key = expression_type_name(module, &first.key, names, index)?;
            let value = expression_type_name(module, &first.value, names, index)?;
            Some(format!("map[{key}, {value}]"))
        }
        _ => None,
    }
}

fn type_annotation_name(annotation: &TypeAnnotation) -> Option<String> {
    match &annotation.kind {
        TypeAnnotationKind::Named { name, arguments } if arguments.is_empty() => Some(name.clone()),
        TypeAnnotationKind::Named { name, arguments } => Some(format!(
            "{name}[{}]",
            arguments
                .iter()
                .map(type_annotation_name)
                .collect::<Option<Vec<_>>>()?
                .join(", ")
        )),
        TypeAnnotationKind::Function { parameters, result } => Some(format!(
            "({}) -> {}",
            parameters
                .iter()
                .map(type_annotation_name)
                .collect::<Option<Vec<_>>>()?
                .join(", "),
            type_annotation_name(result)?
        )),
        TypeAnnotationKind::Union(members) => Some(
            members
                .iter()
                .map(type_annotation_name)
                .collect::<Option<Vec<_>>>()?
                .join(" | "),
        ),
    }
}

fn simple_type_name(annotation: &TypeAnnotation) -> Option<String> {
    annotation.simple_name().map(str::to_owned)
}

fn specialized_type_name(
    annotation: &TypeAnnotation,
    substitution: &Substitution,
) -> Option<String> {
    type_annotation_name(&specialize_annotation(annotation, substitution))
}

pub(super) fn specialize_function(
    function: &severian_ast::FunctionDeclaration,
    substitution: &Substitution,
) -> severian_ast::FunctionDeclaration {
    let mut function = function.clone();
    for parameter in &mut function.parameters {
        parameter.annotation = specialize_annotation(&parameter.annotation, substitution);
    }
    function.result = specialize_annotation(&function.result, substitution);
    function
}

pub(super) fn specialize_signature(
    function: &FunctionDecl,
    substitution: &Substitution,
) -> FunctionDecl {
    FunctionDecl {
        signature: function.signature,
        type_parameters: Vec::new(),
        parameter_names: function.parameter_names.clone(),
        parameter_variadics: function.parameter_variadics.clone(),
        parameters: function
            .parameters
            .iter()
            .map(|annotation| specialize_annotation(annotation, substitution))
            .collect(),
        parameter_defaults: function.parameter_defaults.clone(),
        result: specialize_annotation(&function.result, substitution),
        constraints: function.constraints.clone(),
    }
}

fn specialize_annotation(
    annotation: &TypeAnnotation,
    substitution: &Substitution,
) -> TypeAnnotation {
    let kind = match &annotation.kind {
        TypeAnnotationKind::Named { name, arguments } => TypeAnnotationKind::Named {
            name: substitution
                .get(name)
                .cloned()
                .unwrap_or_else(|| name.clone()),
            arguments: arguments
                .iter()
                .map(|argument| specialize_annotation(argument, substitution))
                .collect(),
        },
        TypeAnnotationKind::Function { parameters, result } => TypeAnnotationKind::Function {
            parameters: parameters
                .iter()
                .map(|parameter| specialize_annotation(parameter, substitution))
                .collect(),
            result: Box::new(specialize_annotation(result, substitution)),
        },
        TypeAnnotationKind::Union(types) => TypeAnnotationKind::Union(
            types
                .iter()
                .map(|ty| specialize_annotation(ty, substitution))
                .collect(),
        ),
    };
    TypeAnnotation {
        kind,
        span: annotation.span,
    }
}
