use super::*;
use std::collections::BTreeSet;

pub(super) type Substitution = BTreeMap<String, String>;
pub(super) type Specializations = BTreeMap<DefId, BTreeSet<Substitution>>;

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
            severian_ast::Statement::Expression(expression)
            | severian_ast::Statement::Return {
                value: Some(expression),
                ..
            } => {
                validate_generic_expression(expression, names, function, index, types)?;
            }
            severian_ast::Statement::Return { value: None, .. } => {}
            severian_ast::Statement::Assert {
                condition, message, ..
            } => {
                validate_generic_expression(condition, names, function, index, types)?;
                if let Some(message) = message {
                    validate_generic_expression(message, names, function, index, types)?;
                }
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
        Expression::Member { object, .. } => {
            validate_generic_expression(object, names, function, index, types)
        }
        Expression::Call { callee, arguments } => {
            validate_generic_expression(callee, names, function, index, types)?;
            for argument in arguments {
                validate_generic_expression(&argument.value, names, function, index, types)?;
            }
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
                    severian_ast::UnaryOperator::Move => return Ok(parameter.clone().into()),
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
                    && matches!(&definition.kind, DefKind::Trait(declaration) if declaration.operators.iter().any(|known| ast_binary_syntax(known.operator) == Some(operator)))
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
        Ast::Add => Universal::Add,
        Ast::Subtract => Universal::Subtract,
        Ast::Multiply => Universal::Multiply,
        Ast::Divide => Universal::Divide,
        Ast::Remainder => Universal::Remainder,
        Ast::Power => Universal::Power,
        Ast::Equal => Universal::Equal,
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
                        if let Some(ty) =
                            expected.or_else(|| expression_type_name(&binding.value, &globals))
                        {
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
                                .and_then(|instances| instances.iter().next().cloned())
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
                    _ => {}
                }
            }
        }
        if specialization_count(&specializations) == before {
            break;
        }
    }
    validate_specializations(index, types, &specializations)?;
    Ok(specializations)
}

fn validate_specializations(
    index: &ProgramIndex,
    types: &severian_universal::TypeContext,
    specializations: &Specializations,
) -> Result<(), Diagnostic> {
    for (definition, instances) in specializations {
        let DefKind::Function(function) = &index.definitions[definition].kind else {
            continue;
        };
        for substitution in instances {
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
                let actual = types.resolve_name(actual_name).ok_or_else(|| {
                    Diagnostic::new(
                        "E000204",
                        format!("unknown inferred generic type `{actual_name}`"),
                        Some(*span),
                    )
                })?;
                let Some((bound_name, _)) = bound.named_parts() else {
                    return Err(Diagnostic::new(
                        "E000217",
                        "a generic capability constraint must name a trait",
                        Some(bound.span),
                    ));
                };
                if satisfies_bound(actual, bound_name, index, types) {
                    continue;
                }
                return Err(Diagnostic::new(
                    "E000217",
                    format!(
                        "`{actual_name}` does not satisfy `{bound_name}` required by `{}`",
                        index.definitions[definition].name
                    ),
                    Some(*span),
                ));
            }
        }
    }
    Ok(())
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
    declaration.operators.iter().all(|operator| {
        use severian_ast::OperatorSyntax as Syntax;
        use severian_universal::{BinaryOperator as Binary, UnaryOperator as Unary};
        match (operator.operator, operator.parameters.is_empty()) {
            (Syntax::Plus, true) => types.supports_unary(Unary::Positive, actual),
            (Syntax::Minus, true) => types.supports_unary(Unary::Negative, actual),
            (Syntax::Not, _) => types.supports_unary(Unary::Not, actual),
            (syntax, _) => {
                let operator = match syntax {
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
    })
}

fn specialization_count(specializations: &Specializations) -> usize {
    specializations.values().map(BTreeSet::len).sum()
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
                if let Some(ty) = expected.or_else(|| expression_type_name(&binding.value, names)) {
                    names.insert(binding.name.clone(), ty);
                }
            }
            severian_ast::Statement::Expression(expression) => {
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
            severian_ast::Statement::Return { value: None, .. } => {}
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
                for definition in resolve_path(module, &path, index) {
                    let DefKind::Function(signature) = &index.definitions[&definition].kind else {
                        continue;
                    };
                    if signature.type_parameters.is_empty()
                        || signature.parameters.len() != arguments.len()
                    {
                        continue;
                    }
                    let mut substitution = Substitution::new();
                    if let Some(expected) = expected {
                        infer_substitution(
                            &signature.result,
                            expected,
                            &signature.type_parameters,
                            &mut substitution,
                        );
                    }
                    for (parameter, argument) in signature.parameters.iter().zip(arguments) {
                        if let Some(actual) = expression_type_name(&argument.value, names) {
                            infer_substitution(
                                parameter,
                                &actual,
                                &signature.type_parameters,
                                &mut substitution,
                            );
                        }
                    }
                    if signature
                        .type_parameters
                        .iter()
                        .all(|parameter| substitution.contains_key(parameter))
                    {
                        specializations
                            .entry(definition)
                            .or_default()
                            .insert(substitution);
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
        severian_ast::ExpressionKind::Unary { operand, .. } => {
            visit_expression_for_specializations(
                module,
                operand,
                expected,
                names,
                index,
                specializations,
            )?;
        }
        severian_ast::ExpressionKind::Binary { left, right, .. } => {
            visit_expression_for_specializations(
                module,
                left,
                expected,
                names,
                index,
                specializations,
            )?;
            visit_expression_for_specializations(
                module,
                right,
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
) {
    if let Some(name) = pattern.simple_name() {
        if parameters.iter().any(|parameter| parameter == name) {
            substitution
                .entry(name.to_owned())
                .or_insert_with(|| actual.to_owned());
        }
    }
}

fn expression_type_name(
    expression: &severian_ast::Expression,
    names: &BTreeMap<String, String>,
) -> Option<String> {
    match &expression.kind {
        severian_ast::ExpressionKind::Name(name) => names.get(name).cloned(),
        severian_ast::ExpressionKind::Literal(literal) => Some(
            match literal {
                severian_ast::Literal::Integer(_) => "int",
                severian_ast::Literal::Float(_) => "float",
                severian_ast::Literal::Boolean(_) => "bool",
                severian_ast::Literal::Character(_) => "char",
                severian_ast::Literal::String(_) => "string",
                severian_ast::Literal::Bytes(_) => "bytes",
                severian_ast::Literal::None => "none",
                severian_ast::Literal::Unit => "unit",
            }
            .to_owned(),
        ),
        _ => None,
    }
}

fn simple_type_name(annotation: &TypeAnnotation) -> Option<String> {
    annotation.simple_name().map(str::to_owned)
}

fn specialized_type_name(
    annotation: &TypeAnnotation,
    substitution: &Substitution,
) -> Option<String> {
    let name = annotation.simple_name()?;
    Some(
        substitution
            .get(name)
            .cloned()
            .unwrap_or_else(|| name.to_owned()),
    )
}

pub(super) fn specialize_function(
    function: &severian_ast::FunctionDeclaration,
    substitution: &Substitution,
) -> severian_ast::FunctionDeclaration {
    let mut function = function.clone();
    function.type_parameters.clear();
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
        parameters: function
            .parameters
            .iter()
            .map(|annotation| specialize_annotation(annotation, substitution))
            .collect(),
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
