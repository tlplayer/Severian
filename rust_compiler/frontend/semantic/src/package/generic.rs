use super::*;
use std::collections::BTreeSet;

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Default)]
pub(crate) struct Substitution {
    kinds: BTreeMap<String, severian_universal::GenericParamKind>,
    types: BTreeMap<String, String>,
    dimensions: BTreeMap<String, severian_universal::DimExpr>,
    shapes: BTreeMap<String, Vec<severian_universal::DimExpr>>,
}

impl Substitution {
    pub(crate) fn new() -> Self {
        Self::default()
    }

    fn for_declaration(function: &FunctionDecl) -> Self {
        let kinds = generic_parameters(&function.type_parameters, &function.constraints)
            .into_iter()
            .map(|parameter| (parameter.name, parameter.kind))
            .collect();
        Self {
            kinds,
            ..Self::default()
        }
    }

    pub(super) fn get(&self, parameter: &str) -> Option<&String> {
        self.types.get(parameter)
    }

    fn dimension(&self, parameter: &str) -> Option<&severian_universal::DimExpr> {
        self.dimensions.get(parameter)
    }

    fn shape(&self, parameter: &str) -> Option<&[severian_universal::DimExpr]> {
        self.shapes.get(parameter).map(Vec::as_slice)
    }

    fn contains_key(&self, parameter: &str) -> bool {
        self.types.contains_key(parameter)
            || self.dimensions.contains_key(parameter)
            || self.shapes.contains_key(parameter)
    }

    pub(super) fn is_empty(&self) -> bool {
        self.types.is_empty() && self.dimensions.is_empty() && self.shapes.is_empty()
    }

    pub(crate) fn insert_type(&mut self, parameter: String, value: String) -> Option<String> {
        self.types.insert(parameter, value)
    }

    fn insert(&mut self, parameter: String, value: String) -> Option<String> {
        self.insert_type(parameter, value)
    }

    pub(super) fn bindings(&self) -> Vec<(String, String)> {
        let mut bindings = self
            .types
            .iter()
            .map(|(parameter, value)| (parameter.clone(), value.clone()))
            .chain(
                self.dimensions
                    .iter()
                    .map(|(parameter, value)| (parameter.clone(), format_dim_expr(value))),
            )
            .chain(
                self.shapes
                    .iter()
                    .map(|(parameter, value)| (parameter.clone(), format_shape(value))),
            )
            .collect::<Vec<_>>();
        bindings.sort();
        bindings
    }

    fn bind_dimension(
        &mut self,
        parameter: &str,
        dimension: severian_universal::DimExpr,
    ) -> Result<(), InferenceConflict> {
        if let Some(known) = self.dimensions.get(parameter) {
            if known == &dimension {
                return Ok(());
            }
            return Err(InferenceConflict {
                parameter: parameter.to_owned(),
                known: format_dim_expr(known),
                inferred: format_dim_expr(&dimension),
            });
        }
        self.dimensions.insert(parameter.to_owned(), dimension);
        Ok(())
    }

    pub(crate) fn insert_dimension(
        &mut self,
        parameter: String,
        dimension: severian_universal::DimExpr,
    ) {
        self.dimensions.insert(parameter, dimension);
    }

    fn bind_shape(
        &mut self,
        parameter: &str,
        dimensions: Vec<severian_universal::DimExpr>,
    ) -> Result<(), InferenceConflict> {
        if let Some(known) = self.shapes.get(parameter) {
            if known == &dimensions {
                return Ok(());
            }
            return Err(InferenceConflict {
                parameter: parameter.to_owned(),
                known: format_shape(known),
                inferred: format_shape(&dimensions),
            });
        }
        self.shapes.insert(parameter.to_owned(), dimensions);
        Ok(())
    }

    fn kind(&self, parameter: &str) -> severian_universal::GenericParamKind {
        self.kinds
            .get(parameter)
            .copied()
            .unwrap_or(severian_universal::GenericParamKind::Type)
    }
}

impl FromIterator<(String, String)> for Substitution {
    fn from_iter<T: IntoIterator<Item = (String, String)>>(iter: T) -> Self {
        Self {
            types: iter.into_iter().collect(),
            ..Self::default()
        }
    }
}

impl Extend<(String, String)> for Substitution {
    fn extend<T: IntoIterator<Item = (String, String)>>(&mut self, iter: T) {
        self.types.extend(iter);
    }
}

fn format_dim_expr(dimension: &severian_universal::DimExpr) -> String {
    match dimension {
        severian_universal::DimExpr::Constant(value) => value.to_string(),
        severian_universal::DimExpr::Parameter(parameter) => format!("p{}", parameter.0),
        severian_universal::DimExpr::Runtime(runtime) => format!("?{}", runtime.0),
        severian_universal::DimExpr::Add(left, right) => {
            format!("({}+{})", format_dim_expr(left), format_dim_expr(right))
        }
        severian_universal::DimExpr::Multiply(left, right) => {
            format!("({}*{})", format_dim_expr(left), format_dim_expr(right))
        }
        severian_universal::DimExpr::DivideExact(left, right) => {
            format!("({}/{})", format_dim_expr(left), format_dim_expr(right))
        }
    }
}

fn format_shape(shape: &[severian_universal::DimExpr]) -> String {
    format!(
        "[{}]",
        shape
            .iter()
            .map(format_dim_expr)
            .collect::<Vec<_>>()
            .join(",")
    )
}
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
            Item::Function(function)
                if !function.compile_time && !function.type_parameters.is_empty() =>
            {
                Some(function)
            }
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
            | severian_ast::Statement::Yield {
                value: expression, ..
            }
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
        Expression::Literal(_) | Expression::Symbol(_) => Ok(None),
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
    let operator = if operator == severian_ast::BinaryOperator::Identity {
        severian_ast::BinaryOperator::Equal
    } else {
        operator
    };
    severian_universal::BinaryOperator::from_stable_id(operator.stable_id())
}

fn ast_binary_syntax(
    operator: severian_ast::OperatorSyntax,
) -> Option<severian_universal::BinaryOperator> {
    (!matches!(
        operator,
        severian_ast::OperatorSyntax::Index
            | severian_ast::OperatorSyntax::If
            | severian_ast::OperatorSyntax::Else
            | severian_ast::OperatorSyntax::Conversion
            | severian_ast::OperatorSyntax::Not
    ))
    .then(|| ast_binary(operator))
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
                    Item::Function(function) => visit_function_for_specializations(
                        module,
                        function,
                        &globals,
                        index,
                        &mut specializations,
                    )?,
                    Item::Class(class) if class.type_parameters.is_empty() => {
                        let mut fields = globals.clone();
                        for field in &class.fields {
                            if let Some(name) = type_annotation_name(&field.annotation) {
                                fields.insert(field.name.clone(), name);
                            }
                        }
                        for function in class.constructors.iter().chain(&class.methods) {
                            visit_function_for_specializations(
                                module,
                                function,
                                &fields,
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

fn visit_function_for_specializations(
    module: &severian_modules::ResolvedModule,
    function: &severian_ast::FunctionDeclaration,
    inherited_names: &BTreeMap<String, String>,
    index: &ProgramIndex,
    specializations: &mut Specializations,
) -> Result<(), Diagnostic> {
    let id = function_def_id(module.package, module.id, &module.ast, function);
    let substitutions = if function.type_parameters.is_empty() {
        vec![Substitution::new()]
    } else {
        specializations
            .get(&id)
            .map(|instances| instances.keys().cloned().collect())
            .unwrap_or_default()
    };
    for substitution in substitutions {
        let mut names = inherited_names.clone();
        install_specialization_environment(&mut names, &substitution);
        for parameter in &function.parameters {
            if let Some(name) = specialized_type_name(&parameter.annotation, &substitution) {
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
                specializations,
            )?;
        }
    }
    Ok(())
}

fn install_specialization_environment(
    names: &mut BTreeMap<String, String>,
    substitution: &Substitution,
) {
    for (parameter, actual) in &substitution.types {
        names.insert(format!("$type:{parameter}"), actual.clone());
    }
    for (parameter, actual) in &substitution.dimensions {
        names.insert(format!("$dimension:{parameter}"), format_dim_expr(actual));
    }
    for (parameter, shape) in &substitution.shapes {
        for (axis, dimension) in shape.iter().enumerate() {
            names.insert(
                format!("$shape:{parameter}:{axis}"),
                format_dim_expr(dimension),
            );
        }
    }
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
            let parameters = generic_parameters(&function.type_parameters, &function.constraints);
            for constraint in &function.constraints {
                let severian_ast::GenericConstraint::Parameter {
                    parameter,
                    bound,
                    span,
                } = constraint
                else {
                    match constraint {
                        severian_ast::GenericConstraint::VariadicPack { .. } => continue,
                        severian_ast::GenericConstraint::Predicate(predicate) => {
                            let Some(constraint) =
                                tensor_dimension_constraint(predicate, substitution)
                            else {
                                return Err(Diagnostic::new(
                                    "E000218",
                                    "generic value predicate is outside the tensor dimension constraint language",
                                    Some(predicate.span),
                                ));
                            };
                            match constraint
                                .resolve(&severian_universal::DimensionBindings::default())
                            {
                                Ok(severian_universal::ConstraintResolution::Proven)
                                | Ok(severian_universal::ConstraintResolution::RuntimeCheck) => {
                                    continue;
                                }
                                Err(error) => {
                                    return Err(Diagnostic::new(
                                        "E000217",
                                        format!(
                                            "cannot specialize `{}`: tensor dimension constraint failed: {error}",
                                            index.definitions[definition].name
                                        ),
                                        Some(*origin),
                                    )
                                    .with_label(*origin, "specialization requested here")
                                    .with_additional([Diagnostic::new(
                                        "E000217",
                                        "tensor dimension constraint declared here",
                                        Some(predicate.span),
                                    )]));
                                }
                            }
                        }
                        severian_ast::GenericConstraint::Parameter { .. } => unreachable!(),
                    }
                };
                if parameters.iter().any(|known| {
                    known.name == *parameter
                        && known.kind != severian_universal::GenericParamKind::Type
                }) {
                    continue;
                }
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
                    || source_class_satisfies_bound(actual_name, bound_name, module_graph, index)
                    || source_trait_satisfies_bound(actual_name, bound_name, index);
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

fn tensor_dimension_constraint(
    predicate: &severian_ast::Expression,
    substitution: &Substitution,
) -> Option<severian_universal::DimensionConstraint> {
    use severian_ast::BinaryOperator;
    use severian_universal::{DimExpr, DimensionConstraint};

    let severian_ast::ExpressionKind::Binary {
        operator,
        left,
        right,
    } = &predicate.kind
    else {
        return None;
    };
    let left_dimension = tensor_dimension_expression(left, substitution);
    let right_dimension = tensor_dimension_expression(right, substitution);
    match *operator {
        BinaryOperator::Equal => {
            if let (
                severian_ast::ExpressionKind::Binary {
                    operator: BinaryOperator::Remainder,
                    left: value,
                    right: factor,
                },
                Some(DimExpr::Constant(0)),
            ) = (&left.kind, right_dimension.as_ref())
            {
                let factor = tensor_dimension_expression(factor, substitution)?;
                let DimExpr::Constant(factor) = factor else {
                    return None;
                };
                return Some(DimensionConstraint::MultipleOf {
                    value: tensor_dimension_expression(value, substitution)?,
                    factor,
                });
            }
            Some(DimensionConstraint::Equal(
                left_dimension?,
                right_dimension?,
            ))
        }
        BinaryOperator::GreaterEqual | BinaryOperator::Greater => {
            let DimExpr::Constant(mut minimum) = right_dimension? else {
                return None;
            };
            if *operator == BinaryOperator::Greater {
                minimum = minimum.checked_add(1)?;
            }
            Some(DimensionConstraint::Range {
                value: left_dimension?,
                minimum: Some(minimum),
                maximum: None,
            })
        }
        BinaryOperator::LessEqual | BinaryOperator::Less => {
            let DimExpr::Constant(mut maximum) = right_dimension? else {
                return None;
            };
            if *operator == BinaryOperator::Less {
                maximum = maximum.checked_sub(1)?;
            }
            Some(DimensionConstraint::Range {
                value: left_dimension?,
                minimum: None,
                maximum: Some(maximum),
            })
        }
        _ => None,
    }
}

fn tensor_dimension_expression(
    expression: &severian_ast::Expression,
    substitution: &Substitution,
) -> Option<severian_universal::DimExpr> {
    use severian_ast::{BinaryOperator, ExpressionKind, Literal};
    use severian_universal::DimExpr;

    match &expression.kind {
        ExpressionKind::Literal(Literal::Integer(value)) => {
            Some(DimExpr::Constant(value.replace('_', "").parse().ok()?))
        }
        ExpressionKind::Name(name) => substitution.dimension(name).cloned(),
        ExpressionKind::Binary {
            operator,
            left,
            right,
        } => {
            let left = Box::new(tensor_dimension_expression(left, substitution)?);
            let right = Box::new(tensor_dimension_expression(right, substitution)?);
            match *operator {
                BinaryOperator::Add => Some(DimExpr::Add(left, right)),
                BinaryOperator::Multiply => Some(DimExpr::Multiply(left, right)),
                BinaryOperator::Divide => Some(DimExpr::DivideExact(left, right)),
                _ => None,
            }
        }
        _ => None,
    }
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
            class.name.rsplit('.').next() == actual_name.rsplit('.').next()
                && class.traits.iter().any(|implemented| {
                    implemented.named_parts().is_some_and(|(name, arguments)| {
                        arguments.is_empty()
                            && source_trait_extends(
                                name.rsplit('.').next().unwrap_or(name),
                                bound_name.rsplit('.').next().unwrap_or(bound_name),
                                index,
                                &mut BTreeSet::new(),
                            )
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
    let trait_name = trait_name.rsplit('.').next().unwrap_or(trait_name);
    let bound_name = bound_name.rsplit('.').next().unwrap_or(bound_name);
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

fn source_trait_satisfies_bound(actual_name: &str, bound_name: &str, index: &ProgramIndex) -> bool {
    let actual_name = actual_name.rsplit('.').next().unwrap_or(actual_name);
    let is_trait = index.definitions.values().any(|definition| {
        definition.name.rsplit('.').next() == Some(actual_name)
            && matches!(definition.kind, DefKind::Trait(_))
    });
    is_trait && source_trait_extends(actual_name, bound_name, index, &mut BTreeSet::new())
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
        definition.name.rsplit('.').next() == bound_name.rsplit('.').next()
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
        use severian_universal::UnaryOperator as Unary;
        match (operator.operator, operator.parameters.is_empty()) {
            (Syntax::Plus, true) => types.supports_unary(Unary::Positive, actual),
            (Syntax::Minus, true) => types.supports_unary(Unary::Negative, actual),
            (Syntax::Not, _) => types.supports_unary(Unary::Not, actual),
            (syntax, _) => ast_binary_syntax(syntax)
                .is_some_and(|operator| types.supports_binary(operator, actual)),
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
        .bindings()
        .into_iter()
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
                let expected = binding
                    .annotation
                    .as_ref()
                    .and_then(|annotation| environment_type_name(annotation, names));
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
            | severian_ast::Statement::Defer { expression, .. }
            | severian_ast::Statement::Yield {
                value: expression, ..
            } => {
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
                        if let Some(ty) = environment_type_name(annotation, names) {
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
                let explicit_arguments =
                    if let severian_ast::ExpressionKind::TypeApplication { arguments, .. } =
                        &callee.kind
                    {
                        arguments
                            .iter()
                            .map(|argument| {
                                let name = type_annotation_name(argument)?;
                                Some(names.get(&format!("$type:{name}")).cloned().unwrap_or(name))
                            })
                            .collect::<Option<Vec<_>>>()
                    } else {
                        None
                    };
                let definitions = resolve_path(module, &path, index);
                if let Some(explicit) = &explicit_arguments {
                    let arities = definitions
                        .iter()
                        .filter_map(|definition| match &index.definitions[definition].kind {
                            DefKind::Function(signature) => Some(signature.type_parameters.len()),
                            _ => None,
                        })
                        .collect::<BTreeSet<_>>();
                    if !arities.is_empty() && !arities.iter().any(|arity| explicit.len() <= *arity)
                    {
                        let expected = arities
                            .iter()
                            .map(usize::to_string)
                            .collect::<Vec<_>>()
                            .join(" or ");
                        return Err(Diagnostic::new(
                            "E000206",
                            format!(
                                "`{path}` expects {expected} generic type argument(s), but received {}",
                                explicit.len()
                            ),
                            Some(callee.span),
                        ));
                    }
                }
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
                    let mut substitution = Substitution::for_declaration(signature);
                    let mut conflict = None;
                    if let Some(explicit) = &explicit_arguments {
                        if explicit.len() > signature.type_parameters.len() {
                            continue;
                        }
                        for ((parameter, value), metadata) in signature
                            .type_parameters
                            .iter()
                            .zip(explicit)
                            .zip(generic_parameters(
                                &signature.type_parameters,
                                &signature.constraints,
                            ))
                        {
                            match metadata.kind {
                                severian_universal::GenericParamKind::Type => {
                                    substitution.insert_type(parameter.clone(), value.clone());
                                }
                                severian_universal::GenericParamKind::Dimension => {
                                    substitution
                                        .bind_dimension(parameter, parse_dim_expr(value, 0))
                                        .map_err(|conflict| {
                                            inference_conflict_diagnostic(
                                                &index.definitions[&definition].name,
                                                &conflict,
                                                expression.span,
                                            )
                                        })?;
                                }
                                severian_universal::GenericParamKind::Shape => {
                                    continue;
                                }
                            }
                        }
                    }
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
            let operand_expected = match *operator {
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
            let operand_expected = match *operator {
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
        severian_ast::ExpressionKind::Literal(_)
        | severian_ast::ExpressionKind::Name(_)
        | severian_ast::ExpressionKind::Symbol(_) => {}
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
        severian_ast::ExpressionKind::TypeApplication { callee, .. } => ast_callable_path(callee),
        _ => None,
    }
}

fn infer_substitution(
    pattern: &TypeAnnotation,
    actual: &str,
    parameters: &[String],
    substitution: &mut Substitution,
) -> Result<(), InferenceConflict> {
    match &pattern.kind {
        TypeAnnotationKind::DimensionConstant(expected) => {
            if parse_dim_expr(actual, 0) == severian_universal::DimExpr::Constant(*expected) {
                return Ok(());
            }
            return Err(InferenceConflict {
                parameter: "dimension".into(),
                known: expected.to_string(),
                inferred: actual.to_owned(),
            });
        }
        TypeAnnotationKind::DimensionRuntime(_) => return Ok(()),
        TypeAnnotationKind::ShapeSpread(parameter) => {
            return substitution.bind_shape(parameter, vec![parse_dim_expr(actual, 0)]);
        }
        _ => {}
    }
    let Some((name, arguments)) = pattern.named_parts() else {
        return Ok(());
    };
    if arguments.is_empty() && parameters.iter().any(|parameter| parameter == name) {
        if substitution.kind(name) == severian_universal::GenericParamKind::Dimension {
            return substitution.bind_dimension(name, parse_dim_expr(actual, 0));
        }
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
    if !same_type_constructor(name, actual_name) {
        return Ok(());
    }
    let spread = arguments
        .iter()
        .position(|argument| matches!(argument.kind, TypeAnnotationKind::ShapeSpread(_)));
    if let Some(spread) = spread {
        let suffix = arguments.len() - spread - 1;
        if actual_arguments.len() < spread + suffix {
            return Ok(());
        }
        for (axis, (pattern, actual)) in arguments[..spread]
            .iter()
            .zip(&actual_arguments[..spread])
            .enumerate()
        {
            infer_substitution(pattern, actual, parameters, substitution)?;
            let _ = axis;
        }
        let TypeAnnotationKind::ShapeSpread(parameter) = &arguments[spread].kind else {
            unreachable!()
        };
        let pack_end = actual_arguments.len() - suffix;
        let shape = actual_arguments[spread..pack_end]
            .iter()
            .enumerate()
            .map(|(axis, dimension)| parse_dim_expr(dimension, spread + axis))
            .collect();
        substitution.bind_shape(parameter, shape)?;
        for (pattern, actual) in arguments[spread + 1..]
            .iter()
            .zip(&actual_arguments[pack_end..])
        {
            infer_substitution(pattern, actual, parameters, substitution)?;
        }
    } else {
        if arguments.len() != actual_arguments.len() {
            return Ok(());
        }
        for (pattern, actual) in arguments.iter().zip(actual_arguments) {
            infer_substitution(pattern, actual, parameters, substitution)?;
        }
    }
    Ok(())
}

fn parse_dim_expr(actual: &str, axis: usize) -> severian_universal::DimExpr {
    if let Ok(value) = actual.parse::<u64>() {
        return severian_universal::DimExpr::Constant(value);
    }
    if let Some(runtime) = actual
        .strip_prefix('?')
        .and_then(|runtime| runtime.parse::<u32>().ok())
    {
        return severian_universal::DimExpr::Runtime(severian_universal::RuntimeDimId(runtime));
    }
    severian_universal::DimExpr::Runtime(severian_universal::RuntimeDimId(super::stable_hash(
        &format!("dimension:{axis}:{actual}"),
    ) as u32))
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
            "f8e4m3fn" | "f8e5m2" | "f16" | "bf16" | "f32" | "f64" | "f80" | "f128"
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
            let operation = path.rsplit('.').next().unwrap_or(&path);
            if matches!(operation, "tensor" | "ranked") && arguments.len() == 2 {
                if let Some(shape) = source_integer_list(&arguments[1].value) {
                    return Some(format!(
                        "Tensor[f64{}]",
                        shape
                            .iter()
                            .map(|dimension| format!(", {dimension}"))
                            .collect::<String>()
                    ));
                }
            }
            let target_element = match operation {
                "f8e4m3fn" | "f8e5m2" | "f16" | "bf16" | "f32" | "f64" | "f80" | "f128" | "i8"
                | "i16" | "i32" | "i64" | "i128" | "u8" | "u16" | "u32" | "u64" | "u128" => {
                    Some(operation)
                }
                "to_f8_e4_m3_fn" => Some("f8e4m3fn"),
                "to_f8_e5_m2" => Some("f8e5m2"),
                "to_f_16" => Some("f16"),
                "to_bf_16" => Some("bf16"),
                "to_f_32" => Some("f32"),
                "to_f_64" => Some("f64"),
                "to_f_80" => Some("f80"),
                "to_i_64" => Some("i64"),
                _ => None,
            };
            if let (Some(target), [argument]) = (target_element, arguments.as_slice()) {
                let source = expression_type_name(module, &argument.value, names, index)?;
                let (constructor, source_arguments) = type_application_parts(&source)?;
                if constructor.rsplit('.').next() == Some("Tensor") && !source_arguments.is_empty()
                {
                    let shape = source_arguments[1..]
                        .iter()
                        .map(|dimension| format!(", {dimension}"))
                        .collect::<String>();
                    return Some(format!("Tensor[{target}{shape}]"));
                }
            }
            let direct = resolve_path(module, &path, index)
                .into_iter()
                .find_map(|definition| match &index.definitions[&definition].kind {
                    DefKind::Function(function) if function.type_parameters.is_empty() => {
                        type_annotation_name(&function.result)
                    }
                    DefKind::Function(function) => {
                        let mut substitution = Substitution::for_declaration(function);
                        if let severian_ast::ExpressionKind::TypeApplication {
                            arguments: explicit,
                            ..
                        } = &callee.kind
                        {
                            for ((parameter, argument), metadata) in
                                function.type_parameters.iter().zip(explicit).zip(
                                    generic_parameters(
                                        &function.type_parameters,
                                        &function.constraints,
                                    ),
                                )
                            {
                                match metadata.kind {
                                    severian_universal::GenericParamKind::Type => {
                                        let value = type_annotation_name(argument)?;
                                        substitution.insert_type(
                                            parameter.clone(),
                                            names
                                                .get(&format!("$type:{value}"))
                                                .cloned()
                                                .unwrap_or(value),
                                        );
                                    }
                                    severian_universal::GenericParamKind::Dimension => {
                                        let value = type_annotation_name(argument)?;
                                        substitution
                                            .bind_dimension(parameter, parse_dim_expr(&value, 0))
                                            .ok()?;
                                    }
                                    severian_universal::GenericParamKind::Shape => {}
                                }
                            }
                        }
                        for (parameter, argument) in function.parameters.iter().zip(arguments) {
                            let Some(actual) =
                                expression_type_name(module, &argument.value, names, index)
                            else {
                                continue;
                            };
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
                    DefKind::Class(class) => {
                        let name = &index.definitions[&definition].name;
                        let explicit = match &callee.kind {
                            severian_ast::ExpressionKind::TypeApplication { arguments, .. } => {
                                arguments
                                    .iter()
                                    .map(type_annotation_name)
                                    .collect::<Option<Vec<_>>>()?
                            }
                            _ => Vec::new(),
                        };
                        if explicit.is_empty() {
                            Some(name.clone())
                        } else if explicit.len() <= class.type_parameters.len() {
                            Some(format!("{name}[{}]", explicit.join(", ")))
                        } else {
                            None
                        }
                    }
                    _ => None,
                });
            if direct.is_some() {
                return direct;
            }
            let severian_ast::ExpressionKind::Member { object, name } = &callee.kind else {
                return None;
            };
            let receiver = expression_type_name(module, object, names, index)?;
            let (owner, owner_arguments) = type_application_parts(&receiver)?;
            index.methods.get(name)?.iter().find_map(|method| {
                if !same_type_constructor(&method.owner, owner)
                    || method.owner_type_parameters.len() != owner_arguments.len()
                    || method.parameters.len() != arguments.len()
                {
                    return None;
                }
                let mut substitution = method
                    .owner_type_parameters
                    .iter()
                    .zip(&owner_arguments)
                    .map(|(parameter, actual)| (parameter.clone(), (*actual).to_owned()))
                    .collect::<Substitution>();
                let mut conflict = None;
                for (parameter, argument) in method.parameters.iter().zip(arguments) {
                    let Some(actual) = expression_type_name(module, &argument.value, names, index)
                    else {
                        continue;
                    };
                    conflict = infer_substitution(
                        parameter,
                        &actual,
                        &method.type_parameters,
                        &mut substitution,
                    )
                    .err();
                    if conflict.is_some() {
                        break;
                    }
                }
                if conflict.is_some()
                    || method
                        .type_parameters
                        .iter()
                        .any(|parameter| !substitution.contains_key(parameter))
                {
                    return None;
                }
                type_annotation_name(&specialize_annotation(&method.result, &substitution))
            })
        }
        severian_ast::ExpressionKind::Unary { operator, operand } => {
            if *operator == severian_ast::UnaryOperator::Not {
                Some("bool".to_owned())
            } else {
                expression_type_name(module, operand, names, index)
            }
        }
        severian_ast::ExpressionKind::Member { object, name } => {
            let receiver = expression_type_name(module, object, names, index)?;
            let (owner, owner_arguments) = type_application_parts(&receiver)?;
            index.fields.get(name)?.iter().find_map(|field| {
                if !same_type_constructor(&field.owner, owner)
                    || field.owner_type_parameters.len() != owner_arguments.len()
                {
                    return None;
                }
                let substitution = field
                    .owner_type_parameters
                    .iter()
                    .zip(&owner_arguments)
                    .map(|(parameter, actual)| (parameter.clone(), (*actual).to_owned()))
                    .collect::<Substitution>();
                type_annotation_name(&specialize_annotation(&field.annotation, &substitution))
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

fn source_integer_list(expression: &severian_ast::Expression) -> Option<Vec<u64>> {
    let severian_ast::ExpressionKind::List(values) = &expression.kind else {
        return None;
    };
    values
        .iter()
        .map(|value| match &value.kind {
            severian_ast::ExpressionKind::Literal(severian_ast::Literal::Integer(value)) => {
                value.replace('_', "").parse::<u64>().ok()
            }
            _ => None,
        })
        .collect()
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
        TypeAnnotationKind::DimensionConstant(value) => Some(value.to_string()),
        TypeAnnotationKind::DimensionRuntime(runtime) => Some(format!("?{runtime}")),
        TypeAnnotationKind::ShapeSpread(name) => Some(format!("*{name}")),
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
    // Preserve applications such as Tensor[bf16]. Generic result inference
    // needs the complete type, not only the outer constructor name.
    type_annotation_name(annotation)
}

fn environment_type_name(
    annotation: &TypeAnnotation,
    names: &BTreeMap<String, String>,
) -> Option<String> {
    let mut substitution = Substitution::new();
    let mut shapes = BTreeMap::<String, BTreeMap<usize, severian_universal::DimExpr>>::new();
    for (name, actual) in names {
        if let Some(parameter) = name.strip_prefix("$type:") {
            substitution.insert_type(parameter.to_owned(), actual.clone());
            continue;
        }
        if let Some(parameter) = name.strip_prefix("$dimension:") {
            substitution
                .bind_dimension(parameter, parse_dim_expr(actual, 0))
                .ok()?;
            continue;
        }
        if let Some(shape_axis) = name.strip_prefix("$shape:") {
            let (parameter, axis) = shape_axis.rsplit_once(':')?;
            let axis = axis.parse::<usize>().ok()?;
            shapes
                .entry(parameter.to_owned())
                .or_default()
                .insert(axis, parse_dim_expr(actual, axis));
        }
    }
    for (parameter, axes) in shapes {
        if axes.keys().copied().eq(0..axes.len()) {
            substitution
                .bind_shape(&parameter, axes.into_values().collect())
                .ok()?;
        }
    }
    specialized_type_name(annotation, &substitution)
}

fn specialized_type_name(
    annotation: &TypeAnnotation,
    substitution: &Substitution,
) -> Option<String> {
    type_annotation_name(&specialize_annotation(annotation, substitution))
}

pub(crate) fn specialize_function(
    function: &severian_ast::FunctionDeclaration,
    substitution: &Substitution,
) -> severian_ast::FunctionDeclaration {
    let mut function = function.clone();
    for parameter in &mut function.parameters {
        parameter.annotation = specialize_annotation(&parameter.annotation, substitution);
        if let Some(default) = &mut parameter.default {
            specialize_expression(default, substitution);
        }
    }
    function.result = specialize_annotation(&function.result, substitution);
    for constraint in &mut function.constraints {
        match constraint {
            severian_ast::GenericConstraint::Parameter { bound, .. } => {
                *bound = specialize_annotation(bound, substitution);
            }
            severian_ast::GenericConstraint::VariadicPack { .. } => {}
            severian_ast::GenericConstraint::Predicate(predicate) => {
                specialize_expression(predicate, substitution);
            }
        }
    }
    for contract in &mut function.contracts {
        specialize_expression(&mut contract.condition, substitution);
        if let Some(failure) = &mut contract.failure {
            specialize_expression(failure, substitution);
        }
    }
    if let Some(hook) = &mut function.hook {
        specialize_statements(&mut hook.with_phase, substitution);
        specialize_statements(&mut hook.without_phase, substitution);
    }
    if let Some(body) = &mut function.body {
        specialize_statements(body, substitution);
    }
    function
}

pub(crate) fn specialize_operator(
    implementation: &severian_ast::OperatorImplementation,
    substitution: &Substitution,
) -> severian_ast::OperatorImplementation {
    let mut implementation = implementation.clone();
    for parameter in &mut implementation.parameters {
        parameter.annotation = specialize_annotation(&parameter.annotation, substitution);
    }
    implementation.result = specialize_annotation(&implementation.result, substitution);
    for constraint in &mut implementation.constraints {
        match constraint {
            severian_ast::GenericConstraint::Parameter { bound, .. } => {
                *bound = specialize_annotation(bound, substitution);
            }
            severian_ast::GenericConstraint::VariadicPack { .. } => {}
            severian_ast::GenericConstraint::Predicate(predicate) => {
                specialize_expression(predicate, substitution);
            }
        }
    }
    for contract in &mut implementation.contracts {
        specialize_expression(&mut contract.condition, substitution);
        if let Some(failure) = &mut contract.failure {
            specialize_expression(failure, substitution);
        }
    }
    specialize_statements(&mut implementation.body, substitution);
    implementation
}

pub(super) fn specialize_signature(
    function: &FunctionDecl,
    substitution: &Substitution,
) -> FunctionDecl {
    let mut parameter_defaults = function.parameter_defaults.clone();
    for default in parameter_defaults.iter_mut().flatten() {
        specialize_expression(default, substitution);
    }
    let mut constraints = function.constraints.clone();
    for constraint in &mut constraints {
        match constraint {
            severian_ast::GenericConstraint::Parameter { bound, .. } => {
                *bound = specialize_annotation(bound, substitution);
            }
            severian_ast::GenericConstraint::VariadicPack { .. } => {}
            severian_ast::GenericConstraint::Predicate(predicate) => {
                specialize_expression(predicate, substitution);
            }
        }
    }
    let mut generic_body = function.generic_body.clone();
    if let Some(body) = &mut generic_body {
        specialize_statements(body, substitution);
    }
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
        parameter_defaults,
        result: specialize_annotation(&function.result, substitution),
        constraints,
        generic_body,
    }
}

fn specialize_statements(statements: &mut [severian_ast::Statement], substitution: &Substitution) {
    for statement in statements {
        specialize_statement(statement, substitution);
    }
}

fn specialize_binding(binding: &mut severian_ast::Binding, substitution: &Substitution) {
    if let Some(annotation) = &mut binding.annotation {
        *annotation = specialize_annotation(annotation, substitution);
    }
    specialize_expression(&mut binding.value, substitution);
}

fn specialize_statement(statement: &mut severian_ast::Statement, substitution: &Substitution) {
    use severian_ast::Statement;
    match statement {
        Statement::Binding(binding) => specialize_binding(binding, substitution),
        Statement::Destructure { value, .. }
        | Statement::Expression(value)
        | Statement::Defer {
            expression: value, ..
        }
        | Statement::Return {
            value: Some(value), ..
        }
        | Statement::Yield { value, .. }
        | Statement::FallibleElse { value, .. } => {
            specialize_expression(value, substitution);
        }
        Statement::FieldAssignment { object, value, .. } => {
            specialize_expression(object, substitution);
            specialize_expression(value, substitution);
        }
        Statement::IndexAssignment {
            object,
            index,
            value,
            ..
        } => {
            specialize_expression(object, substitution);
            specialize_expression(index, substitution);
            specialize_expression(value, substitution);
        }
        Statement::Return { value: None, .. }
        | Statement::Break { .. }
        | Statement::Continue { .. } => {}
        Statement::Assert {
            condition, message, ..
        } => {
            specialize_expression(condition, substitution);
            if let Some(message) = message {
                specialize_expression(message, substitution);
            }
        }
        Statement::Unsafe { body, .. } | Statement::Placement { body, .. } => {
            specialize_statements(body, substitution);
        }
        Statement::Try {
            body,
            catch_annotation,
            catch_body,
            ..
        } => {
            specialize_statements(body, substitution);
            if let Some(annotation) = catch_annotation {
                *annotation = specialize_annotation(annotation, substitution);
            }
            specialize_statements(catch_body, substitution);
        }
        Statement::If {
            condition,
            then_block,
            else_block,
            ..
        } => {
            specialize_expression(condition, substitution);
            specialize_statements(then_block, substitution);
            specialize_statements(else_block, substitution);
        }
        Statement::While {
            condition,
            initializer,
            guards,
            body,
            ..
        } => {
            specialize_expression(condition, substitution);
            if let Some(initializer) = initializer {
                specialize_binding(initializer, substitution);
            }
            for guard in guards {
                specialize_expression(&mut guard.condition, substitution);
            }
            specialize_statements(body, substitution);
        }
        Statement::For {
            iterable,
            initializer,
            body,
            ..
        } => {
            specialize_expression(iterable, substitution);
            if let Some(initializer) = initializer {
                specialize_binding(initializer, substitution);
            }
            specialize_statements(body, substitution);
        }
        Statement::Match { subject, cases, .. } => {
            specialize_expression(subject, substitution);
            for case in cases {
                if let Some(annotation) = &mut case.annotation {
                    *annotation = specialize_annotation(annotation, substitution);
                }
                specialize_statements(&mut case.body, substitution);
            }
        }
        Statement::Select {
            limit,
            cases,
            error_body,
            ..
        } => {
            specialize_expression(limit, substitution);
            for case in cases {
                specialize_expression(&mut case.channel, substitution);
                specialize_statements(&mut case.body, substitution);
            }
            specialize_statements(error_body, substitution);
        }
    }
    if let Statement::FallibleElse { body, .. } = statement {
        specialize_statements(body, substitution);
    }
}

fn specialize_expression(expression: &mut severian_ast::Expression, substitution: &Substitution) {
    use severian_ast::ExpressionKind;
    match &mut expression.kind {
        ExpressionKind::Name(name) => {
            if let Some(severian_universal::DimExpr::Constant(value)) = substitution.dimension(name)
            {
                expression.kind =
                    ExpressionKind::Literal(severian_ast::Literal::Integer(value.to_string()));
            } else if let Some(replacement) = substitution.get(name) {
                name.clone_from(replacement);
            }
        }
        ExpressionKind::Literal(_) | ExpressionKind::Symbol(_) => {}
        ExpressionKind::List(values)
        | ExpressionKind::Set(values)
        | ExpressionKind::Tuple(values) => {
            for value in values {
                specialize_expression(value, substitution);
            }
        }
        ExpressionKind::Map(entries) => {
            for entry in entries {
                specialize_expression(&mut entry.key, substitution);
                specialize_expression(&mut entry.value, substitution);
            }
        }
        ExpressionKind::ListComprehension { value, clauses }
        | ExpressionKind::SetComprehension { value, clauses } => {
            specialize_expression(value, substitution);
            specialize_comprehension_clauses(clauses, substitution);
        }
        ExpressionKind::MapComprehension {
            key,
            value,
            clauses,
        } => {
            specialize_expression(key, substitution);
            specialize_expression(value, substitution);
            specialize_comprehension_clauses(clauses, substitution);
        }
        ExpressionKind::Mock { cases, fallback } => {
            for case in cases {
                specialize_expression(&mut case.call, substitution);
                specialize_expression(&mut case.result, substitution);
            }
            specialize_expression(fallback, substitution);
        }
        ExpressionKind::Lambda { body, .. }
        | ExpressionKind::Member { object: body, .. }
        | ExpressionKind::Async {
            expression: body, ..
        }
        | ExpressionKind::Await { expression: body }
        | ExpressionKind::Throw { error: body }
        | ExpressionKind::Unary { operand: body, .. } => {
            specialize_expression(body, substitution);
        }
        ExpressionKind::Index { object, index } => {
            specialize_expression(object, substitution);
            specialize_expression(index, substitution);
        }
        ExpressionKind::Slice {
            object,
            start,
            end,
            step,
            ..
        } => {
            specialize_expression(object, substitution);
            for bound in [start, end, step].into_iter().flatten() {
                specialize_expression(bound, substitution);
            }
        }
        ExpressionKind::TypeApplication { callee, arguments } => {
            specialize_expression(callee, substitution);
            for argument in arguments {
                *argument = specialize_annotation(argument, substitution);
            }
        }
        ExpressionKind::Call { callee, arguments } => {
            specialize_expression(callee, substitution);
            for argument in arguments {
                specialize_expression(&mut argument.value, substitution);
                if let Some(expected_error) = &mut argument.expected_error {
                    specialize_expression(expected_error, substitution);
                }
            }
        }
        ExpressionKind::Conditional {
            value,
            condition,
            fallback,
        } => {
            specialize_expression(value, substitution);
            specialize_expression(condition, substitution);
            specialize_expression(fallback, substitution);
        }
        ExpressionKind::Fallback { value, fallback }
        | ExpressionKind::Binary {
            left: value,
            right: fallback,
            ..
        } => {
            specialize_expression(value, substitution);
            specialize_expression(fallback, substitution);
        }
    }
}

pub(crate) fn specialize_property(
    property: &severian_ast::PropertyDeclaration,
    substitution: &Substitution,
) -> severian_ast::PropertyDeclaration {
    let mut property = property.clone();
    property.annotation = specialize_annotation(&property.annotation, substitution);
    if let Some(default) = &mut property.default {
        specialize_expression(default, substitution);
    }
    property
}

fn specialize_comprehension_clauses(
    clauses: &mut [severian_ast::ComprehensionClause],
    substitution: &Substitution,
) {
    for clause in clauses {
        specialize_expression(&mut clause.iterable, substitution);
        if let Some(condition) = &mut clause.condition {
            specialize_expression(condition, substitution);
        }
    }
}

fn specialize_annotation(
    annotation: &TypeAnnotation,
    substitution: &Substitution,
) -> TypeAnnotation {
    let kind = match &annotation.kind {
        TypeAnnotationKind::Named { name, arguments } => {
            if arguments.is_empty() {
                if let Some(dimension) = substitution.dimension(name) {
                    dim_expr_annotation(dimension)
                } else if let Some(replacement) = substitution.get(name) {
                    TypeAnnotationKind::Named {
                        name: replacement.clone(),
                        arguments: Vec::new(),
                    }
                } else {
                    TypeAnnotationKind::Named {
                        name: name.clone(),
                        arguments: Vec::new(),
                    }
                }
            } else {
                TypeAnnotationKind::Named {
                    name: substitution
                        .get(name)
                        .cloned()
                        .unwrap_or_else(|| name.clone()),
                    arguments: arguments
                        .iter()
                        .flat_map(|argument| {
                            if let TypeAnnotationKind::ShapeSpread(parameter) = &argument.kind {
                                if let Some(shape) = substitution.shape(parameter) {
                                    return shape
                                        .iter()
                                        .map(|dimension| TypeAnnotation {
                                            kind: dim_expr_annotation(dimension),
                                            span: argument.span,
                                        })
                                        .collect::<Vec<_>>();
                                }
                            }
                            vec![specialize_annotation(argument, substitution)]
                        })
                        .collect(),
                }
            }
        }
        TypeAnnotationKind::DimensionConstant(value) => {
            TypeAnnotationKind::DimensionConstant(*value)
        }
        TypeAnnotationKind::DimensionRuntime(runtime) => {
            TypeAnnotationKind::DimensionRuntime(*runtime)
        }
        TypeAnnotationKind::ShapeSpread(name) => TypeAnnotationKind::ShapeSpread(name.clone()),
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

fn dim_expr_annotation(dimension: &severian_universal::DimExpr) -> TypeAnnotationKind {
    match dimension {
        severian_universal::DimExpr::Constant(value) => {
            TypeAnnotationKind::DimensionConstant(*value)
        }
        severian_universal::DimExpr::Runtime(runtime) => {
            TypeAnnotationKind::DimensionRuntime(runtime.0)
        }
        severian_universal::DimExpr::Parameter(parameter) => TypeAnnotationKind::Named {
            name: format!("__dim_parameter_{}", parameter.0),
            arguments: Vec::new(),
        },
        severian_universal::DimExpr::Add(_, _)
        | severian_universal::DimExpr::Multiply(_, _)
        | severian_universal::DimExpr::DivideExact(_, _) => TypeAnnotationKind::Named {
            name: format_dim_expr(dimension),
            arguments: Vec::new(),
        },
    }
}

#[cfg(test)]
mod environment_tests {
    use super::*;

    #[test]
    fn nested_specialization_retains_element_dimensions_and_shape_packs() {
        let source = severian_source::SourceFile::virtual_source(
            "ranked-environment.sev",
            "def preserve[T: TensorElement, B: Dim, *Tail: Dim](value: Tensor[T, B, *Tail]) -> Tensor[T, B, *Tail]:\n    return value\n",
        );
        let tokens = severian_lexer::scan(&source).unwrap();
        let ast = severian_parser::parse(&tokens).unwrap();
        let severian_ast::Item::Function(function) = &ast.items[0] else {
            panic!("expected function")
        };

        let mut substitution = Substitution::new();
        substitution.insert_type("T".into(), "bf16".into());
        substitution
            .bind_dimension("B", severian_universal::DimExpr::Constant(2))
            .unwrap();
        substitution
            .bind_shape(
                "Tail",
                vec![
                    severian_universal::DimExpr::Constant(16),
                    severian_universal::DimExpr::Constant(128),
                ],
            )
            .unwrap();

        let mut names = BTreeMap::new();
        install_specialization_environment(&mut names, &substitution);
        assert_eq!(
            environment_type_name(&function.parameters[0].annotation, &names).as_deref(),
            Some("Tensor[bf16, 2, 16, 128]")
        );
    }
}
