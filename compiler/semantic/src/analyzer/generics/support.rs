use super::*;

pub(super) fn generic_class_expression(expression: &Expr) -> Option<(String, Vec<Type>, Span)> {
    let Expr::Index(index) = expression else {
        return None;
    };
    let class = expression_path(index.object.as_ref())?;
    let arguments = match index.index.as_ref() {
        Expr::Tuple(tuple) => tuple
            .elements
            .iter()
            .map(expression_as_type)
            .collect::<Option<Vec<_>>>()?,
        argument => vec![expression_as_type(argument)?],
    };
    Some((class, arguments, index.span))
}

fn expression_path(expression: &Expr) -> Option<String> {
    match expression {
        Expr::Identifier(identifier) => Some(identifier.name.clone()),
        Expr::Member(member) => Some(format!(
            "{}.{}",
            expression_path(member.object.as_ref())?,
            member.member.name
        )),
        _ => None,
    }
}

fn expression_as_type(expression: &Expr) -> Option<Type> {
    let (span, segments) = match expression {
        Expr::Identifier(identifier) => (identifier.span, vec![identifier.clone()]),
        Expr::Member(member) => {
            let Expr::Identifier(module) = member.object.as_ref() else {
                return None;
            };
            (member.span, vec![module.clone(), member.member.clone()])
        }
        _ => return None,
    };
    Some(Type::Named(severian_ast::TypePath {
        span,
        segments,
        args: Vec::new(),
    }))
}

pub(super) fn contains_generic(ty: &Type, generics: &HashSet<String>) -> bool {
    match ty {
        Type::Named(path) => {
            path.segments
                .first()
                .is_some_and(|segment| generics.contains(&segment.name))
                || path.args.iter().any(|argument| {
                    argument
                        .as_type()
                        .is_some_and(|argument| contains_generic(argument, generics))
                })
        }
        Type::List { element, .. }
        | Type::Set { element, .. }
        | Type::Option { some: element, .. }
        | Type::Future {
            output: element, ..
        }
        | Type::Reference { inner: element, .. } => contains_generic(element, generics),
        Type::Tuple { elements, .. }
        | Type::Union {
            alternatives: elements,
            ..
        } => elements
            .iter()
            .any(|element| contains_generic(element, generics)),
        Type::Map { key, value, .. }
        | Type::Result {
            ok: key,
            err: value,
            ..
        } => contains_generic(key, generics) || contains_generic(value, generics),
        Type::Function {
            params, returns, ..
        } => {
            params
                .iter()
                .any(|parameter| contains_generic(parameter, generics))
                || contains_generic(returns, generics)
        }
    }
}

pub(super) fn specialization_name(template: &str, arguments: &[Type]) -> String {
    let suffix = arguments
        .iter()
        .map(declaration_type_key)
        .map(|name| {
            name.chars()
                .map(|character| {
                    if character.is_ascii_alphanumeric() {
                        character
                    } else {
                        '_'
                    }
                })
                .collect::<String>()
        })
        .collect::<Vec<_>>()
        .join("__");
    let template = template
        .chars()
        .map(|character| {
            if character.is_ascii_alphanumeric() {
                character
            } else {
                '_'
            }
        })
        .collect::<String>();
    format!("{template}__{suffix}")
}

pub(super) fn dtype_argument(ty: &Type) -> Option<TensorElementType> {
    let Type::Named(path) = ty else { return None };
    (path.args.is_empty() && path.segments.len() == 1)
        .then(|| TensorElementType::parse(&path.segments[0].name))?
}

pub(super) fn set_type_span(ty: &mut Type, span: Span) {
    match ty {
        Type::Named(path) => path.span = span,
        Type::List { span: current, .. }
        | Type::Tuple { span: current, .. }
        | Type::Union { span: current, .. }
        | Type::Map { span: current, .. }
        | Type::Set { span: current, .. }
        | Type::Result { span: current, .. }
        | Type::Option { span: current, .. }
        | Type::Function { span: current, .. }
        | Type::Future { span: current, .. }
        | Type::Reference { span: current, .. } => *current = span,
    }
}

pub(super) fn callable_types_match(
    method: &severian_ast::FunctionDecl,
    required: &severian_ast::TraitMethod,
    self_type: &Type,
) -> bool {
    let method_params = &method.params[usize::from(
        method
            .params
            .first()
            .is_some_and(|parameter| parameter.name.name == "self"),
    )..];
    let required_params = &required.params[usize::from(
        required
            .params
            .first()
            .is_some_and(|parameter| parameter.name.name == "self"),
    )..];
    if method_params.len() != required_params.len() {
        return false;
    }
    let substitutions = HashMap::from([("Self".to_owned(), self_type.clone())]);
    method_params
        .iter()
        .zip(required_params)
        .all(|(actual, expected)| {
            actual.ty.as_ref().map(declaration_type_key)
                == expected
                    .ty
                    .as_ref()
                    .map(|ty| substitute_declared_type(ty, &substitutions))
                    .as_ref()
                    .map(declaration_type_key)
        })
        && method.return_type.as_ref().map(declaration_type_key)
            == required
                .return_type
                .as_ref()
                .map(|ty| substitute_declared_type(ty, &substitutions))
                .as_ref()
                .map(declaration_type_key)
}
