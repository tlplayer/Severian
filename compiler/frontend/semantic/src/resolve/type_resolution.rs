use super::*;
use severian_package::TypeResolutionPolicy;
use std::collections::BTreeSet;

/// Enforce the package's type-resolution boundary after metadata attachment and
/// before ownership checking or MIR construction.
pub fn enforce_type_resolution_policy(
    module: &Module,
    program: &Program,
    policy: TypeResolutionPolicy,
) -> Result<(), SemanticError> {
    let specialized = specialize_generic_classes(module)?;
    enforce_specialized_type_resolution_policy(&specialized, program, policy)
}

fn enforce_specialized_type_resolution_policy(
    module: &Module,
    program: &Program,
    policy: TypeResolutionPolicy,
) -> Result<(), SemanticError> {
    if policy.is_permissive() {
        return Ok(());
    }

    if policy.deny_any || policy.deny_inferred_fallback {
        if let Some((span, name, kind)) = first_inferred_declaration(module) {
            return Err(policy_error(
                span,
                AnyOrigin::InferenceFallback,
                "Any",
                format!("{kind} `{name}` has no declared type"),
                if policy.deny_inferred_fallback {
                    "compiler.type_resolution.deny_inferred_fallback"
                } else {
                    "compiler.type_resolution.deny_any"
                },
            ));
        }
    }

    if policy.deny_unresolved {
        if let Some(span) = first_unresolved_definition(module, program) {
            return Err(policy_error(
                span,
                AnyOrigin::UnresolvedType,
                "unresolved",
                "a declared type could not be resolved".into(),
                "compiler.type_resolution.deny_unresolved",
            ));
        }
    }

    let violation = program
        .metadata
        .expression_any_origins
        .iter()
        .filter_map(|(id, origin)| {
            let ty = program.metadata.expression_types.get(id)?;
            let kind = program.metadata.types.get(*ty)?;
            let dynamic = match kind {
                TypeKind::Any => "Any",
                TypeKind::TensorAny => "Tensor[Any]",
                _ => return None,
            };
            let denied_by = denied_by(policy, dynamic, *origin)?;
            let span = program.metadata.sources.expression_span(*id)?;
            Some((span, dynamic, *origin, denied_by))
        })
        .min_by_key(|(span, ..)| (span.file, span.range.start, span.range.end));
    if let Some((span, dynamic, origin, denied_by)) = violation {
        return Err(policy_error(
            Span::new(span.range.start, span.range.end),
            origin,
            dynamic,
            origin_explanation(origin).into(),
            denied_by,
        ));
    }
    Ok(())
}

fn denied_by(
    policy: TypeResolutionPolicy,
    dynamic: &str,
    origin: AnyOrigin,
) -> Option<&'static str> {
    if origin == AnyOrigin::Explicit {
        return None;
    }
    if origin == AnyOrigin::InferenceFallback && policy.deny_inferred_fallback {
        return Some("compiler.type_resolution.deny_inferred_fallback");
    }
    if matches!(
        origin,
        AnyOrigin::UnresolvedType | AnyOrigin::UnresolvedGeneric
    ) && policy.deny_unresolved
    {
        return Some("compiler.type_resolution.deny_unresolved");
    }
    if origin == AnyOrigin::LostTypeInformation && policy.deny_lost_type_information {
        return Some("compiler.type_resolution.deny_lost_type_information");
    }
    if dynamic == "Tensor[Any]" && policy.deny_tensor_any {
        return Some("compiler.type_resolution.deny_tensor_any");
    }
    if dynamic == "Any" && policy.deny_any {
        return Some("compiler.type_resolution.deny_any");
    }
    None
}

fn policy_error(
    span: Span,
    origin: AnyOrigin,
    dynamic: &str,
    reason: String,
    setting: &str,
) -> SemanticError {
    error(
        span,
        format!(
            "E000207: unresolved type escaped semantic analysis\n\
             type inference produced `{dynamic}` because {reason}\n\
             package policy forbids this type-resolution origin ({origin:?}):\n  {setting} = true"
        ),
    )
}

fn origin_explanation(origin: AnyOrigin) -> &'static str {
    match origin {
        AnyOrigin::Explicit => "the source explicitly requested a dynamic type",
        AnyOrigin::InferenceFallback => "a declaration or inference path had no concrete type",
        AnyOrigin::UnresolvedType => "a named type did not resolve",
        AnyOrigin::UnresolvedGeneric => "a generic tensor element type remained unbound",
        AnyOrigin::LostTypeInformation => {
            "an operation discarded previously available type information"
        }
    }
}

fn first_inferred_declaration(module: &Module) -> Option<(Span, &str, &'static str)> {
    for item in &module.items {
        match item {
            Item::Function(function) => {
                if let Some(parameter) = function
                    .params
                    .iter()
                    .find(|parameter| parameter.ty.is_none())
                {
                    return Some((parameter.name.span, &parameter.name.name, "parameter"));
                }
            }
            Item::Class(class) => {
                if let Some(field) = class.fields.iter().find(|field| field.ty.is_none()) {
                    return Some((field.name.span, &field.name.name, "field"));
                }
                for function in &class.constructors {
                    if let Some(parameter) = function
                        .params
                        .iter()
                        .find(|parameter| parameter.ty.is_none())
                    {
                        return Some((parameter.name.span, &parameter.name.name, "parameter"));
                    }
                }
                for function in &class.methods {
                    if let Some(parameter) = function
                        .params
                        .iter()
                        .find(|parameter| parameter.ty.is_none())
                    {
                        return Some((parameter.name.span, &parameter.name.name, "parameter"));
                    }
                }
            }
            Item::Trait(declaration) => {
                for parameters in declaration
                    .methods
                    .iter()
                    .map(|method| &method.params)
                    .chain(
                        declaration
                            .operators
                            .iter()
                            .map(|operator| &operator.params),
                    )
                {
                    if let Some(parameter) =
                        parameters.iter().find(|parameter| parameter.ty.is_none())
                    {
                        return Some((parameter.name.span, &parameter.name.name, "parameter"));
                    }
                }
            }
            Item::Enum(enumeration) => {
                for variant in &enumeration.variants {
                    if let Some(field) = variant.fields.iter().find(|field| field.ty.is_none()) {
                        return Some((field.name.span, &field.name.name, "variant field"));
                    }
                }
            }
            Item::Import(_) | Item::Statement(_) => {}
        }
    }
    None
}

fn first_unresolved_definition(module: &Module, program: &Program) -> Option<Span> {
    let types = &program.metadata.types;
    for (id, function) in &program.metadata.functions {
        if function
            .parameters
            .iter()
            .chain(std::iter::once(&function.returns))
            .any(|ty| contains_unresolved(types, *ty, &mut BTreeSet::new()))
        {
            let span = program
                .metadata
                .sources
                .definition_span(DefinitionId::Function(*id))?;
            return Some(Span::new(span.range.start, span.range.end));
        }
    }
    for (id, class) in &program.metadata.classes {
        if class
            .fields
            .iter()
            .any(|field| contains_unresolved(types, field.ty, &mut BTreeSet::new()))
        {
            let span = program
                .metadata
                .sources
                .definition_span(DefinitionId::Type(*id))?;
            return Some(Span::new(span.range.start, span.range.end));
        }
    }
    for (id, enumeration) in &program.metadata.enums {
        if enumeration.variants.iter().any(|variant| {
            variant
                .fields
                .iter()
                .any(|field| contains_unresolved(types, *field, &mut BTreeSet::new()))
        }) {
            let span = program
                .metadata
                .sources
                .definition_span(DefinitionId::Type(*id))?;
            return Some(Span::new(span.range.start, span.range.end));
        }
    }
    for (name, ty) in &program.metadata.globals {
        if contains_unresolved(types, *ty, &mut BTreeSet::new()) {
            let span = module.items.iter().find_map(|item| match item {
                Item::Statement(Stmt::Let(binding)) if binding.name.name == *name => {
                    Some(binding.name.span)
                }
                _ => None,
            });
            return span.or(Some(Span::dummy()));
        }
    }
    None
}

fn contains_unresolved(types: &TypeTable, id: TypeId, seen: &mut BTreeSet<TypeId>) -> bool {
    if !seen.insert(id) {
        return false;
    }
    match types.get(id) {
        Some(TypeKind::Unresolved { .. }) => true,
        Some(
            TypeKind::List(inner)
            | TypeKind::Set(inner)
            | TypeKind::Channel(inner)
            | TypeKind::Option(inner)
            | TypeKind::Future(inner)
            | TypeKind::Reference { inner, .. },
        ) => contains_unresolved(types, *inner, seen),
        Some(TypeKind::Tuple(elements) | TypeKind::Union(elements)) => elements
            .iter()
            .any(|element| contains_unresolved(types, *element, seen)),
        Some(
            TypeKind::Map { key, value }
            | TypeKind::Result {
                ok: key,
                error: value,
            },
        ) => contains_unresolved(types, *key, seen) || contains_unresolved(types, *value, seen),
        Some(TypeKind::Function {
            parameters,
            returns,
        }) => {
            parameters
                .iter()
                .any(|parameter| contains_unresolved(types, *parameter, seen))
                || contains_unresolved(types, *returns, seen)
        }
        Some(TypeKind::Named { arguments, .. }) => arguments
            .iter()
            .any(|argument| contains_unresolved(types, *argument, seen)),
        Some(
            TypeKind::Int
            | TypeKind::Float
            | TypeKind::Bool
            | TypeKind::String
            | TypeKind::Unit
            | TypeKind::Any
            | TypeKind::Tensor(_)
            | TypeKind::TensorAny,
        )
        | None => false,
    }
}
