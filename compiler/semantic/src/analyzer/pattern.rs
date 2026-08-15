use super::*;

pub(super) fn lower_pattern(
    pattern: &Pattern,
    scope: &mut HashMap<String, Binding>,
    aliases: &HashMap<String, String>,
) -> Result<MatchPattern, SemanticError> {
    match pattern {
        Pattern::Wildcard(_) => Ok(MatchPattern::Wildcard),
        Pattern::Identifier(name) => {
            if aliases.contains_key(&format!("__variant_fields.{}", name.name)) {
                return Ok(MatchPattern::Constructor {
                    name: name.name.clone(),
                    fields: Vec::new(),
                });
            }
            let reference = source_binding(name);
            scope.insert(
                name.name.clone(),
                Binding {
                    reference: reference.clone(),
                    ty: ValueType::Any,
                    class: None,
                    enum_variant: None,
                    function_return: None,
                    collection_len: None,
                    mutable: false,
                    field: false,
                    integer_max: None,
                    known_integer: None,
                    any_origin: Some(AnyOrigin::LostTypeInformation),
                },
            );
            Ok(MatchPattern::Bind(reference))
        }
        Pattern::Literal(Literal::Integer { value, .. }) => Ok(MatchPattern::Integer(*value)),
        Pattern::Literal(Literal::Float { value, .. }) => Ok(MatchPattern::Float(value.to_bits())),
        Pattern::Literal(Literal::Boolean { value, .. }) => Ok(MatchPattern::Boolean(*value)),
        Pattern::Literal(Literal::String { value, .. }) => Ok(MatchPattern::String(value.clone())),
        Pattern::Constructor { span, name, fields } => {
            let Type::Named(path) = name else {
                return Err(error(name.span(), "invalid constructor pattern"));
            };
            let name = path.segments.first().unwrap().name.clone();
            if fields.is_empty() {
                if let Some(field_names) = aliases
                    .get(&format!("__variant_fields.{name}"))
                    .or_else(|| aliases.get(&format!("__class_fields.{name}")))
                {
                    let fields = if field_names.is_empty() {
                        Vec::new()
                    } else {
                        field_names
                            .split(',')
                            .enumerate()
                            .map(|(index, field)| {
                                let reference =
                                    BindingRef::source(field, span.start + index, span.end);
                                scope.insert(
                                    field.into(),
                                    Binding {
                                        reference: reference.clone(),
                                        ty: ValueType::Any,
                                        class: None,
                                        enum_variant: None,
                                        function_return: None,
                                        collection_len: None,
                                        mutable: false,
                                        field: false,
                                        integer_max: None,
                                        known_integer: None,
                                        any_origin: Some(AnyOrigin::LostTypeInformation),
                                    },
                                );
                                MatchPattern::Bind(reference)
                            })
                            .collect()
                    };
                    return Ok(MatchPattern::Constructor { name, fields });
                }
            }
            if fields.is_empty()
                && name.as_bytes().first().is_some_and(u8::is_ascii_lowercase)
                && !matches!(name.as_str(), "absent" | "ok" | "failure" | "present")
            {
                let reference = BindingRef::source(&name, span.start, span.end);
                scope.insert(
                    name.clone(),
                    Binding {
                        reference: reference.clone(),
                        ty: ValueType::Any,
                        class: None,
                        enum_variant: None,
                        function_return: None,
                        collection_len: None,
                        mutable: false,
                        field: false,
                        integer_max: None,
                        known_integer: None,
                        any_origin: Some(AnyOrigin::LostTypeInformation),
                    },
                );
                return Ok(MatchPattern::Bind(reference));
            }
            let fields = fields
                .iter()
                .map(|field| lower_pattern(field, scope, aliases))
                .collect::<Result<Vec<_>, _>>()?;
            Ok(MatchPattern::Constructor { name, fields })
        }
        Pattern::Tuple { elements, .. } => {
            let fields = elements
                .iter()
                .map(|field| lower_pattern(field, scope, aliases))
                .collect::<Result<Vec<_>, _>>()?;
            Ok(MatchPattern::Constructor {
                name: "tuple".into(),
                fields,
            })
        }
        _ => Err(error(pattern.span(), "pattern is not supported yet")),
    }
}
