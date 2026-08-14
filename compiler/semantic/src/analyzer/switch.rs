use super::*;

pub(super) fn validate_exhaustive_enum_switch(
    statement: &severian_ast::SwitchStmt,
    scope: &HashMap<String, Binding>,
    aliases: &HashMap<String, String>,
) -> Result<(), SemanticError> {
    if statement.values.len() != 1
        || statement.repeat_condition.is_some()
        || statement.setup.is_some()
        || statement.arms.iter().any(|arm| arm.source.is_some())
    {
        return Ok(());
    }
    let Expr::Identifier(value) = &statement.values[0] else {
        return Ok(());
    };
    let Some(enum_name) = scope
        .get(&value.name)
        .and_then(|binding| binding.class.as_deref())
    else {
        return Ok(());
    };
    let Some(variants) = aliases.get(&format!("__enum_variants.{enum_name}")) else {
        return Ok(());
    };
    let variants = variants.split(',').collect::<Vec<_>>();
    let mut covered = HashSet::new();
    for arm in statement.arms.iter().filter(|arm| arm.guard.is_none()) {
        match &arm.pattern {
            Pattern::Wildcard(_) => return Ok(()),
            Pattern::Identifier(identifier) if variants.contains(&identifier.name.as_str()) => {
                covered.insert(identifier.name.as_str());
            }
            Pattern::Identifier(identifier)
                if identifier
                    .name
                    .as_bytes()
                    .first()
                    .is_some_and(u8::is_ascii_lowercase) =>
            {
                return Ok(());
            }
            Pattern::Constructor { name, .. } => {
                if let Type::Named(path) = name {
                    if let Some(name) = path.segments.first().map(|segment| segment.name.as_str()) {
                        if variants.contains(&name) {
                            covered.insert(name);
                        }
                    }
                }
            }
            _ => {}
        }
    }
    let missing = variants
        .into_iter()
        .filter(|variant| !covered.contains(variant))
        .collect::<Vec<_>>();
    if missing.is_empty() {
        return Ok(());
    }
    Err(error(
        statement.values[0].span(),
        format!(
            "E000206: non-exhaustive switch on `{enum_name}`; missing {}",
            missing
                .iter()
                .map(|variant| format!("`{enum_name}.{variant}`"))
                .collect::<Vec<_>>()
                .join(", ")
        ),
    ))
}
