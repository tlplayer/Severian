use crate::analyzer::*;

const PRIMITIVES_PACKAGE: &str = "core.primitives";

#[derive(Debug, Clone)]
pub(in crate::analyzer) struct PrimitiveCatalog {
    by_name: HashMap<String, PrimitiveDefinition>,
    by_id: HashMap<PrimitiveId, PrimitiveDefinition>,
    defaults: HashMap<PrimitiveCategory, PrimitiveId>,
}

impl PrimitiveCatalog {
    pub(in crate::analyzer) fn from_interfaces(
        interfaces: &[PackageInterface],
    ) -> Result<Self, SemanticError> {
        let mut catalog = Self {
            by_name: HashMap::new(),
            by_id: HashMap::new(),
            defaults: HashMap::new(),
        };

        for interface in interfaces.iter().filter(|interface| {
            interface.name == PRIMITIVES_PACKAGE
                || interface.export_package.as_deref() == Some(PRIMITIVES_PACKAGE)
        }) {
            for declaration in interface.module.items.iter().filter_map(|item| match item {
                Item::Trait(declaration) if composes_primitive(declaration) => Some(declaration),
                _ => None,
            }) {
                let definition = read_definition(declaration)?;
                if catalog.by_name.contains_key(&definition.name) {
                    return Err(error(
                        declaration.name.span,
                        format!("duplicate core primitive declaration `{}`", definition.name),
                    ));
                }
                if definition.default_literal {
                    if catalog
                        .defaults
                        .insert(definition.category, definition.id)
                        .is_some()
                    {
                        return Err(error(
                            declaration.name.span,
                            format!(
                                "primitive category `{:?}` has more than one default literal type",
                                definition.category
                            ),
                        ));
                    }
                }
                catalog.by_id.insert(definition.id, definition.clone());
                catalog.by_name.insert(definition.name.clone(), definition);
            }
        }

        if catalog.by_name.is_empty() {
            return Err(error(
                Span::dummy(),
                "compiler bootstrap failed: `core.primitives` was not loaded",
            ));
        }

        Ok(catalog)
    }

    pub(in crate::analyzer) fn contains(&self, name: &str) -> bool {
        self.by_name.contains_key(name)
    }

    pub(in crate::analyzer) fn resolve(&self, name: &str) -> Option<PrimitiveId> {
        self.by_name.get(name).map(|definition| definition.id)
    }

    pub(in crate::analyzer) fn definition(
        &self,
        id: PrimitiveId,
    ) -> Option<&PrimitiveDefinition> {
        self.by_id.get(&id)
    }

    pub(in crate::analyzer) fn default_for(
        &self,
        category: PrimitiveCategory,
    ) -> Option<PrimitiveId> {
        self.defaults.get(&category).copied()
    }

    pub(in crate::analyzer) fn type_kind(&self, name: &str) -> Option<TypeKind> {
        self.resolve(name).map(TypeKind::Primitive)
    }

    /// Compatibility adapter for execution-oriented HIR while it is migrated
    /// from `ValueType` to `TypeId`. Exact identity remains in TypeKind.
    pub(in crate::analyzer) fn value_type(&self, name: &str) -> Option<ValueType> {
        let definition = self.by_name.get(name)?;
        Some(match definition.category {
            PrimitiveCategory::Boolean => ValueType::Bool,
            PrimitiveCategory::Integer => ValueType::Int,
            PrimitiveCategory::Float => ValueType::Float,
            PrimitiveCategory::Text => ValueType::String,
            PrimitiveCategory::Bytes => ValueType::Any,
            PrimitiveCategory::Absence | PrimitiveCategory::Unit => ValueType::Unit,
        })
    }

    pub(in crate::analyzer) fn install_in(&self, metadata: &mut ProgramMetadata) {
        metadata
            .primitives
            .extend(self.by_id.iter().map(|(id, definition)| (*id, definition.clone())));
    }
}

fn read_definition(
    declaration: &severian_ast::TraitDecl,
) -> Result<PrimitiveDefinition, SemanticError> {
    let category = match required_string(declaration, "category")? {
        "boolean" => PrimitiveCategory::Boolean,
        "integer" => PrimitiveCategory::Integer,
        "float" => PrimitiveCategory::Float,
        "text" => PrimitiveCategory::Text,
        "bytes" => PrimitiveCategory::Bytes,
        "absence" => PrimitiveCategory::Absence,
        "unit" => PrimitiveCategory::Unit,
        category => {
            return Err(error(
                declaration.name.span,
                format!("unknown primitive category `{category}`"),
            ))
        }
    };
    let representation = required_string(declaration, "representation")?.to_owned();
    let bits = integer_property(declaration, "bits").unwrap_or(0);
    let bit_width = if bits == 0 {
        None
    } else {
        Some(u16::try_from(bits).map_err(|_| {
            error(declaration.name.span, "primitive bit width is out of range")
        })?)
    };
    let name = declaration.name.name.clone();
    let id = PrimitiveId(TypeDefinitionId::from_name(&format!(
        "{PRIMITIVES_PACKAGE}.{name}"
    )));

    Ok(PrimitiveDefinition {
        id,
        name,
        category,
        representation,
        bit_width,
        signed: (category == PrimitiveCategory::Integer)
            .then(|| bool_property(declaration, "signed").unwrap_or(false)),
        default_literal: bool_property(declaration, "default_literal").unwrap_or(false),
    })
}

fn composes_primitive(declaration: &severian_ast::TraitDecl) -> bool {
    declaration.composed_traits.iter().any(|ty| {
        matches!(
            ty,
            Type::Named(path)
                if path.segments.last().is_some_and(|segment| segment.name == "Primitive")
        )
    })
}

fn required_string<'a>(
    declaration: &'a severian_ast::TraitDecl,
    property: &str,
) -> Result<&'a str, SemanticError> {
    string_property(declaration, property).ok_or_else(|| {
        error(
            declaration.name.span,
            format!(
                "primitive `{}` must declare `{property}`",
                declaration.name.name
            ),
        )
    })
}

fn string_property<'a>(
    declaration: &'a severian_ast::TraitDecl,
    name: &str,
) -> Option<&'a str> {
    match declaration
        .properties
        .iter()
        .find(|property| property.name.name == name)?
        .default
        .as_ref()?
    {
        Expr::Literal(Literal::String { value, .. }) => Some(value),
        _ => None,
    }
}

fn integer_property(declaration: &severian_ast::TraitDecl, name: &str) -> Option<i64> {
    match declaration
        .properties
        .iter()
        .find(|property| property.name.name == name)?
        .default
        .as_ref()?
    {
        Expr::Literal(Literal::Integer { value, .. }) => Some(*value),
        _ => None,
    }
}

fn bool_property(declaration: &severian_ast::TraitDecl, name: &str) -> Option<bool> {
    match declaration
        .properties
        .iter()
        .find(|property| property.name.name == name)?
        .default
        .as_ref()?
    {
        Expr::Literal(Literal::Boolean { value, .. }) => Some(*value),
        _ => None,
    }
}
