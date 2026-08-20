use super::*;

const PRIMITIVES_PACKAGE: &str = "core.primitives";

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(in crate::analyzer) enum PrimitiveKind {
    Integer,
    Float,
    Bool,
    String,
    None,
}

impl PrimitiveKind {
    pub(in crate::analyzer) fn value_type(self) -> ValueType {
        match self {
            Self::Integer => ValueType::Int,
            Self::Float => ValueType::Float,
            Self::Bool => ValueType::Bool,
            Self::String => ValueType::String,
            Self::None => ValueType::Unit,
        }
    }

    pub(in crate::analyzer) fn type_kind(self) -> TypeKind {
        match self {
            Self::Integer => TypeKind::Int,
            Self::Float => TypeKind::Float,
            Self::Bool => TypeKind::Bool,
            Self::String => TypeKind::String,
            Self::None => TypeKind::Unit,
        }
    }
}

#[derive(Debug, Clone)]
pub(in crate::analyzer) struct PrimitiveDefinition {
    pub name: String,
    pub kind: PrimitiveKind,
    pub default: bool,
}

#[derive(Debug, Clone, Default)]
pub(in crate::analyzer) struct PrimitiveCatalog {
    definitions: HashMap<String, PrimitiveDefinition>,
}

impl PrimitiveCatalog {
    pub(in crate::analyzer) fn from_interfaces(
        interfaces: &[PackageInterface],
    ) -> Result<Self, SemanticError> {
        let mut catalog = Self::default();

        for interface in interfaces {
            let belongs_to_primitives =
                interface.name == PRIMITIVES_PACKAGE
                    || interface.export_package.as_deref()
                        == Some(PRIMITIVES_PACKAGE);

            if !belongs_to_primitives {
                continue;
            }

            for item in &interface.module.items {
                let Item::Trait(declaration) = item else {
                    continue;
                };

                if !composes_primitive(declaration) {
                    continue;
                }

                let kind = primitive_kind(declaration)?;
                let default = bool_property(declaration, "default")
                    .unwrap_or(false);

                let definition = PrimitiveDefinition {
                    name: declaration.name.name.clone(),
                    kind,
                    default,
                };

                catalog
                    .definitions
                    .insert(definition.name.clone(), definition);
            }
        }

        if catalog.definitions.is_empty() {
            return Err(error(
                Span::dummy(),
                "core primitive definitions were not loaded",
            ));
        }

        Ok(catalog)
    }

    pub(in crate::analyzer) fn contains(&self, name: &str) -> bool {
        self.definitions.contains_key(name)
    }

    pub(in crate::analyzer) fn value_type(&self, name: &str) -> Option<ValueType> {
        self.definitions
            .get(name)
            .map(|primitive| primitive.kind.value_type())
    }

    pub(in crate::analyzer) fn type_kind(&self, name: &str) -> Option<TypeKind> {
        self.definitions
            .get(name)
            .map(|primitive| primitive.kind.type_kind())
    }

    pub(in crate::analyzer) fn kind(&self, name: &str) -> Option<PrimitiveKind> {
        self.definitions.get(name).map(|primitive| primitive.kind)
    }
}

fn composes_primitive(
    declaration: &severian_ast::TraitDecl,
) -> bool {
    declaration.composed_traits.iter().any(|ty| {
        matches!(
            ty,
            Type::Named(path)
                if path.segments
                    .last()
                    .is_some_and(|segment| segment.name == "Primitive")
        )
    })
}

fn primitive_kind(
    declaration: &severian_ast::TraitDecl,
) -> Result<PrimitiveKind, SemanticError> {
    let kind = string_property(declaration, "kind").ok_or_else(|| {
        error(
            declaration.name.span,
            format!(
                "primitive `{}` must declare `kind`",
                declaration.name.name
            ),
        )
    })?;

    match kind {
        "integer" => Ok(PrimitiveKind::Integer),
        "float" => Ok(PrimitiveKind::Float),
        "bool" => Ok(PrimitiveKind::Bool),
        "string" => Ok(PrimitiveKind::String),
        "none" => Ok(PrimitiveKind::None),
        other => Err(error(
            declaration.name.span,
            format!("unknown primitive kind `{other}`"),
        )),
    }
}

fn string_property<'a>(
    declaration: &'a severian_ast::TraitDecl,
    name: &str,
) -> Option<&'a str> {
    let property = declaration
        .properties
        .iter()
        .find(|property| property.name.name == name)?;

    match property.default.as_ref()? {
        Expr::Literal(Literal::String { value, .. }) => Some(value),
        _ => None,
    }
}

fn bool_property(
    declaration: &severian_ast::TraitDecl,
    name: &str,
) -> Option<bool> {
    let property = declaration
        .properties
        .iter()
        .find(|property| property.name.name == name)?;

    match property.default.as_ref()? {
        Expr::Literal(Literal::Boolean { value, .. }) => Some(*value),
        _ => None,
    }
}