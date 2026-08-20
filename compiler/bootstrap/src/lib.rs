#![forbid(unsafe_code)]

use severian_ast::{
    Expression, ExpressionKind, Literal as AstLiteral, OperatorDeclaration, OperatorSyntax,
    TraitDeclaration, TypeAnnotation, TypeAnnotationKind,
};
use severian_source::{SourceFile, SourceId};
use severian_universal::{
    BinaryOperator, OperatorSignature, PrimitiveCategory, PrimitiveRepresentation, TargetSpec,
    TypeContext, TypeId, TypePattern, UnaryOperator, UniversalContext,
};
use std::collections::{BTreeMap, HashMap};
use std::fmt;

const PACKAGE_PATH: &str = "core.primitives";

pub fn load(target: TargetSpec) -> Result<UniversalContext, BootstrapError> {
    build_from_sources(
        severian_primitives::SOURCES
            .iter()
            .map(|source| (source.path, source.source)),
        target,
    )
}

fn build_from_sources<'a>(
    sources: impl IntoIterator<Item = (&'a str, &'a str)>,
    target: TargetSpec,
) -> Result<UniversalContext, BootstrapError> {
    let mut declarations = BTreeMap::<String, TraitDeclaration>::new();
    for (index, (path, text)) in sources.into_iter().enumerate() {
        let source = SourceFile {
            id: SourceId(index as u32),
            path: path.into(),
            text: text.to_owned(),
        };
        let tokens = severian_lexer::scan(&source).map_err(BootstrapError::Parse)?;
        let module = severian_parser::parse(&tokens).map_err(BootstrapError::Parse)?;
        for declaration in module.traits {
            let name = declaration.name.clone();
            if declarations.insert(name.clone(), declaration).is_some() {
                return Err(BootstrapError::DuplicateDeclaration(name));
            }
        }
    }

    // Pass 1: every source declaration receives its stable path identity before
    // bases, metadata, or signatures are interpreted.
    let mut types = TypeContext::new();
    for name in declarations.keys() {
        types
            .register_declaration(format!("{PACKAGE_PATH}.{name}"), name)
            .map_err(|error| BootstrapError::Type(error.to_string()))?;
    }

    // Pass 2a: resolve the Primitive base and its typed metadata.
    let protocol = declarations
        .get("Primitive")
        .ok_or_else(|| BootstrapError::MissingDeclaration("Primitive".into()))?;
    for declaration in declarations.values().filter(|declaration| {
        declaration
            .bases
            .iter()
            .any(|base| base.simple_name() == Some("Primitive"))
    }) {
        let category = string_property(declaration, protocol, "category")?;
        let representation = string_property(declaration, protocol, "representation")?;
        let bits = integer_property(declaration, protocol, "bits")?
            .filter(|value| *value != 0)
            .map(|value| {
                u16::try_from(value).map_err(|_| BootstrapError::InvalidProperty {
                    declaration: declaration.name.clone(),
                    property: "bits".into(),
                })
            })
            .transpose()?;
        let signed = boolean_property(declaration, protocol, "signed")?.unwrap_or(false);
        let default_literal =
            boolean_property(declaration, protocol, "default_literal")?.unwrap_or(false);
        let category = PrimitiveCategory::from_contract(&category)
            .map_err(|error| BootstrapError::Type(error.to_string()))?;
        let representation = PrimitiveRepresentation::from_contract(&representation, bits, signed)
            .map_err(|error| BootstrapError::Type(error.to_string()))?;
        let type_id = types
            .resolve_name(&declaration.name)
            .ok_or_else(|| BootstrapError::MissingDeclaration(declaration.name.clone()))?;
        types
            .define_primitive(type_id, category, representation, default_literal)
            .map_err(|error| BootstrapError::Type(error.to_string()))?;
    }

    // Pass 2b: expand typed operators after all result/operand names exist.
    for declaration in declarations.values().filter(|declaration| {
        declaration
            .bases
            .iter()
            .any(|base| base.simple_name() == Some("Primitive"))
    }) {
        let owner = types
            .resolve_name(&declaration.name)
            .ok_or_else(|| BootstrapError::MissingDeclaration(declaration.name.clone()))?;
        let mut operators = Vec::new();
        collect_operators(declaration, &HashMap::new(), &declarations, &mut operators)?;
        for operator in operators {
            add_operator(&mut types, owner, operator)?;
        }
    }

    Ok(UniversalContext::new(types, target))
}

#[derive(Debug, Clone)]
struct ResolvedOperator {
    operator: OperatorSyntax,
    parameters: Vec<TypeAnnotation>,
    result: TypeAnnotation,
}

fn collect_operators(
    declaration: &TraitDeclaration,
    substitutions: &HashMap<String, TypeAnnotation>,
    declarations: &BTreeMap<String, TraitDeclaration>,
    output: &mut Vec<ResolvedOperator>,
) -> Result<(), BootstrapError> {
    for operator in &declaration.operators {
        output.push(resolve_operator(operator, substitutions));
    }
    for base in declaration
        .bases
        .iter()
        .filter(|base| base.simple_name() != Some("Primitive"))
    {
        let Some((base_name, arguments)) = base.named_parts() else {
            return Err(BootstrapError::UnsupportedTypeAnnotation(base.clone()));
        };
        let inherited = declarations
            .get(base_name)
            .ok_or_else(|| BootstrapError::MissingDeclaration(base_name.to_owned()))?;
        if inherited.type_parameters.len() != arguments.len() {
            return Err(BootstrapError::GenericArity(base_name.to_owned()));
        }
        let inherited_substitutions = inherited
            .type_parameters
            .iter()
            .zip(arguments)
            .map(|(parameter, argument)| {
                (
                    parameter.clone(),
                    substitute_type(argument, substitutions),
                )
            })
            .collect();
        collect_operators(inherited, &inherited_substitutions, declarations, output)?;
    }
    Ok(())
}

fn resolve_operator(
    operator: &OperatorDeclaration,
    substitutions: &HashMap<String, TypeAnnotation>,
) -> ResolvedOperator {
    ResolvedOperator {
        operator: operator.operator,
        parameters: operator
            .parameters
            .iter()
            .map(|parameter| substitute_type(&parameter.annotation, substitutions))
            .collect(),
        result: substitute_type(&operator.result, substitutions),
    }
}

fn substitute_type(
    annotation: &TypeAnnotation,
    substitutions: &HashMap<String, TypeAnnotation>,
) -> TypeAnnotation {
    match &annotation.kind {
        TypeAnnotationKind::Named { name, arguments } if arguments.is_empty() => substitutions
            .get(name)
            .cloned()
            .unwrap_or_else(|| annotation.clone()),
        TypeAnnotationKind::Named { name, arguments } => TypeAnnotation {
            kind: TypeAnnotationKind::Named {
                name: name.clone(),
                arguments: arguments
                    .iter()
                    .map(|argument| substitute_type(argument, substitutions))
                    .collect(),
            },
            span: annotation.span,
        },
        TypeAnnotationKind::Union(members) => TypeAnnotation {
            kind: TypeAnnotationKind::Union(
                members
                    .iter()
                    .map(|member| substitute_type(member, substitutions))
                    .collect(),
            ),
            span: annotation.span,
        },
    }
}

fn add_operator(
    types: &mut TypeContext,
    owner: TypeId,
    operator: ResolvedOperator,
) -> Result<(), BootstrapError> {
    let result = resolve_source_type(types, &operator.result)?;
    match operator.parameters.as_slice() {
        [] => {
            let unary = universal_unary(operator.operator)
                .ok_or(BootstrapError::UnknownOperator(operator.operator))?;
            types.add_unary(unary, owner, result);
        }
        [right] => {
            let binary = universal_binary(operator.operator)
                .ok_or(BootstrapError::UnknownOperator(operator.operator))?;
            let right = resolve_source_type(types, right)?;
            types.add_binary(OperatorSignature {
                operator: binary,
                left: TypePattern::Exact(owner),
                right: TypePattern::Exact(right),
                result: TypePattern::Exact(result),
            });
        }
        _ => return Err(BootstrapError::OperatorArity(operator.operator)),
    }
    Ok(())
}

fn resolve_source_type(
    types: &TypeContext,
    annotation: &TypeAnnotation,
) -> Result<TypeId, BootstrapError> {
    let name = annotation
        .simple_name()
        .ok_or_else(|| BootstrapError::UnsupportedTypeAnnotation(annotation.clone()))?;
    types
        .resolve_name(name)
        .ok_or_else(|| BootstrapError::MissingDeclaration(name.to_owned()))
}

fn universal_unary(operator: OperatorSyntax) -> Option<UnaryOperator> {
    Some(match operator {
        OperatorSyntax::Plus => UnaryOperator::Positive,
        OperatorSyntax::Minus => UnaryOperator::Negative,
        OperatorSyntax::Not => UnaryOperator::Not,
        _ => return None,
    })
}

fn universal_binary(operator: OperatorSyntax) -> Option<BinaryOperator> {
    Some(match operator {
        OperatorSyntax::Plus => BinaryOperator::Add,
        OperatorSyntax::Minus => BinaryOperator::Subtract,
        OperatorSyntax::Multiply => BinaryOperator::Multiply,
        OperatorSyntax::Divide => BinaryOperator::Divide,
        OperatorSyntax::Remainder => BinaryOperator::Remainder,
        OperatorSyntax::Power => BinaryOperator::Power,
        OperatorSyntax::Equal => BinaryOperator::Equal,
        OperatorSyntax::NotEqual => BinaryOperator::NotEqual,
        OperatorSyntax::Less => BinaryOperator::Less,
        OperatorSyntax::LessEqual => BinaryOperator::LessEqual,
        OperatorSyntax::Greater => BinaryOperator::Greater,
        OperatorSyntax::GreaterEqual => BinaryOperator::GreaterEqual,
        OperatorSyntax::And => BinaryOperator::And,
        OperatorSyntax::Or => BinaryOperator::Or,
        OperatorSyntax::Not => return None,
    })
}

fn property<'a>(
    declaration: &'a TraitDeclaration,
    protocol: &'a TraitDeclaration,
    name: &str,
) -> Option<&'a Expression> {
    declaration
        .properties
        .iter()
        .find(|property| property.name == name)
        .and_then(|property| property.default.as_ref())
        .or_else(|| {
            protocol
                .properties
                .iter()
                .find(|property| property.name == name)
                .and_then(|property| property.default.as_ref())
        })
}

fn string_property(
    declaration: &TraitDeclaration,
    protocol: &TraitDeclaration,
    name: &str,
) -> Result<String, BootstrapError> {
    match property(declaration, protocol, name) {
        Some(Expression {
            kind: ExpressionKind::Literal(AstLiteral::String(value)),
            ..
        }) => Ok(value.clone()),
        _ => Err(BootstrapError::InvalidProperty {
            declaration: declaration.name.clone(),
            property: name.into(),
        }),
    }
}

fn integer_property(
    declaration: &TraitDeclaration,
    protocol: &TraitDeclaration,
    name: &str,
) -> Result<Option<u64>, BootstrapError> {
    match property(declaration, protocol, name) {
        Some(Expression {
            kind: ExpressionKind::Literal(AstLiteral::Integer(value)),
            ..
        }) => {
            value
                .parse()
                .map(Some)
                .map_err(|_| BootstrapError::InvalidProperty {
                    declaration: declaration.name.clone(),
                    property: name.into(),
                })
        }
        None => Ok(None),
        _ => Err(BootstrapError::InvalidProperty {
            declaration: declaration.name.clone(),
            property: name.into(),
        }),
    }
}

fn boolean_property(
    declaration: &TraitDeclaration,
    protocol: &TraitDeclaration,
    name: &str,
) -> Result<Option<bool>, BootstrapError> {
    match property(declaration, protocol, name) {
        Some(Expression {
            kind: ExpressionKind::Literal(AstLiteral::Boolean(value)),
            ..
        }) => Ok(Some(*value)),
        None => Ok(None),
        _ => Err(BootstrapError::InvalidProperty {
            declaration: declaration.name.clone(),
            property: name.into(),
        }),
    }
}

#[derive(Debug)]
pub enum BootstrapError {
    Parse(severian_diagnostics::Diagnostic),
    DuplicateDeclaration(String),
    MissingDeclaration(String),
    InvalidProperty {
        declaration: String,
        property: String,
    },
    GenericArity(String),
    OperatorArity(OperatorSyntax),
    UnknownOperator(OperatorSyntax),
    UnsupportedTypeAnnotation(TypeAnnotation),
    Type(String),
}

impl fmt::Display for BootstrapError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{self:?}")
    }
}

impl std::error::Error for BootstrapError {}

#[cfg(test)]
mod tests {
    use super::*;
    use severian_universal::{IntegerWidth, LiteralKind, TypeConstraint};

    #[test]
    fn loads_sources_through_the_real_parser() {
        let context = load(TargetSpec::host()).unwrap();
        let i32 = context.types.resolve_name("i32").unwrap();
        let definition = context.types.primitive(i32).unwrap();
        assert_eq!(
            definition.representation,
            PrimitiveRepresentation::Integer {
                bits: IntegerWidth::Fixed(32),
                signed: true,
            }
        );
    }

    #[test]
    fn inherited_operator_constraints_are_symmetric() {
        let context = load(TargetSpec::host()).unwrap();
        let i32 = context.types.resolve_name("i32").unwrap();
        for operands in [
            (
                TypeConstraint::Known(i32),
                TypeConstraint::Literal(LiteralKind::Integer),
            ),
            (
                TypeConstraint::Literal(LiteralKind::Integer),
                TypeConstraint::Known(i32),
            ),
        ] {
            assert_eq!(
                context
                    .types
                    .resolve_binary(BinaryOperator::Add, operands.0, operands.1, None)
                    .unwrap()
                    .result,
                i32
            );
        }
    }

    #[test]
    fn primitive_ids_do_not_depend_on_source_order() {
        let forward = build_from_sources(
            severian_primitives::SOURCES
                .iter()
                .map(|source| (source.path, source.source)),
            TargetSpec::host(),
        )
        .unwrap();
        let reverse = build_from_sources(
            severian_primitives::SOURCES
                .iter()
                .rev()
                .map(|source| (source.path, source.source)),
            TargetSpec::host(),
        )
        .unwrap();
        let id = |context: &UniversalContext| {
            context
                .types
                .primitive(context.types.resolve_name("i32").unwrap())
                .unwrap()
                .id
        };
        assert_eq!(id(&forward), id(&reverse));
    }

    #[test]
    fn a_new_float_family_member_needs_no_semantic_branch() {
        const F128: &str = "trait f128: Primitive + Floating[f128]:\n    property category: string = \"float\"\n    property representation: string = \"ieee-float\"\n    property bits: int = 128\n";
        let context = build_from_sources(
            severian_primitives::SOURCES
                .iter()
                .map(|source| (source.path, source.source))
                .chain(std::iter::once(("src/f128.sev", F128))),
            TargetSpec::host(),
        )
        .unwrap();
        let f128 = context.types.resolve_name("f128").unwrap();
        assert_eq!(
            context.types.primitive(f128).unwrap().representation,
            PrimitiveRepresentation::Float {
                format: severian_universal::FloatFormat::Ieee(128)
            }
        );
        assert_eq!(
            context
                .types
                .resolve_binary(
                    BinaryOperator::Add,
                    TypeConstraint::Known(f128),
                    TypeConstraint::Literal(LiteralKind::Float),
                    None,
                )
                .unwrap()
                .result,
            f128
        );
    }
}
