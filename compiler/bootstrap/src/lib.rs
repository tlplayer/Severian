#![forbid(unsafe_code)]

use severian_ast::{
    Expression, ExpressionKind, Literal as AstLiteral, OperatorDeclaration, OperatorSyntax,
    TraitDeclaration, TypeAnnotation, TypeAnnotationKind,
};
use severian_source::{SourceFile, SourceId};
use severian_universal::{
    BinaryOperator, OperatorSignature, PrimitiveCategory, PrimitiveRepresentation,
    TypeContextBuilder, TypeId, TypePattern, UnaryOperator, UniversalContext,
};
use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::fmt;

#[cfg(test)]
const PRIMITIVE_PACKAGE_PATH: &str = "core.primitives";

pub fn load() -> Result<UniversalContext, BootstrapError> {
    build_from_packages(
        severian_compile_protocol::SOURCES
            .iter()
            .map(|source| {
                (
                    severian_compile_protocol::PACKAGE_NAME,
                    source.path,
                    source.source,
                )
            })
            .chain(severian_primitives::SOURCES.iter().map(|source| {
                (
                    severian_primitives::PACKAGE_NAME,
                    source.path,
                    source.source,
                )
            })),
    )
}

#[cfg(test)]
fn build_from_sources<'a>(
    sources: impl IntoIterator<Item = (&'a str, &'a str)>,
) -> Result<UniversalContext, BootstrapError> {
    build_from_packages(
        sources
            .into_iter()
            .map(|(path, source)| (PRIMITIVE_PACKAGE_PATH, path, source)),
    )
}

fn build_from_packages<'a>(
    sources: impl IntoIterator<Item = (&'a str, &'a str, &'a str)>,
) -> Result<UniversalContext, BootstrapError> {
    let mut declarations = BTreeMap::<String, TraitDeclaration>::new();
    let mut declaration_packages = BTreeMap::<String, String>::new();
    for (index, (package, path, text)) in sources.into_iter().enumerate() {
        let source = SourceFile {
            id: SourceId(index as u32),
            path: format!("{package}/{path}").into(),
            text: text.to_owned(),
        };
        let tokens = severian_lexer::scan(&source).map_err(BootstrapError::Parse)?;
        let module = severian_parser::parse(&tokens).map_err(BootstrapError::Parse)?;
        for declaration in module.items.into_iter().filter_map(|item| match item {
            severian_ast::Item::Trait(declaration) => Some(declaration),
            _ => None,
        }) {
            let name = declaration.name.clone();
            if declarations.insert(name.clone(), declaration).is_some() {
                return Err(BootstrapError::DuplicateDeclaration(name));
            }
            declaration_packages.insert(name, package.to_owned());
        }
    }

    // Pass 1: every source declaration receives its stable path identity before
    // bases, metadata, or signatures are interpreted.
    let mut types = TypeContextBuilder::new();
    for name in declarations.keys() {
        let package = declaration_packages
            .get(name)
            .expect("every declaration records its source package");
        types
            .register_generic_declaration(
                format!("{package}.{name}"),
                name,
                declarations[name].type_parameters.len(),
            )
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

    // Every nominal declaration records the protocols it implements. Generic
    // arguments are checked by semantic inference; capability identity is the
    // declaration at the head of the applied trait.
    for declaration in declarations.values() {
        let owner = types
            .resolve_name(&declaration.name)
            .ok_or_else(|| BootstrapError::MissingDeclaration(declaration.name.clone()))?;
        let mut capabilities = BTreeSet::new();
        collect_capabilities(declaration, &declarations, &mut capabilities);
        for capability in capabilities {
            let trait_id = types
                .resolve_name(&capability)
                .ok_or_else(|| BootstrapError::MissingDeclaration(capability.clone()))?;
            types
                .add_capability(owner, trait_id)
                .map_err(|error| BootstrapError::Type(error.to_string()))?;
        }
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

    for declaration in declarations.values() {
        let trait_id = types
            .resolve_name(&declaration.name)
            .ok_or_else(|| BootstrapError::MissingDeclaration(declaration.name.clone()))?;
        let mut operators = Vec::new();
        collect_operators(declaration, &HashMap::new(), &declarations, &mut operators)?;
        for operator in operators {
            if operator.parameters.is_empty() {
                if let Some(operator) = universal_unary(operator.operator) {
                    types.add_trait_unary(trait_id, operator);
                }
            } else if let Some(operator) = universal_binary(operator.operator) {
                types.add_trait_binary(trait_id, operator);
            }
        }
    }

    // Pass 2c: interpret the ordinary core CompileType protocol. Universal
    // receives only stable declaration identities; it knows no source names.
    if declarations.contains_key("CompileType") {
        if !declarations.contains_key("Compiler") {
            return Err(BootstrapError::MissingDeclaration("Compiler".into()));
        }
        for declaration in declarations.values() {
            for base in &declaration.bases {
                let Some((name, arguments)) = base.named_parts() else {
                    continue;
                };
                if name != "CompileType" {
                    continue;
                }
                let [compiler] = arguments else {
                    return Err(BootstrapError::GenericArity("CompileType".into()));
                };
                let compiler_name = compiler
                    .simple_name()
                    .ok_or_else(|| BootstrapError::UnsupportedTypeAnnotation(compiler.clone()))?;
                let compiler_declaration = declarations
                    .get(compiler_name)
                    .ok_or_else(|| BootstrapError::MissingDeclaration(compiler_name.into()))?;
                if !inherits_protocol(
                    compiler_declaration,
                    "Compiler",
                    &declarations,
                    &mut BTreeSet::new(),
                ) {
                    return Err(BootstrapError::InvalidCompiler(compiler_name.into()));
                }
                let type_id = types
                    .resolve_name(&declaration.name)
                    .ok_or_else(|| BootstrapError::MissingDeclaration(declaration.name.clone()))?;
                let compiler_type = types
                    .resolve_name(compiler_name)
                    .ok_or_else(|| BootstrapError::MissingDeclaration(compiler_name.into()))?;
                let compiler_id = types
                    .compiler_id(compiler_type)
                    .map_err(|error| BootstrapError::Type(error.to_string()))?;
                types
                    .set_compile_route(type_id, compiler_id)
                    .map_err(|error| BootstrapError::Type(error.to_string()))?;
            }
        }
    }

    Ok(UniversalContext::new(types.build()))
}

fn collect_capabilities(
    declaration: &TraitDeclaration,
    declarations: &BTreeMap<String, TraitDeclaration>,
    output: &mut BTreeSet<String>,
) {
    for base in &declaration.bases {
        let Some((name, _)) = base.named_parts() else {
            continue;
        };
        if !output.insert(name.to_owned()) {
            continue;
        }
        if let Some(base) = declarations.get(name) {
            collect_capabilities(base, declarations, output);
        }
    }
}

fn inherits_protocol(
    declaration: &TraitDeclaration,
    protocol: &str,
    declarations: &BTreeMap<String, TraitDeclaration>,
    visiting: &mut BTreeSet<String>,
) -> bool {
    if !visiting.insert(declaration.name.clone()) {
        return false;
    }
    let found = declaration.bases.iter().any(|base| {
        let Some((name, _)) = base.named_parts() else {
            return false;
        };
        name == protocol
            || declarations
                .get(name)
                .is_some_and(|base| inherits_protocol(base, protocol, declarations, visiting))
    });
    visiting.remove(&declaration.name);
    found
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
                (parameter.clone(), substitute_type(argument, substitutions))
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
        TypeAnnotationKind::Function { parameters, result } => TypeAnnotation {
            kind: TypeAnnotationKind::Function {
                parameters: parameters
                    .iter()
                    .map(|parameter| substitute_type(parameter, substitutions))
                    .collect(),
                result: Box::new(substitute_type(result, substitutions)),
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
    types: &mut TypeContextBuilder,
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
    types: &TypeContextBuilder,
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
        OperatorSyntax::Pipe => BinaryOperator::BitwiseOr,
        OperatorSyntax::BitwiseAnd => BinaryOperator::BitwiseAnd,
        OperatorSyntax::BitwiseXor => BinaryOperator::BitwiseXor,
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
        OperatorSyntax::Contains => BinaryOperator::Contains,
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
        }) => value
            .parse()
            .map(Some)
            .map_err(|_| BootstrapError::InvalidProperty {
                declaration: declaration.name.clone(),
                property: name.into(),
            }),
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
    InvalidCompiler(String),
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
    use severian_universal::{CompileRoute, IntegerWidth, LiteralKind, TypeConstraint};

    #[test]
    fn loads_sources_through_the_real_parser() {
        let context = load().unwrap();
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
    fn character_literals_are_defined_by_the_core_primitive_contract() {
        let context = load().unwrap();
        let character = context.types.resolve_name("char").unwrap();
        let definition = context.types.primitive(character).unwrap();
        assert_eq!(definition.category, PrimitiveCategory::Character);
        assert_eq!(
            definition.representation,
            PrimitiveRepresentation::Character
        );
        assert_eq!(
            context.types.resolve_literal(
                &severian_universal::LiteralValue::Character('\u{03bb}'),
                None,
            ),
            Ok(character)
        );
    }

    #[test]
    fn inherited_operator_constraints_are_symmetric() {
        let context = load().unwrap();
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
    fn integer_primitives_declare_bitwise_algebra() {
        let context = load().unwrap();
        let integer = context.types.resolve_name("u128").unwrap();
        let floating = context.types.resolve_name("f64").unwrap();
        for operator in [
            BinaryOperator::BitwiseOr,
            BinaryOperator::BitwiseAnd,
            BinaryOperator::BitwiseXor,
        ] {
            assert!(context.types.supports_binary(operator, integer));
            assert!(!context.types.supports_binary(operator, floating));
        }
    }

    #[test]
    fn primitive_ids_do_not_depend_on_source_order() {
        let forward = build_from_sources(
            severian_primitives::SOURCES
                .iter()
                .map(|source| (source.path, source.source)),
        )
        .unwrap();
        let reverse = build_from_sources(
            severian_primitives::SOURCES
                .iter()
                .rev()
                .map(|source| (source.path, source.source)),
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
        const F128: &str = "trait f128: Primitive + Floating[f128]\n    property category: string = \"float\"\n    property representation: string = \"ieee-float\"\n    property bits: int = 128\n";
        let context = build_from_sources(
            severian_primitives::SOURCES
                .iter()
                .map(|source| (source.path, source.source))
                .chain(std::iter::once(("src/f128.sev", F128))),
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

    #[test]
    fn core_compile_protocol_resolves_to_stable_universal_routes() {
        const SPECIAL: &str = "trait TestCompiler: Compiler\n    pass\n\ntrait TestIR[T]: CompileType[TestCompiler]\n    pass\n";
        let context = build_from_packages(
            severian_compile_protocol::SOURCES
                .iter()
                .map(|source| {
                    (
                        severian_compile_protocol::PACKAGE_NAME,
                        source.path,
                        source.source,
                    )
                })
                .chain(severian_primitives::SOURCES.iter().map(|source| {
                    (
                        severian_primitives::PACKAGE_NAME,
                        source.path,
                        source.source,
                    )
                }))
                .chain(std::iter::once(("test.compile", "src/mod.sev", SPECIAL))),
        )
        .unwrap();
        let compiler_type = context.types.resolve_name("TestCompiler").unwrap();
        let compiler = context.types.compiler_id(compiler_type).unwrap();
        let special = context.types.resolve_name("TestIR").unwrap();
        assert_eq!(
            context.types.compile_route(special).unwrap(),
            CompileRoute::Compiler(compiler)
        );
        assert_eq!(
            context.types.definition(compiler_type).unwrap().declaration,
            severian_universal::DeclarationId::from_path("test.compile.TestCompiler")
        );
    }
}
