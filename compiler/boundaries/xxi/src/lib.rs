#![forbid(unsafe_code)]

use severian_abi::{AbiTarget, AbiType, CallingConvention, ScalarType, Symbol};
use severian_ast::{
    Decorator as Attribute, DecoratorValue as AttributeValue,
    FunctionDeclaration as ExternalFunctionDeclaration, ImportSubject, Item, Module,
    TypeAnnotation, TypeAnnotationKind, TypeDeclaration as ExternalTypeDeclaration,
};
use severian_ffi::{
    lower_function, AbiSelection, BoundaryPlan, ForeignFunction, ForeignModule, ForeignParameter,
    ForeignTypeDeclaration, ForeignTypeRef, Lifetime, Ownership, ParameterMode, ValueContract,
};
use severian_universal::TypeContext;
use std::{collections::BTreeSet, fmt};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ExternalLanguage {
    C,
    Rust,
    System,
    Custom(String),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExternalImport {
    pub language: ExternalLanguage,
    pub provider: String,
    pub alias: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResolvedExternalModule {
    pub imports: Vec<ExternalImport>,
    pub foreign: ForeignModule,
    pub plans: Vec<BoundaryPlan>,
    pub declarations: Vec<ResolvedExternalFunction>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResolvedExternalFunction {
    pub span_start: u32,
    pub span_end: u32,
    pub function: ForeignFunction,
}

/// Resolves source-facing XXI declarations into validated FFI declarations and
/// fully classified ABI plans. XXI never performs ABI layout itself.
pub fn resolve(
    module: &Module,
    types: &TypeContext,
    target: &AbiTarget,
) -> Result<ResolvedExternalModule, XxiError> {
    for item in &module.items {
        let decorators = match item {
            Item::Function(declaration) => &declaration.decorators,
            Item::Type(declaration) => &declaration.decorators,
            _ => continue,
        };
        if decorators.iter().any(Attribute::is_compile_policy)
            && decorators.iter().any(is_external_attribute)
        {
            return Err(XxiError::MixedCompilerAndExternalAttributes);
        }
    }
    let mut imports = Vec::new();
    for import in module.items.iter().filter_map(|item| match item {
        Item::Import(import) => Some(import),
        _ => None,
    }) {
        let (ImportSubject::Locator(path), None, Some(alias)) =
            (&import.subject, &import.source, &import.alias)
        else {
            continue;
        };
        if let Some(import) = external_import(path, alias) {
            imports.push(import?);
        }
    }
    let mut foreign = ForeignModule::default();
    let mut declarations = Vec::new();
    let hook_decorators = semantic_hook_decorators(module);
    for declaration in module.items.iter().filter_map(|item| match item {
        Item::Type(declaration)
            if declaration
                .decorators
                .iter()
                .any(|decorator| !decorator.is_compile_policy()) =>
        {
            Some(declaration)
        }
        _ => None,
    }) {
        if foreign
            .types
            .iter()
            .any(|known| known.name == declaration.name)
        {
            return Err(XxiError::DuplicateDeclaration(declaration.name.clone()));
        }
        foreign.types.push(resolve_external_type(declaration)?);
    }
    for declaration in module.items.iter().filter_map(|item| match item {
        Item::Function(declaration)
            if declaration
                .decorators
                .iter()
                .any(|decorator| !decorator.is_compile_policy())
                && !semantic_operator_declaration(declaration)
                && !declaration
                    .decorators
                    .iter()
                    .any(|decorator| hook_decorators.contains(decorator.name.as_str())) =>
        {
            Some(declaration)
        }
        _ => None,
    }) {
        let resolved = resolve_function(declaration, &foreign, &imports, types)?;
        if foreign.functions.iter().any(|known| {
            known.name == resolved.name
                && same_parameter_contract(&known.parameters, &resolved.parameters)
        }) {
            return Err(XxiError::DuplicateDeclaration(declaration.name.clone()));
        }
        declarations.push(ResolvedExternalFunction {
            span_start: declaration.span.start,
            span_end: declaration.span.end,
            function: resolved.clone(),
        });
        foreign.functions.push(resolved);
    }
    let plans = foreign
        .functions
        .iter()
        .map(|function| {
            lower_function(function, &foreign, types, target)
                .map_err(|error| XxiError::Ffi(error.to_string()))
        })
        .collect::<Result<Vec<_>, _>>()?;
    Ok(ResolvedExternalModule {
        imports,
        foreign,
        plans,
        declarations,
    })
}

fn semantic_hook_decorators(module: &Module) -> BTreeSet<&str> {
    let mut hooks = BTreeSet::new();
    for declaration in module.items.iter().filter_map(|item| match item {
        Item::Trait(declaration) => Some(declaration),
        _ => None,
    }) {
        hooks.extend(
            declaration
                .namespaces
                .iter()
                .map(|decorator| decorator.name.as_str()),
        );
        hooks.extend(
            declaration
                .methods
                .iter()
                .filter(|method| method.hook.is_some())
                .flat_map(|method| &method.decorators)
                .filter(|decorator| decorator.arguments.is_empty())
                .map(|decorator| decorator.name.as_str()),
        );
    }
    hooks
}

fn semantic_operator_declaration(declaration: &ExternalFunctionDeclaration) -> bool {
    declaration.decorators.iter().any(|decorator| {
        decorator.arguments.iter().any(|argument| {
            argument.name.is_none()
                && matches!(&argument.value, AttributeValue::Name(value)
                    if matches!(value.as_str(),
                        "|" | "+" | "-" | "*" | "/" | "%" | "**" | "==" | "!=" | "<"
                            | "<=" | ">" | ">=" | "in" | "and" | "or")
                        || value.chars().next().is_some_and(|character| character.is_ascii_uppercase()))
        })
    })
}

fn is_external_attribute(attribute: &Attribute) -> bool {
    matches!(attribute.name.as_str(), "c" | "rust" | "system")
        || attribute.arguments.iter().any(|argument| {
            matches!(
                argument.name.as_deref(),
                Some("abi" | "library" | "provider" | "repr" | "symbol" | "variadic")
            )
        })
}

fn same_parameter_contract(left: &[ForeignParameter], right: &[ForeignParameter]) -> bool {
    left.len() == right.len()
        && left
            .iter()
            .zip(right)
            .all(|(left, right)| left.contract == right.contract && left.mode == right.mode)
}

fn external_import(path: &str, alias: &str) -> Option<Result<ExternalImport, XxiError>> {
    let (language, provider) = path.split_once(':')?;
    Some(language_from_name(language).map(|language| ExternalImport {
        language,
        provider: provider.to_owned(),
        alias: alias.to_owned(),
    }))
}

fn resolve_external_type(
    declaration: &ExternalTypeDeclaration,
) -> Result<ForeignTypeDeclaration, XxiError> {
    if !declaration.type_parameters.is_empty() {
        return Err(XxiError::GenericExternalType(declaration.name.clone()));
    }
    let (_, language_attribute) = language_attribute(&declaration.decorators)?;
    let representation = match named_argument(language_attribute, "repr") {
        None | Some("opaque") => AbiType::Opaque {
            name: declaration.name.clone(),
        },
        Some("pointer") => AbiType::pointer_to(
            AbiType::Opaque {
                name: declaration.name.clone(),
            },
            true,
        ),
        Some("bool") => AbiType::Scalar(ScalarType::Boolean),
        Some(value) => parse_scalar_representation(value)
            .ok_or_else(|| XxiError::UnknownRepresentation(value.to_owned()))?,
    };
    Ok(ForeignTypeDeclaration {
        name: declaration.name.clone(),
        representation,
    })
}

fn resolve_function(
    declaration: &ExternalFunctionDeclaration,
    foreign: &ForeignModule,
    imports: &[ExternalImport],
    types: &TypeContext,
) -> Result<ForeignFunction, XxiError> {
    let (language, attribute) = language_attribute(&declaration.decorators)?;
    let symbol = named_argument(attribute, "symbol").unwrap_or(&declaration.name);
    let abi = named_argument(attribute, "abi")
        .map(parse_abi)
        .transpose()?
        .unwrap_or_else(|| language_abi(&language));
    let variadic = boolean_argument(attribute, "variadic")?.unwrap_or(false);
    let provider = named_argument(attribute, "provider")
        .or_else(|| named_argument(attribute, "library"))
        .map(str::to_owned)
        .or_else(|| {
            let mut matching = imports.iter().filter(|import| import.language == language);
            let provider = matching.next().map(|import| import.provider.clone());
            (matching.next().is_none()).then_some(provider).flatten()
        });
    let parameters = declaration
        .parameters
        .iter()
        .map(|parameter| {
            let (contract, mode) = resolve_contract(&parameter.annotation, foreign, types, false)?;
            Ok(ForeignParameter {
                name: parameter.name.clone(),
                contract,
                mode,
            })
        })
        .collect::<Result<Vec<_>, XxiError>>()?;
    let (result, result_mode) = resolve_contract(&declaration.result, foreign, types, true)?;
    if result_mode != ParameterMode::In {
        return Err(XxiError::InvalidResultMode(declaration.name.clone()));
    }
    Ok(ForeignFunction {
        name: declaration.name.clone(),
        provider,
        symbol: Symbol::imported_function(symbol)
            .map_err(|error| XxiError::Symbol(error.to_string()))?,
        parameters,
        result,
        abi,
        variadic,
    })
}

fn resolve_contract(
    annotation: &TypeAnnotation,
    foreign: &ForeignModule,
    types: &TypeContext,
    result: bool,
) -> Result<(ValueContract, ParameterMode), XxiError> {
    let mut ownership = Ownership::Copy;
    let mut nullable = false;
    let mut mode = ParameterMode::In;
    let mut current = annotation;
    while let Some((name, arguments)) = current.named_parts() {
        let wrapper = match (name, arguments) {
            ("borrowed", [inner]) => Some((
                Ownership::Borrowed(if result {
                    Lifetime::Return
                } else {
                    Lifetime::Call
                }),
                ParameterMode::In,
                false,
                inner,
            )),
            ("owned", [inner]) => Some((Ownership::Owned, ParameterMode::In, false, inner)),
            ("transferred", [inner]) => {
                Some((Ownership::Transferred, ParameterMode::In, false, inner))
            }
            ("out", [inner]) => Some((Ownership::Owned, ParameterMode::Out, false, inner)),
            ("inout", [inner]) => Some((Ownership::Owned, ParameterMode::InOut, false, inner)),
            ("nullable", [inner]) => Some((ownership.clone(), mode, true, inner)),
            _ => None,
        };
        let Some((new_ownership, new_mode, new_nullable, inner)) = wrapper else {
            break;
        };
        ownership = new_ownership;
        mode = new_mode;
        nullable |= new_nullable;
        current = inner;
    }
    let ty = resolve_type_ref(current, foreign, types)?;
    Ok((
        ValueContract {
            ty,
            ownership,
            nullable,
        },
        mode,
    ))
}

fn resolve_type_ref(
    annotation: &TypeAnnotation,
    foreign: &ForeignModule,
    types: &TypeContext,
) -> Result<ForeignTypeRef, XxiError> {
    let TypeAnnotationKind::Named { name, arguments } = &annotation.kind else {
        return Err(XxiError::UnsupportedType(
            "union types cannot cross an external boundary".into(),
        ));
    };
    match (name.as_str(), arguments.as_slice()) {
        ("ptr", [pointee]) => Ok(ForeignTypeRef::Pointer {
            pointee: Box::new(resolve_type_ref(pointee, foreign, types)?),
            mutable: false,
        }),
        ("mut_ptr", [pointee]) => Ok(ForeignTypeRef::Pointer {
            pointee: Box::new(resolve_type_ref(pointee, foreign, types)?),
            mutable: true,
        }),
        (_, []) => {
            if let Some(id) = types.resolve_name(name) {
                Ok(ForeignTypeRef::Severian(id))
            } else if foreign.type_declaration(name).is_some() {
                Ok(ForeignTypeRef::External(name.clone()))
            } else {
                Err(XxiError::UnknownType(name.clone()))
            }
        }
        _ => Err(XxiError::UnsupportedType(name.clone())),
    }
}

fn language_attribute(
    attributes: &[Attribute],
) -> Result<(ExternalLanguage, &Attribute), XxiError> {
    let mut found = attributes
        .iter()
        .filter(|attribute| !attribute.is_compile_policy());
    let attribute = found.next().ok_or(XxiError::MissingLanguageAttribute)?;
    if found.next().is_some() {
        return Err(XxiError::MultipleLanguageAttributes);
    }
    Ok((language_from_name(&attribute.name)?, attribute))
}

fn language_from_name(name: &str) -> Result<ExternalLanguage, XxiError> {
    match name {
        "c" => Ok(ExternalLanguage::C),
        "rust" => Ok(ExternalLanguage::Rust),
        "system" => Ok(ExternalLanguage::System),
        value if !value.is_empty() => Ok(ExternalLanguage::Custom(value.to_owned())),
        _ => Err(XxiError::UnknownLanguage(name.to_owned())),
    }
}

fn language_abi(language: &ExternalLanguage) -> AbiSelection {
    match language {
        ExternalLanguage::C => AbiSelection::C,
        ExternalLanguage::Rust => AbiSelection::Rust,
        ExternalLanguage::System | ExternalLanguage::Custom(_) => AbiSelection::System,
    }
}

fn parse_abi(value: &str) -> Result<AbiSelection, XxiError> {
    Ok(match value {
        "c" | "c-v1" => AbiSelection::C,
        "system" => AbiSelection::System,
        "rust" => AbiSelection::Rust,
        "sysv64" => AbiSelection::Explicit(CallingConvention::SysV64),
        "win64" => AbiSelection::Explicit(CallingConvention::Win64),
        "aapcs64" => AbiSelection::Explicit(CallingConvention::Aapcs64),
        _ => return Err(XxiError::UnknownAbi(value.to_owned())),
    })
}

fn parse_scalar_representation(value: &str) -> Option<AbiType> {
    let (signed, digits) = if let Some(digits) = value.strip_prefix('i') {
        (true, digits)
    } else if let Some(digits) = value.strip_prefix('u') {
        (false, digits)
    } else if let Some(digits) = value.strip_prefix('f') {
        return digits.parse().ok().map(AbiType::float);
    } else {
        return None;
    };
    digits
        .parse()
        .ok()
        .map(|bits| AbiType::integer(bits, signed))
}

fn named_argument<'a>(attribute: &'a Attribute, name: &str) -> Option<&'a str> {
    attribute.arguments.iter().find_map(|argument| {
        if argument.name.as_deref() == Some(name) {
            match &argument.value {
                AttributeValue::Name(value)
                | AttributeValue::String(value)
                | AttributeValue::Integer(value) => Some(value.as_str()),
                AttributeValue::Boolean(_) => None,
            }
        } else {
            None
        }
    })
}

fn boolean_argument(attribute: &Attribute, name: &str) -> Result<Option<bool>, XxiError> {
    attribute
        .arguments
        .iter()
        .find(|argument| argument.name.as_deref() == Some(name))
        .map(|argument| match argument.value {
            AttributeValue::Boolean(value) => Ok(value),
            _ => Err(XxiError::InvalidAttributeArgument(name.to_owned())),
        })
        .transpose()
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum XxiError {
    DuplicateDeclaration(String),
    Ffi(String),
    GenericExternalType(String),
    InvalidAttributeArgument(String),
    InvalidResultMode(String),
    MissingLanguageAttribute,
    MixedCompilerAndExternalAttributes,
    MultipleLanguageAttributes,
    Symbol(String),
    UnknownAbi(String),
    UnknownLanguage(String),
    UnknownRepresentation(String),
    UnknownType(String),
    UnsupportedType(String),
}

impl fmt::Display for XxiError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "invalid external interface declaration: {self:?}"
        )
    }
}

impl std::error::Error for XxiError {}

#[cfg(test)]
mod tests {
    use super::*;
    use severian_ast::Item;
    use severian_lexer::scan;
    use severian_parser::parse;
    use severian_source::SourceFile;
    use severian_target::TargetSpec;

    fn target() -> AbiTarget {
        AbiTarget::derive(&TargetSpec::host())
    }

    #[test]
    fn maps_c_source_declarations_through_ffi_to_abi() {
        let context = severian_bootstrap::load().unwrap();
        let source = SourceFile::virtual_source(
            "ffi.sev",
            "import \"c:libc\" as libc\n@c(repr = \"opaque\")\ntype FILE\n@c(symbol = \"sev_write\")\ndef write(value: borrowed[string], output: out[FILE]) -> i32\n",
        );
        let module = parse(&scan(&source).unwrap()).unwrap();
        assert!(matches!(module.items[1], Item::Type(_)));
        let resolved = resolve(&module, &context.types, &target()).unwrap();
        assert_eq!(resolved.imports[0].provider, "libc");
        assert_eq!(resolved.plans[0].provider.as_deref(), Some("libc"));
        assert_eq!(resolved.plans[0].symbol.name.as_str(), "sev_write");
        assert_eq!(
            resolved.plans[0].parameters[1].conversion,
            severian_ffi::Conversion::OutPointer
        );
    }

    #[test]
    fn canonical_xxi_import_allows_c_declarations_without_a_provider() {
        let context = severian_bootstrap::load().unwrap();
        let source = SourceFile::virtual_source(
            "ffi.sev",
            "import c from xxi\n@c(symbol = \"strlen\")\ndef c_strlen(text: borrowed[string]) -> usize\n",
        );
        let module = parse(&scan(&source).unwrap()).unwrap();
        let resolved = resolve(&module, &context.types, &target()).unwrap();
        assert!(resolved.imports.is_empty());
        assert_eq!(resolved.declarations.len(), 1);
        assert_eq!(resolved.declarations[0].function.symbol.name.as_str(), "strlen");
    }

    #[test]
    fn rust_selects_the_rust_calling_convention() {
        let context = severian_bootstrap::load().unwrap();
        let source =
            SourceFile::virtual_source("ffi.sev", "@rust\ndef identity(value: i32) -> i32\n");
        let module = parse(&scan(&source).unwrap()).unwrap();
        let resolved = resolve(&module, &context.types, &target()).unwrap();
        assert_eq!(
            resolved.plans[0].signature.convention,
            CallingConvention::Rust
        );
    }

    #[test]
    fn custom_language_attributes_use_the_system_boundary_by_default() {
        let context = severian_bootstrap::load().unwrap();
        let source = SourceFile::virtual_source(
            "ffi.sev",
            "@swift(symbol = \"foreign_identity\")\ndef identity(value: i32) -> i32\n",
        );
        let module = parse(&scan(&source).unwrap()).unwrap();
        let resolved = resolve(&module, &context.types, &target()).unwrap();
        assert_eq!(resolved.foreign.functions[0].abi, AbiSelection::System);
    }

    #[test]
    fn external_functions_may_overload_by_parameter_contract() {
        let context = severian_bootstrap::load().unwrap();
        let source = SourceFile::virtual_source(
            "overloads.sev",
            "@c(symbol = \"print_text\")\ndef print(value: string) -> i32\n@c(symbol = \"print_int\")\ndef print(value: int) -> i32\n",
        );
        let module = parse(&scan(&source).unwrap()).unwrap();
        let resolved = resolve(&module, &context.types, &target()).unwrap();
        assert_eq!(resolved.foreign.functions.len(), 2);
        assert_eq!(resolved.plans.len(), 2);
    }

    #[test]
    fn external_functions_reject_duplicate_parameter_contracts() {
        let context = severian_bootstrap::load().unwrap();
        let source = SourceFile::virtual_source(
            "duplicates.sev",
            "@c(symbol = \"first\")\ndef print(value: int) -> i32\n@c(symbol = \"second\")\ndef print(other: int) -> i32\n",
        );
        let module = parse(&scan(&source).unwrap()).unwrap();
        assert_eq!(
            resolve(&module, &context.types, &target()).unwrap_err(),
            XxiError::DuplicateDeclaration("print".into())
        );
    }

    #[test]
    fn semantic_operator_decorators_do_not_create_foreign_functions() {
        let context = severian_bootstrap::load().unwrap();
        let source = SourceFile::virtual_source(
            "operators.sev",
            "@strings(|)\ndef combine(left: string, right: string) -> string:\n    return left | right\n",
        );
        let module = parse(&scan(&source).unwrap()).unwrap();
        let resolved = resolve(&module, &context.types, &target()).unwrap();
        assert!(resolved.foreign.functions.is_empty());
        assert!(resolved.plans.is_empty());
    }

    #[test]
    fn compiler_policy_decorators_never_create_foreign_functions() {
        let context = severian_bootstrap::load().unwrap();
        let source = SourceFile::virtual_source(
            "compiler.sev",
            "@compile(mlir, stablehlo, xla)\ndef add(left: i32, right: i32) -> i32\n@mlir\ndef shape(value: i32) -> i32\n",
        );
        let module = parse(&scan(&source).unwrap()).unwrap();
        let resolved = resolve(&module, &context.types, &target()).unwrap();
        assert!(resolved.foreign.functions.is_empty());
        assert!(resolved.plans.is_empty());
    }

    #[test]
    fn compiler_policy_cannot_be_mixed_with_a_foreign_boundary() {
        let context = severian_bootstrap::load().unwrap();
        let source = SourceFile::virtual_source(
            "mixed.sev",
            "@c(symbol = \"add\")\n@compile(mlir)\ndef add(left: i32, right: i32) -> i32\n",
        );
        let module = parse(&scan(&source).unwrap()).unwrap();
        assert_eq!(
            resolve(&module, &context.types, &target()).unwrap_err(),
            XxiError::MixedCompilerAndExternalAttributes
        );
    }

    #[test]
    fn named_symbol_pack_operators_do_not_create_foreign_functions() {
        let context = severian_bootstrap::load().unwrap();
        let source = SourceFile::virtual_source(
            "operators.sev",
            "@tensor(X)\ndef contract(left: string, right: string) -> string:\n    return left X right\n",
        );
        let module = parse(&scan(&source).unwrap()).unwrap();
        let resolved = resolve(&module, &context.types, &target()).unwrap();
        assert!(resolved.foreign.functions.is_empty());
        assert!(resolved.plans.is_empty());
    }

    #[test]
    fn semantic_hook_decorators_do_not_create_foreign_functions() {
        let context = severian_bootstrap::load().unwrap();
        let source = SourceFile::virtual_source(
            "hooks.sev",
            "trait Monitor:\n    @monitor_error\n    def monitor_error(context: HookContext) -> None with context\n\n@monitor_error\ndef search() -> int:\n    return 10\n",
        );
        let module = parse(&scan(&source).unwrap()).unwrap();
        let resolved = resolve(&module, &context.types, &target()).unwrap();
        assert!(resolved.foreign.functions.is_empty());
        assert!(resolved.plans.is_empty());
    }

    #[test]
    fn composed_hook_namespaces_do_not_create_foreign_functions() {
        let context = severian_bootstrap::load().unwrap();
        let source = SourceFile::virtual_source(
            "hooks.sev",
            "trait Monitor:\n    @monitor\n\n    @monitor_error\n    def monitor_error(context: HookContext) -> None with context\n\n@monitor(monitor_error)\ndef search() -> int:\n    return 10\n",
        );
        let module = parse(&scan(&source).unwrap()).unwrap();
        let resolved = resolve(&module, &context.types, &target()).unwrap();
        assert!(resolved.foreign.functions.is_empty());
        assert!(resolved.plans.is_empty());
    }

    #[test]
    fn core_native_declarations_use_xxi_instead_of_legacy_extern_syntax() {
        let context = severian_bootstrap::load().unwrap();
        for (path, text) in [
            (
                "core/random/src/ffi.sev",
                include_str!("../../../../library/core/random/src/ffi.sev"),
            ),
            (
                "core/regex/src/ffi.sev",
                include_str!("../../../../library/core/regex/src/ffi.sev"),
            ),
        ] {
            let source = SourceFile::virtual_source(path, text);
            let module = parse(&scan(&source).unwrap()).unwrap();
            let resolved = resolve(&module, &context.types, &target()).unwrap();
            assert!(!resolved.plans.is_empty());
        }
    }
}
