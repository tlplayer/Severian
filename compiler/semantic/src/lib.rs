#![forbid(unsafe_code)]

use severian_ast::{
    AssignOp as AstAssignOp, BinaryOp as AstBinaryOp, Block, ElseBranch, Expr, ImportKind, Item,
    LetKind, Literal, Module, OwnershipOp as AstOwnershipOp, Pattern, Span, Stmt, Type, TypeArg,
    UnaryOp as AstUnaryOp,
};
use severian_hir::{
    AssignmentOp, BinaryOp, CallTarget, ChaosAction as HirChaosAction, Class, ClassDefinition,
    ComprehensionClause as HirComprehensionClause, Decorator as HirDecorator, DefinitionId,
    DetailedFunctionType, EnumDefinition, Expression, FieldDefinition, Function,
    FunctionContract as HirFunctionContract, FunctionId, Global, HirId, Instruction, MatchPattern,
    OwnershipOp, Parameter, Program, ProgramMetadata, SourceRange, SourceSpan,
    SwitchArm as HirSwitchArm, TaskPlacement, TensorDimension, TensorElementType, TensorType, Test,
    TestMode as HirTestMode, TypeDefinitionId, TypeId, TypeKind, TypeTable, UnaryOp, ValueType,
    VariantDefinition, VariantId,
};
use severian_package::{local_import_exposed_name, local_import_module_name, PackageInterface};
use std::collections::{HashMap, HashSet};
use std::fmt;
use std::path::PathBuf;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SemanticError {
    pub span: Span,
    pub message: String,
}

impl fmt::Display for SemanticError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "{} at bytes {}..{}",
            self.message, self.span.start, self.span.end
        )
    }
}

impl std::error::Error for SemanticError {}

#[derive(Clone)]
struct Signature {
    target: CallTarget,
    params: Vec<SignatureParameter>,
    returns: ValueType,
}

#[derive(Clone)]
struct SignatureParameter {
    name: String,
    ty: ValueType,
    function_return: Option<ValueType>,
    default: Option<Expr>,
}

#[derive(Clone, Copy)]
struct Binding {
    ty: ValueType,
    function_return: Option<ValueType>,
    collection_len: Option<usize>,
    mutable: bool,
    field: bool,
    integer_max: Option<i64>,
    known_integer: Option<i64>,
}

pub fn analyze(module: &Module) -> Result<Program, SemanticError> {
    analyze_with_interfaces(module, &[])
}

pub fn analyze_with_interfaces(
    module: &Module,
    interfaces: &[(String, Module)],
) -> Result<Program, SemanticError> {
    let interfaces = interfaces
        .iter()
        .map(|(name, module)| PackageInterface {
            name: name.clone(),
            module: module.clone(),
            compiler: Default::default(),
            source_path: PathBuf::from(format!("<interface:{name}>")),
            source: String::new(),
        })
        .collect::<Vec<_>>();
    analyze_with_packages(module, &interfaces)
}

pub fn analyze_with_packages(
    module: &Module,
    interfaces: &[PackageInterface],
) -> Result<Program, SemanticError> {
    let mut aliases = collect_imports(module);
    let imported_modules = collect_imported_modules(module);
    for interface in interfaces {
        for (symbol, function) in &interface.compiler.symbols {
            aliases.insert(
                format!("__symbol_alias.{}.{}", interface.name, symbol),
                format!("{}.{}", interface.name, function),
            );
        }
        for function in &interface.compiler.external_functions {
            aliases.insert(format!("__external_function.{function}"), String::new());
        }
        for rule in &interface.compiler.fusion_rules {
            aliases.insert(
                format!("__external_function.{}", rule.function),
                String::new(),
            );
        }
        for alias in &interface.compiler.fusion_aliases {
            aliases.insert(
                format!("__external_function.{}", alias.function),
                String::new(),
            );
        }
    }
    for item in &module.items {
        if let Item::Enum(enumeration) = item {
            if !is_upper_camel_case(&enumeration.name.name) {
                return Err(error(
                    enumeration.name.span,
                    format!("enum `{}` must use PascalCase", enumeration.name.name),
                ));
            }
            for variant in &enumeration.variants {
                if !is_upper_camel_case(&variant.name.name) {
                    return Err(error(
                        variant.name.span,
                        format!("enum variant `{}` must use PascalCase", variant.name.name),
                    ));
                }
                aliases.insert(
                    format!("__variant_fields.{}", variant.name.name),
                    variant
                        .fields
                        .iter()
                        .map(|field| field.name.name.as_str())
                        .collect::<Vec<_>>()
                        .join(","),
                );
            }
        }
        if let Item::Class(class) = item {
            aliases.insert(
                class.name.name.clone(),
                format!("__class.{}", class.name.name),
            );
            aliases.insert(
                format!("__class_fields.{}", class.name.name),
                class
                    .fields
                    .iter()
                    .map(|field| field.name.name.as_str())
                    .collect::<Vec<_>>()
                    .join(","),
            );
        }
    }
    let mut signatures = HashMap::new();
    for interface in interfaces {
        let module_name = &interface.name;
        for item in &interface.module.items {
            if let Item::Class(class) = item {
                let exported = format!("{module_name}.{}", class.name.name);
                aliases.insert(
                    format!("__module_class.{exported}"),
                    class.name.name.clone(),
                );
                aliases
                    .entry(format!("__class_fields.{}", class.name.name))
                    .or_insert_with(|| {
                        class
                            .fields
                            .iter()
                            .map(|field| field.name.name.as_str())
                            .collect::<Vec<_>>()
                            .join(",")
                    });
                if imports_entire_module(module, module_name) {
                    aliases
                        .entry(class.name.name.clone())
                        .or_insert_with(|| format!("__class.{}", class.name.name));
                }
                continue;
            }
            let (name, native_symbol, params, return_type) = match item {
                Item::Function(function) => (
                    &function.name,
                    function.native_symbol.as_deref(),
                    function.params.as_slice(),
                    function.return_type.as_ref(),
                ),
                _ => continue,
            };
            let key = format!("{module_name}.{}", name.name);
            let signature = lower_signature(&key, native_symbol, params, return_type)?;
            if signatures.insert(key.clone(), signature.clone()).is_some() {
                return Err(error(
                    name.span,
                    format!("duplicate exported function `{key}`"),
                ));
            }
            if imports_entire_module(module, module_name) {
                aliases.entry(name.name.clone()).or_insert(key);
            }
        }
    }
    for item in &module.items {
        let (name, native_symbol, params, return_type) = match item {
            Item::Function(function) => (
                &function.name,
                function.native_symbol.as_deref(),
                function.params.as_slice(),
                function.return_type.as_ref(),
            ),
            _ => continue,
        };
        let signature = lower_signature(&name.name, native_symbol, params, return_type)?;
        if signatures.insert(name.name.clone(), signature).is_some() {
            return Err(error(
                name.span,
                format!("duplicate function `{}`", name.name),
            ));
        }
    }

    let mut global_scope = HashMap::new();
    let mut globals = Vec::new();
    for item in &module.items {
        if let Item::Statement(Stmt::Let(binding)) = item {
            let declared = binding.ty.as_ref().map(lower_type).transpose()?;
            let source = binding
                .value
                .as_ref()
                .ok_or_else(|| error(binding.span, "global requires a value"))?;
            let (value, inferred) = lower_expression(source, &global_scope, &signatures, &aliases)?;
            if let Some(declared) = declared {
                compatible(binding.span, inferred, declared)?;
            }
            let ty = declared.unwrap_or(inferred);
            global_scope.insert(
                binding.name.name.clone(),
                Binding {
                    ty,
                    function_return: None,
                    collection_len: None,
                    mutable: false,
                    field: false,
                    integer_max: None,
                    known_integer: None,
                },
            );
            globals.push(Global {
                name: binding.name.name.clone(),
                value,
            });
        }
    }

    let mut functions = Vec::new();
    for item in &module.items {
        let Item::Function(function) = item else {
            continue;
        };
        let decorators = lower_decorators(&function.decorators, &imported_modules)?;
        let function_aliases = aliases_with_decorators(&aliases, &function.decorators);
        let signature = signatures.get(&function.name.name).unwrap();
        let mut scope = global_scope.clone();
        let mut params = Vec::new();
        for parameter in &signature.params {
            let default = parameter
                .default
                .as_ref()
                .map(|value| {
                    let (value, ty) =
                        lower_expression(value, &scope, &signatures, &function_aliases)?;
                    compatible(value_span(&parameter.default), ty, parameter.ty)?;
                    Ok(value)
                })
                .transpose()?;
            scope.insert(
                parameter.name.clone(),
                Binding {
                    ty: parameter.ty,
                    function_return: parameter.function_return,
                    collection_len: None,
                    mutable: false,
                    field: false,
                    integer_max: None,
                    known_integer: None,
                },
            );
            params.push(Parameter {
                name: parameter.name.clone(),
                ty: parameter.ty,
                default,
            });
        }
        let instructions = lower_block(
            &function.body,
            &mut scope,
            signature.returns,
            &signatures,
            &function_aliases,
        )?;
        if function.native_symbol.is_none()
            && signature.returns != ValueType::Unit
            && !always_returns(&instructions)
        {
            return Err(error(
                function.span,
                format!("function `{}` must return a value", function.name.name),
            ));
        }
        let mut tests = Vec::new();
        for test in &function.tests {
            let mut test_scope = global_scope.clone();
            add_test_bindings(&mut test_scope, &test.modes);
            tests.push(Test {
                name: test.name.as_ref().map(|name| name.name.clone()),
                modes: lower_test_modes(&test.modes),
                instructions: lower_block(
                    &test.body,
                    &mut test_scope,
                    ValueType::Unit,
                    &signatures,
                    &function_aliases,
                )?,
            });
        }
        functions.push(Function {
            id: FunctionId::from_name(&function.name.name),
            name: function.name.name.clone(),
            native_symbol: function.native_symbol.clone(),
            decorators,
            contract: lower_function_contract(
                function.contract.as_ref(),
                &scope,
                &signatures,
                &function_aliases,
            )?,
            params,
            return_type: signature.returns,
            instructions,
            tests,
        });
    }
    let mut classes = Vec::new();
    for item in &module.items {
        let Item::Class(class) = item else { continue };
        let class_decorators = lower_decorators(&class.decorators, &imported_modules)?;
        let fields = class
            .fields
            .iter()
            .map(|field| field.name.name.clone())
            .collect::<Vec<_>>();
        let field_defaults = class
            .fields
            .iter()
            .map(|field| {
                if let Some(default) = &field.default {
                    return lower_expression(default, &global_scope, &signatures, &aliases)
                        .map(|(default, _)| Some(default));
                }
                let default = match field.ty.as_ref().map(lower_type).transpose()? {
                    Some(ValueType::List) => Some(Expression::List(Vec::new())),
                    Some(ValueType::Map) => Some(Expression::Map(Vec::new())),
                    Some(ValueType::Set) => Some(Expression::Set(Vec::new())),
                    _ => None,
                };
                Ok(default)
            })
            .collect::<Result<Vec<_>, SemanticError>>()?;
        let mut constructors = Vec::new();
        for constructor in &class.constructors {
            lower_decorators(&constructor.decorators, &imported_modules)?;
            constructors.push(lower_class_function(
                &class.name.name,
                &fields,
                &constructor.name.name,
                &constructor.decorators,
                &constructor.params,
                constructor.contract.as_ref(),
                &constructor.body,
                &constructor.tests,
                ValueType::Unit,
                &global_scope,
                &signatures,
                &aliases,
            )?);
        }
        let mut methods = Vec::new();
        for method in &class.methods {
            lower_decorators(&method.decorators, &imported_modules)?;
            let returns = method
                .return_type
                .as_ref()
                .map(lower_type)
                .transpose()?
                .unwrap_or(ValueType::Unit);
            methods.push(lower_class_function(
                &class.name.name,
                &fields,
                &method.name.name,
                &method.decorators,
                &method.params,
                method.contract.as_ref(),
                &method.body,
                &method.tests,
                returns,
                &global_scope,
                &signatures,
                &aliases,
            )?);
        }
        classes.push(Class {
            id: TypeDefinitionId::from_name(&class.name.name),
            name: class.name.name.clone(),
            decorators: class_decorators,
            fields,
            field_types: class
                .fields
                .iter()
                .map(|field| field.ty.as_ref().map(lower_type).transpose())
                .collect::<Result<Vec<_>, _>>()?
                .into_iter()
                .map(|ty| ty.unwrap_or(ValueType::Any))
                .collect(),
            field_classes: class
                .fields
                .iter()
                .map(|field| field.ty.as_ref().and_then(class_type_name))
                .collect(),
            field_defaults,
            constructors,
            methods,
        });
    }
    Ok(Program {
        metadata: Default::default(),
        globals,
        classes,
        functions,
    })
}

/// Attaches the HIR-v2 metadata wire without changing semantic execution.
/// Existing lowering continues to consume the compact `ValueType`; detailed
/// types and source provenance live in this sidecar until consumers migrate.
pub fn attach_module_metadata(
    module: &Module,
    program: &mut Program,
    path: impl Into<PathBuf>,
    source: impl Into<String>,
    namespace: Option<&str>,
) {
    let mut metadata = std::mem::take(&mut program.metadata);
    attach_module_metadata_to(module, program, &mut metadata, path, source, namespace);
    program.metadata = metadata;
}

pub fn attach_module_metadata_to(
    module: &Module,
    program: &mut Program,
    metadata: &mut ProgramMetadata,
    path: impl Into<PathBuf>,
    source: impl Into<String>,
    namespace: Option<&str>,
) {
    let file = program.attach_source_file_to(metadata, path, source);
    let known_types = module
        .items
        .iter()
        .filter_map(|item| match item {
            Item::Class(class) => Some((
                class.name.name.clone(),
                metadata_type_id(namespace, &class.name.name),
            )),
            Item::Trait(trait_) => Some((
                trait_.name.name.clone(),
                metadata_type_id(namespace, &trait_.name.name),
            )),
            Item::Enum(enumeration) => Some((
                enumeration.name.name.clone(),
                metadata_type_id(namespace, &enumeration.name.name),
            )),
            _ => None,
        })
        .collect::<HashMap<_, _>>();
    let mut variant_owners = HashMap::new();

    for item in &module.items {
        match item {
            Item::Function(function) => {
                register_function_metadata(function, None, namespace, file, &known_types, metadata);
            }
            Item::Class(class) => {
                let id = metadata_type_id(namespace, &class.name.name);
                metadata
                    .sources
                    .record_definition(DefinitionId::Type(id), source_span(file, class.name.span));
                let fields = class
                    .fields
                    .iter()
                    .map(|field| FieldDefinition {
                        name: field.name.name.clone(),
                        ty: intern_optional_type(
                            field.ty.as_ref(),
                            &known_types,
                            &mut metadata.types,
                        ),
                        default: field.default.as_ref().map(|default| {
                            HirId::from_source_span(
                                file,
                                SourceRange {
                                    start: default.span().start,
                                    end: default.span().end,
                                },
                            )
                        }),
                    })
                    .collect();
                metadata.classes.insert(
                    id,
                    ClassDefinition {
                        id,
                        name: class.name.name.clone(),
                        fields,
                    },
                );
                for constructor in &class.constructors {
                    register_constructor_metadata(
                        constructor,
                        &class.name.name,
                        namespace,
                        file,
                        &known_types,
                        metadata,
                    );
                }
                for method in &class.methods {
                    register_function_metadata(
                        method,
                        Some(&class.name.name),
                        namespace,
                        file,
                        &known_types,
                        metadata,
                    );
                }
            }
            Item::Enum(enumeration) => {
                let id = metadata_type_id(namespace, &enumeration.name.name);
                metadata.sources.record_definition(
                    DefinitionId::Type(id),
                    source_span(file, enumeration.name.span),
                );
                let variants = enumeration
                    .variants
                    .iter()
                    .map(|variant| {
                        let variant_id = VariantId::from_name(&variant.name.name);
                        variant_owners.insert(variant.name.name.clone(), id);
                        metadata.sources.record_definition(
                            DefinitionId::Variant {
                                owner: id,
                                variant: variant_id,
                            },
                            source_span(file, variant.name.span),
                        );
                        VariantDefinition {
                            id: variant_id,
                            name: variant.name.name.clone(),
                            fields: variant
                                .fields
                                .iter()
                                .map(|field| {
                                    intern_optional_type(
                                        field.ty.as_ref(),
                                        &known_types,
                                        &mut metadata.types,
                                    )
                                })
                                .collect(),
                        }
                    })
                    .collect();
                metadata.enums.insert(
                    id,
                    EnumDefinition {
                        id,
                        name: enumeration.name.name.clone(),
                        variants,
                    },
                );
            }
            Item::Statement(Stmt::Let(binding)) => {
                let ty =
                    intern_optional_type(binding.ty.as_ref(), &known_types, &mut metadata.types);
                metadata.globals.insert(binding.name.name.clone(), ty);
            }
            Item::Trait(_) | Item::Import(_) | Item::Statement(_) => {}
        }
    }

    program.visit_expressions_mut(&mut |expression| {
        if let Expression::Variant { type_id, name, .. } = expression.kind() {
            if type_id.is_none() {
                if let Some(owner) = variant_owners.get(name) {
                    let Expression::Typed { expression, .. } = expression else {
                        return;
                    };
                    if let Expression::Variant { type_id, .. } = expression.as_mut() {
                        *type_id = Some(*owner);
                    }
                }
            }
        }
    });
}

fn register_constructor_metadata(
    constructor: &severian_ast::ConstructorDecl,
    class: &str,
    namespace: Option<&str>,
    file: severian_hir::SourceFileId,
    known_types: &HashMap<String, TypeDefinitionId>,
    metadata: &mut ProgramMetadata,
) {
    let class = qualified_name(namespace, class);
    let id = FunctionId::from_name(&format!("{class}.{}", constructor.name.name));
    metadata.sources.record_definition(
        DefinitionId::Function(id),
        source_span(file, constructor.name.span),
    );
    let parameters = constructor
        .params
        .iter()
        .map(|parameter| {
            intern_optional_type(parameter.ty.as_ref(), known_types, &mut metadata.types)
        })
        .collect();
    let returns = metadata.types.intern(TypeKind::Unit);
    metadata.functions.insert(
        id,
        DetailedFunctionType {
            parameters,
            returns,
        },
    );
}

fn register_function_metadata(
    function: &severian_ast::FunctionDecl,
    class: Option<&str>,
    namespace: Option<&str>,
    file: severian_hir::SourceFileId,
    known_types: &HashMap<String, TypeDefinitionId>,
    metadata: &mut ProgramMetadata,
) {
    let name = if let Some(class) = class {
        format!(
            "{}.{}",
            qualified_name(namespace, class),
            function.name.name
        )
    } else if let Some(namespace) = namespace {
        format!("{namespace}.{}", function.name.name)
    } else {
        function.name.name.clone()
    };
    let id = FunctionId::from_name(&name);
    metadata.sources.record_definition(
        DefinitionId::Function(id),
        source_span(file, function.name.span),
    );
    let parameters = function
        .params
        .iter()
        .map(|parameter| {
            intern_optional_type(parameter.ty.as_ref(), known_types, &mut metadata.types)
        })
        .collect();
    let returns = match &function.return_type {
        Some(ty) => intern_type(ty, known_types, &mut metadata.types),
        None => metadata.types.intern(TypeKind::Unit),
    };
    metadata.functions.insert(
        id,
        DetailedFunctionType {
            parameters,
            returns,
        },
    );
}

fn intern_optional_type(
    ty: Option<&Type>,
    known_types: &HashMap<String, TypeDefinitionId>,
    types: &mut TypeTable,
) -> TypeId {
    match ty {
        Some(ty) => intern_type(ty, known_types, types),
        None => types.intern(TypeKind::Any),
    }
}

fn intern_type(
    ty: &Type,
    known_types: &HashMap<String, TypeDefinitionId>,
    types: &mut TypeTable,
) -> TypeId {
    match ty {
        Type::Named(path) => {
            let name = path
                .segments
                .iter()
                .map(|segment| segment.name.as_str())
                .collect::<Vec<_>>()
                .join(".");
            let arguments = path
                .args
                .iter()
                .filter_map(TypeArg::as_type)
                .map(|argument| intern_type(argument, known_types, types))
                .collect::<Vec<_>>();
            match name.as_str() {
                "int" | "u8" | "u16" | "u32" | "u64" | "usize" | "i32" | "i64" => {
                    types.intern(TypeKind::Int)
                }
                "float" | "f32" | "f64" => types.intern(TypeKind::Float),
                "bool" => types.intern(TypeKind::Bool),
                "string" => types.intern(TypeKind::String),
                "unit" => types.intern(TypeKind::Unit),
                "list" => {
                    let element = arguments
                        .first()
                        .copied()
                        .unwrap_or_else(|| types.intern(TypeKind::Any));
                    types.intern(TypeKind::List(element))
                }
                "map" => {
                    let any = types.intern(TypeKind::Any);
                    let key = arguments.first().copied().unwrap_or(any);
                    let value = arguments.get(1).copied().unwrap_or(any);
                    types.intern(TypeKind::Map { key, value })
                }
                "set" => {
                    let element = arguments
                        .first()
                        .copied()
                        .unwrap_or_else(|| types.intern(TypeKind::Any));
                    types.intern(TypeKind::Set(element))
                }
                "Tensor" => types
                    .intern(TypeKind::Tensor(lower_tensor_type(path).unwrap_or_else(
                        |_| TensorType::dynamic(TensorElementType::F64),
                    ))),
                "Channel" => {
                    let element = arguments
                        .first()
                        .copied()
                        .unwrap_or_else(|| types.intern(TypeKind::Any));
                    types.intern(TypeKind::Channel(element))
                }
                "Result" => {
                    let any = types.intern(TypeKind::Any);
                    let ok = arguments.first().copied().unwrap_or(any);
                    let error = arguments.get(1).copied().unwrap_or(any);
                    types.intern(TypeKind::Result { ok, error })
                }
                "Option" => {
                    let some = arguments
                        .first()
                        .copied()
                        .unwrap_or_else(|| types.intern(TypeKind::Any));
                    types.intern(TypeKind::Option(some))
                }
                "fn" => {
                    let any = types.intern(TypeKind::Any);
                    let (parameters, returns) = arguments
                        .split_last()
                        .map_or((Vec::new(), any), |(returns, parameters)| {
                            (parameters.to_vec(), *returns)
                        });
                    types.intern(TypeKind::Function {
                        parameters,
                        returns,
                    })
                }
                _ if known_types.contains_key(&name) => types.intern(TypeKind::Named {
                    definition: known_types[&name],
                    name,
                    arguments,
                }),
                _ => types.intern(TypeKind::Unresolved { name, arguments }),
            }
        }
        Type::List { element, .. } => {
            let element = intern_type(element, known_types, types);
            types.intern(TypeKind::List(element))
        }
        Type::Tuple { elements, .. } => {
            let elements = elements
                .iter()
                .map(|element| intern_type(element, known_types, types))
                .collect();
            types.intern(TypeKind::Tuple(elements))
        }
        Type::Union { alternatives, .. } => {
            let alternatives = alternatives
                .iter()
                .map(|alternative| intern_type(alternative, known_types, types))
                .collect();
            types.intern(TypeKind::Union(alternatives))
        }
        Type::Map { key, value, .. } => {
            let key = intern_type(key, known_types, types);
            let value = intern_type(value, known_types, types);
            types.intern(TypeKind::Map { key, value })
        }
        Type::Set { element, .. } => {
            let element = intern_type(element, known_types, types);
            types.intern(TypeKind::Set(element))
        }
        Type::Result { ok, err, .. } => {
            let ok = intern_type(ok, known_types, types);
            let error = intern_type(err, known_types, types);
            types.intern(TypeKind::Result { ok, error })
        }
        Type::Option { some, .. } => {
            let some = intern_type(some, known_types, types);
            types.intern(TypeKind::Option(some))
        }
        Type::Function {
            params, returns, ..
        } => {
            let parameters = params
                .iter()
                .map(|parameter| intern_type(parameter, known_types, types))
                .collect();
            let returns = intern_type(returns, known_types, types);
            types.intern(TypeKind::Function {
                parameters,
                returns,
            })
        }
        Type::Future { output, .. } => {
            let output = intern_type(output, known_types, types);
            types.intern(TypeKind::Future(output))
        }
        Type::Reference { mutable, inner, .. } => {
            let inner = intern_type(inner, known_types, types);
            types.intern(TypeKind::Reference {
                mutable: *mutable,
                inner,
            })
        }
    }
}

fn metadata_type_id(namespace: Option<&str>, name: &str) -> TypeDefinitionId {
    TypeDefinitionId::from_name(&qualified_name(namespace, name))
}

fn qualified_name(namespace: Option<&str>, name: &str) -> String {
    namespace.map_or_else(
        || name.to_owned(),
        |namespace| format!("{namespace}.{name}"),
    )
}

fn source_span(file: severian_hir::SourceFileId, span: Span) -> SourceSpan {
    SourceSpan {
        file,
        range: SourceRange {
            start: span.start,
            end: span.end,
        },
    }
}

fn class_type_name(ty: &Type) -> Option<String> {
    let Type::Named(path) = ty else { return None };
    let name = path.segments.first()?.name.as_str();
    if matches!(
        name,
        "int"
            | "u8"
            | "u16"
            | "u32"
            | "u64"
            | "usize"
            | "float"
            | "f32"
            | "f64"
            | "bool"
            | "string"
            | "unit"
            | "list"
            | "map"
            | "set"
            | "Tensor"
            | "Channel"
            | "fn"
            | "Result"
            | "Option"
    ) {
        None
    } else {
        Some(name.to_owned())
    }
}

fn imports_entire_module(module: &Module, module_name: &str) -> bool {
    module.items.iter().any(|item| {
        let Item::Import(import) = item else {
            return false;
        };
        match &import.kind {
            ImportKind::Local { path, .. } => {
                local_import_module_name(path).as_deref() == Some(module_name)
            }
            ImportKind::Module { path, .. } => {
                path.iter()
                    .map(|part| part.name.as_str())
                    .collect::<Vec<_>>()
                    .join(".")
                    == module_name
            }
            ImportKind::From { .. } => false,
        }
    })
}

#[allow(clippy::too_many_arguments)]
fn lower_class_function(
    class_name: &str,
    fields: &[String],
    name: &str,
    source_decorators: &[severian_ast::Decorator],
    source_params: &[severian_ast::Parameter],
    source_contract: Option<&severian_ast::FunctionContract>,
    body: &Block,
    source_tests: &[severian_ast::TestBlock],
    return_type: ValueType,
    global_scope: &HashMap<String, Binding>,
    signatures: &HashMap<String, Signature>,
    aliases: &HashMap<String, String>,
) -> Result<Function, SemanticError> {
    let mut scope = global_scope.clone();
    for field in fields {
        scope.insert(
            field.clone(),
            Binding {
                ty: ValueType::Any,
                function_return: None,
                collection_len: None,
                mutable: true,
                field: true,
                integer_max: None,
                known_integer: None,
            },
        );
    }
    let mut params = Vec::new();
    for param in source_params {
        let ty = param
            .ty
            .as_ref()
            .map(lower_type)
            .transpose()?
            .unwrap_or(ValueType::Any);
        let default = param
            .default
            .as_ref()
            .map(|value| {
                lower_expression(value, &scope, signatures, aliases).map(|(value, _)| value)
            })
            .transpose()?;
        scope.insert(
            param.name.name.clone(),
            Binding {
                ty,
                function_return: function_return_type(param.ty.as_ref()),
                collection_len: None,
                mutable: false,
                field: false,
                integer_max: None,
                known_integer: None,
            },
        );
        params.push(Parameter {
            name: param.name.name.clone(),
            ty,
            default,
        });
    }
    let instructions = lower_block(body, &mut scope, return_type, signatures, aliases)?;
    if return_type != ValueType::Unit && !always_returns(&instructions) {
        return Err(error(
            body.span,
            format!("method `{class_name}.{name}` must return a value"),
        ));
    }
    let mut tests = Vec::new();
    for test in source_tests {
        let mut test_scope = global_scope.clone();
        add_test_bindings(&mut test_scope, &test.modes);
        tests.push(Test {
            name: test.name.as_ref().map(|name| name.name.clone()),
            modes: lower_test_modes(&test.modes),
            instructions: lower_block(
                &test.body,
                &mut test_scope,
                ValueType::Unit,
                signatures,
                aliases,
            )?,
        });
    }
    Ok(Function {
        id: FunctionId::from_name(&format!("{class_name}.{name}")),
        name: name.into(),
        native_symbol: None,
        decorators: decorator_metadata(source_decorators),
        contract: lower_function_contract(source_contract, &scope, signatures, aliases)?,
        params,
        return_type,
        instructions,
        tests,
    })
}

fn lower_function_contract(
    contract: Option<&severian_ast::FunctionContract>,
    scope: &HashMap<String, Binding>,
    signatures: &HashMap<String, Signature>,
    aliases: &HashMap<String, String>,
) -> Result<Option<HirFunctionContract>, SemanticError> {
    contract
        .map(|contract| {
            let requirements = contract
                .requirements
                .iter()
                .map(|requirement| {
                    let (requirement, ty) =
                        lower_expression(requirement, scope, signatures, aliases)?;
                    compatible(contract.span, ty, ValueType::Bool)?;
                    Ok(requirement)
                })
                .collect::<Result<Vec<_>, SemanticError>>()?;
            let capabilities = contract
                .capabilities
                .iter()
                .map(|capability| {
                    lower_expression(
                        &Expr::Identifier(capability.clone()),
                        scope,
                        signatures,
                        aliases,
                    )
                    .map(|(capability, _)| capability)
                })
                .collect::<Result<Vec<_>, _>>()?;
            Ok(HirFunctionContract {
                requirements,
                capabilities,
            })
        })
        .transpose()
}

fn collect_imports(module: &Module) -> HashMap<String, String> {
    let mut aliases = HashMap::new();
    for item in &module.items {
        let Item::Import(import) = item else { continue };
        match &import.kind {
            ImportKind::Local { path, alias } => {
                let Some(canonical) = local_import_module_name(path) else {
                    continue;
                };
                let exposed = alias
                    .as_ref()
                    .map(|alias| alias.name.clone())
                    .or_else(|| local_import_exposed_name(path));
                if let Some(exposed) = exposed {
                    aliases.insert(exposed, canonical);
                }
            }
            ImportKind::Module { path, alias } => {
                let canonical = path
                    .iter()
                    .map(|part| part.name.as_str())
                    .collect::<Vec<_>>()
                    .join(".");
                let exposed = alias
                    .as_ref()
                    .unwrap_or_else(|| path.first().unwrap())
                    .name
                    .clone();
                aliases.insert(exposed, canonical);
            }
            ImportKind::From { module, names } => {
                let module = module
                    .iter()
                    .map(|part| part.name.as_str())
                    .collect::<Vec<_>>()
                    .join(".");
                for name in names {
                    let exposed = name.alias.as_ref().unwrap_or(&name.name).name.clone();
                    aliases.insert(exposed, format!("{module}.{}", name.name.name));
                }
            }
        }
    }
    aliases
}

fn aliases_with_decorators(
    aliases: &HashMap<String, String>,
    decorators: &[severian_ast::Decorator],
) -> HashMap<String, String> {
    let mut aliases = aliases.clone();
    for decorator in decorators {
        let package = decorator
            .name
            .segments
            .iter()
            .map(|segment| segment.name.as_str())
            .collect::<Vec<_>>()
            .join(".");
        for symbol in &decorator.symbols {
            aliases.insert(format!("__symbol.{}", symbol.spelling), package.clone());
            if let Some(function) =
                aliases.get(&format!("__symbol_alias.{package}.{}", symbol.spelling))
            {
                aliases.insert(symbol.spelling.clone(), function.clone());
            }
        }
    }
    aliases
}

fn collect_imported_modules(module: &Module) -> HashSet<String> {
    module
        .items
        .iter()
        .filter_map(|item| {
            let Item::Import(import) = item else {
                return None;
            };
            match &import.kind {
                ImportKind::Local { path, alias } => alias
                    .as_ref()
                    .map(|alias| alias.name.clone())
                    .or_else(|| local_import_exposed_name(path)),
                ImportKind::Module { path, alias } => Some(
                    alias
                        .as_ref()
                        .unwrap_or_else(|| path.first().unwrap())
                        .name
                        .clone(),
                ),
                ImportKind::From { .. } => None,
            }
        })
        .collect()
}

fn lower_decorators(
    decorators: &[severian_ast::Decorator],
    imported_modules: &HashSet<String>,
) -> Result<Vec<HirDecorator>, SemanticError> {
    for decorator in decorators {
        let root = &decorator.name.segments.first().unwrap().name;
        if !imported_modules.contains(root) {
            return Err(error(
                decorator.name.span,
                format!("decorator package `{root}` must be imported"),
            ));
        }
        let mut seen = HashSet::new();
        for symbol in &decorator.symbols {
            if !seen.insert(&symbol.spelling) {
                return Err(error(
                    symbol.span,
                    format!("duplicate decorator symbol `{}`", symbol.spelling),
                ));
            }
        }
    }
    Ok(decorator_metadata(decorators))
}

fn decorator_metadata(decorators: &[severian_ast::Decorator]) -> Vec<HirDecorator> {
    decorators
        .iter()
        .map(|decorator| HirDecorator {
            package: decorator
                .name
                .segments
                .iter()
                .map(|segment| segment.name.as_str())
                .collect::<Vec<_>>()
                .join("."),
            symbols: decorator
                .symbols
                .iter()
                .map(|symbol| symbol.spelling.clone())
                .collect(),
        })
        .collect()
}

fn lower_block(
    block: &Block,
    scope: &mut HashMap<String, Binding>,
    return_type: ValueType,
    signatures: &HashMap<String, Signature>,
    aliases: &HashMap<String, String>,
) -> Result<Vec<Instruction>, SemanticError> {
    let mut instructions = Vec::new();
    for statement in &block.statements {
        match statement {
            Stmt::Let(binding) => {
                let source = binding
                    .value
                    .as_ref()
                    .ok_or_else(|| error(binding.span, "binding requires a value"))?;
                if binding.ty.as_ref().is_some_and(|ty| {
                    matches!(ty, Type::Named(path) if path.segments.first().is_some_and(|segment| segment.name == "u8"))
                }) && constant_integer(source).is_some_and(|value| !(0..=u8::MAX as i64).contains(&value))
                {
                    return Err(error(
                        source.span(),
                        "E0501: Checked integer arithmetic cannot produce a value outside the destination type.",
                    ));
                }
                if checked_integer_overflow(source, scope) {
                    return Err(error(
                        source.span(),
                        "E0501: Checked integer arithmetic cannot produce a value outside the destination type.",
                    ));
                }
                let (value, inferred) = lower_expression(source, scope, signatures, aliases)?;
                let declared = binding.ty.as_ref().map(lower_type).transpose()?;
                if let Some(declared) = declared {
                    compatible(binding.span, inferred, declared)?;
                }
                let ty = declared.unwrap_or(inferred);
                let integer_max = binding
                    .ty
                    .as_ref()
                    .filter(|ty| named_type_is(ty, "u8"))
                    .map(|_| u8::MAX as i64);
                let known_integer = constant_integer(source);
                if scope
                    .get(&binding.name.name)
                    .is_some_and(|existing| existing.field || existing.mutable)
                {
                    instructions.push(Instruction::Assign {
                        target: Expression::Variable(binding.name.name.clone()),
                        op: AssignmentOp::Assign,
                        value,
                    });
                    continue;
                }
                if scope
                    .insert(
                        binding.name.name.clone(),
                        Binding {
                            ty,
                            function_return: None,
                            collection_len: binding.value.as_ref().and_then(collection_length),
                            mutable: binding.kind == LetKind::Changeable,
                            field: false,
                            integer_max,
                            known_integer,
                        },
                    )
                    .is_some()
                {
                    return Err(error(
                        binding.name.span,
                        format!("duplicate binding `{}`", binding.name.name),
                    ));
                }
                instructions.push(Instruction::Let {
                    name: binding.name.name.clone(),
                    value,
                });
            }
            Stmt::DestructureLet(binding) => {
                let (value, _) = lower_expression(&binding.value, scope, signatures, aliases)?;
                let temporary = format!("__destructure_{}", binding.span.start);
                instructions.push(Instruction::Let {
                    name: temporary.clone(),
                    value,
                });
                for (index, name) in binding.names.iter().enumerate() {
                    scope.insert(
                        name.name.clone(),
                        Binding {
                            ty: ValueType::Any,
                            function_return: None,
                            collection_len: None,
                            mutable: false,
                            field: false,
                            integer_max: None,
                            known_integer: None,
                        },
                    );
                    instructions.push(Instruction::Let {
                        name: name.name.clone(),
                        value: Expression::Index {
                            object: Box::new(Expression::Variable(temporary.clone())),
                            index: Box::new(Expression::Integer(index as i64)),
                        },
                    });
                }
            }
            Stmt::Assign(assignment) => {
                let (target, target_type) =
                    lower_expression(&assignment.target, scope, signatures, aliases)?;
                if let Expr::Identifier(name) = &assignment.target {
                    if !scope.get(&name.name).is_some_and(|binding| binding.mutable) {
                        return Err(error(
                            name.span,
                            format!("binding `{}` is not changeable", name.name),
                        ));
                    }
                } else if !matches!(assignment.target, Expr::Index(_)) {
                    return Err(error(
                        assignment.target.span(),
                        "assignment target is not mutable",
                    ));
                }
                let (value, value_type) =
                    lower_expression(&assignment.value, scope, signatures, aliases)?;
                if target_type != ValueType::Any && value_type != ValueType::Any {
                    compatible(assignment.span, value_type, target_type)?;
                }
                instructions.push(Instruction::Assign {
                    target,
                    op: match assignment.op {
                        AstAssignOp::Assign => AssignmentOp::Assign,
                        AstAssignOp::AddAssign => AssignmentOp::Add,
                        AstAssignOp::SubAssign => AssignmentOp::Sub,
                        AstAssignOp::MulAssign => AssignmentOp::Mul,
                        AstAssignOp::DivAssign => AssignmentOp::Div,
                        AstAssignOp::ModAssign => AssignmentOp::Mod,
                    },
                    value,
                });
            }
            Stmt::TryBind(binding) => {
                let (value, _) = lower_expression(&binding.value, scope, signatures, aliases)?;
                scope.insert(
                    binding.name.name.clone(),
                    Binding {
                        ty: ValueType::Any,
                        function_return: None,
                        collection_len: None,
                        mutable: false,
                        field: false,
                        integer_max: None,
                        known_integer: None,
                    },
                );
                instructions.push(Instruction::TryLet {
                    name: binding.name.name.clone(),
                    value,
                });
            }
            Stmt::Expr(expression) => {
                let (expression, expression_type) =
                    lower_expression(expression, scope, signatures, aliases)?;
                if let Expression::MethodCall { object, method, .. } = expression.kind() {
                    if collection_shape_mutating_method(method) {
                        if let Expression::Variable(name) = object.kind() {
                            if let Some(binding) = scope.get_mut(name) {
                                binding.collection_len = None;
                            }
                        }
                    }
                }
                if expression_type == ValueType::Result {
                    return Err(error(
                        statement.span(),
                        "E0801: A recoverable error must be propagated, handled, or explicitly discarded with a reason.",
                    ));
                }
                match expression.kind() {
                    Expression::Call { target, args } if target.name == "print" => {
                        let mut args = args.clone();
                        let value = if args.len() == 1 {
                            args.remove(0)
                        } else {
                            Expression::Typed {
                                id: HirId::from_source_range(
                                    statement.span().start,
                                    statement.span().end,
                                ),
                                ty: ValueType::Tuple,
                                expression: Box::new(Expression::PrintArgs(args)),
                            }
                        };
                        instructions.push(Instruction::Print(value));
                    }
                    _ => instructions.push(Instruction::Evaluate(expression)),
                }
            }
            Stmt::Assert(assertion) => {
                let (condition, ty) =
                    lower_expression(&assertion.condition, scope, signatures, aliases)?;
                compatible(assertion.condition.span(), ty, ValueType::Bool)?;
                instructions.push(Instruction::Assert(condition));
            }
            Stmt::Return(statement) => {
                let value = statement
                    .value
                    .as_ref()
                    .map(|value| lower_expression(value, scope, signatures, aliases))
                    .transpose()?;
                let actual = value.as_ref().map_or(ValueType::Unit, |(_, ty)| *ty);
                let value = value.map(|(value, _)| value);
                let value = match (return_type, actual, value) {
                    (ValueType::Result, ValueType::Result, value) => value,
                    (ValueType::Result, _, Some(value)) => Some(Expression::Variant {
                        type_id: Some(TypeDefinitionId::from_name("Result")),
                        variant_id: VariantId::from_name("ok"),
                        name: "ok".into(),
                        fields: vec![value],
                    }),
                    (ValueType::Option, ValueType::Option, value) => value,
                    (_, _, value) => {
                        compatible(statement.span, actual, return_type)?;
                        value
                    }
                };
                instructions.push(Instruction::Return(value));
            }
            Stmt::If(statement) => {
                let (condition, ty) =
                    lower_expression(&statement.condition, scope, signatures, aliases)?;
                compatible(statement.condition.span(), ty, ValueType::Bool)?;
                let mut then_scope = scope.clone();
                let then_instructions = lower_block(
                    &statement.then_block,
                    &mut then_scope,
                    return_type,
                    signatures,
                    aliases,
                )?;
                let else_instructions = match &statement.else_branch {
                    None => Vec::new(),
                    Some(ElseBranch::Block(block)) => {
                        let mut else_scope = scope.clone();
                        lower_block(block, &mut else_scope, return_type, signatures, aliases)?
                    }
                    Some(ElseBranch::If(branch)) => {
                        let mut else_scope = scope.clone();
                        lower_block(
                            &Block {
                                span: branch.span,
                                statements: vec![Stmt::If((**branch).clone())],
                            },
                            &mut else_scope,
                            return_type,
                            signatures,
                            aliases,
                        )?
                    }
                };
                instructions.push(Instruction::If {
                    condition,
                    then_instructions,
                    else_instructions,
                });
            }
            Stmt::While(statement) => {
                let setup = statement
                    .setup
                    .as_ref()
                    .map(|setup| {
                        let lowered = lower_block(
                            &Block {
                                span: setup.span(),
                                statements: vec![(**setup).clone()],
                            },
                            scope,
                            return_type,
                            signatures,
                            aliases,
                        )?;
                        Ok(Box::new(lowered.into_iter().next().unwrap()))
                    })
                    .transpose()?;
                let (condition, ty) =
                    lower_expression(&statement.condition, scope, signatures, aliases)?;
                compatible(statement.condition.span(), ty, ValueType::Bool)?;
                let capabilities = statement
                    .capabilities
                    .iter()
                    .map(|capability| {
                        lower_expression(capability, scope, signatures, aliases)
                            .map(|(capability, _)| capability)
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                let mut body_scope = scope.clone();
                let body = lower_block(
                    &statement.body,
                    &mut body_scope,
                    return_type,
                    signatures,
                    aliases,
                )?;
                propagate_unknown_collection_shapes(scope, &body_scope);
                instructions.push(Instruction::While {
                    setup,
                    capabilities,
                    condition,
                    instructions: body,
                });
            }
            Stmt::For(statement) => {
                let setup = statement
                    .setup
                    .as_ref()
                    .map(|setup| {
                        let lowered = lower_block(
                            &Block {
                                span: setup.span(),
                                statements: vec![(**setup).clone()],
                            },
                            scope,
                            return_type,
                            signatures,
                            aliases,
                        )?;
                        Ok(Box::new(lowered.into_iter().next().unwrap()))
                    })
                    .transpose()?;
                if inclusive_collection_range(&statement.iterable) {
                    return Err(error(
                        statement.iterable.span(),
                        "E0402: An inclusive range ending at a collection's element count includes one invalid index.",
                    ));
                }
                let (iterable, _) =
                    lower_expression(&statement.iterable, scope, signatures, aliases)?;
                let mut body_scope = scope.clone();
                let pattern = lower_pattern(&statement.pattern, &mut body_scope, aliases)?;
                let body = lower_block(
                    &statement.body,
                    &mut body_scope,
                    return_type,
                    signatures,
                    aliases,
                )?;
                propagate_unknown_collection_shapes(scope, &body_scope);
                instructions.push(Instruction::For {
                    setup,
                    pattern,
                    iterable,
                    instructions: body,
                });
            }
            Stmt::Switch(statement) => {
                let setup = statement
                    .setup
                    .as_ref()
                    .map(|setup| {
                        let lowered = lower_block(
                            &Block {
                                span: setup.span(),
                                statements: vec![(**setup).clone()],
                            },
                            scope,
                            return_type,
                            signatures,
                            aliases,
                        )?;
                        Ok(Box::new(lowered.into_iter().next().unwrap()))
                    })
                    .transpose()?;
                let repeat_condition = statement
                    .repeat_condition
                    .as_ref()
                    .map(|condition| {
                        let (condition, ty) =
                            lower_expression(condition, scope, signatures, aliases)?;
                        compatible(statement.span, ty, ValueType::Bool)?;
                        Ok(condition)
                    })
                    .transpose()?;
                let mut arms = Vec::new();
                for arm in &statement.arms {
                    let mut arm_scope = scope.clone();
                    let pattern = lower_pattern(&arm.pattern, &mut arm_scope, aliases)?;
                    let source = arm
                        .source
                        .as_ref()
                        .map(|source| {
                            lower_expression(source, &arm_scope, signatures, aliases)
                                .map(|(source, _)| source)
                        })
                        .transpose()?;
                    let guard = arm
                        .guard
                        .as_ref()
                        .map(|guard| {
                            lower_expression(guard, &arm_scope, signatures, aliases)
                                .map(|(guard, _)| guard)
                        })
                        .transpose()?;
                    let arm_instructions =
                        lower_block(&arm.body, &mut arm_scope, return_type, signatures, aliases)?;
                    arms.push(HirSwitchArm {
                        source,
                        pattern,
                        guard,
                        instructions: arm_instructions,
                    });
                }
                if statement.values.len() == 1
                    && statement.repeat_condition.is_none()
                    && statement.setup.is_none()
                    && statement.arms.iter().all(|arm| arm.source.is_none())
                {
                    let value =
                        lower_expression(&statement.values[0], scope, signatures, aliases)?.0;
                    instructions.push(Instruction::Switch { value, arms });
                } else {
                    let channels = statement
                        .values
                        .iter()
                        .map(|channel| {
                            lower_expression(channel, scope, signatures, aliases)
                                .map(|(channel, _)| channel)
                        })
                        .collect::<Result<Vec<_>, _>>()?;
                    instructions.push(Instruction::ChannelSwitch {
                        channels,
                        setup,
                        repeat_condition,
                        arms,
                    });
                }
            }
            Stmt::Unsafe(block) => {
                instructions.extend(lower_block(
                    &block.body,
                    scope,
                    return_type,
                    signatures,
                    aliases,
                )?);
            }
            Stmt::With(block) => {
                let mut resources = Vec::new();
                let mut placement = TaskPlacement::Default;
                for resource in &block.resources {
                    if let Expr::Identifier(identifier) = resource {
                        match identifier.name.as_str() {
                            "gpu" | "simd" => {
                                if !aliases.values().any(|module| module == "parallel") {
                                    return Err(error(
                                        identifier.span,
                                        format!(
                                            "execution placement `{}` requires `import parallel`",
                                            identifier.name
                                        ),
                                    ));
                                }
                                let requested = if identifier.name == "gpu" {
                                    TaskPlacement::Gpu
                                } else {
                                    TaskPlacement::Simd
                                };
                                if placement != TaskPlacement::Default {
                                    return Err(error(
                                        identifier.span,
                                        "an execution region accepts only one placement",
                                    ));
                                }
                                placement = requested;
                                continue;
                            }
                            "self" | "runtime" | "local" | "simt" => continue,
                            _ => {}
                        }
                    }
                    resources.push(lower_expression(resource, scope, signatures, aliases)?.0);
                }
                let mut with_scope = scope.clone();
                let body = lower_block(
                    &block.body,
                    &mut with_scope,
                    return_type,
                    signatures,
                    aliases,
                )?;
                instructions.push(Instruction::With {
                    placement,
                    resources,
                    instructions: body,
                });
            }
            Stmt::Break(_) => instructions.push(Instruction::Break),
            Stmt::Continue(_) => instructions.push(Instruction::Continue),
        }
    }
    Ok(instructions)
}

fn lower_expression(
    expression: &Expr,
    scope: &HashMap<String, Binding>,
    signatures: &HashMap<String, Signature>,
    aliases: &HashMap<String, String>,
) -> Result<(Expression, ValueType), SemanticError> {
    let (lowered, ty) = lower_expression_kind(expression, scope, signatures, aliases)?;
    let span = expression.span();
    Ok((
        Expression::Typed {
            id: HirId::from_source_range(span.start, span.end),
            ty,
            expression: Box::new(lowered),
        },
        ty,
    ))
}

fn lower_expression_kind(
    expression: &Expr,
    scope: &HashMap<String, Binding>,
    signatures: &HashMap<String, Signature>,
    aliases: &HashMap<String, String>,
) -> Result<(Expression, ValueType), SemanticError> {
    match expression {
        Expr::Literal(Literal::Integer { value, .. }) => {
            Ok((Expression::Integer(*value), ValueType::Int))
        }
        Expr::Literal(Literal::Float { value, .. }) => {
            Ok((Expression::Float(value.to_bits()), ValueType::Float))
        }
        Expr::Literal(Literal::Boolean { value, .. }) => {
            Ok((Expression::Boolean(*value), ValueType::Bool))
        }
        Expr::Literal(Literal::String { value, .. }) => {
            Ok((Expression::String(value.clone()), ValueType::String))
        }
        Expr::Identifier(identifier) => {
            if let Some(binding) = scope.get(&identifier.name) {
                Ok((Expression::Variable(identifier.name.clone()), binding.ty))
            } else if signatures.contains_key(&identifier.name) {
                Ok((
                    Expression::Function(identifier.name.clone()),
                    ValueType::Function,
                ))
            } else if identifier.name == "invalid" {
                Ok((
                    Expression::Variant {
                        type_id: None,
                        variant_id: VariantId::from_name("invalid"),
                        name: "invalid".into(),
                        fields: Vec::new(),
                    },
                    ValueType::Any,
                ))
            } else if identifier.name == "absent" {
                Ok((
                    Expression::Variant {
                        type_id: Some(TypeDefinitionId::from_name("Option")),
                        variant_id: VariantId::from_name("absent"),
                        name: "absent".into(),
                        fields: Vec::new(),
                    },
                    ValueType::Option,
                ))
            } else if identifier.name == "None" {
                Ok((
                    Expression::Variant {
                        type_id: Some(TypeDefinitionId::from_name("Option")),
                        variant_id: VariantId::from_name("None"),
                        name: "None".into(),
                        fields: Vec::new(),
                    },
                    ValueType::Option,
                ))
            } else if identifier
                .name
                .as_bytes()
                .first()
                .is_some_and(u8::is_ascii_uppercase)
            {
                Ok((
                    Expression::Variant {
                        type_id: None,
                        variant_id: VariantId::from_name(&identifier.name),
                        name: identifier.name.clone(),
                        fields: Vec::new(),
                    },
                    ValueType::Any,
                ))
            } else {
                Err(error(
                    identifier.span,
                    format!("unknown binding `{}`", identifier.name),
                ))
            }
        }
        Expr::Unary(unary) => {
            let (expression, ty) = lower_expression(&unary.expr, scope, signatures, aliases)?;
            let (op, expected, result) = match unary.op {
                AstUnaryOp::Negate => (UnaryOp::Negate, ty, ty),
                AstUnaryOp::Not => (UnaryOp::Not, ValueType::Bool, ValueType::Bool),
            };
            compatible(unary.span, ty, expected)?;
            Ok((
                Expression::Unary {
                    op,
                    expression: Box::new(expression),
                },
                result,
            ))
        }
        Expr::Binary(binary) => {
            let (left, left_type) = lower_expression(&binary.left, scope, signatures, aliases)?;
            let (right, right_type) = lower_expression(&binary.right, scope, signatures, aliases)?;
            let (op, result_type) = match binary.op {
                AstBinaryOp::Add => (
                    BinaryOp::Add,
                    merge_numeric(left_type, right_type, binary.span)?,
                ),
                AstBinaryOp::Sub => (
                    BinaryOp::Sub,
                    merge_numeric(left_type, right_type, binary.span)?,
                ),
                AstBinaryOp::Mul => (
                    BinaryOp::Mul,
                    merge_numeric(left_type, right_type, binary.span)?,
                ),
                AstBinaryOp::Div => (
                    BinaryOp::Div,
                    merge_numeric(left_type, right_type, binary.span)?,
                ),
                AstBinaryOp::Mod => (
                    BinaryOp::Mod,
                    merge_numeric(left_type, right_type, binary.span)?,
                ),
                AstBinaryOp::Power => {
                    let result = power_type(left_type, right_type, binary.span)?;
                    (BinaryOp::Power, result)
                }
                AstBinaryOp::MatMul => {
                    let package = aliases.get("__symbol.X").map(String::as_str);
                    if package != Some("tensor") {
                        return Err(error(
                            binary.span,
                            "operator `X` requires `@tensor(X)` on this function",
                        ));
                    }
                    return Ok((
                        Expression::Call {
                            target: CallTarget::source("tensor.rankedMatmul"),
                            args: vec![left, right],
                        },
                        left_type,
                    ));
                }
                AstBinaryOp::Cross => {
                    return Err(error(binary.span, "operator `^` is not supported"));
                }
                AstBinaryOp::Equal => (BinaryOp::Equal, ValueType::Bool),
                AstBinaryOp::NotEqual => (BinaryOp::NotEqual, ValueType::Bool),
                AstBinaryOp::Less => (BinaryOp::Less, ValueType::Bool),
                AstBinaryOp::LessEqual => (BinaryOp::LessEqual, ValueType::Bool),
                AstBinaryOp::Greater => (BinaryOp::Greater, ValueType::Bool),
                AstBinaryOp::GreaterEqual => (BinaryOp::GreaterEqual, ValueType::Bool),
                AstBinaryOp::And => (BinaryOp::And, ValueType::Bool),
                AstBinaryOp::Or => (BinaryOp::Or, ValueType::Bool),
                AstBinaryOp::In => (BinaryOp::In, ValueType::Bool),
            };
            Ok((
                Expression::Binary {
                    left: Box::new(left),
                    op,
                    right: Box::new(right),
                },
                result_type,
            ))
        }
        Expr::Call(call) => lower_call(call, scope, signatures, aliases),
        Expr::List(collection) => {
            lower_collection(&collection.elements, scope, signatures, aliases)
                .map(|elements| (Expression::List(elements), ValueType::List))
        }
        Expr::Tuple(collection) => {
            lower_collection(&collection.elements, scope, signatures, aliases)
                .map(|elements| (Expression::Tuple(elements), ValueType::Tuple))
        }
        Expr::Set(collection) => lower_collection(&collection.elements, scope, signatures, aliases)
            .map(|elements| (Expression::Set(elements), ValueType::Set)),
        Expr::Map(map) => {
            let entries = map
                .entries
                .iter()
                .map(|entry| {
                    Ok((
                        lower_expression(&entry.key, scope, signatures, aliases)?.0,
                        lower_expression(&entry.value, scope, signatures, aliases)?.0,
                    ))
                })
                .collect::<Result<Vec<_>, SemanticError>>()?;
            Ok((Expression::Map(entries), ValueType::Map))
        }
        Expr::Index(index) => {
            if let Expr::Literal(Literal::Integer {
                value: element_index,
                span,
            }) = index.index.as_ref()
            {
                let length = match index.object.as_ref() {
                    Expr::List(collection) => Some(collection.elements.len()),
                    Expr::Identifier(identifier) => scope
                        .get(&identifier.name)
                        .and_then(|binding| binding.collection_len),
                    _ => None,
                };
                if length
                    .is_some_and(|length| *element_index < 0 || *element_index as usize >= length)
                {
                    return Err(error(
                        *span,
                        format!(
                            "E0401: An index known to be outside a fixed-length collection is rejected at compile time. (index {element_index}, length {})",
                            length.unwrap()
                        ),
                    ));
                }
            }
            let (object, object_type) =
                lower_expression(&index.object, scope, signatures, aliases)?;
            let index_value = lower_expression(&index.index, scope, signatures, aliases)?.0;
            Ok((
                Expression::Index {
                    object: Box::new(object),
                    index: Box::new(index_value),
                },
                if object_type == ValueType::String {
                    ValueType::String
                } else {
                    ValueType::Any
                },
            ))
        }
        Expr::Slice(slice) => {
            let (object, object_type) =
                lower_expression(&slice.object, scope, signatures, aliases)?;
            let lower_bound = |bound: &Option<Box<Expr>>| {
                bound
                    .as_ref()
                    .map(|bound| {
                        let span = bound.span();
                        let (bound, ty) = lower_expression(bound, scope, signatures, aliases)?;
                        compatible(span, ty, ValueType::Int)?;
                        Ok(Box::new(bound))
                    })
                    .transpose()
            };
            Ok((
                Expression::Slice {
                    object: Box::new(object),
                    start: lower_bound(&slice.start)?,
                    end: lower_bound(&slice.end)?,
                    step: lower_bound(&slice.step)?,
                },
                object_type,
            ))
        }
        Expr::Member(member) => {
            let object = lower_expression(&member.object, scope, signatures, aliases)?.0;
            Ok((
                Expression::Member {
                    object: Box::new(object),
                    member: member.member.name.clone(),
                },
                ValueType::Any,
            ))
        }
        Expr::ListComprehension(comprehension) => {
            let mut inner_scope = scope.clone();
            let clauses = lower_comprehension_clauses(
                &comprehension.clauses,
                &mut inner_scope,
                signatures,
                aliases,
            )?;
            let element =
                lower_expression(&comprehension.element, &inner_scope, signatures, aliases)?.0;
            Ok((
                Expression::ListComprehension {
                    element: Box::new(element),
                    clauses,
                },
                ValueType::List,
            ))
        }
        Expr::SetComprehension(comprehension) => {
            let mut inner_scope = scope.clone();
            let clauses = lower_comprehension_clauses(
                &comprehension.clauses,
                &mut inner_scope,
                signatures,
                aliases,
            )?;
            let element =
                lower_expression(&comprehension.element, &inner_scope, signatures, aliases)?.0;
            Ok((
                Expression::SetComprehension {
                    element: Box::new(element),
                    clauses,
                },
                ValueType::Set,
            ))
        }
        Expr::MapComprehension(comprehension) => {
            let mut inner_scope = scope.clone();
            let clauses = lower_comprehension_clauses(
                &comprehension.clauses,
                &mut inner_scope,
                signatures,
                aliases,
            )?;
            let key = lower_expression(&comprehension.key, &inner_scope, signatures, aliases)?.0;
            let value =
                lower_expression(&comprehension.value, &inner_scope, signatures, aliases)?.0;
            Ok((
                Expression::MapComprehension {
                    key: Box::new(key),
                    value: Box::new(value),
                    clauses,
                },
                ValueType::Map,
            ))
        }
        Expr::If(conditional) => {
            let (condition, condition_type) =
                lower_expression(&conditional.condition, scope, signatures, aliases)?;
            compatible(
                conditional.condition.span(),
                condition_type,
                ValueType::Bool,
            )?;
            let (then_expression, then_type) =
                lower_expression(&conditional.then_expr, scope, signatures, aliases)?;
            let (else_expression, else_type) =
                lower_expression(&conditional.else_expr, scope, signatures, aliases)?;
            let result_type = if then_type == else_type {
                then_type
            } else if then_type == ValueType::Any || else_type == ValueType::Any {
                ValueType::Any
            } else {
                return Err(error(
                    conditional.span,
                    format!(
                        "conditional branches have incompatible types `{then_type:?}` and `{else_type:?}`"
                    ),
                ));
            };
            Ok((
                Expression::Conditional {
                    condition: Box::new(condition),
                    then_expression: Box::new(then_expression),
                    else_expression: Box::new(else_expression),
                },
                result_type,
            ))
        }
        Expr::Async(task) => {
            if let Expr::Call(call) = task.value.as_ref() {
                if let Expr::Member(member) = call.callee.as_ref() {
                    if let Expr::Identifier(object) = member.object.as_ref() {
                        if scope
                            .get(&object.name)
                            .is_some_and(|binding| binding.mutable)
                            && !task.captures.iter().any(|capture| capture.name == "lock")
                        {
                            return Err(error(
                                task.span,
                                "E0601: Mutable method calls across an async boundary require transferring the value's `lock` capability.",
                            ));
                        }
                    }
                }
            }
            let placement = match task.placement {
                severian_ast::TaskPlacement::Default => TaskPlacement::Default,
                severian_ast::TaskPlacement::Local => {
                    if !aliases.values().any(|module| module == "distributed") {
                        return Err(error(
                            task.span,
                            "task placement `local` requires `import distributed`",
                        ));
                    }
                    TaskPlacement::Local
                }
                severian_ast::TaskPlacement::Gpu
                | severian_ast::TaskPlacement::Simd
                | severian_ast::TaskPlacement::Simt => {
                    if !aliases.values().any(|module| module == "parallel") {
                        return Err(error(
                            task.span,
                            format!(
                                "task placement `{}` requires `import parallel`",
                                match task.placement {
                                    severian_ast::TaskPlacement::Gpu => "gpu",
                                    severian_ast::TaskPlacement::Simd => "simd",
                                    severian_ast::TaskPlacement::Simt => "simt",
                                    _ => unreachable!(),
                                }
                            ),
                        ));
                    }
                    match task.placement {
                        severian_ast::TaskPlacement::Gpu => TaskPlacement::Gpu,
                        severian_ast::TaskPlacement::Simd => TaskPlacement::Simd,
                        severian_ast::TaskPlacement::Simt => TaskPlacement::Simt,
                        _ => unreachable!(),
                    }
                }
            };
            let (value, _) = lower_expression(&task.value, scope, signatures, aliases)?;
            Ok((
                Expression::Task {
                    value: Box::new(value),
                    placement,
                },
                ValueType::Any,
            ))
        }
        Expr::Await(task) => {
            let (value, _) = lower_expression(&task.value, scope, signatures, aliases)?;
            Ok((Expression::Await(Box::new(value)), ValueType::Any))
        }
        Expr::Channel(channel) => {
            let capacity = lower_expression(&channel.capacity, scope, signatures, aliases)?.0;
            Ok((Expression::Channel(Box::new(capacity)), ValueType::Channel))
        }
        Expr::Send(send) => {
            let value = lower_expression(&send.value, scope, signatures, aliases)?.0;
            let channel = lower_expression(&send.channel, scope, signatures, aliases)?.0;
            Ok((
                Expression::Send {
                    value: Box::new(value),
                    channel: Box::new(channel),
                },
                ValueType::Unit,
            ))
        }
        Expr::Ownership(ownership) => {
            let (value, ty) = lower_expression(&ownership.value, scope, signatures, aliases)?;
            let op = match ownership.op {
                AstOwnershipOp::View => OwnershipOp::View,
                AstOwnershipOp::Borrow => OwnershipOp::Borrow,
                AstOwnershipOp::Clone => OwnershipOp::Clone,
                AstOwnershipOp::Move => OwnershipOp::Move,
                AstOwnershipOp::AddressOf => OwnershipOp::AddressOf,
            };
            Ok((
                Expression::Ownership {
                    op,
                    value: Box::new(value),
                },
                ty,
            ))
        }
        Expr::Lambda(lambda) => {
            let severian_ast::LambdaBody::Expr(body) = &lambda.body else {
                return Err(error(
                    lambda.span,
                    "lambda blocks require an expression body",
                ));
            };
            let mut lambda_scope = scope.clone();
            let mut params = Vec::new();
            for parameter in &lambda.params {
                lambda_scope.insert(
                    parameter.name.name.clone(),
                    Binding {
                        ty: ValueType::Any,
                        function_return: None,
                        collection_len: None,
                        mutable: false,
                        field: false,
                        integer_max: None,
                        known_integer: None,
                    },
                );
                params.push(parameter.name.name.clone());
            }
            let body = lower_expression(body, &lambda_scope, signatures, aliases)?.0;
            Ok((
                Expression::Lambda {
                    params,
                    body: Box::new(body),
                },
                ValueType::Function,
            ))
        }
        Expr::ChaosRule(rule) => {
            let (function, return_type) =
                lower_expression(&rule.function, scope, signatures, aliases)?;
            let Expression::Function(function) = function.into_kind() else {
                return Err(error(
                    rule.function.span(),
                    "chaos injection target must be a function",
                ));
            };
            let (value, value_type) = lower_expression(&rule.value, scope, signatures, aliases)?;
            if rule.action == severian_ast::ChaosAction::Return {
                let declared_return = signatures
                    .get(&function)
                    .map_or(return_type, |signature| signature.returns);
                compatible(rule.value.span(), value_type, declared_return)?;
            }
            Ok((
                Expression::ChaosRule {
                    function,
                    action: match rule.action {
                        severian_ast::ChaosAction::Return => HirChaosAction::Return,
                        severian_ast::ChaosAction::Throw => HirChaosAction::Throw,
                    },
                    value: Box::new(value),
                },
                ValueType::Any,
            ))
        }
        _ => Err(error(
            expression.span(),
            "expression is not supported in this compiler slice yet",
        )),
    }
}

fn add_test_bindings(scope: &mut HashMap<String, Binding>, modes: &[severian_ast::TestMode]) {
    scope.insert(
        "chaos".into(),
        Binding {
            ty: ValueType::List,
            function_return: None,
            collection_len: None,
            mutable: false,
            field: false,
            integer_max: None,
            known_integer: None,
        },
    );
    if modes.contains(&severian_ast::TestMode::Integration) {
        for name in ["stdout", "stderr"] {
            scope.insert(
                name.into(),
                Binding {
                    ty: ValueType::String,
                    function_return: None,
                    collection_len: None,
                    mutable: false,
                    field: false,
                    integer_max: None,
                    known_integer: None,
                },
            );
        }
    }
}

fn lower_call(
    call: &severian_ast::CallExpr,
    scope: &HashMap<String, Binding>,
    signatures: &HashMap<String, Signature>,
    aliases: &HashMap<String, String>,
) -> Result<(Expression, ValueType), SemanticError> {
    if let Expr::Index(index) = call.callee.as_ref() {
        if let Expr::Identifier(callee) = index.object.as_ref() {
            let imported = aliases
                .get(&callee.name)
                .map(String::as_str)
                .unwrap_or(&callee.name);
            let canonical = resolve_linked_function(imported, signatures);
            if let Some(signature) = signatures.get(canonical) {
                return lower_declared_call(call, canonical, signature, scope, signatures, aliases);
            }
        }
        if let Expr::Member(member) = index.object.as_ref() {
            if let Expr::Identifier(object) = member.object.as_ref() {
                if let Some(module) = aliases.get(&object.name) {
                    let function = format!("{module}.{}", member.member.name);
                    let canonical = resolve_linked_function(&function, signatures);
                    let signature = signatures.get(canonical).ok_or_else(|| {
                        error(call.span, format!("unknown exported function `{function}`"))
                    })?;
                    return lower_declared_call(
                        call, canonical, signature, scope, signatures, aliases,
                    );
                }
            }
        }
    }
    if let Expr::Member(member) = call.callee.as_ref() {
        if let Expr::Identifier(object) = member.object.as_ref() {
            if object.name == "int" && member.member.name == "parse" {
                let args = call
                    .args
                    .iter()
                    .map(|arg| {
                        lower_expression(&arg.value, scope, signatures, aliases).map(|(arg, _)| arg)
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                return Ok((
                    Expression::Call {
                        target: CallTarget::source("int.parse"),
                        args,
                    },
                    ValueType::Result,
                ));
            }
            if object.name == "http" && member.member.name == "get" {
                let args = call
                    .args
                    .iter()
                    .map(|arg| {
                        lower_expression(&arg.value, scope, signatures, aliases).map(|(arg, _)| arg)
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                return Ok((
                    Expression::Call {
                        target: CallTarget::source("http.get"),
                        args,
                    },
                    ValueType::Result,
                ));
            }
            if member.member.name == "zero" && !scope.contains_key(&object.name) {
                return Ok((
                    Expression::Call {
                        target: CallTarget::source("Number.zero"),
                        args: Vec::new(),
                    },
                    ValueType::Any,
                ));
            }
            if let Some(module) = aliases.get(&object.name) {
                let function = format!("{module}.{}", member.member.name);
                let canonical = resolve_linked_function(&function, signatures);
                if let Some(signature) = signatures.get(canonical) {
                    return lower_declared_call(
                        call, canonical, signature, scope, signatures, aliases,
                    );
                }
                if let Some(class) = aliases.get(&format!("__module_class.{function}")) {
                    let args = call
                        .args
                        .iter()
                        .map(|arg| {
                            lower_expression(&arg.value, scope, signatures, aliases)
                                .map(|(arg, _)| arg)
                        })
                        .collect::<Result<Vec<_>, _>>()?;
                    return Ok((
                        Expression::Construct {
                            type_id: TypeDefinitionId::from_name(class),
                            class: class.clone(),
                            args,
                        },
                        ValueType::Any,
                    ));
                }
                return Err(error(
                    call.span,
                    format!("unknown exported function or class `{function}`"),
                ));
            }
        }
        let (object, object_type) = lower_expression(&member.object, scope, signatures, aliases)?;
        let lowered_args = call
            .args
            .iter()
            .map(|arg| lower_expression(&arg.value, scope, signatures, aliases))
            .collect::<Result<Vec<_>, _>>()?;
        validate_builtin_method(
            call.span,
            object_type,
            &member.member.name,
            &lowered_args.iter().map(|(_, ty)| *ty).collect::<Vec<_>>(),
        )?;
        let args = lowered_args.into_iter().map(|(arg, _)| arg).collect();
        let return_type = method_return_type(object_type, &member.member.name);
        return Ok((
            Expression::MethodCall {
                object: Box::new(object),
                method: member.member.name.clone(),
                args,
            },
            return_type,
        ));
    }
    let Expr::Identifier(callee) = call.callee.as_ref() else {
        let (callee, ty) = lower_expression(&call.callee, scope, signatures, aliases)?;
        if ty != ValueType::Function {
            return Err(error(call.callee.span(), "value is not callable"));
        }
        let args = call
            .args
            .iter()
            .map(|arg| lower_expression(&arg.value, scope, signatures, aliases).map(|(arg, _)| arg))
            .collect::<Result<Vec<_>, _>>()?;
        return Ok((
            Expression::CallValue {
                callee: Box::new(callee),
                args,
                return_type: ValueType::Any,
            },
            ValueType::Any,
        ));
    };
    let imported = if signatures.contains_key(&callee.name) {
        callee.name.as_str()
    } else {
        aliases
            .get(&callee.name)
            .map(String::as_str)
            .unwrap_or(&callee.name)
    };
    // Path dependencies are compiled into the same translation unit. Their
    // functions therefore have source-level names, while `from package import`
    // records a package-qualified alias. Prefer a real qualified interface when
    // one exists, then fall back to the linked source implementation.
    let canonical = resolve_linked_function(imported, signatures);
    if let Some(class) = aliases.get(&format!("__module_class.{imported}")) {
        let args = call
            .args
            .iter()
            .map(|arg| lower_expression(&arg.value, scope, signatures, aliases).map(|(arg, _)| arg))
            .collect::<Result<Vec<_>, _>>()?;
        return Ok((
            Expression::Construct {
                type_id: TypeDefinitionId::from_name(class),
                class: class.clone(),
                args,
            },
            ValueType::Any,
        ));
    }
    if let Some(class) = canonical.strip_prefix("__class.") {
        let args = call
            .args
            .iter()
            .map(|arg| lower_expression(&arg.value, scope, signatures, aliases).map(|(arg, _)| arg))
            .collect::<Result<Vec<_>, _>>()?;
        return Ok((
            Expression::Construct {
                type_id: TypeDefinitionId::from_name(class),
                class: class.into(),
                args,
            },
            ValueType::Any,
        ));
    }
    let builtin = match canonical {
        "print" | "io.print" => Some(("print", ValueType::Unit)),
        "panic" => Some(("panic", ValueType::Unit)),
        "float" => Some(("float", ValueType::Float)),
        "string" => Some(("string", ValueType::String)),
        "range" => Some(("range", ValueType::List)),
        "indices" => Some(("indices", ValueType::List)),
        "enumerate" => Some(("enumerate", ValueType::List)),
        "zip" => Some(("zip", ValueType::List)),
        "any" => Some(("any", ValueType::Bool)),
        "all" => Some(("all", ValueType::Bool)),
        "abs" | "min" | "max" => Some((canonical, ValueType::Any)),
        "divmod" => Some(("divmod", ValueType::Tuple)),
        "read" if !signatures.contains_key(&callee.name) => Some(("read", ValueType::Result)),
        "len" | "size" | "bytes" | "bits" | "capacity" => Some((canonical, ValueType::Int)),
        "present" => {
            let fields = call
                .args
                .iter()
                .map(|arg| {
                    lower_expression(&arg.value, scope, signatures, aliases).map(|(arg, _)| arg)
                })
                .collect::<Result<Vec<_>, _>>()?;
            return Ok((
                Expression::Variant {
                    type_id: Some(TypeDefinitionId::from_name("Option")),
                    variant_id: VariantId::from_name("present"),
                    name: "present".into(),
                    fields,
                },
                ValueType::Option,
            ));
        }
        "failure" => {
            let fields = call
                .args
                .iter()
                .map(|arg| {
                    lower_expression(&arg.value, scope, signatures, aliases).map(|(arg, _)| arg)
                })
                .collect::<Result<Vec<_>, _>>()?;
            return Ok((
                Expression::Variant {
                    type_id: Some(TypeDefinitionId::from_name("Result")),
                    variant_id: VariantId::from_name("failure"),
                    name: "failure".into(),
                    fields,
                },
                ValueType::Result,
            ));
        }
        "__format" => {
            let Expr::Literal(Literal::String { value, .. }) = &call.args[0].value else {
                unreachable!()
            };
            let (args, arg_types) = lower_format_args(value, scope, call.span)?;
            return Ok((
                Expression::Format {
                    template: value.clone(),
                    args,
                    arg_types,
                },
                ValueType::String,
            ));
        }
        _ => None,
    };
    if let Some((name, returns)) = builtin {
        let lowered = call
            .args
            .iter()
            .map(|arg| lower_expression(&arg.value, scope, signatures, aliases))
            .collect::<Result<Vec<_>, _>>()?;
        let types = lowered.iter().map(|(_, ty)| *ty).collect::<Vec<_>>();
        let valid_arity = match name {
            "range" => (1..=3).contains(&lowered.len()),
            "zip" => lowered.len() == 2,
            "min" | "max" | "divmod" => lowered.len() == 2,
            "print" | "panic" => !lowered.is_empty(),
            _ => lowered.len() == 1,
        };
        if !valid_arity {
            return Err(error(
                call.span,
                format!("builtin `{name}` received an invalid number of arguments"),
            ));
        }
        if name == "range"
            && types
                .iter()
                .any(|ty| !matches!(ty, ValueType::Int | ValueType::Any))
        {
            return Err(error(call.span, "`range` arguments must be integers"));
        }
        if matches!(name, "enumerate" | "any" | "all")
            && !matches!(
                types[0],
                ValueType::List | ValueType::Tuple | ValueType::Set
            )
        {
            return Err(error(call.span, format!("`{name}` expects an iterable")));
        }
        if name == "zip"
            && types
                .iter()
                .any(|ty| !matches!(ty, ValueType::List | ValueType::Tuple | ValueType::Set))
        {
            return Err(error(call.span, "`zip` expects two iterables"));
        }
        let args = lowered.into_iter().map(|(arg, _)| arg).collect();
        return Ok((
            Expression::Call {
                target: CallTarget::source(name),
                args,
            },
            returns,
        ));
    }
    if let Some(signature) = signatures.get(canonical) {
        return lower_declared_call(call, canonical, signature, scope, signatures, aliases);
    }
    if scope
        .get(&callee.name)
        .is_some_and(|binding| binding.ty == ValueType::Function)
    {
        let return_type = scope
            .get(&callee.name)
            .and_then(|binding| binding.function_return)
            .unwrap_or(ValueType::Any);
        let args = call
            .args
            .iter()
            .map(|arg| lower_expression(&arg.value, scope, signatures, aliases).map(|(arg, _)| arg))
            .collect::<Result<Vec<_>, _>>()?;
        return Ok((
            Expression::CallValue {
                callee: Box::new(Expression::Variable(callee.name.clone())),
                args,
                return_type,
            },
            return_type,
        ));
    }
    if callee
        .name
        .as_bytes()
        .first()
        .is_some_and(u8::is_ascii_uppercase)
    {
        let fields = call
            .args
            .iter()
            .map(|arg| lower_expression(&arg.value, scope, signatures, aliases).map(|(arg, _)| arg))
            .collect::<Result<Vec<_>, _>>()?;
        return Ok((
            Expression::Variant {
                type_id: None,
                variant_id: VariantId::from_name(&callee.name),
                name: callee.name.clone(),
                fields,
            },
            ValueType::Any,
        ));
    }
    Err(error(
        callee.span,
        format!("unknown function `{}`", callee.name),
    ))
}

fn resolve_linked_function<'a>(
    imported: &'a str,
    signatures: &HashMap<String, Signature>,
) -> &'a str {
    if signatures.contains_key(imported) {
        imported
    } else {
        imported
            .rsplit_once('.')
            .map(|(_, function)| function)
            .filter(|function| signatures.contains_key(*function))
            .unwrap_or(imported)
    }
}

fn method_return_type(object: ValueType, method: &str) -> ValueType {
    match (object, method) {
        (ValueType::String, "characters" | "words" | "split")
        | (ValueType::List, "reversed" | "sorted" | "map" | "filter")
        | (ValueType::Map, "keys" | "values")
        | (ValueType::Set, "toList") => ValueType::List,
        (ValueType::String | ValueType::List, "frequencies") => ValueType::Map,
        (ValueType::List, "toSet") | (ValueType::Set, "difference") => ValueType::Set,
        (ValueType::List, "join") => ValueType::String,
        (ValueType::List, "reduce") => ValueType::Any,
        (ValueType::String, "length" | "find" | "count") => ValueType::Int,
        (
            ValueType::String
            | ValueType::List
            | ValueType::Tuple
            | ValueType::Map
            | ValueType::Set
            | ValueType::Tensor(_)
            | ValueType::TensorAny,
            "len" | "size" | "bytes" | "bits" | "capacity",
        ) => ValueType::Int,
        (ValueType::String, "startsWith" | "endsWith") => ValueType::Bool,
        (ValueType::String, "strip" | "lower" | "upper" | "replace") => ValueType::String,
        (
            ValueType::List,
            "append" | "appendleft" | "extend" | "insert" | "remove" | "heapPush",
        ) => ValueType::Unit,
        (ValueType::Set, "union" | "intersection" | "symmetricDifference") => ValueType::Set,
        _ => ValueType::Any,
    }
}

fn collection_shape_mutating_method(method: &str) -> bool {
    matches!(
        method,
        "append"
            | "appendleft"
            | "extend"
            | "insert"
            | "remove"
            | "pop"
            | "popleft"
            | "heapPush"
            | "heapPop"
            | "clear"
    )
}

fn validate_builtin_method(
    span: Span,
    object: ValueType,
    method: &str,
    args: &[ValueType],
) -> Result<(), SemanticError> {
    let arity = match (object, method) {
        (ValueType::List, "pop") => Some(0..=1),
        (ValueType::List, "sorted") => Some(0..=2),
        (ValueType::List, "reduce") => Some(1..=2),
        (
            ValueType::List,
            "append" | "appendleft" | "extend" | "remove" | "heapPush" | "join" | "map" | "filter",
        ) => Some(1..=1),
        (ValueType::List, "insert") => Some(2..=2),
        (
            ValueType::List,
            "popleft" | "heapPop" | "last" | "reversed" | "sum" | "minimum" | "maximum"
            | "frequencies" | "toSet",
        ) => Some(0..=0),
        (ValueType::Set, "union" | "intersection" | "difference" | "symmetricDifference") => {
            Some(1..=1)
        }
        (ValueType::Set, "toList") => Some(0..=0),
        (ValueType::Map, "get" | "setDefault") => Some(2..=2),
        (ValueType::Map, "keys" | "values") => Some(0..=0),
        (
            ValueType::String,
            "characters" | "words" | "frequencies" | "strip" | "lower" | "upper" | "length",
        ) => Some(0..=0),
        (ValueType::String, "split" | "startsWith" | "endsWith" | "find" | "count") => Some(1..=1),
        (ValueType::String, "replace") => Some(2..=2),
        (
            ValueType::String
            | ValueType::List
            | ValueType::Tuple
            | ValueType::Map
            | ValueType::Set
            | ValueType::Tensor(_)
            | ValueType::TensorAny,
            "len" | "size" | "bytes" | "bits" | "capacity",
        ) => Some(0..=0),
        _ => None,
    };
    if let Some(arity) = arity {
        if !arity.contains(&args.len()) {
            return Err(error(
                span,
                format!("method `{method}` received an invalid number of arguments"),
            ));
        }
    }
    if object == ValueType::List && matches!(method, "map" | "filter" | "reduce") {
        if !matches!(args.first(), Some(ValueType::Function | ValueType::Any)) {
            return Err(error(span, format!("method `{method}` expects a callable")));
        }
    }
    if object == ValueType::List && method == "sorted" && !args.is_empty() {
        if !matches!(
            args[0],
            ValueType::Bool | ValueType::Function | ValueType::Any
        ) {
            return Err(error(
                span,
                "method `sorted` expects a reverse flag or key callable",
            ));
        }
        if args.len() == 2 && args[1] != ValueType::Bool && args[1] != ValueType::Any {
            return Err(error(
                span,
                "method `sorted` expects a boolean reverse flag",
            ));
        }
    }
    Ok(())
}

fn propagate_unknown_collection_shapes(
    outer: &mut HashMap<String, Binding>,
    inner: &HashMap<String, Binding>,
) {
    for (name, binding) in outer.iter_mut() {
        if binding.collection_len.is_some()
            && inner
                .get(name)
                .is_some_and(|inner| inner.collection_len.is_none())
        {
            binding.collection_len = None;
        }
    }
}

fn lower_declared_call(
    call: &severian_ast::CallExpr,
    function: &str,
    signature: &Signature,
    scope: &HashMap<String, Binding>,
    signatures: &HashMap<String, Signature>,
    aliases: &HashMap<String, String>,
) -> Result<(Expression, ValueType), SemanticError> {
    let mut supplied: Vec<Option<Expression>> = vec![None; signature.params.len()];
    let mut positional = 0;
    for argument in &call.args {
        let index = if let Some(name) = &argument.name {
            signature
                .params
                .iter()
                .position(|param| param.name == name.name)
                .ok_or_else(|| error(name.span, format!("unknown argument `{}`", name.name)))?
        } else {
            let index = positional;
            positional += 1;
            index
        };
        if index >= supplied.len() || supplied[index].is_some() {
            return Err(error(
                argument.span,
                format!("invalid arguments for `{function}`"),
            ));
        }
        let (value, ty) = lower_expression(&argument.value, scope, signatures, aliases)?;
        compatible(argument.span, ty, signature.params[index].ty)?;
        supplied[index] = Some(value);
    }
    let args = supplied
        .into_iter()
        .zip(&signature.params)
        .map(|(value, param)| {
            if let Some(value) = value {
                Ok(value)
            } else if let Some(default) = &param.default {
                lower_expression(default, scope, signatures, aliases).map(|(value, _)| value)
            } else {
                Err(error(
                    call.span,
                    format!("missing argument `{}`", param.name),
                ))
            }
        })
        .collect::<Result<Vec<_>, _>>()?;
    let external_operation = aliases.contains_key(&format!("__external_function.{function}"));
    let linked_function = if external_operation {
        function
    } else {
        function
            .rsplit_once('.')
            .map(|(_, name)| name)
            .filter(|name| signatures.contains_key(*name))
            .unwrap_or(function)
    };
    let runtime_function = match linked_function {
        // The MLIR backend currently exposes this intrinsic under its C symbol.
        // Its type comes from library/math, not from this mapping.
        "math.sqrt" => "sqrt",
        _ => linked_function,
    };
    Ok((
        Expression::Call {
            target: if runtime_function == linked_function {
                signature.target.clone()
            } else {
                CallTarget::source(runtime_function)
            },
            args,
        },
        signature.returns,
    ))
}

fn collection_length(expression: &Expr) -> Option<usize> {
    match expression {
        Expr::List(collection) | Expr::Tuple(collection) => Some(collection.elements.len()),
        _ => None,
    }
}

fn constant_integer(expression: &Expr) -> Option<i64> {
    match expression {
        Expr::Literal(Literal::Integer { value, .. }) => Some(*value),
        Expr::Binary(binary) => {
            let left = constant_integer(&binary.left)?;
            let right = constant_integer(&binary.right)?;
            match binary.op {
                AstBinaryOp::Add => left.checked_add(right),
                AstBinaryOp::Sub => left.checked_sub(right),
                AstBinaryOp::Mul => left.checked_mul(right),
                _ => None,
            }
        }
        _ => None,
    }
}

fn named_type_is(ty: &Type, expected: &str) -> bool {
    matches!(ty, Type::Named(path) if path.segments.first().is_some_and(|segment| segment.name == expected))
}

fn checked_integer_overflow(expression: &Expr, scope: &HashMap<String, Binding>) -> bool {
    let Expr::Binary(binary) = expression else {
        return false;
    };
    let (binding, constant, operation) = match (binary.left.as_ref(), binary.right.as_ref()) {
        (Expr::Identifier(identifier), right) => (
            scope.get(&identifier.name),
            constant_integer(right),
            binary.op,
        ),
        _ => return false,
    };
    let (Some(constant), Some(value), Some(maximum)) = (
        constant,
        binding.and_then(|binding| binding.known_integer),
        binding.and_then(|binding| binding.integer_max),
    ) else {
        return false;
    };
    let result = match operation {
        AstBinaryOp::Add => value.checked_add(constant),
        AstBinaryOp::Sub => value.checked_sub(constant),
        AstBinaryOp::Mul => value.checked_mul(constant),
        _ => return false,
    };
    result.is_none_or(|result| !(0..=maximum).contains(&result))
}

fn inclusive_collection_range(expression: &Expr) -> bool {
    let Expr::Call(call) = expression else {
        return false;
    };
    let Expr::Identifier(callee) = call.callee.as_ref() else {
        return false;
    };
    if callee.name != "range" || call.args.len() != 2 {
        return false;
    }
    let Expr::Binary(end) = &call.args[1].value else {
        return false;
    };
    if end.op != AstBinaryOp::Add || constant_integer(&end.right) != Some(1) {
        return false;
    }
    let Expr::Call(size) = end.left.as_ref() else {
        return false;
    };
    matches!(size.callee.as_ref(), Expr::Identifier(name) if name.name == "size")
}

fn lower_format_args(
    template: &str,
    scope: &HashMap<String, Binding>,
    span: Span,
) -> Result<(Vec<Expression>, Vec<ValueType>), SemanticError> {
    let mut args = Vec::new();
    let mut arg_types = Vec::new();
    let mut remainder = template;

    while let Some(open) = remainder.find('{') {
        remainder = &remainder[open + 1..];
        let close = remainder
            .find('}')
            .ok_or_else(|| error(span, "formatted string has an unmatched `{`"))?;
        let name = &remainder[..close];
        if name.is_empty()
            || !name.bytes().enumerate().all(|(index, byte)| {
                byte == b'_'
                    || byte.is_ascii_alphanumeric() && (index > 0 || !byte.is_ascii_digit())
            })
        {
            return Err(error(
                span,
                format!("unsupported formatted string field `{{{name}}}`"),
            ));
        }
        let binding = scope
            .get(name)
            .ok_or_else(|| error(span, format!("unknown formatted string field `{name}`")))?;
        args.push(Expression::Variable(name.into()));
        arg_types.push(binding.ty);
        remainder = &remainder[close + 1..];
    }

    if remainder.contains('}') {
        return Err(error(span, "formatted string has an unmatched `}`"));
    }
    Ok((args, arg_types))
}

fn lower_comprehension_clauses(
    clauses: &[severian_ast::ComprehensionClause],
    scope: &mut HashMap<String, Binding>,
    signatures: &HashMap<String, Signature>,
    aliases: &HashMap<String, String>,
) -> Result<Vec<HirComprehensionClause>, SemanticError> {
    let mut lowered = Vec::new();
    for clause in clauses {
        let iterable = lower_expression(&clause.iterable, scope, signatures, aliases)?.0;
        let pattern = lower_pattern(&clause.pattern, scope, aliases)?;
        let condition = clause
            .condition
            .as_ref()
            .map(|condition| {
                let (condition, ty) = lower_expression(condition, scope, signatures, aliases)?;
                compatible(clause.iterable.span(), ty, ValueType::Bool)?;
                Ok(condition)
            })
            .transpose()?;
        lowered.push(HirComprehensionClause {
            pattern,
            iterable,
            condition,
        });
    }
    Ok(lowered)
}

fn lower_collection(
    elements: &[Expr],
    scope: &HashMap<String, Binding>,
    signatures: &HashMap<String, Signature>,
    aliases: &HashMap<String, String>,
) -> Result<Vec<Expression>, SemanticError> {
    elements
        .iter()
        .map(|element| {
            lower_expression(element, scope, signatures, aliases).map(|(element, _)| element)
        })
        .collect()
}

fn lower_signature(
    name: &str,
    native_symbol: Option<&str>,
    params: &[severian_ast::Parameter],
    return_type: Option<&Type>,
) -> Result<Signature, SemanticError> {
    let params = params
        .iter()
        .map(|param| {
            Ok(SignatureParameter {
                name: param.name.name.clone(),
                ty: param
                    .ty
                    .as_ref()
                    .ok_or_else(|| error(param.span, "parameters require a type"))
                    .and_then(lower_type)?,
                function_return: function_return_type(param.ty.as_ref()),
                default: param.default.clone(),
            })
        })
        .collect::<Result<Vec<_>, SemanticError>>()?;
    let returns = return_type
        .map(lower_type)
        .transpose()?
        .unwrap_or(ValueType::Unit);
    let target = match native_symbol {
        Some(symbol) => CallTarget::native(name, symbol),
        None => CallTarget::source(name),
    }
    .with_signature(params.iter().map(|param| param.ty), returns);
    Ok(Signature {
        target,
        params,
        returns,
    })
}

fn function_return_type(ty: Option<&Type>) -> Option<ValueType> {
    let Type::Named(path) = ty? else {
        return None;
    };
    if path.segments.first()?.name != "fn" {
        return None;
    }
    path.args
        .last()
        .and_then(TypeArg::as_type)
        .and_then(|argument| lower_type(argument).ok())
}

fn lower_type(ty: &Type) -> Result<ValueType, SemanticError> {
    match ty {
        Type::Union { .. } => Ok(ValueType::Any),
        Type::Named(path) => {
            let name = path
                .segments
                .first()
                .map(|segment| segment.name.as_str())
                .unwrap_or("");
            match name {
                "int" | "u8" | "u16" | "u32" | "u64" | "usize" => Ok(ValueType::Int),
                "float" | "f32" | "f64" => Ok(ValueType::Float),
                "bool" => Ok(ValueType::Bool),
                "string" => Ok(ValueType::String),
                "unit" => Ok(ValueType::Unit),
                "list" => Ok(ValueType::List),
                "map" => Ok(ValueType::Map),
                "set" => Ok(ValueType::Set),
                "Tensor"
                    if matches!(
                        path.args.first().and_then(TypeArg::as_type),
                        Some(Type::Named(element))
                            if element.segments.first().is_some_and(|part| part.name == "type")
                    ) =>
                {
                    Ok(ValueType::TensorAny)
                }
                "Tensor" => Ok(ValueType::Tensor(lower_tensor_type(path)?)),
                "Channel" => Ok(ValueType::Channel),
                "fn" => Ok(ValueType::Function),
                "Result" => Ok(ValueType::Result),
                "Option" => Ok(ValueType::Option),
                _ => Ok(ValueType::Any),
            }
        }
        _ => Err(error(ty.span(), "type is not supported yet")),
    }
}

fn lower_tensor_type(path: &severian_ast::TypePath) -> Result<TensorType, SemanticError> {
    let element = match path.args.first().and_then(TypeArg::as_type) {
        Some(Type::Named(element)) => match element.segments.first().map(|part| part.name.as_str())
        {
            Some("bf16" | "bfloat16") => TensorElementType::BF16,
            Some("f32") => TensorElementType::F32,
            Some("f64" | "float") => TensorElementType::F64,
            Some("i32") => TensorElementType::I32,
            Some("i64" | "int") => TensorElementType::I64,
            _ => {
                return Err(error(
                    path.span,
                    "tensor elements must be bf16, f32, f64, i32, or i64",
                ))
            }
        },
        None if path.args.is_empty() => TensorElementType::F64,
        _ => {
            return Err(error(
                path.span,
                "the first Tensor argument must be an element type",
            ))
        }
    };
    if path.args.len() <= 1 {
        return Ok(TensorType::dynamic(element));
    }
    let mut dimensions = Vec::with_capacity(path.args.len() - 1);
    for argument in &path.args[1..] {
        match argument {
            TypeArg::Dimension { size, .. } => dimensions.push(TensorDimension::Static(*size)),
            TypeArg::Type { ty, .. } if matches!(ty.as_ref(), Type::Named(name) if name.segments.first().is_some_and(|part| part.name == "dynamic")) => {
                dimensions.push(TensorDimension::Dynamic)
            }
            _ => {
                return Err(error(
                    argument.span(),
                    "tensor dimensions must be integers or `dynamic`",
                ))
            }
        }
    }
    TensorType::ranked(element, &dimensions).map_err(|message| error(path.span, message))
}

fn merge_numeric(
    left: ValueType,
    right: ValueType,
    span: Span,
) -> Result<ValueType, SemanticError> {
    if left == ValueType::Any || right == ValueType::Any {
        return Ok(ValueType::Any);
    }
    if left == right && matches!(left, ValueType::Int | ValueType::Float | ValueType::String) {
        Ok(left)
    } else {
        Err(error(span, "operator requires matching numeric values"))
    }
}

fn power_type(
    base: ValueType,
    exponent: ValueType,
    span: Span,
) -> Result<ValueType, SemanticError> {
    if base == ValueType::Any || exponent == ValueType::Any {
        return Ok(ValueType::Any);
    }
    if base == ValueType::Int && exponent == ValueType::Int {
        return Ok(ValueType::Int);
    }
    if matches!(base, ValueType::Int | ValueType::Float)
        && matches!(exponent, ValueType::Int | ValueType::Float)
    {
        return Ok(ValueType::Float);
    }
    Err(error(span, "power requires numeric values"))
}

fn compatible(span: Span, actual: ValueType, expected: ValueType) -> Result<(), SemanticError> {
    if actual == expected
        || actual == ValueType::Any
        || expected == ValueType::Any
        || matches!(
            (actual, expected),
            (ValueType::Tensor(_), ValueType::TensorAny)
        )
        || matches!((actual, expected), (ValueType::Tensor(actual), ValueType::Tensor(expected)) if actual.is_compatible_with(expected))
        || (expected == ValueType::Result && actual != ValueType::Unit)
    {
        Ok(())
    } else {
        Err(error(
            span,
            format!("expected {expected:?}, found {actual:?}"),
        ))
    }
}

fn always_returns(instructions: &[Instruction]) -> bool {
    instructions.iter().any(|instruction| match instruction {
        Instruction::Return(_) => true,
        Instruction::If {
            then_instructions,
            else_instructions,
            ..
        } => always_returns(then_instructions) && always_returns(else_instructions),
        Instruction::Switch { arms, .. } => {
            !arms.is_empty() && arms.iter().all(|arm| always_returns(&arm.instructions))
        }
        Instruction::With { instructions, .. } => always_returns(instructions),
        Instruction::While { condition, .. }
            if matches!(condition.kind(), Expression::Boolean(true)) =>
        {
            true
        }
        _ => false,
    })
}

fn is_upper_camel_case(name: &str) -> bool {
    name.as_bytes().first().is_some_and(u8::is_ascii_uppercase)
        && name.bytes().all(|byte| byte.is_ascii_alphanumeric())
}

fn value_span(value: &Option<Expr>) -> Span {
    value.as_ref().map_or(Span::dummy(), Expr::span)
}

fn lower_test_modes(modes: &[severian_ast::TestMode]) -> Vec<HirTestMode> {
    modes
        .iter()
        .map(|mode| match mode {
            severian_ast::TestMode::Property => HirTestMode::Property,
            severian_ast::TestMode::Bench => HirTestMode::Bench,
            severian_ast::TestMode::Chaos => HirTestMode::Chaos,
            severian_ast::TestMode::Integration => HirTestMode::Integration,
        })
        .collect()
}

fn error(span: Span, message: impl Into<String>) -> SemanticError {
    SemanticError {
        span,
        message: message.into(),
    }
}

fn lower_pattern(
    pattern: &Pattern,
    scope: &mut HashMap<String, Binding>,
    aliases: &HashMap<String, String>,
) -> Result<MatchPattern, SemanticError> {
    match pattern {
        Pattern::Wildcard(_) => Ok(MatchPattern::Wildcard),
        Pattern::Identifier(name) => {
            scope.insert(
                name.name.clone(),
                Binding {
                    ty: ValueType::Any,
                    function_return: None,
                    collection_len: None,
                    mutable: false,
                    field: false,
                    integer_max: None,
                    known_integer: None,
                },
            );
            Ok(MatchPattern::Bind(name.name.clone()))
        }
        Pattern::Literal(Literal::Integer { value, .. }) => Ok(MatchPattern::Integer(*value)),
        Pattern::Literal(Literal::Float { value, .. }) => Ok(MatchPattern::Float(value.to_bits())),
        Pattern::Literal(Literal::Boolean { value, .. }) => Ok(MatchPattern::Boolean(*value)),
        Pattern::Literal(Literal::String { value, .. }) => Ok(MatchPattern::String(value.clone())),
        Pattern::Constructor { name, fields, .. } => {
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
                            .map(|field| {
                                scope.insert(
                                    field.into(),
                                    Binding {
                                        ty: ValueType::Any,
                                        function_return: None,
                                        collection_len: None,
                                        mutable: false,
                                        field: false,
                                        integer_max: None,
                                        known_integer: None,
                                    },
                                );
                                MatchPattern::Bind(field.into())
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
                scope.insert(
                    name.clone(),
                    Binding {
                        ty: ValueType::Any,
                        function_return: None,
                        collection_len: None,
                        mutable: false,
                        field: false,
                        integer_max: None,
                        known_integer: None,
                    },
                );
                return Ok(MatchPattern::Bind(name));
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
