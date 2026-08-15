use super::*;

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
    attach_module_metadata_with_packages(module, program, path, source, namespace, &[]);
}

pub fn attach_module_metadata_with_packages(
    module: &Module,
    program: &mut Program,
    path: impl Into<PathBuf>,
    source: impl Into<String>,
    namespace: Option<&str>,
    interfaces: &[PackageInterface],
) {
    let mut metadata = std::mem::take(&mut program.metadata);
    attach_module_metadata_to_with_packages(
        module,
        program,
        &mut metadata,
        path,
        source,
        namespace,
        interfaces,
    );
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
    attach_module_metadata_to_with_packages(
        module,
        program,
        metadata,
        path,
        source,
        namespace,
        &[],
    );
}

pub fn attach_module_metadata_to_with_packages(
    module: &Module,
    program: &mut Program,
    metadata: &mut ProgramMetadata,
    path: impl Into<PathBuf>,
    source: impl Into<String>,
    namespace: Option<&str>,
    interfaces: &[PackageInterface],
) {
    let specialized = specialize_generic_classes_with_interfaces(module, interfaces)
        .expect("semantic analysis already validated generic class specializations");
    attach_specialized_module_metadata_to_with_packages(
        &specialized,
        program,
        metadata,
        path,
        source,
        namespace,
        interfaces,
    );
}

fn attach_specialized_module_metadata_to_with_packages(
    module: &Module,
    program: &mut Program,
    metadata: &mut ProgramMetadata,
    path: impl Into<PathBuf>,
    source: impl Into<String>,
    namespace: Option<&str>,
    interfaces: &[PackageInterface],
) {
    let file = program.attach_source_file_to(metadata, path, source);
    let mut known_types = module
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
    let imports = collect_imports(module);
    for interface in interfaces {
        for item in &interface.module.items {
            let type_name = match item {
                Item::Class(class) => &class.name.name,
                Item::Trait(trait_) => &trait_.name.name,
                Item::Enum(enumeration) => &enumeration.name.name,
                _ => continue,
            };
            let canonical = format!("{}.{}", interface.name, type_name);
            let id = TypeDefinitionId::from_name(&canonical);
            known_types.insert(canonical.clone(), id);
            if let Some(package) = &interface.export_package {
                known_types.insert(format!("{package}.{type_name}"), id);
            }
            for (exposed, target) in &imports {
                if target == &canonical {
                    known_types.insert(exposed.clone(), id);
                } else if target == &interface.name
                    || interface
                        .export_package
                        .as_ref()
                        .is_some_and(|package| target == package)
                {
                    known_types.insert(format!("{exposed}.{type_name}"), id);
                }
            }
        }
    }
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
                            transitions: variant
                                .transitions
                                .iter()
                                .map(|target| VariantId::from_name(&target.name))
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

pub(super) fn register_constructor_metadata(
    constructor: &severian_ast::ConstructorDecl,
    class: &str,
    namespace: Option<&str>,
    file: severian_hir::SourceFileId,
    known_types: &HashMap<String, TypeDefinitionId>,
    metadata: &mut ProgramMetadata,
) {
    let id = constructor_id(class, &constructor.name.name, constructor.span);
    let id = namespace.map_or(id, |namespace| id.in_namespace(namespace));
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

pub(super) fn register_function_metadata(
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

pub(super) fn intern_optional_type(
    ty: Option<&Type>,
    known_types: &HashMap<String, TypeDefinitionId>,
    types: &mut TypeTable,
) -> TypeId {
    match ty {
        Some(ty) => intern_type(ty, known_types, types),
        None => types.intern(TypeKind::Any),
    }
}

pub(super) fn intern_type(
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
                "int" | "i8" | "i16" | "i32" | "i64" | "isize" | "u8" | "u16" | "u32" | "u64"
                | "usize" => types.intern(TypeKind::Int),
                "float" | "f32" | "f64" => types.intern(TypeKind::Float),
                "bool" => types.intern(TypeKind::Bool),
                "string" => types.intern(TypeKind::String),
                "unit" => types.intern(TypeKind::Unit),
                "Any" | "any" => types.intern(TypeKind::Any),
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
                "Function" => {
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

pub(super) fn metadata_type_id(namespace: Option<&str>, name: &str) -> TypeDefinitionId {
    TypeDefinitionId::from_name(&qualified_name(namespace, name))
}

pub(super) fn qualified_name(namespace: Option<&str>, name: &str) -> String {
    namespace.map_or_else(
        || name.to_owned(),
        |namespace| format!("{namespace}.{name}"),
    )
}

pub(super) fn source_span(file: severian_hir::SourceFileId, span: Span) -> SourceSpan {
    SourceSpan {
        file,
        range: SourceRange {
            start: span.start,
            end: span.end,
        },
    }
}

pub(super) fn declared_receiver_type(
    ty: &Type,
    aliases: &HashMap<String, String>,
) -> Option<ReceiverType> {
    let name = resolved_class_type_name(ty, aliases)?;
    let methods = aliases
        .get(&format!("__class_methods.{name}"))?
        .split(',')
        .filter(|method| !method.is_empty())
        .map(str::to_owned)
        .collect();
    Some(ReceiverType {
        concrete: !aliases.contains_key(&format!("__trait.{name}")),
        name,
        methods,
    })
}

pub(super) fn register_class_field_aliases(
    aliases: &mut HashMap<String, String>,
    class: &str,
    fields: &[severian_ast::Field],
) -> Result<(), SemanticError> {
    aliases.insert(
        format!("__class_default_fields.{class}"),
        fields
            .iter()
            .filter(|field| field.default.is_some())
            .map(|field| field.name.name.as_str())
            .collect::<Vec<_>>()
            .join(","),
    );
    for field in fields {
        if let Some(ty) = &field.ty {
            aliases.insert(
                format!("__class_field_type.{class}.{}", field.name.name),
                encode_field_type(declared_value_type(ty, aliases)).to_owned(),
            );
            if let Some(field_class) = class_type_name(ty) {
                aliases.insert(
                    format!("__class_field_class.{class}.{}", field.name.name),
                    field_class,
                );
            }
        }
    }
    Ok(())
}

pub(super) fn register_method_return_alias(
    aliases: &mut HashMap<String, String>,
    class: &str,
    method: &str,
    return_type: Option<&Type>,
) -> Result<(), SemanticError> {
    let ty = return_type
        .map(|ty| declared_value_type(ty, aliases))
        .unwrap_or(ValueType::Unit);
    aliases.insert(
        format!("__class_method_return.{class}.{method}"),
        encode_field_type(ty).to_owned(),
    );
    if let Some(return_class) = return_type.and_then(class_type_name) {
        aliases.insert(
            format!("__class_method_return_class.{class}.{method}"),
            return_class,
        );
    }
    Ok(())
}

pub(super) fn register_class_method_signature_alias(
    aliases: &mut HashMap<String, String>,
    class: &str,
    method: &severian_ast::FunctionDecl,
) {
    aliases.insert(
        format!("__class_method_signature.{class}.{}", method.name.name),
        callable_signature(&method.params, method.return_type.as_ref()),
    );
}

pub(super) fn register_trait_aliases(
    aliases: &mut HashMap<String, String>,
    declaration: &severian_ast::TraitDecl,
) {
    aliases.insert(format!("__trait.{}", declaration.name.name), String::new());
    aliases.insert(
        format!("__trait_generic_params.{}", declaration.name.name),
        declaration
            .generic_params
            .iter()
            .map(|parameter| parameter.name.name.as_str())
            .collect::<Vec<_>>()
            .join(","),
    );
    aliases
        .entry(format!("__class_fields.{}", declaration.name.name))
        .or_default();
    aliases
        .entry(format!("__class_methods.{}", declaration.name.name))
        .or_insert_with(|| {
            declaration
                .methods
                .iter()
                .map(|method| method.name.name.as_str())
                .collect::<Vec<_>>()
                .join(",")
        });
    for method in &declaration.methods {
        aliases.insert(
            format!(
                "__trait_method_signature.{}.{}",
                declaration.name.name, method.name.name
            ),
            callable_signature(&method.params, method.return_type.as_ref()),
        );
    }
}

pub(super) fn callable_signature(
    parameters: &[severian_ast::Parameter],
    returns: Option<&Type>,
) -> String {
    let parameters = parameters
        .iter()
        .skip(usize::from(
            parameters
                .first()
                .is_some_and(|parameter| parameter.name.name == "self"),
        ))
        .map(|parameter| {
            parameter
                .ty
                .as_ref()
                .map(declaration_type_key)
                .unwrap_or_else(|| "Any".into())
        })
        .collect::<Vec<_>>()
        .join(";");
    let returns = returns
        .map(declaration_type_key)
        .unwrap_or_else(|| "unit".into());
    format!("{parameters}->{returns}")
}

pub(super) fn encode_field_type(ty: ValueType) -> String {
    match ty {
        ValueType::Int => "int".into(),
        ValueType::Float => "float".into(),
        ValueType::Bool => "bool".into(),
        ValueType::String => "string".into(),
        ValueType::List => "list".into(),
        ValueType::Tuple => "tuple".into(),
        ValueType::Map => "map".into(),
        ValueType::Set => "set".into(),
        ValueType::Tensor(tensor) => encode_tensor_type(tensor),
        ValueType::TensorAny => "tensor".into(),
        ValueType::Channel => "channel".into(),
        ValueType::Function => "function".into(),
        ValueType::Result => "result".into(),
        ValueType::Option => "option".into(),
        ValueType::Interface(definition) => format!("interface:{}", definition.0),
        ValueType::Any => "any".into(),
        ValueType::Unit => "unit".into(),
    }
}

fn encode_tensor_type(tensor: TensorType) -> String {
    let element = tensor.element.name();
    let Some(rank) = tensor.rank else {
        return format!("tensor:{element}:*");
    };
    let dimensions = tensor.dimensions[..rank as usize]
        .iter()
        .map(|dimension| match dimension {
            TensorDimension::Static(size) => size.to_string(),
            TensorDimension::Dynamic => "?".into(),
        })
        .collect::<Vec<_>>()
        .join(",");
    format!("tensor:{element}:{rank}:{dimensions}")
}

pub(super) fn decode_field_type(value: &str) -> Option<ValueType> {
    if value.starts_with("tensor:") {
        return decode_tensor_type(value).map(ValueType::Tensor);
    }
    if let Some(identity) = value.strip_prefix("interface:") {
        return identity
            .parse::<u64>()
            .ok()
            .map(TypeDefinitionId)
            .map(ValueType::Interface);
    }
    Some(match value {
        "int" => ValueType::Int,
        "float" => ValueType::Float,
        "bool" => ValueType::Bool,
        "string" => ValueType::String,
        "list" => ValueType::List,
        "tuple" => ValueType::Tuple,
        "map" => ValueType::Map,
        "set" => ValueType::Set,
        "tensor" => ValueType::TensorAny,
        "channel" => ValueType::Channel,
        "function" => ValueType::Function,
        "result" => ValueType::Result,
        "option" => ValueType::Option,
        "any" => ValueType::Any,
        "unit" => ValueType::Unit,
        _ => return None,
    })
}

fn decode_tensor_type(value: &str) -> Option<TensorType> {
    let mut parts = value.splitn(4, ':');
    (parts.next()? == "tensor").then_some(())?;
    let element = TensorElementType::parse(parts.next()?)?;
    let rank = parts.next()?;
    if rank == "*" {
        return Some(TensorType::dynamic(element));
    }
    let rank = rank.parse::<usize>().ok()?;
    let encoded_dimensions = parts.next().unwrap_or_default();
    let dimensions = if encoded_dimensions.is_empty() {
        Vec::new()
    } else {
        encoded_dimensions
            .split(',')
            .map(|dimension| match dimension {
                "?" => Some(TensorDimension::Dynamic),
                size => size.parse::<u64>().ok().map(TensorDimension::Static),
            })
            .collect::<Option<Vec<_>>>()?
    };
    (dimensions.len() == rank)
        .then(|| TensorType::ranked(element, &dimensions).ok())
        .flatten()
}
