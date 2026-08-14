use super::*;

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
            export_package: None,
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
    let module = specialize_generic_classes_with_interfaces(module, interfaces)?;
    analyze_specialized(&module, interfaces)
}

fn analyze_specialized(
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
    validate_trait_implementations(module, interfaces, &aliases)?;
    for item in &module.items {
        if let Item::Trait(declaration) = item {
            register_trait_aliases(&mut aliases, declaration);
            for method in &declaration.methods {
                register_method_return_alias(
                    &mut aliases,
                    &declaration.name.name,
                    &method.name.name,
                    method.return_type.as_ref(),
                )?;
            }
        }
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
            aliases.insert(
                format!("__enum_variants.{}", enumeration.name.name),
                enumeration
                    .variants
                    .iter()
                    .map(|variant| variant.name.name.as_str())
                    .collect::<Vec<_>>()
                    .join(","),
            );
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
            aliases.insert(
                format!("__class_methods.{}", class.name.name),
                class
                    .methods
                    .iter()
                    .map(|method| method.name.name.as_str())
                    .collect::<Vec<_>>()
                    .join(","),
            );
            aliases.insert(
                format!("__class_constructor_arities.{}", class.name.name),
                class
                    .constructors
                    .iter()
                    .map(|constructor| constructor.params.len().to_string())
                    .collect::<Vec<_>>()
                    .join(","),
            );
            aliases.insert(
                format!("__class_traits.{}", class.name.name),
                class
                    .traits
                    .iter()
                    .filter_map(declaration_type_name)
                    .collect::<Vec<_>>()
                    .join(","),
            );
            register_class_field_aliases(&mut aliases, &class.name.name, &class.fields)?;
            for method in &class.methods {
                register_method_return_alias(
                    &mut aliases,
                    &class.name.name,
                    &method.name.name,
                    method.return_type.as_ref(),
                )?;
                register_class_method_signature_alias(&mut aliases, &class.name.name, method);
            }
        }
    }
    let mut signatures = HashMap::new();
    for interface in interfaces {
        let module_name = &interface.name;
        for item in &interface.module.items {
            if let Item::Trait(declaration) = item {
                register_trait_aliases(&mut aliases, declaration);
                for method in &declaration.methods {
                    register_method_return_alias(
                        &mut aliases,
                        &declaration.name.name,
                        &method.name.name,
                        method.return_type.as_ref(),
                    )?;
                }
                continue;
            }
            if let Item::Class(class) = item {
                let exported = format!("{module_name}.{}", class.name.name);
                aliases.insert(
                    format!("__module_class.{exported}"),
                    class.name.name.clone(),
                );
                if let Some(package) = &interface.export_package {
                    aliases.insert(
                        format!("__module_class.{package}.{}", class.name.name),
                        class.name.name.clone(),
                    );
                }
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
                aliases
                    .entry(format!("__class_methods.{}", class.name.name))
                    .or_insert_with(|| {
                        class
                            .methods
                            .iter()
                            .map(|method| method.name.name.as_str())
                            .collect::<Vec<_>>()
                            .join(",")
                    });
                aliases
                    .entry(format!("__class_constructor_arities.{}", class.name.name))
                    .or_insert_with(|| {
                        class
                            .constructors
                            .iter()
                            .map(|constructor| constructor.params.len().to_string())
                            .collect::<Vec<_>>()
                            .join(",")
                    });
                aliases
                    .entry(format!("__class_traits.{}", class.name.name))
                    .or_insert_with(|| {
                        class
                            .traits
                            .iter()
                            .filter_map(declaration_type_name)
                            .collect::<Vec<_>>()
                            .join(",")
                    });
                register_class_field_aliases(&mut aliases, &class.name.name, &class.fields)?;
                for method in &class.methods {
                    register_method_return_alias(
                        &mut aliases,
                        &class.name.name,
                        &method.name.name,
                        method.return_type.as_ref(),
                    )?;
                    register_class_method_signature_alias(&mut aliases, &class.name.name, method);
                }
                if imports_entire_module(module, module_name)
                    || interface
                        .export_package
                        .as_deref()
                        .is_some_and(|package| imports_entire_module(module, package))
                {
                    aliases
                        .entry(class.name.name.clone())
                        .or_insert_with(|| format!("__class.{}", class.name.name));
                }
                continue;
            }
            let (name, native_symbol, generic_params, params, return_type) = match item {
                Item::Function(function) => (
                    &function.name,
                    function.native_symbol.as_deref(),
                    function.generic_params.as_slice(),
                    function.params.as_slice(),
                    function.return_type.as_ref(),
                ),
                _ => continue,
            };
            let key = format!("{module_name}.{}", name.name);
            if let Some(class) = return_type.and_then(class_type_name) {
                aliases.insert(format!("__function_return_class.{key}"), class.clone());
                if imports_entire_module(module, module_name) {
                    aliases
                        .entry(format!("__function_return_class.{}", name.name))
                        .or_insert(class);
                }
            }
            let signature =
                lower_signature(&key, native_symbol, generic_params, params, return_type)?;
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
    register_concrete_trait_aliases(&mut aliases, module, interfaces)?;
    for item in &module.items {
        let (name, native_symbol, generic_params, params, return_type) = match item {
            Item::Function(function) => (
                &function.name,
                function.native_symbol.as_deref(),
                function.generic_params.as_slice(),
                function.params.as_slice(),
                function.return_type.as_ref(),
            ),
            _ => continue,
        };
        if let Some(class) = return_type.and_then(class_type_name) {
            aliases.insert(format!("__function_return_class.{}", name.name), class);
        }
        let signature = lower_signature(
            &name.name,
            native_symbol,
            generic_params,
            params,
            return_type,
        )?;
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
            let any_origin =
                declared_any_origin(binding.ty.as_ref(), ty).or_else(|| value.any_origin());
            global_scope.insert(
                binding.name.name.clone(),
                Binding {
                    reference: source_binding(&binding.name),
                    ty,
                    class: expression_class(source, &global_scope, &aliases),
                    function_return: None,
                    collection_len: None,
                    mutable: false,
                    field: false,
                    integer_max: None,
                    known_integer: None,
                    any_origin,
                },
            );
            globals.push(Global {
                name: global_scope[&binding.name.name].reference.clone(),
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
        for (index, parameter) in signature.params.iter().enumerate() {
            let parameter_type = parameter.ty.erased();
            let default = parameter
                .default
                .as_ref()
                .map(|value| {
                    let (value, ty) =
                        lower_expression(value, &scope, &signatures, &function_aliases)?;
                    compatible(value_span(&parameter.default), ty, parameter_type)?;
                    Ok(value)
                })
                .transpose()?;
            scope.insert(
                parameter.name.clone(),
                Binding {
                    reference: source_binding(&function.params[index].name),
                    ty: parameter_type,
                    class: function
                        .params
                        .get(index)
                        .and_then(|parameter| parameter.ty.as_ref())
                        .and_then(class_type_name),
                    function_return: parameter.function_return,
                    collection_len: None,
                    mutable: false,
                    field: false,
                    integer_max: None,
                    known_integer: None,
                    any_origin: parameter.any_origin,
                },
            );
            params.push(Parameter {
                name: scope[&parameter.name].reference.clone(),
                ty: parameter_type,
                default,
                receiver: function
                    .params
                    .get(index)
                    .and_then(|parameter| parameter.ty.as_ref())
                    .and_then(|ty| declared_receiver_type(ty, &function_aliases)),
            });
        }
        let mut instructions = lower_block(
            &function.body,
            &mut scope,
            signature.returns.erased(),
            &signatures,
            &function_aliases,
        )?;
        if function.native_symbol.is_none()
            && signature.returns.erased() != ValueType::Unit
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
            let contract = lower_function_contract(
                test.contract.as_ref(),
                &test_scope,
                &signatures,
                &function_aliases,
            )?;
            let mut test_instructions = lower_block(
                &test.body,
                &mut test_scope,
                ValueType::Unit,
                &signatures,
                &function_aliases,
            )?;
            if !test.modes.contains(&severian_ast::TestMode::Profile) {
                enforce_function_contract(&mut test_instructions, contract.as_ref());
            }
            tests.push(Test {
                name: test.name.as_ref().map(|name| name.name.clone()),
                modes: lower_test_modes(&test.modes),
                return_type: lower_test_return_type(test)?,
                contract,
                instructions: test_instructions,
            });
        }
        let contract = lower_function_contract(
            function.contract.as_ref(),
            &scope,
            &signatures,
            &function_aliases,
        )?;
        enforce_function_contract(&mut instructions, contract.as_ref());
        functions.push(Function {
            id: FunctionId::from_name(&function.name.name),
            name: function.name.name.clone(),
            native_symbol: function.native_symbol.clone(),
            decorators,
            contract,
            params,
            return_type: signature.returns.erased(),
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
        let mut constraint_scope = global_scope.clone();
        for field in &class.fields {
            let ty = field
                .ty
                .as_ref()
                .map(lower_type)
                .transpose()?
                .unwrap_or(ValueType::Any);
            constraint_scope.insert(
                field.name.name.clone(),
                Binding {
                    reference: named_binding(
                        &field.name.name,
                        format!("{}.{}", class.name.name, field.name.name),
                    ),
                    ty,
                    class: field.ty.as_ref().and_then(class_type_name),
                    function_return: None,
                    collection_len: None,
                    mutable: false,
                    field: true,
                    integer_max: None,
                    known_integer: None,
                    any_origin: declared_any_origin(field.ty.as_ref(), ty),
                },
            );
        }
        let field_constraints = class
            .fields
            .iter()
            .flat_map(|field| field.constraints.iter())
            .map(|constraint| {
                let (condition, ty) =
                    lower_expression(constraint, &constraint_scope, &signatures, &aliases)?;
                compatible(constraint.span(), ty, ValueType::Bool)?;
                Ok(condition)
            })
            .collect::<Result<Vec<_>, SemanticError>>()?;
        let mut constructors = Vec::new();
        for constructor in &class.constructors {
            lower_decorators(&constructor.decorators, &imported_modules)?;
            constructors.push(lower_class_function(
                constructor_id(&class.name.name, &constructor.name.name, constructor.span),
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
                FunctionId::from_name(&format!("{}.{}", class.name.name, method.name.name)),
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
            field_constraints,
            constructors,
            methods,
            method_return_classes: class
                .methods
                .iter()
                .map(|method| method.return_type.as_ref().and_then(class_type_name))
                .collect(),
        });
    }
    Ok(Program {
        metadata: Default::default(),
        globals,
        classes,
        functions,
    })
}
