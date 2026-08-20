use super::*;

mod native;
pub use native::load_manifest_native_units;
use native::native_units;

pub fn load_path_dependency_interfaces(
    manifest_path: &Path,
) -> Result<Vec<PackageInterface>, PackageError> {
    load_dependency_interfaces(resolve_dependencies(manifest_path)?)
}

pub fn load_transient_dependency_interfaces(
    manifest_path: &Path,
) -> Result<Vec<PackageInterface>, PackageError> {
    load_dependency_interfaces(resolve_dependencies_transient(manifest_path)?)
}

fn load_dependency_interfaces(
    resolution: Resolution,
) -> Result<Vec<PackageInterface>, PackageError> {
    let mut interfaces = Vec::new();
    for dependency in resolution.dependencies {
        interfaces.extend(load_interface_tree_as(
            &dependency.import_name,
            &dependency.package_name,
            &dependency.root,
        )?);
    }
    interfaces.sort_by(|left, right| left.name.cmp(&right.name));
    Ok(interfaces)
}

/// Loads quoted source imports from a project root. Quoted imports are never
/// resolved through the package registry or the official library directory.
pub fn load_local_interfaces(
    module: &Module,
    project_root: &Path,
) -> Result<Vec<PackageInterface>, PackageError> {
    let has_local_imports = module.items.iter().any(|item| {
        matches!(
            item,
            Item::Import(import) if matches!(import.kind, ImportKind::Local { .. })
        )
    });
    if !has_local_imports {
        return Ok(Vec::new());
    }
    let project_root = if project_root.as_os_str().is_empty() {
        Path::new(".")
    } else {
        project_root
    };
    let project_root = project_root.canonicalize().map_err(|error| {
        PackageError::Manifest(format!(
            "local import root {} is invalid: {error}",
            project_root.display()
        ))
    })?;
    let mut visited = HashSet::new();
    let mut names = HashMap::new();
    let mut interfaces = Vec::new();
    collect_local_interfaces(
        module,
        &project_root,
        &mut visited,
        &mut names,
        &mut interfaces,
    )?;
    interfaces.sort_by(|left, right| left.name.cmp(&right.name));
    Ok(interfaces)
}

fn collect_local_interfaces(
    module: &Module,
    project_root: &Path,
    visited: &mut HashSet<PathBuf>,
    names: &mut HashMap<String, PathBuf>,
    interfaces: &mut Vec<PackageInterface>,
) -> Result<(), PackageError> {
    for item in &module.items {
        let Item::Import(import) = item else {
            continue;
        };
        let ImportKind::Local { path, alias } = &import.kind else {
            continue;
        };
        let module_name = local_import_module_name(path).ok_or_else(|| {
            PackageError::Manifest(format!("local import path `{path}` is empty or invalid"))
        })?;
        if alias.is_none() && local_import_exposed_name(path).is_none() {
            return Err(PackageError::Manifest(format!(
                "local import `{path}` needs an identifier alias"
            )));
        }
        let source_path = resolve_local_import(project_root, path)?;
        if let Some(existing) = names.get(&module_name) {
            if existing != &source_path {
                return Err(PackageError::Manifest(format!(
                    "local imports {} and {} both resolve to module `{module_name}`",
                    existing.display(),
                    source_path.display()
                )));
            }
        } else {
            names.insert(module_name.clone(), source_path.clone());
        }
        if !visited.insert(source_path.clone()) {
            continue;
        }
        let source = std::fs::read_to_string(&source_path).map_err(|error| {
            PackageError::Manifest(format!(
                "could not read local import {}: {error}",
                source_path.display()
            ))
        })?;
        let tokens = severian_lexer::lex(&source).map_err(|error| PackageError::Frontend {
            package: module_name.clone(),
            stage: "lexer",
            span: error.span,
            message: error.message,
            source_path: Some(source_path.clone()),
            source: Some(source.clone()),
        })?;
        let imported = severian_parser::parse(&tokens).map_err(|error| PackageError::Frontend {
            package: module_name.clone(),
            stage: "parser",
            span: error.span,
            message: error.message,
            source_path: Some(source_path.clone()),
            source: Some(source.clone()),
        })?;
        collect_local_interfaces(&imported, project_root, visited, names, interfaces)?;
        interfaces.push(PackageInterface {
            name: module_name,
            export_package: None,
            module: imported,
            compiler: CompilerMetadata::default(),
            native_units: Vec::new(),
            native_assets: Vec::new(),
            source_path,
            source,
        });
    }
    Ok(())
}

fn resolve_local_import(project_root: &Path, import: &str) -> Result<PathBuf, PackageError> {
    let relative = Path::new(import);
    if relative.is_absolute()
        || relative.components().any(|component| {
            matches!(
                component,
                std::path::Component::ParentDir
                    | std::path::Component::RootDir
                    | std::path::Component::Prefix(_)
            )
        })
    {
        return Err(PackageError::Manifest(format!(
            "local import `{import}` must stay within the project root"
        )));
    }
    let mut candidate = project_root.join(relative);
    if candidate.extension().is_none() {
        candidate.set_extension("sev");
    }
    let canonical = candidate.canonicalize().map_err(|error| {
        PackageError::Manifest(format!(
            "local import `{import}` does not resolve to {}: {error}",
            candidate.display()
        ))
    })?;
    if !canonical.starts_with(project_root) || !canonical.is_file() {
        return Err(PackageError::Manifest(format!(
            "local import `{import}` must resolve to a .sev file within {}",
            project_root.display()
        )));
    }
    Ok(canonical)
}

pub fn local_import_module_name(path: &str) -> Option<String> {
    let path = path.replace('\\', "/");
    let path = path.strip_suffix(".sev").unwrap_or(&path);
    let parts = path
        .split('/')
        .filter(|part| !part.is_empty() && *part != ".")
        .collect::<Vec<_>>();
    (!parts.is_empty()).then(|| parts.join("."))
}

pub fn local_import_exposed_name(path: &str) -> Option<String> {
    let module = local_import_module_name(path)?;
    let name = module.rsplit('.').next()?;
    let mut bytes = name.bytes();
    let valid = bytes
        .next()
        .is_some_and(|byte| byte == b'_' || byte.is_ascii_alphabetic())
        && bytes.all(|byte| byte == b'_' || byte.is_ascii_alphanumeric());
    valid.then(|| name.to_owned())
}

pub fn load_official_interfaces(
    module: &Module,
    library_root: &Path,
) -> Result<Vec<PackageInterface>, PackageError> {
    let mut pending = imported_packages(module)
        .into_iter()
        .collect::<BTreeSet<_>>();
    let mut loaded = HashSet::new();
    let mut interfaces = Vec::new();
    while let Some(name) = pending.pop_first() {
        if !loaded.insert(name.clone()) {
            continue;
        }
        let Some((name, directory)) = ({
            let directory = name
                .split('.')
                .fold(library_root.to_path_buf(), |path, segment| {
                    path.join(segment)
                });
            manifest_in(&directory).map(|_| (name, directory))
        }) else {
            continue;
        };
        for interface in load_interface_tree_as(&name, &name, &directory)? {
            pending.extend(imported_packages(&interface.module));
            interfaces.push(interface);
        }
    }
    interfaces.sort_by(|left, right| left.name.cmp(&right.name));
    Ok(interfaces)
}

/// Loads the mandatory language bootstrap package independently of user
/// imports. Absence is an error: the compiler must never fabricate fallback
/// primitive declarations.
pub fn load_core_primitive_interfaces(
    library_root: &Path,
) -> Result<Vec<PackageInterface>, PackageError> {
    const PACKAGE: &str = "core.primitives";
    let directory = PACKAGE
        .split('.')
        .fold(library_root.to_path_buf(), |path, segment| path.join(segment));
    if manifest_in(&directory).is_none() {
        return Err(PackageError::Manifest(format!(
            "compiler bootstrap failed: `{PACKAGE}` is unavailable at {}",
            directory.display()
        )));
    }
    load_interface_tree_as(PACKAGE, PACKAGE, &directory)
}

/// Loads standard packages embedded in the compiler binary. This keeps named
/// imports available when `sev` is installed or relocated without its source
/// checkout. An explicit `SEVERIAN_LIBRARY_PATH` can still select editable
/// on-disk packages during compiler and standard-library development.
pub fn load_embedded_official_interfaces(
    module: &Module,
    packages: &[EmbeddedOfficialPackage<'_>],
) -> Result<Vec<PackageInterface>, PackageError> {
    load_embedded_named_interfaces(imported_packages(module), packages)
}

pub fn load_embedded_core_primitive_interfaces(
    packages: &[EmbeddedOfficialPackage<'_>],
) -> Result<Vec<PackageInterface>, PackageError> {
    load_embedded_named_interfaces(["core.primitives".to_owned()], packages)
}

fn load_embedded_named_interfaces(
    names: impl IntoIterator<Item = String>,
    packages: &[EmbeddedOfficialPackage<'_>],
) -> Result<Vec<PackageInterface>, PackageError> {
    let mut pending = names.into_iter().collect::<BTreeSet<_>>();
    let mut loaded = HashSet::new();
    let mut interfaces = Vec::new();
    while let Some(name) = pending.pop_first() {
        if !loaded.insert(name.clone()) {
            continue;
        }
        let Some(package) = packages.iter().find(|package| package.name == name) else {
            continue;
        };
        let manifest_path = PathBuf::from("<severian-stdlib>")
            .join(package.name)
            .join(MANIFEST_FILE);
        let source_path = PathBuf::from("<severian-stdlib>")
            .join(package.name)
            .join("src/lib.sev");
        let manifest = toml::from_str::<toml::Value>(package.manifest).map_err(|error| {
            PackageError::Manifest(format!(
                "invalid embedded manifest for package `{}`: {error}",
                package.name
            ))
        })?;
        let declared_name = package_name(&manifest, &manifest_path)?;
        if declared_name != package.name {
            return Err(PackageError::Manifest(format!(
                "embedded package `{}` declares package `{declared_name}`",
                package.name
            )));
        }
        let mut interface = load_interface_source(
            &name,
            &manifest,
            &manifest_path,
            source_path,
            package.source.to_owned(),
        )?;
        interface.native_assets = package
            .native_assets
            .iter()
            .map(|asset| EmbeddedNativeAsset {
                path: PathBuf::from("<severian-stdlib>")
                    .join(package.name)
                    .join(asset.path),
                contents: asset.contents.to_vec(),
            })
            .collect();
        pending.extend(imported_packages(&interface.module));
        interfaces.push(interface);
        for embedded in package.modules {
            let module_name = local_import_module_name(embedded.path).ok_or_else(|| {
                PackageError::Manifest(format!(
                    "embedded module path `{}` in package `{}` is invalid",
                    embedded.path, package.name
                ))
            })?;
            let module_path = PathBuf::from("<severian-stdlib>")
                .join(package.name)
                .join(embedded.path);
            let mut interface = load_interface_source(
                &module_name,
                &manifest,
                &manifest_path,
                module_path,
                embedded.source.to_owned(),
            )?;
            interface.export_package = Some(name.clone());
            pending.extend(imported_packages(&interface.module));
            interfaces.push(interface);
        }
    }
    interfaces.sort_by(|left, right| left.name.cmp(&right.name));
    Ok(interfaces)
}

fn imported_packages(module: &Module) -> HashSet<String> {
    module
        .items
        .iter()
        .filter_map(|item| {
            let Item::Import(import) = item else {
                return None;
            };
            Some(match &import.kind {
                ImportKind::Local { .. } => Vec::new(),
                ImportKind::Module { path, .. } => vec![path
                    .iter()
                    .map(|part| part.name.as_str())
                    .collect::<Vec<_>>()
                    .join(".")],
                ImportKind::From { module, names } => {
                    let module = module
                        .iter()
                        .map(|part| part.name.as_str())
                        .collect::<Vec<_>>()
                        .join(".");
                    let mut packages = vec![module.clone()];
                    packages.extend(
                        names
                            .iter()
                            .map(|name| format!("{module}.{}", name.name.name)),
                    );
                    packages
                }
            })
        })
        .flatten()
        .collect()
}

fn load_interface_tree_as(
    import_name: &str,
    expected_package: &str,
    directory: &Path,
) -> Result<Vec<PackageInterface>, PackageError> {
    let root = load_interface_as(import_name, expected_package, directory)?;
    let manifest_path = manifest_in(directory).unwrap_or_else(|| directory.join(MANIFEST_FILE));
    let manifest = parse_manifest(&manifest_path)?;
    let mut local = load_local_interfaces(&root.module, directory)?;
    for interface in &mut local {
        let tokens =
            severian_lexer::lex(&interface.source).map_err(|error| PackageError::Frontend {
                package: expected_package.into(),
                stage: "lexer",
                span: error.span,
                message: error.message,
                source_path: Some(interface.source_path.clone()),
                source: Some(interface.source.clone()),
            })?;
        if let Some(token) = tokens
            .iter()
            .find(|token| token.kind == severian_lexer::TokenKind::Unsafe)
        {
            enforce_manifest_unsafe_policy(
                &manifest,
                &manifest_path,
                &interface.source_path,
                &tokens,
                token.span,
                true,
            )
            .map_err(|error| {
                with_frontend_source(error, &interface.source_path, &interface.source)
            })?;
        }
        interface.native_units = root.native_units.clone();
        interface.export_package = Some(import_name.to_owned());
    }
    let mut interfaces = Vec::with_capacity(local.len() + 1);
    interfaces.push(root);
    interfaces.extend(local);
    Ok(interfaces)
}

fn load_interface_as(
    import_name: &str,
    expected_package: &str,
    directory: &Path,
) -> Result<PackageInterface, PackageError> {
    let manifest_path = manifest_in(directory).unwrap_or_else(|| directory.join(MANIFEST_FILE));
    let manifest = parse_manifest(&manifest_path)?;
    let source_path = directory.join(library_path(&manifest));
    let source = std::fs::read_to_string(&source_path).map_err(|error| {
        PackageError::Manifest(format!(
            "could not read package `{expected_package}` at {}: {error}",
            source_path.display()
        ))
    })?;
    let declared_name = package_name(&manifest, &manifest_path)?;
    if declared_name != expected_package {
        return Err(PackageError::Manifest(format!(
            "dependency `{import_name}` expects package `{expected_package}` but resolves to `{declared_name}`"
        )));
    }
    load_interface_source(import_name, &manifest, &manifest_path, source_path, source)
}

fn load_interface_source(
    name: &str,
    manifest: &toml::Value,
    manifest_path: &Path,
    source_path: PathBuf,
    source: String,
) -> Result<PackageInterface, PackageError> {
    let tokens = severian_lexer::lex(&source).map_err(|error| PackageError::Frontend {
        package: name.into(),
        stage: "lexer",
        span: error.span,
        message: error.message,
        source_path: Some(source_path.clone()),
        source: Some(source.clone()),
    })?;
    if let Some(token) = tokens
        .iter()
        .find(|token| token.kind == severian_lexer::TokenKind::Unsafe)
    {
        enforce_manifest_unsafe_policy(
            manifest,
            manifest_path,
            &source_path,
            &tokens,
            token.span,
            true,
        )
        .map_err(|error| with_frontend_source(error, &source_path, &source))?;
    }
    let module = severian_parser::parse(&tokens).map_err(|error| PackageError::Frontend {
        package: name.into(),
        stage: "parser",
        span: error.span,
        message: error.message,
        source_path: Some(source_path.clone()),
        source: Some(source.clone()),
    })?;
    Ok(PackageInterface {
        name: name.into(),
        export_package: None,
        module,
        compiler: compiler_metadata(name, manifest, manifest_path)?,
        native_units: native_units(name, manifest, manifest_path)?,
        native_assets: Vec::new(),
        source_path,
        source,
    })
}

fn compiler_metadata(
    package: &str,
    manifest: &toml::Value,
    manifest_path: &Path,
) -> Result<CompilerMetadata, PackageError> {
    let Some(compiler) = manifest
        .get("package")
        .and_then(toml::Value::as_table)
        .and_then(|package| package.get("metadata"))
        .and_then(toml::Value::as_table)
        .and_then(|metadata| metadata.get("compiler"))
        .and_then(toml::Value::as_table)
    else {
        return Ok(CompilerMetadata::default());
    };

    let symbols = compiler
        .get("symbols")
        .and_then(toml::Value::as_table)
        .map(|symbols| {
            symbols
                .iter()
                .map(|(symbol, function)| {
                    function
                        .as_str()
                        .map(|function| (symbol.clone(), function.to_owned()))
                        .ok_or_else(|| {
                            metadata_error(
                                manifest_path,
                                format!("symbol `{symbol}` must name a function"),
                            )
                        })
                })
                .collect::<Result<HashMap<_, _>, _>>()
        })
        .transpose()?
        .unwrap_or_default();

    let mut external_functions = compiler
        .get("external-functions")
        .and_then(toml::Value::as_array)
        .map(|functions| {
            functions
                .iter()
                .map(|function| {
                    function
                        .as_str()
                        .map(|function| format!("{package}.{function}"))
                        .ok_or_else(|| {
                            metadata_error(manifest_path, "external-functions must be strings")
                        })
                })
                .collect::<Result<Vec<_>, _>>()
        })
        .transpose()?
        .unwrap_or_default();
    external_functions.sort();
    external_functions.dedup();

    let mut fusion_aliases = compiler
        .get("fusion-aliases")
        .and_then(toml::Value::as_table)
        .map(|aliases| {
            aliases
                .iter()
                .map(|(function, target)| {
                    let target = target.as_str().ok_or_else(|| {
                        metadata_error(
                            manifest_path,
                            format!("fusion alias `{function}` must name a target function"),
                        )
                    })?;
                    Ok(FusionAlias {
                        function: format!("{package}.{function}"),
                        target: if target.contains('.') {
                            target.to_owned()
                        } else {
                            format!("{package}.{target}")
                        },
                    })
                })
                .collect::<Result<Vec<_>, PackageError>>()
        })
        .transpose()?
        .unwrap_or_default();
    fusion_aliases.sort_by(|left, right| left.function.cmp(&right.function));

    let mut fusion_rules = Vec::new();
    if let Some(fusion) = compiler.get("fusion").and_then(toml::Value::as_table) {
        let runtime_symbol = required_string(fusion, "runtime-symbol", manifest_path)?;
        if !valid_symbol(&runtime_symbol) {
            return Err(metadata_error(
                manifest_path,
                "fusion runtime-symbol is not a valid native symbol",
            ));
        }
        let packing_bits = optional_integer(fusion, "packing-bits", 4, manifest_path)?;
        let max_chain = optional_integer(fusion, "max-chain", 16, manifest_path)?;
        if !(1..=8).contains(&packing_bits) || max_chain == 0 || max_chain > 64 / packing_bits {
            return Err(metadata_error(
                manifest_path,
                "fusion packing-bits must be 1..=8 and its max-chain must fit in 64 bits",
            ));
        }
        let operations = fusion
            .get("operations")
            .and_then(toml::Value::as_table)
            .ok_or_else(|| metadata_error(manifest_path, "fusion.operations is required"))?;
        for (function, opcode) in operations {
            let opcode = opcode.as_integer().ok_or_else(|| {
                metadata_error(
                    manifest_path,
                    format!("fusion opcode for `{function}` must be an integer"),
                )
            })?;
            let maximum_opcode = (1u16 << packing_bits) - 1;
            if opcode <= 0 || opcode > i64::from(maximum_opcode) {
                return Err(metadata_error(
                    manifest_path,
                    format!("fusion opcode for `{function}` does not fit packing-bits"),
                ));
            }
            fusion_rules.push(FusionRule {
                function: format!("{package}.{function}"),
                runtime_symbol: runtime_symbol.clone(),
                opcode: opcode as u8,
                packing_bits: packing_bits as u8,
                max_chain: max_chain as usize,
            });
        }
    }
    fusion_rules.sort_by(|left, right| left.function.cmp(&right.function));

    let mut graph_rules = compiler
        .get("graph-operations")
        .and_then(toml::Value::as_table)
        .map(|operations| {
            operations
                .iter()
                .map(|(function, operation)| {
                    let operation = operation.as_str().ok_or_else(|| {
                        metadata_error(
                            manifest_path,
                            format!("graph operation `{function}` must name an operation kind"),
                        )
                    })?;
                    let operation = match operation {
                        "input" => GraphOperation::Input,
                        "relu" => GraphOperation::Relu,
                        "add" => GraphOperation::Add,
                        "matmul" => GraphOperation::Matmul,
                        "transpose" => GraphOperation::Transpose,
                        "scale" => GraphOperation::Scale,
                        "softmax-rows" => GraphOperation::SoftmaxRows,
                        "layer-norm" => GraphOperation::LayerNorm,
                        "run" => GraphOperation::Run,
                        other => {
                            return Err(metadata_error(
                                manifest_path,
                                format!("unknown graph operation kind `{other}`"),
                            ));
                        }
                    };
                    Ok(GraphRule {
                        function: format!("{package}.{function}"),
                        operation,
                    })
                })
                .collect::<Result<Vec<_>, PackageError>>()
        })
        .transpose()?
        .unwrap_or_default();
    graph_rules.sort_by(|left, right| left.function.cmp(&right.function));
    Ok(CompilerMetadata {
        symbols,
        external_functions,
        fusion_rules,
        fusion_aliases,
        graph_rules,
    })
}

fn valid_symbol(symbol: &str) -> bool {
    let mut characters = symbol.chars();
    characters
        .next()
        .is_some_and(|first| first == '_' || first.is_ascii_alphabetic())
        && characters.all(|character| {
            character == '_' || character == '.' || character.is_ascii_alphanumeric()
        })
}

fn required_string(table: &toml::Table, key: &str, path: &Path) -> Result<String, PackageError> {
    table
        .get(key)
        .and_then(toml::Value::as_str)
        .map(str::to_owned)
        .ok_or_else(|| metadata_error(path, format!("compiler fusion requires `{key}`")))
}

fn optional_integer(
    table: &toml::Table,
    key: &str,
    default: i64,
    path: &Path,
) -> Result<i64, PackageError> {
    match table.get(key) {
        Some(value) => value
            .as_integer()
            .ok_or_else(|| metadata_error(path, format!("`{key}` must be an integer"))),
        None => Ok(default),
    }
}

fn metadata_error(path: &Path, message: impl Into<String>) -> PackageError {
    PackageError::Manifest(format!(
        "invalid compiler metadata in {}: {}",
        path.display(),
        message.into()
    ))
}
