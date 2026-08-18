use super::*;

pub(super) fn interfaces_with_root(
    interfaces: &[PackageInterface],
    ast: &AstModule,
    source_path: &Path,
    source: &str,
) -> Result<Vec<PackageInterface>, CompileError> {
    let mut planned = interfaces.to_vec();
    if source_path.to_string_lossy().starts_with('<') {
        return Ok(planned);
    }
    let Some(manifest_path) = severian_package::find_manifest(source_path) else {
        return Ok(planned);
    };
    let units = severian_package::load_manifest_native_units(&manifest_path)
        .map_err(|error| CompileError::Package(error.to_string()))?;
    if let Some(unit) = units.first() {
        planned.push(PackageInterface {
            name: unit.package.clone(),
            export_package: None,
            module: ast.clone(),
            compiler: Default::default(),
            native_units: units,
            native_assets: Vec::new(),
            source_path: source_path.to_path_buf(),
            source: source.to_owned(),
        });
    }
    Ok(planned)
}

pub(super) fn build(
    interfaces: &[PackageInterface],
) -> Result<
    (
        Vec<severian_package::NativeUnit>,
        Vec<severian_package::EmbeddedNativeAsset>,
        std::collections::BTreeMap<String, severian_abi::ExternalFunction>,
    ),
    CompileError,
> {
    let mut units = std::collections::BTreeMap::new();
    let mut assets = std::collections::BTreeMap::new();
    let mut functions = std::collections::BTreeMap::new();
    for interface in interfaces {
        for unit in &interface.native_units {
            let key = (unit.package.clone(), unit.name.clone());
            if let Some(existing) = units.insert(key.clone(), unit.clone()) {
                if existing != *unit {
                    return Err(CompileError::Package(format!(
                        "native unit `{}.{}` resolves to conflicting definitions",
                        key.0, key.1
                    )));
                }
            }
        }
        for asset in &interface.native_assets {
            assets.insert(asset.path.clone(), asset.clone());
        }
        let validated = severian_semantic::validate_native_abi(interface).map_err(|error| {
            CompileError::Frontend {
                stage: "native ABI",
                span: error.span,
                message: error.message,
                source_path: interface.source_path.clone(),
                source: interface.source.clone(),
            }
        })?;
        for function in validated {
            if let Some(existing) = functions.insert(function.symbol.name.clone(), function.clone())
            {
                return Err(CompileError::Frontend {
                    stage: "native ABI",
                    span: severian_ast::Span::empty(0),
                    message: format!(
                        "E0803: native symbol `{}` is declared by both `{}` and `{}`",
                        function.symbol, existing.function, function.function
                    ),
                    source_path: interface.source_path.clone(),
                    source: interface.source.clone(),
                });
            }
        }
    }
    Ok((
        units.into_values().collect(),
        assets.into_values().collect(),
        functions,
    ))
}
