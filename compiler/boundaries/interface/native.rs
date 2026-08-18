use super::*;

pub(super) fn native_units(
    package: &str,
    manifest: &toml::Value,
    manifest_path: &Path,
) -> Result<Vec<NativeUnit>, PackageError> {
    let Some(entries) = manifest
        .get("ffi")
        .and_then(toml::Value::as_table)
        .and_then(|ffi| ffi.get("c"))
        .and_then(toml::Value::as_array)
    else {
        return Ok(Vec::new());
    };
    let root = manifest_path.parent().unwrap_or_else(|| Path::new("."));
    let embedded = manifest_path
        .components()
        .next()
        .is_some_and(|component| component.as_os_str() == "<severian-stdlib>");
    let mut names = HashSet::new();
    entries
        .iter()
        .map(|entry| {
            let table = entry.as_table().ok_or_else(|| {
                ffi_manifest_error(manifest_path, "each [[ffi.c]] entry must be a table")
            })?;
            for key in table.keys() {
                if !matches!(
                    key.as_str(),
                    "name" | "abi" | "targets" | "sources" | "include" | "libraries"
                ) {
                    return Err(ffi_manifest_error(
                        manifest_path,
                        format!(
                            "unsupported [[ffi.c]] key `{key}`; compiler and linker flags are not allowed"
                        ),
                    ));
                }
            }
            let name = ffi_required_string(table, "name", manifest_path)?;
            if !valid_native_name(&name) || !names.insert(name.clone()) {
                return Err(ffi_manifest_error(
                    manifest_path,
                    format!("native unit name `{name}` is invalid or duplicated"),
                ));
            }
            let abi = match ffi_required_string(table, "abi", manifest_path)?.as_str() {
                "c-v1" => severian_abi::AbiVersion::CV1,
                other => {
                    return Err(ffi_manifest_error(
                        manifest_path,
                        format!("unsupported native ABI `{other}`; use `c-v1`"),
                    ))
                }
            };
            let sources = ffi_string_array(table, "sources", manifest_path, false)?
                .into_iter()
                .map(|source| resolve_native_path(root, &source, manifest_path, embedded, false))
                .collect::<Result<Vec<_>, _>>()?;
            if sources.is_empty() {
                return Err(ffi_manifest_error(
                    manifest_path,
                    format!("native unit `{name}` must declare at least one C source"),
                ));
            }
            if let Some(source) = sources
                .iter()
                .find(|source| source.extension().and_then(|value| value.to_str()) != Some("c"))
            {
                return Err(ffi_manifest_error(
                    manifest_path,
                    format!("native source {} must use the .c extension", source.display()),
                ));
            }
            let include_directories = ffi_string_array(table, "include", manifest_path, true)?
                .into_iter()
                .map(|include| resolve_native_path(root, &include, manifest_path, embedded, true))
                .collect::<Result<Vec<_>, _>>()?;
            let libraries = ffi_string_array(table, "libraries", manifest_path, true)?;
            if let Some(library) = libraries.iter().find(|library| {
                library.is_empty()
                    || library.starts_with('-')
                    || !library.bytes().all(|byte| {
                        byte.is_ascii_alphanumeric() || matches!(byte, b'_' | b'-' | b'+')
                    })
            }) {
                return Err(ffi_manifest_error(
                    manifest_path,
                    format!("native library `{library}` is not a safe system library name"),
                ));
            }
            let targets = ffi_string_array(table, "targets", manifest_path, false)?
                .into_iter()
                .map(|target| {
                    if target.is_empty()
                        || !target.bytes().all(|byte| {
                            byte.is_ascii_alphanumeric() || matches!(byte, b'_' | b'-' | b'*')
                        })
                    {
                        return Err(ffi_manifest_error(
                            manifest_path,
                            format!("native target pattern `{target}` is invalid"),
                        ));
                    }
                    Ok(TargetPattern(target))
                })
                .collect::<Result<Vec<_>, PackageError>>()?;
            if targets.is_empty() {
                return Err(ffi_manifest_error(
                    manifest_path,
                    format!("native unit `{name}` must declare at least one target"),
                ));
            }
            Ok(NativeUnit {
                package: package.to_owned(),
                name,
                abi,
                language: NativeLanguage::C,
                sources,
                include_directories,
                libraries,
                targets,
            })
        })
        .collect()
}

pub fn load_manifest_native_units(manifest_path: &Path) -> Result<Vec<NativeUnit>, PackageError> {
    let manifest = parse_manifest(manifest_path)?;
    let package = package_name(&manifest, manifest_path)?;
    native_units(package, &manifest, manifest_path)
}

fn ffi_required_string(
    table: &toml::Table,
    key: &str,
    path: &Path,
) -> Result<String, PackageError> {
    table
        .get(key)
        .and_then(toml::Value::as_str)
        .filter(|value| !value.is_empty())
        .map(str::to_owned)
        .ok_or_else(|| ffi_manifest_error(path, format!("[[ffi.c]].{key} must be a string")))
}

fn ffi_string_array(
    table: &toml::Table,
    key: &str,
    path: &Path,
    optional: bool,
) -> Result<Vec<String>, PackageError> {
    let Some(value) = table.get(key) else {
        return if optional {
            Ok(Vec::new())
        } else {
            Err(ffi_manifest_error(
                path,
                format!("[[ffi.c]].{key} is required"),
            ))
        };
    };
    value
        .as_array()
        .ok_or_else(|| ffi_manifest_error(path, format!("[[ffi.c]].{key} must be an array")))?
        .iter()
        .map(|value| {
            value.as_str().map(str::to_owned).ok_or_else(|| {
                ffi_manifest_error(path, format!("[[ffi.c]].{key} must contain strings"))
            })
        })
        .collect()
}

fn resolve_native_path(
    root: &Path,
    relative: &str,
    manifest_path: &Path,
    embedded: bool,
    directory: bool,
) -> Result<PathBuf, PackageError> {
    let relative = Path::new(relative);
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
        return Err(ffi_manifest_error(
            manifest_path,
            format!(
                "native path `{}` must stay inside the package",
                relative.display()
            ),
        ));
    }
    let candidate = root.join(relative);
    if embedded {
        return Ok(candidate);
    }
    let canonical_root = root.canonicalize().map_err(PackageError::Io)?;
    let canonical = candidate.canonicalize().map_err(|error| {
        ffi_manifest_error(
            manifest_path,
            format!(
                "native path {} is unavailable: {error}",
                candidate.display()
            ),
        )
    })?;
    if !canonical.starts_with(&canonical_root)
        || (directory && !canonical.is_dir())
        || (!directory && !canonical.is_file())
    {
        return Err(ffi_manifest_error(
            manifest_path,
            format!(
                "native path {} has the wrong kind or escapes the package",
                candidate.display()
            ),
        ));
    }
    Ok(canonical)
}

fn valid_native_name(name: &str) -> bool {
    !name.is_empty()
        && name
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'_' | b'-'))
}

fn ffi_manifest_error(path: &Path, message: impl Into<String>) -> PackageError {
    PackageError::Manifest(format!(
        "invalid native FFI configuration in {}: {}",
        path.display(),
        message.into()
    ))
}
