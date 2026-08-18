use crate::{
    toolchain::{find_required_tool, run_tool, tool_version, TemporaryFiles, Tool},
    BackendError,
};
use severian_hir::Program;
use severian_package::{EmbeddedNativeAsset, NativeUnit};
use sha2::{Digest, Sha256};
use std::{
    collections::{BTreeMap, BTreeSet, HashMap},
    ffi::OsString,
    path::{Component, Path, PathBuf},
    process::Command,
};

pub(crate) struct PackageNativeObjects {
    pub objects: Vec<PathBuf>,
    pub system_libraries: Vec<String>,
}

pub(crate) fn compile_package_native_units(
    program: &Program,
    units: &[NativeUnit],
    assets: &[EmbeddedNativeAsset],
    temporary: &TemporaryFiles,
    optimization: u8,
) -> Result<PackageNativeObjects, BackendError> {
    let required = program
        .metadata
        .external_functions
        .values()
        .map(|function| function.package.clone())
        .collect::<BTreeSet<_>>();
    if required.is_empty() {
        return Ok(PackageNativeObjects {
            objects: Vec::new(),
            system_libraries: Vec::new(),
        });
    }
    let mut selected = Vec::new();
    for package in &required {
        let candidates = units
            .iter()
            .filter(|unit| &unit.package == package)
            .filter_map(|unit| {
                unit.targets
                    .iter()
                    .filter_map(|target| target.specificity_for_host())
                    .max()
                    .map(|specificity| (specificity, unit))
            })
            .collect::<Vec<_>>();
        let Some(best) = candidates.iter().map(|(specificity, _)| *specificity).max() else {
            return Err(backend_error(format!(
                "E0804: package `{package}` has no c-v1 native provider for {}-{}",
                std::env::consts::ARCH,
                std::env::consts::OS
            )));
        };
        let matches = candidates
            .into_iter()
            .filter(|(specificity, _)| *specificity == best)
            .map(|(_, unit)| unit)
            .collect::<Vec<_>>();
        if matches.len() != 1 {
            return Err(backend_error(format!(
                "E0804: package `{package}` has ambiguous c-v1 native providers for {}-{}: {}",
                std::env::consts::ARCH,
                std::env::consts::OS,
                matches
                    .iter()
                    .map(|unit| unit.name.as_str())
                    .collect::<Vec<_>>()
                    .join(", ")
            )));
        }
        selected.push(matches[0]);
    }

    let materialized = materialize_assets(assets, temporary)?;
    let clang = find_required_tool(Tool::Clang)?;
    let clang_version = tool_version(&clang).unwrap_or_else(|| "unknown-clang".into());
    let cache = std::env::temp_dir().join("severian-native-cache-c-v1");
    std::fs::create_dir_all(&cache)?;
    let mut objects = Vec::new();
    let mut system_libraries = BTreeSet::new();
    let mut headers = BTreeMap::new();

    for unit in selected {
        let functions = program
            .metadata
            .external_functions
            .values()
            .filter(|function| function.package == unit.package)
            .collect::<Vec<_>>();
        let header = headers.entry(unit.package.clone()).or_insert_with(|| {
            let path = temporary.path(&format!("{}-c-v1.h", unit.package));
            (path, severian_abi::c_v1_header(functions))
        });
        std::fs::write(&header.0, &header.1)?;
        system_libraries.extend(unit.libraries.iter().cloned());

        for source in &unit.sources {
            let source = materialized.get(source).unwrap_or(source);
            let mut digest = Sha256::new();
            digest.update(env!("CARGO_PKG_VERSION"));
            digest.update(std::env::consts::ARCH);
            digest.update(std::env::consts::OS);
            digest.update(&clang_version);
            digest.update(format!("{:?}", unit));
            digest.update(std::fs::read(source)?);
            digest.update(header.1.as_bytes());
            for include in &unit.include_directories {
                let include = materialized.get(include).unwrap_or(include);
                hash_tree(include, &mut digest)?;
            }
            let key = format!("{:x}", digest.finalize());
            let cached = cache.join(format!("{key}.o"));
            if !cached.is_file() {
                let pending = temporary.path(&format!("ffi-{key}.o"));
                let mut arguments = vec![
                    OsString::from("-std=c11"),
                    OsString::from("-D_POSIX_C_SOURCE=200809L"),
                    OsString::from(format!("-O{}", optimization.min(3))),
                    OsString::from("-fPIC"),
                    OsString::from("-c"),
                    source.as_os_str().to_owned(),
                    OsString::from("-o"),
                    pending.as_os_str().to_owned(),
                    OsString::from("-include"),
                    header.0.as_os_str().to_owned(),
                ];
                for include in &unit.include_directories {
                    let include = materialized.get(include).unwrap_or(include);
                    arguments.push(OsString::from(format!("-I{}", include.display())));
                }
                run_tool(&clang, &arguments).map_err(|error| {
                    backend_error(format!(
                        "E0806: native unit `{}.{}` failed ABI compilation: {error}",
                        unit.package, unit.name
                    ))
                })?;
                match std::fs::rename(&pending, &cached) {
                    Ok(()) => {}
                    Err(_) if cached.is_file() => {}
                    Err(error) => return Err(error.into()),
                }
            }
            objects.push(cached);
        }
    }
    verify_provider_symbols(program, &objects)?;
    Ok(PackageNativeObjects {
        objects,
        system_libraries: system_libraries.into_iter().collect(),
    })
}

fn materialize_assets(
    assets: &[EmbeddedNativeAsset],
    temporary: &TemporaryFiles,
) -> Result<HashMap<PathBuf, PathBuf>, BackendError> {
    let root = temporary.directory("native-assets");
    let mut output = HashMap::new();
    for asset in assets {
        let relative = asset
            .path
            .components()
            .filter_map(|component| match component {
                Component::Normal(name) if name != "<severian-stdlib>" => Some(name),
                _ => None,
            })
            .collect::<PathBuf>();
        let destination = root.join(relative);
        if let Some(parent) = destination.parent() {
            std::fs::create_dir_all(parent)?;
        }
        std::fs::write(&destination, &asset.contents)?;
        output.insert(asset.path.clone(), destination);
    }
    // Include directories do not themselves occur in the asset map. Add every
    // ancestor so manifest paths resolve to their materialized counterpart.
    let entries = output.clone();
    for (source, destination) in entries {
        let mut source = source.parent();
        let mut destination = destination.parent();
        while let (Some(from), Some(to)) = (source, destination) {
            output
                .entry(from.to_path_buf())
                .or_insert_with(|| to.to_path_buf());
            source = from.parent();
            destination = to.parent();
        }
    }
    Ok(output)
}

fn hash_tree(path: &Path, digest: &mut Sha256) -> Result<(), BackendError> {
    if path.is_file() {
        digest.update(path.to_string_lossy().as_bytes());
        digest.update(std::fs::read(path)?);
        return Ok(());
    }
    let mut entries = std::fs::read_dir(path)?.collect::<Result<Vec<_>, _>>()?;
    entries.sort_by_key(std::fs::DirEntry::path);
    for entry in entries {
        hash_tree(&entry.path(), digest)?;
    }
    Ok(())
}

fn verify_provider_symbols(program: &Program, objects: &[PathBuf]) -> Result<(), BackendError> {
    let nm = find_required_tool(Tool::Nm)?;
    let output = Command::new(&nm)
        .arg("-g")
        .arg("--defined-only")
        .args(objects)
        .output()?;
    if !output.status.success() {
        return Err(backend_error(format!(
            "{} failed while inspecting package-native providers: {}",
            nm.display(),
            String::from_utf8_lossy(&output.stderr).trim()
        )));
    }
    let symbols = String::from_utf8_lossy(&output.stdout);
    for function in program.metadata.external_functions.values() {
        if !symbols
            .lines()
            .any(|line| line.split_whitespace().last() == Some(function.symbol.as_str()))
        {
            return Err(backend_error(format!(
                "E0805: package `{}` native provider does not define `{}`",
                function.package, function.symbol
            )));
        }
    }
    Ok(())
}

fn backend_error(message: impl Into<String>) -> BackendError {
    BackendError(std::io::Error::other(message.into()))
}
