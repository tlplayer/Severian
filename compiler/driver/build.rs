use std::{env, fs, path::PathBuf, process::Command};

fn main() {
    let out = PathBuf::from(env::var_os("OUT_DIR").expect("OUT_DIR is set by Cargo"));
    let manifest = PathBuf::from(env::var_os("CARGO_MANIFEST_DIR").unwrap());
    let xla_manifest = manifest.join("../xla/Cargo.toml");
    let target = out.join("xla-runtime-target");
    let output_profile = env::var("PROFILE").unwrap_or_else(|_| "debug".into());
    let cargo_profile = if output_profile == "debug" {
        "dev"
    } else {
        output_profile.as_str()
    };
    let status = Command::new(env::var_os("CARGO").unwrap_or_else(|| "cargo".into()))
        .args(["build", "--manifest-path"])
        .arg(&xla_manifest)
        .args(["--lib", "--profile", cargo_profile, "--target-dir"])
        .arg(&target)
        .status()
        .expect("starting the XLA runtime build must succeed");
    if !status.success() {
        panic!("building the embedded XLA runtime failed with {status}");
    }
    let runtime = target.join(&output_profile).join("libseverian_xla.a");
    fs::copy(&runtime, out.join("libseverian_xla.a"))
        .expect("copying the XLA runtime into the sev binary assets must succeed");
    embed_official_libraries(&manifest, &out);
    println!("cargo:rerun-if-changed={}", xla_manifest.display());
    println!(
        "cargo:rerun-if-changed={}",
        manifest.join("../xla/src").display()
    );
}

fn embed_official_libraries(manifest: &std::path::Path, out: &std::path::Path) {
    let library_root = manifest.join("../../library");
    let mut directories = Vec::new();
    collect_package_directories(&library_root, &mut directories);
    directories.sort();

    let mut generated = String::from(
        "pub(crate) const EMBEDDED_OFFICIAL_PACKAGES: &[severian_package::EmbeddedOfficialPackage<'static>] = &[\n",
    );
    for directory in directories {
        let manifest_path = directory.join("package.toml");
        let manifest_source = fs::read_to_string(&manifest_path)
            .unwrap_or_else(|error| panic!("reading {} failed: {error}", manifest_path.display()));
        let parsed = toml::from_str::<toml::Value>(&manifest_source)
            .unwrap_or_else(|error| panic!("parsing {} failed: {error}", manifest_path.display()));
        let name = parsed
            .get("package")
            .and_then(|package| package.get("name"))
            .and_then(toml::Value::as_str)
            .unwrap_or_else(|| panic!("{} has no package.name", manifest_path.display()));
        let library_path = parsed
            .get("lib")
            .and_then(|library| library.get("path"))
            .and_then(toml::Value::as_str)
            .unwrap_or("src/lib.sev");
        let source_path = directory.join(library_path);
        let source = fs::read_to_string(&source_path)
            .unwrap_or_else(|error| panic!("reading {} failed: {error}", source_path.display()));
        let canonical_source = source_path
            .canonicalize()
            .unwrap_or_else(|error| panic!("resolving {} failed: {error}", source_path.display()));
        let mut module_paths = Vec::new();
        collect_package_modules(&directory, &directory, &canonical_source, &mut module_paths);
        module_paths.sort();
        let modules = module_paths
            .iter()
            .map(|path| {
                let module_source = fs::read_to_string(path)
                    .unwrap_or_else(|error| panic!("reading {} failed: {error}", path.display()));
                let relative = path
                    .strip_prefix(&directory)
                    .expect("module source is inside its package")
                    .to_string_lossy();
                println!("cargo:rerun-if-changed={}", path.display());
                format!(
                    "severian_package::EmbeddedOfficialModule {{ path: {relative:?}, source: {module_source:?} }}"
                )
            })
            .collect::<Vec<_>>()
            .join(", ");
        generated.push_str(&format!(
            "    severian_package::EmbeddedOfficialPackage {{ name: {name:?}, manifest: {manifest_source:?}, source: {source:?}, modules: &[{modules}] }},\n"
        ));
        println!("cargo:rerun-if-changed={}", manifest_path.display());
        println!("cargo:rerun-if-changed={}", source_path.display());
    }
    generated.push_str("];\n");
    fs::write(out.join("official_libraries.rs"), generated)
        .expect("writing embedded Severian library assets must succeed");
}

fn collect_package_directories(directory: &std::path::Path, packages: &mut Vec<PathBuf>) {
    let mut entries = fs::read_dir(directory)
        .unwrap_or_else(|error| panic!("reading {} failed: {error}", directory.display()))
        .map(|entry| {
            entry
                .expect("standard library directory entry is readable")
                .path()
        })
        .filter(|path| path.is_dir() && !ignored_directory(path))
        .collect::<Vec<_>>();
    entries.sort();
    for path in entries {
        if path.join("package.toml").is_file() {
            packages.push(path.clone());
        }
        collect_package_directories(&path, packages);
    }
}

fn collect_package_modules(
    package_root: &std::path::Path,
    directory: &std::path::Path,
    library_source: &std::path::Path,
    modules: &mut Vec<PathBuf>,
) {
    let mut entries = fs::read_dir(directory)
        .unwrap_or_else(|error| panic!("reading {} failed: {error}", directory.display()))
        .map(|entry| {
            entry
                .expect("standard library source entry is readable")
                .path()
        })
        .collect::<Vec<_>>();
    entries.sort();
    for path in entries {
        if path.is_dir() {
            if ignored_directory(&path) {
                continue;
            }
            if path != package_root && path.join("package.toml").is_file() {
                continue;
            }
            collect_package_modules(package_root, &path, library_source, modules);
        } else if path.extension().and_then(|extension| extension.to_str()) == Some("sev")
            && path
                .canonicalize()
                .ok()
                .is_none_or(|candidate| candidate != library_source)
        {
            modules.push(path);
        }
    }
}

fn ignored_directory(path: &std::path::Path) -> bool {
    path.file_name()
        .and_then(|name| name.to_str())
        .is_some_and(|name| name == "target" || name.starts_with('.'))
}
