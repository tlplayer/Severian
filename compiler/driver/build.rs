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
    let mut directories = fs::read_dir(&library_root)
        .unwrap_or_else(|error| {
            panic!(
                "reading the Severian standard library at {} failed: {error}",
                library_root.display()
            )
        })
        .map(|entry| {
            entry
                .expect("standard library directory entry is readable")
                .path()
        })
        .filter(|path| path.join("package.toml").is_file())
        .collect::<Vec<_>>();
    directories.sort();

    let mut generated = String::from(
        "pub(crate) const EMBEDDED_OFFICIAL_PACKAGES: &[severian_package::EmbeddedOfficialPackage<'static>] = &[\n",
    );
    for directory in directories {
        let name = directory
            .file_name()
            .and_then(|name| name.to_str())
            .expect("standard library package names are UTF-8");
        let manifest_path = directory.join("package.toml");
        let source_path = directory.join("src/lib.sev");
        let manifest_source = fs::read_to_string(&manifest_path)
            .unwrap_or_else(|error| panic!("reading {} failed: {error}", manifest_path.display()));
        let source = fs::read_to_string(&source_path)
            .unwrap_or_else(|error| panic!("reading {} failed: {error}", source_path.display()));
        generated.push_str(&format!(
            "    severian_package::EmbeddedOfficialPackage {{ name: {name:?}, manifest: {manifest_source:?}, source: {source:?} }},\n"
        ));
        println!("cargo:rerun-if-changed={}", manifest_path.display());
        println!("cargo:rerun-if-changed={}", source_path.display());
    }
    generated.push_str("];\n");
    fs::write(out.join("official_libraries.rs"), generated)
        .expect("writing embedded Severian library assets must succeed");
}
