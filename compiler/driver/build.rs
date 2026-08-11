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
    println!("cargo:rerun-if-changed={}", xla_manifest.display());
    println!("cargo:rerun-if-changed={}", manifest.join("../xla/src").display());
}
