use std::env;
use std::process::Command;

fn main() {
    println!("cargo:rerun-if-env-changed=SEVERIAN_LLVM_CONFIG");
    if !cfg!(target_os = "linux") {
        return;
    }
    let llvm_config = env::var("SEVERIAN_LLVM_CONFIG")
        .ok()
        .filter(|path| !path.is_empty())
        .unwrap_or_else(|| "llvm-config-21".into());
    let output = Command::new(&llvm_config)
        .arg("--libdir")
        .output()
        .unwrap_or_else(|error| panic!("failed to invoke {llvm_config}: {error}"));
    assert!(
        output.status.success(),
        "{llvm_config} --libdir failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let libdir = String::from_utf8(output.stdout)
        .expect("llvm-config libdir was not UTF-8")
        .trim()
        .to_owned();
    println!("cargo:rustc-link-arg=-Wl,-rpath,{libdir}");
}
