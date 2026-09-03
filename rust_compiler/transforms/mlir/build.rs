use std::env;
use std::path::Path;
use std::process::Command;

const CAPI_LIBRARIES: [&str; 14] = [
    "MLIRCAPIIR",
    "MLIRCAPIArith",
    "MLIRCAPIRegisterEverything",
    "MLIRCAPIAsync",
    "MLIRCAPIControlFlow",
    "MLIRCAPIFunc",
    "MLIRCAPIGPU",
    "MLIRCAPILinalg",
    "MLIRCAPILLVM",
    "MLIRCAPIMath",
    "MLIRCAPIROCDL",
    "MLIRCAPISCF",
    "MLIRCAPITensor",
    "MLIRCAPIVector",
];

fn main() {
    println!("cargo:rerun-if-env-changed=SEVERIAN_LLVM_CONFIG");

    let llvm_config = env::var("SEVERIAN_LLVM_CONFIG")
        .ok()
        .filter(|path| !path.is_empty())
        .unwrap_or_else(find_llvm_config);
    let libdir = output(&llvm_config, &["--libdir"]);
    let version = output(&llvm_config, &["--version"]);
    let major = version
        .split('.')
        .next()
        .expect("llvm-config returned an empty version");
    let llvm_libraries = output(&llvm_config, &["--link-shared", "--libnames"]);
    let llvm_library = llvm_libraries
        .split_whitespace()
        .next()
        .map(library_name)
        .expect("llvm-config did not report a shared LLVM library");

    if !cfg!(target_os = "linux") && !cfg!(target_os = "macos") {
        panic!("the direct MLIR C API bridge currently supports Linux and macOS hosts");
    }

    // Link the small upstream C API shims directly against the shared MLIR
    // library. The former Linux path used `cc -r --whole-archive` to merge
    // every static MLIR component into one object, consuming gigabytes before
    // the Severian compiler itself could be built.
    println!("cargo:rustc-link-search=native={libdir}");
    for library in CAPI_LIBRARIES {
        println!("cargo:rustc-link-lib=static={library}");
    }
    println!("cargo:rustc-link-lib=dylib=MLIR");
    println!("cargo:rustc-link-lib=dylib={llvm_library}");
    println!("cargo:rustc-link-lib=dylib=stdc++");
    if cfg!(target_os = "linux") {
        println!("cargo:rustc-link-arg=-Wl,-rpath,{libdir}");
    }

    let extension = if cfg!(target_os = "macos") {
        "dylib"
    } else {
        "so"
    };
    let mlir = Path::new(&libdir).join(format!("libMLIR.{extension}"));
    assert!(
        mlir.exists(),
        "LLVM {major} at {libdir} does not provide the shared MLIR library"
    );
}

fn find_llvm_config() -> String {
    for candidate in ["llvm-config-21", "llvm-config"] {
        if Command::new(candidate)
            .arg("--version")
            .output()
            .is_ok_and(|output| output.status.success())
        {
            return candidate.to_owned();
        }
    }
    panic!("MLIR integration requires llvm-config (set SEVERIAN_LLVM_CONFIG)");
}

fn output(program: &str, arguments: &[&str]) -> String {
    let output = Command::new(program)
        .args(arguments)
        .output()
        .unwrap_or_else(|error| panic!("failed to run {program}: {error}"));
    assert!(
        output.status.success(),
        "{program} {} failed: {}",
        arguments.join(" "),
        String::from_utf8_lossy(&output.stderr)
    );
    String::from_utf8(output.stdout)
        .expect("llvm-config output was not UTF-8")
        .trim()
        .to_owned()
}

fn library_name(file: &str) -> &str {
    file.strip_prefix("lib")
        .and_then(|name| name.split('.').next())
        .unwrap_or(file)
}
