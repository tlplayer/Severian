use std::env;
use std::fs;
use std::path::Path;
use std::process::Command;

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

    if cfg!(target_os = "linux") {
        build_linux_bridge(&libdir, llvm_library);
    } else if cfg!(target_os = "macos") {
        println!("cargo:rustc-link-search=native={libdir}");
        println!("cargo:rustc-link-lib=static=MLIRCAPIIR");
        println!("cargo:rustc-link-lib=static=MLIRCAPIArith");
        println!("cargo:rustc-link-lib=static=MLIRCAPIRegisterEverything");
        println!("cargo:rustc-link-lib=static=MLIRCAPIControlFlow");
        println!("cargo:rustc-link-lib=static=MLIRCAPIFunc");
        println!("cargo:rustc-link-lib=static=MLIRCAPIGPU");
        println!("cargo:rustc-link-lib=static=MLIRCAPILinalg");
        println!("cargo:rustc-link-lib=static=MLIRCAPIMath");
        println!("cargo:rustc-link-lib=static=MLIRCAPIROCDL");
        println!("cargo:rustc-link-lib=static=MLIRCAPISCF");
        println!("cargo:rustc-link-lib=static=MLIRCAPITensor");
        println!("cargo:rustc-link-lib=static=MLIRCAPIVector");
        println!("cargo:rustc-link-lib=dylib=MLIR");
        println!("cargo:rustc-link-lib=dylib={llvm_library}");
        println!("cargo:rustc-link-lib=dylib=c++");
    } else {
        panic!("the direct MLIR C API bridge currently supports Linux and macOS hosts");
    }

    if cfg!(target_os = "macos") {
        let mlir = Path::new(&libdir).join("libMLIR.dylib");
        assert!(
            mlir.exists(),
            "LLVM {major} at {libdir} does not provide the shared MLIR library"
        );
    }
}

fn build_linux_bridge(libdir: &str, llvm_library: &str) {
    let output_dir = env::var("OUT_DIR").expect("Cargo did not provide OUT_DIR");
    let object = Path::new(&output_dir).join("severian_mlir_capi.o");
    let archive = Path::new(&output_dir).join("libseverian_mlir_capi.a");
    let capi_archives = [
        "libMLIRCAPIIR.a",
        "libMLIRCAPIArith.a",
        "libMLIRCAPIRegisterEverything.a",
        "libMLIRCAPIAsync.a",
        "libMLIRCAPIControlFlow.a",
        "libMLIRCAPIFunc.a",
        "libMLIRCAPIGPU.a",
        "libMLIRCAPILinalg.a",
        "libMLIRCAPILLVM.a",
        "libMLIRCAPIMath.a",
        "libMLIRCAPIROCDL.a",
        "libMLIRCAPISCF.a",
        "libMLIRCAPITensor.a",
        "libMLIRCAPIVector.a",
    ];
    let mut mlir_archives = fs::read_dir(libdir)
        .expect("could not inspect the MLIR library directory")
        .map(|entry| entry.expect("could not inspect an MLIR archive").path())
        .filter(|path| {
            path.file_name()
                .and_then(|name| name.to_str())
                .is_some_and(|name| {
                    name.starts_with("libMLIR")
                        && !name.starts_with("libMLIRCAPI")
                        && name.ends_with(".a")
                })
        })
        .collect::<Vec<_>>();
    mlir_archives.sort();
    assert!(
        !mlir_archives.is_empty(),
        "LLVM at {libdir} does not provide static MLIR component archives"
    );

    let mut linker = Command::new("cc");
    linker
        .args(["-r", "-o"])
        .arg(&object)
        .arg("-Wl,--whole-archive")
        .args(capi_archives.map(|name| Path::new(libdir).join(name)))
        .arg("-Wl,--no-whole-archive")
        .arg("-Wl,--start-group")
        .args(mlir_archives)
        .arg("-Wl,--end-group");
    let status = linker
        .status()
        .unwrap_or_else(|error| panic!("failed to invoke the native linker: {error}"));
    assert!(status.success(), "failed to combine the MLIR C API bridge");

    let status = Command::new("ar")
        .args(["crs"])
        .arg(&archive)
        .arg(&object)
        .status()
        .unwrap_or_else(|error| panic!("failed to invoke the native archiver: {error}"));
    assert!(status.success(), "failed to archive the MLIR C API bridge");
    assert!(
        fs::metadata(&archive).is_ok(),
        "the native archiver did not produce {}",
        archive.display()
    );
    println!("cargo:rustc-link-search=native={output_dir}");
    println!("cargo:rustc-link-search=native={libdir}");
    println!("cargo:rustc-link-lib=static=severian_mlir_capi");
    println!("cargo:rustc-link-lib=dylib={llvm_library}");
    println!("cargo:rustc-link-lib=dylib=stdc++");
    println!("cargo:rustc-link-lib=dylib=m");
    println!("cargo:rustc-link-lib=dylib=dl");
    println!("cargo:rustc-link-lib=dylib=rt");
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
