#![forbid(unsafe_code)]

mod pipeline;
pub use pipeline::{compile_file, compile_source, CompileError};

#[cfg(test)]
mod tests {
    use super::*;
    use severian_source::SourceFile;

    #[test]
    fn compiles_integer_addition_to_runnable_executable() {
        let source = SourceFile::virtual_source("addition.sev", "b = 2\na = 1 + b\n");
        let output =
            std::env::temp_dir().join(format!("severian-int-slice-{}", std::process::id()));
        let artifact = compile_source(&source, &output).unwrap();
        assert!(artifact.path.exists());
        assert!(std::process::Command::new(&artifact.path)
            .status()
            .unwrap()
            .success());
        std::fs::remove_file(output).unwrap();
    }
}
