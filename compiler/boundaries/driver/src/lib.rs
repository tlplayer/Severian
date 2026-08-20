#![forbid(unsafe_code)]

mod pipeline;

pub use pipeline::{compile_file, compile_source, CompileError, Compiler};

#[cfg(test)]
mod tests {
    use super::*;
    use severian_source::SourceFile;

    #[test]
    fn compiles_symmetric_i32_additions_to_a_runnable_executable() {
        let source =
            SourceFile::virtual_source("addition.sev", "x: i32 = 10\na = x + 1\nb = 1 + x\n");
        let output = std::env::temp_dir().join(format!(
            "severian-universal-pipeline-{}",
            std::process::id()
        ));
        let artifact = compile_source(&source, &output).unwrap();
        assert!(artifact.path.exists());
        assert!(std::process::Command::new(&artifact.path)
            .status()
            .unwrap()
            .success());
        std::fs::remove_file(output).unwrap();
    }
}
