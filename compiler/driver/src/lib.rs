#![forbid(unsafe_code)]

use severian_ast::Span;
use severian_hir::{Instruction, Program};
use severian_mlir::Module;
use std::fmt;
use std::path::{Path, PathBuf};
use std::process::Command;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Compilation {
    pub hir: Program,
    pub mlir: Module,
}

#[derive(Debug)]
pub enum CompileError {
    Io(std::io::Error),
    Frontend {
        stage: &'static str,
        span: Span,
        message: String,
    },
    Ownership(String),
}

impl fmt::Display for CompileError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CompileError::Io(error) => error.fmt(formatter),
            CompileError::Frontend {
                stage,
                span,
                message,
            } => write!(
                formatter,
                "{stage} error at bytes {}..{}: {message}",
                span.start, span.end
            ),
            CompileError::Ownership(message) => write!(formatter, "ownership error: {message}"),
        }
    }
}

impl std::error::Error for CompileError {}

impl From<std::io::Error> for CompileError {
    fn from(error: std::io::Error) -> Self {
        Self::Io(error)
    }
}

pub fn compile_source(source: &str) -> Result<Compilation, CompileError> {
    let tokens = severian_lexer::lex(source).map_err(|error| CompileError::Frontend {
        stage: "lexer",
        span: error.span,
        message: error.message,
    })?;
    let ast = severian_parser::parse(&tokens).map_err(|error| CompileError::Frontend {
        stage: "parser",
        span: error.span,
        message: error.message,
    })?;
    let hir = severian_semantic::analyze(&ast).map_err(|error| CompileError::Frontend {
        stage: "semantic",
        span: error.span,
        message: error.message,
    })?;
    severian_ownership::check(&hir).map_err(|error| CompileError::Ownership(error.message))?;
    let mlir = severian_lowering::lower(&hir);

    Ok(Compilation { hir, mlir })
}

pub fn compile_path(path: &Path) -> Result<Compilation, CompileError> {
    compile_source(&std::fs::read_to_string(path)?)
}

pub fn compile_native(compilation: &Compilation, output: &Path) -> Result<(), CompileError> {
    let prefix = std::env::temp_dir().join(format!(
        "severian-compile-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("system time must follow the Unix epoch")
            .as_nanos()
    ));
    let source_mlir = prefix.with_extension("mlir");
    let checked_mlir = prefix.with_extension("checked.mlir");
    let llvm_ir = prefix.with_extension("ll");

    let result = (|| {
        std::fs::write(&source_mlir, compilation.mlir.as_str())?;
        run_tool(
            find_tool(&["mlir-opt", "mlir-opt-21", "/usr/lib/llvm-21/bin/mlir-opt"])
                .ok_or_else(|| missing_tool("mlir-opt"))?,
            &[
                source_mlir.as_path(),
                Path::new("-o"),
                checked_mlir.as_path(),
            ],
        )?;
        run_tool(
            find_tool(&[
                "mlir-translate",
                "mlir-translate-21",
                "/usr/lib/llvm-21/bin/mlir-translate",
            ])
            .ok_or_else(|| missing_tool("mlir-translate"))?,
            &[
                Path::new("--mlir-to-llvmir"),
                checked_mlir.as_path(),
                Path::new("-o"),
                llvm_ir.as_path(),
            ],
        )?;
        run_tool(
            find_tool(&["clang", "clang-21", "/usr/bin/clang-21"])
                .ok_or_else(|| missing_tool("clang"))?,
            &[llvm_ir.as_path(), Path::new("-o"), output],
        )
    })();

    for temporary in [&source_mlir, &checked_mlir, &llvm_ir] {
        let _ = std::fs::remove_file(temporary);
    }

    result
}

pub fn run(program: &Program, mut write_line: impl FnMut(&str)) -> Result<(), CompileError> {
    let main = program.main().ok_or_else(|| CompileError::Frontend {
        stage: "semantic",
        span: Span::default(),
        message: "program must define `main`".into(),
    })?;

    for instruction in &main.instructions {
        match instruction {
            Instruction::Print(value) => write_line(value),
        }
    }
    Ok(())
}

fn find_tool(candidates: &[&str]) -> Option<PathBuf> {
    for candidate in candidates {
        let path = Path::new(candidate);
        if path.components().count() > 1 && path.is_file() {
            return Some(path.into());
        }

        if let Some(paths) = std::env::var_os("PATH") {
            for directory in std::env::split_paths(&paths) {
                let executable = directory.join(candidate);
                if executable.is_file() {
                    return Some(executable);
                }
            }
        }
    }
    None
}

fn run_tool(tool: PathBuf, args: &[&Path]) -> Result<(), CompileError> {
    let output = Command::new(&tool).args(args).output()?;
    if output.status.success() {
        return Ok(());
    }

    let stderr = String::from_utf8_lossy(&output.stderr).trim().to_owned();
    Err(CompileError::Io(std::io::Error::other(format!(
        "{} failed: {stderr}",
        tool.display()
    ))))
}

fn missing_tool(name: &str) -> CompileError {
    CompileError::Io(std::io::Error::new(
        std::io::ErrorKind::NotFound,
        format!("required tool `{name}` was not found"),
    ))
}
