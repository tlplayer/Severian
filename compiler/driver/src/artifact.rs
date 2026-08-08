use crate::options::EmitKind;
use severian_hir::Program;
use severian_mlir::Module;
use std::{fs, io, path::{Path, PathBuf}};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ArtifactKind {
    Hir, Mlir, StableHlo, LlvmIr, Object, Executable, SharedLibrary, PjrtExecutable,
}

#[derive(Debug, Clone)]
pub struct Artifact {
    pub kind: ArtifactKind,
    pub path: PathBuf,
}

impl Artifact {
    pub fn new(kind: ArtifactKind, path: impl Into<PathBuf>) -> Self {
        Self { kind, path: path.into() }
    }
    pub fn exists(&self) -> bool { self.path.exists() }
}

#[derive(Debug, Clone)]
pub struct ArtifactLayout {
    pub output: PathBuf,
    pub working_directory: PathBuf,
    pub source_mlir: PathBuf,
    pub stablehlo: PathBuf,
    pub llvm_mlir: PathBuf,
    pub llvm_ir: PathBuf,
    pub object: PathBuf,
}

impl ArtifactLayout {
    pub fn new(output: impl Into<PathBuf>) -> io::Result<Self> {
        let output = output.into();
        let directory = output.parent().unwrap_or_else(|| Path::new("."));
        let stem = output.file_stem().and_then(|v| v.to_str()).unwrap_or("severian");
        let working_directory = directory.join(format!(".{stem}.severian"));
        fs::create_dir_all(&working_directory)?;

        Ok(Self {
            source_mlir: working_directory.join(format!("{stem}.mlir")),
            stablehlo: working_directory.join(format!("{stem}.stablehlo.mlir")),
            llvm_mlir: working_directory.join(format!("{stem}.llvm.mlir")),
            llvm_ir: working_directory.join(format!("{stem}.ll")),
            object: working_directory.join(format!("{stem}.o")),
            output,
            working_directory,
        })
    }

    pub fn final_kind(&self, emit: EmitKind) -> ArtifactKind {
        match emit {
            EmitKind::Executable => ArtifactKind::Executable,
            EmitKind::Mlir => ArtifactKind::Mlir,
            EmitKind::LlvmIr => ArtifactKind::LlvmIr,
            EmitKind::StableHlo => ArtifactKind::StableHlo,
            EmitKind::Object => ArtifactKind::Object,
            EmitKind::SharedLibrary => ArtifactKind::SharedLibrary,
        }
    }

    pub fn cleanup(&self) { let _ = fs::remove_dir_all(&self.working_directory); }
}

pub fn write_mlir(module: &Module, path: impl AsRef<Path>) -> io::Result<Artifact> {
    let path = path.as_ref();
    fs::write(path, module.as_str())?;
    Ok(Artifact::new(ArtifactKind::Mlir, path))
}

pub fn write_stablehlo(
    module: &severian_xla::StableHloModule,
    path: impl AsRef<Path>,
) -> io::Result<Artifact> {
    let path = path.as_ref();
    fs::write(path, module.bytes())?;
    Ok(Artifact::new(ArtifactKind::StableHlo, path))
}

pub fn write_hir_debug(program: &Program, path: impl AsRef<Path>) -> io::Result<Artifact> {
    let path = path.as_ref();
    fs::write(path, format!("{program:#?}"))?;
    Ok(Artifact::new(ArtifactKind::Hir, path))
}
