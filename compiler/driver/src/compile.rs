use crate::{
    artifact::{write_mlir, write_stablehlo, Artifact, ArtifactKind, ArtifactLayout},
    options::{CompileOptions, EmitKind},
    pipeline::PipelinePlan,
    target::{BackendFamily, DriverTarget},
    Compilation, CompileError,
};
use std::path::Path;

#[derive(Debug, Clone)]
pub struct CompileRequest<'a> {
    pub source: CompileInput<'a>,
    pub options: CompileOptions,
}

#[derive(Debug, Clone)]
pub enum CompileInput<'a> {
    Source(&'a str),
    Path(&'a Path),
}

#[derive(Debug, Clone)]
pub struct CompileOutput {
    pub compilation: Compilation,
    pub pipeline: PipelinePlan,
    pub artifact: Option<Artifact>,
}

pub fn compile(request: CompileRequest<'_>) -> Result<CompileOutput, CompileError> {
    let pipeline = PipelinePlan::build(&request.options);

    let compilation = match request.source {
        CompileInput::Source(source) => crate::compile_source(source)?,
        CompileInput::Path(path) => crate::compile_path(path)?,
    };

    let source_path = match request.source {
        CompileInput::Path(path) => Some(path),
        CompileInput::Source(_) => None,
    };

    let output = request.options.output_path(source_path);
    let layout = ArtifactLayout::new(&output)?;

    let artifact = emit_compilation(&compilation, &request.options, &layout)?;

    if !request.options.keep_intermediates {
        layout.cleanup();
    }

    Ok(CompileOutput {
        compilation,
        pipeline,
        artifact,
    })
}

fn emit_compilation(
    compilation: &Compilation,
    options: &CompileOptions,
    layout: &ArtifactLayout,
) -> Result<Option<Artifact>, CompileError> {
    match options.emit {
        EmitKind::Mlir => {
            std::fs::write(&layout.output, compilation.mlir.as_str())?;
            return Ok(Some(Artifact::new(ArtifactKind::Mlir, &layout.output)));
        }

        EmitKind::StableHlo if !options.target.is_xla() => {
            return Err(CompileError::Optimization(
                "StableHLO emission requires an XLA target".into(),
            ));
        }

        _ => {}
    }

    match options.target.family() {
        BackendFamily::Native | BackendFamily::Llvm => {
            emit_native(compilation, options, layout)
        }
        BackendFamily::Amd => emit_amd(compilation, options, layout),
        BackendFamily::Nvidia => emit_nvidia(compilation, options, layout),
        BackendFamily::Xla => emit_xla(compilation, options, layout),
        BackendFamily::Spirv => emit_spirv(compilation, options, layout),
    }
}

fn emit_native(
    compilation: &Compilation,
    options: &CompileOptions,
    layout: &ArtifactLayout,
) -> Result<Option<Artifact>, CompileError> {
    match options.emit {
        EmitKind::Executable => {
            crate::compile_native(compilation, &layout.output)?;
            Ok(Some(Artifact::new(ArtifactKind::Executable, &layout.output)))
        }

        EmitKind::Mlir => unreachable!(),

        EmitKind::LlvmIr | EmitKind::Object | EmitKind::SharedLibrary => {
            let _ = write_mlir(&compilation.mlir, &layout.source_mlir)?;
            Err(CompileError::Optimization(format!(
                "{:?} emission needs the backend intermediate-artifact API",
                options.emit
            )))
        }

        EmitKind::StableHlo => Err(CompileError::Optimization(
            "native target does not emit StableHLO".into(),
        )),
    }
}

fn emit_amd(
    compilation: &Compilation,
    options: &CompileOptions,
    layout: &ArtifactLayout,
) -> Result<Option<Artifact>, CompileError> {
    if options.run_iree_passes {
        let _plan = severian_passes::iree::IreePlan::analyze(&compilation.optimized_hir);
    }
    let DriverTarget::Amd { architecture, .. } = &options.target else {
        unreachable!();
    };

    let chip = architecture
        .clone()
        .or_else(crate::detect_amd_gpu_chip)
        .ok_or_else(|| {
            CompileError::Optimization(
                "AMD target requires a gfx architecture; pass `rocm:gfx1100` or configure detection"
                    .into(),
            )
        })?;

    match options.emit {
        EmitKind::Executable => {
            crate::compile_rocm(compilation, &layout.output, &chip)?;
            Ok(Some(Artifact::new(ArtifactKind::Executable, &layout.output)))
        }

        EmitKind::Mlir => unreachable!(),

        EmitKind::LlvmIr | EmitKind::Object | EmitKind::SharedLibrary => {
            let module = crate::lower_to_rocdl(compilation, &chip)?;
            std::fs::write(&layout.output, module.as_str())?;
            Ok(Some(Artifact::new(ArtifactKind::Mlir, &layout.output)))
        }

        EmitKind::StableHlo => Err(CompileError::Optimization(
            "direct AMD target does not emit StableHLO; use `xla:gpu` for the XLA path".into(),
        )),
    }
}

fn emit_nvidia(
    compilation: &Compilation,
    options: &CompileOptions,
    layout: &ArtifactLayout,
) -> Result<Option<Artifact>, CompileError> {
    if options.run_iree_passes {
        let _plan = severian_passes::iree::IreePlan::analyze(&compilation.optimized_hir);
    }
    let _ = write_mlir(&compilation.mlir, &layout.source_mlir)?;

    Err(CompileError::Optimization(
        "NVIDIA target selected, but severian-backend still needs the NVVM/CUDA artifact implementation"
            .into(),
    ))
}

fn emit_xla(
    compilation: &Compilation,
    options: &CompileOptions,
    layout: &ArtifactLayout,
) -> Result<Option<Artifact>, CompileError> {
    let DriverTarget::Xla {
        platform,
        device_ordinal,
    } = &options.target
    else {
        unreachable!();
    };

    let destination = match (platform, device_ordinal) {
        (Some(platform), Some(device)) => format!("{platform}@{device}"),
        (Some(platform), None) => platform.clone(),
        (None, Some(device)) => format!("device {device}"),
        (None, None) => "default PJRT device".into(),
    };

    let mut xla_hir = compilation.optimized_hir.clone();
    severian_xla::optimization_pipeline()
        .run(&mut xla_hir)
        .map_err(|error| CompileError::Optimization(error.to_string()))?;
    let _plan = severian_xla::XlaOptimizationPlan::analyze(&xla_hir);
    let lowered = severian_lowering::stablehlo::lower_program(&xla_hir)
        .map_err(|error| CompileError::Optimization(error.to_string()))?;
    let stablehlo = severian_xla::StableHloModule::from_text(lowered.as_str());

    if options.emit == EmitKind::StableHlo {
        return write_stablehlo(&stablehlo, &layout.output)
            .map(Some)
            .map_err(CompileError::Io);
    }

    let _ = write_stablehlo(&stablehlo, &layout.stablehlo)?;
    Err(CompileError::Optimization(format!(
        "StableHLO for XLA target `{destination}` is ready, but selecting/loading a PJRT plugin requires an explicit plugin path that DriverTarget does not yet represent"
    )))
}

fn emit_spirv(
    compilation: &Compilation,
    options: &CompileOptions,
    layout: &ArtifactLayout,
) -> Result<Option<Artifact>, CompileError> {
    if options.run_iree_passes {
        let _plan = severian_passes::iree::IreePlan::analyze(&compilation.optimized_hir);
    }
    let _ = write_mlir(&compilation.mlir, &layout.source_mlir)?;

    Err(CompileError::Optimization(
        "SPIR-V target selected, but the backend still needs SPIR-V binary/link emission".into(),
    ))
}
