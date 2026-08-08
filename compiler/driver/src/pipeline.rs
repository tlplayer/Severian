use crate::{
    options::{CompileOptions, EmitKind},
    target::{BackendFamily, DriverTarget},
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PipelineStage {
    Lex,
    Parse,
    Semantic,
    Ownership,
    EffectAnalysis,
    TypeAnalysis,
    GenericOptimization,
    XlaOptimization,
    IreePlanning,
    LowerLinalg,
    LowerStableHlo,
    LowerGpu,
    Bufferize,
    LowerLlvm,
    CompileXla,
    LoadPjrt,
    LinkRuntime,
    LinkNative,
}

#[derive(Debug, Clone)]
pub struct PipelinePlan {
    pub target: DriverTarget,
    pub stages: Vec<PipelineStage>,
}

impl PipelinePlan {
    pub fn build(options: &CompileOptions) -> Self {
        let mut stages = vec![
            PipelineStage::Lex,
            PipelineStage::Parse,
            PipelineStage::Semantic,
            PipelineStage::Ownership,
            PipelineStage::EffectAnalysis,
            PipelineStage::TypeAnalysis,
        ];

        if options.run_generic_passes {
            stages.push(PipelineStage::GenericOptimization);
        }

        match options.target.family() {
            BackendFamily::Native | BackendFamily::Llvm => {
                stages.extend([
                    PipelineStage::LowerLinalg,
                    PipelineStage::Bufferize,
                    PipelineStage::LowerLlvm,
                ]);

                if matches!(options.emit, EmitKind::Executable | EmitKind::SharedLibrary) {
                    stages.extend([
                        PipelineStage::LinkRuntime,
                        PipelineStage::LinkNative,
                    ]);
                }
            }

            BackendFamily::Xla => {
                if options.run_xla_passes {
                    stages.push(PipelineStage::XlaOptimization);
                }
                stages.extend([
                    PipelineStage::LowerStableHlo,
                    PipelineStage::CompileXla,
                    PipelineStage::LoadPjrt,
                ]);
            }

            BackendFamily::Nvidia | BackendFamily::Amd | BackendFamily::Spirv => {
                if options.run_iree_passes {
                    stages.push(PipelineStage::IreePlanning);
                }

                stages.extend([
                    PipelineStage::LowerGpu,
                    PipelineStage::Bufferize,
                    PipelineStage::LowerLlvm,
                ]);

                if matches!(options.emit, EmitKind::Executable | EmitKind::SharedLibrary) {
                    stages.extend([
                        PipelineStage::LinkRuntime,
                        PipelineStage::LinkNative,
                    ]);
                }
            }
        }

        Self {
            target: options.target.clone(),
            stages,
        }
    }

    pub fn contains(&self, stage: PipelineStage) -> bool {
        self.stages.contains(&stage)
    }

    pub fn description(&self) -> String {
        self.stages
            .iter()
            .map(|stage| stage.name())
            .collect::<Vec<_>>()
            .join(" -> ")
    }
}

impl PipelineStage {
    pub const fn name(self) -> &'static str {
        match self {
            Self::Lex => "lex",
            Self::Parse => "parse",
            Self::Semantic => "semantic",
            Self::Ownership => "ownership",
            Self::EffectAnalysis => "effects",
            Self::TypeAnalysis => "types",
            Self::GenericOptimization => "optimize",
            Self::XlaOptimization => "xla-optimize",
            Self::IreePlanning => "iree-plan",
            Self::LowerLinalg => "linalg",
            Self::LowerStableHlo => "stablehlo",
            Self::LowerGpu => "gpu",
            Self::Bufferize => "bufferize",
            Self::LowerLlvm => "llvm",
            Self::CompileXla => "xla-compile",
            Self::LoadPjrt => "pjrt",
            Self::LinkRuntime => "runtime",
            Self::LinkNative => "link",
        }
    }
}
