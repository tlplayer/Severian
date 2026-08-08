use super::features::{detect_features, maximum_fixed_vector_bits, CpuFeature};
use crate::target::{
    host_architecture, host_operating_system, Architecture, Backend, OperatingSystem,
};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CpuTarget {
    pub architecture: Architecture,
    pub operating_system: OperatingSystem,
    pub backend: Backend,
    pub llvm_triple: String,
    pub cpu: String,
    pub features: Vec<CpuFeature>,
    pub pointer_bits: u8,
}

impl CpuTarget {
    pub fn detect() -> Self {
        let architecture = host_architecture();
        let operating_system = host_operating_system();
        let features = detect_features();

        Self {
            architecture,
            operating_system,
            backend: Backend::Native,
            llvm_triple: native_llvm_triple(architecture, operating_system),
            cpu: "native".into(),
            features,
            pointer_bits: usize::BITS as u8,
        }
    }

    pub fn generic(
        architecture: Architecture,
        operating_system: OperatingSystem,
    ) -> Self {
        Self {
            architecture,
            operating_system,
            backend: Backend::Llvm,
            llvm_triple: native_llvm_triple(architecture, operating_system),
            cpu: "generic".into(),
            features: Vec::new(),
            pointer_bits: if matches!(architecture, Architecture::Wasm32) {
                32
            } else {
                64
            },
        }
    }

    pub fn has(&self, feature: CpuFeature) -> bool {
        self.features.contains(&feature)
    }

    pub fn maximum_fixed_vector_bits(&self) -> Option<u16> {
        maximum_fixed_vector_bits(&self.features)
    }

    pub fn llvm_features(&self) -> String {
        self.features
            .iter()
            .map(|feature| format!("+{}", feature.llvm_name()))
            .collect::<Vec<_>>()
            .join(",")
    }

    pub fn supports_simd(&self) -> bool {
        self.maximum_fixed_vector_bits().is_some()
            || self.has(CpuFeature::Sve)
            || self.has(CpuFeature::Sve2)
    }
}

fn native_llvm_triple(
    architecture: Architecture,
    operating_system: OperatingSystem,
) -> String {
    let architecture = match architecture {
        Architecture::X86_64 => "x86_64",
        Architecture::Aarch64 => "aarch64",
        Architecture::Riscv64 => "riscv64",
        Architecture::Wasm32 => "wasm32",
        Architecture::Wasm64 => "wasm64",
        Architecture::Other => std::env::consts::ARCH,
    };

    let suffix = match operating_system {
        OperatingSystem::Linux => "unknown-linux-gnu",
        OperatingSystem::Windows => "pc-windows-msvc",
        OperatingSystem::MacOs => "apple-darwin",
        OperatingSystem::FreeBsd => "unknown-freebsd",
        OperatingSystem::Android => "linux-android",
        OperatingSystem::Ios => "apple-ios",
        OperatingSystem::Wasi => "wasi",
        OperatingSystem::Other => "unknown-unknown",
    };

    format!("{architecture}-{suffix}")
}
