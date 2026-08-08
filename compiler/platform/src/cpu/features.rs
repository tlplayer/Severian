#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum CpuFeature {
    Sse2,
    Sse3,
    Ssse3,
    Sse41,
    Sse42,
    Avx,
    Avx2,
    Avx512F,
    Avx512Bw,
    Avx512Dq,
    Avx512Vl,
    Avx512Vnni,
    Fma,
    Bmi1,
    Bmi2,
    Aes,
    Sha,
    Neon,
    Sve,
    Sve2,
    DotProd,
    Fp16,
}

impl CpuFeature {
    pub const fn llvm_name(self) -> &'static str {
        match self {
            Self::Sse2 => "sse2",
            Self::Sse3 => "sse3",
            Self::Ssse3 => "ssse3",
            Self::Sse41 => "sse4.1",
            Self::Sse42 => "sse4.2",
            Self::Avx => "avx",
            Self::Avx2 => "avx2",
            Self::Avx512F => "avx512f",
            Self::Avx512Bw => "avx512bw",
            Self::Avx512Dq => "avx512dq",
            Self::Avx512Vl => "avx512vl",
            Self::Avx512Vnni => "avx512vnni",
            Self::Fma => "fma",
            Self::Bmi1 => "bmi",
            Self::Bmi2 => "bmi2",
            Self::Aes => "aes",
            Self::Sha => "sha",
            Self::Neon => "neon",
            Self::Sve => "sve",
            Self::Sve2 => "sve2",
            Self::DotProd => "dotprod",
            Self::Fp16 => "fullfp16",
        }
    }

    pub const fn vector_bits(self) -> Option<u16> {
        match self {
            Self::Sse2
            | Self::Sse3
            | Self::Ssse3
            | Self::Sse41
            | Self::Sse42
            | Self::Neon => Some(128),
            Self::Avx | Self::Avx2 => Some(256),
            Self::Avx512F
            | Self::Avx512Bw
            | Self::Avx512Dq
            | Self::Avx512Vl
            | Self::Avx512Vnni => Some(512),
            Self::Sve | Self::Sve2 => None,
            Self::Fma
            | Self::Bmi1
            | Self::Bmi2
            | Self::Aes
            | Self::Sha
            | Self::DotProd
            | Self::Fp16 => None,
        }
    }
}

pub fn detect_features() -> Vec<CpuFeature> {
    let mut features = Vec::new();

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        macro_rules! detect {
            ($name:literal, $feature:expr) => {
                if std::arch::is_x86_feature_detected!($name) {
                    features.push($feature);
                }
            };
        }

        detect!("sse2", CpuFeature::Sse2);
        detect!("sse3", CpuFeature::Sse3);
        detect!("ssse3", CpuFeature::Ssse3);
        detect!("sse4.1", CpuFeature::Sse41);
        detect!("sse4.2", CpuFeature::Sse42);
        detect!("avx", CpuFeature::Avx);
        detect!("avx2", CpuFeature::Avx2);
        detect!("avx512f", CpuFeature::Avx512F);
        detect!("avx512bw", CpuFeature::Avx512Bw);
        detect!("avx512dq", CpuFeature::Avx512Dq);
        detect!("avx512vl", CpuFeature::Avx512Vl);
        detect!("avx512vnni", CpuFeature::Avx512Vnni);
        detect!("fma", CpuFeature::Fma);
        detect!("bmi1", CpuFeature::Bmi1);
        detect!("bmi2", CpuFeature::Bmi2);
        detect!("aes", CpuFeature::Aes);
        detect!("sha", CpuFeature::Sha);
    }

    #[cfg(target_arch = "aarch64")]
    {
        // AArch64 has NEON/Advanced SIMD as a baseline architectural feature in
        // Rust's target model. Optional features are represented with cfg
        // target_feature until a portable runtime detection macro is available.
        features.push(CpuFeature::Neon);

        if cfg!(target_feature = "sve") {
            features.push(CpuFeature::Sve);
        }
        if cfg!(target_feature = "sve2") {
            features.push(CpuFeature::Sve2);
        }
        if cfg!(target_feature = "dotprod") {
            features.push(CpuFeature::DotProd);
        }
        if cfg!(target_feature = "fp16") {
            features.push(CpuFeature::Fp16);
        }
        if cfg!(target_feature = "aes") {
            features.push(CpuFeature::Aes);
        }
        if cfg!(target_feature = "sha2") {
            features.push(CpuFeature::Sha);
        }
    }

    features.sort();
    features.dedup();
    features
}

pub fn maximum_fixed_vector_bits(features: &[CpuFeature]) -> Option<u16> {
    features.iter().filter_map(|feature| feature.vector_bits()).max()
}
