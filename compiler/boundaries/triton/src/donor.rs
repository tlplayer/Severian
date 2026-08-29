//! Rust ports of target policy from the pinned Triton donor.
//!
//! Source: `third_party/triton-donor/third_party/amd/lib/Dialect/
//! TritonAMDGPU/IR/TargetFeatures.cpp`.

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AmdIsaFamily {
    Unknown,
    Gcn51,
    Cdna1,
    Cdna2,
    Cdna3,
    Cdna4,
    Rdna1,
    Rdna2,
    Rdna3,
    Rdna4Mobile,
    Rdna4,
    Gfx1250,
}

impl AmdIsaFamily {
    pub fn is_cdna(self) -> bool {
        matches!(
            self,
            Self::Cdna1 | Self::Cdna2 | Self::Cdna3 | Self::Cdna4 | Self::Gfx1250
        )
    }

    pub fn is_rdna(self) -> bool {
        matches!(
            self,
            Self::Rdna1 | Self::Rdna2 | Self::Rdna3 | Self::Rdna4Mobile | Self::Rdna4
        )
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LdsTransLoadParameters {
    pub instruction_bits: u32,
    pub tile_elements: u32,
    pub leading_register_bases: u32,
    pub leading_lane_bases: u32,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AmdTargetFeatures {
    architecture: String,
    family: AmdIsaFamily,
}

impl AmdTargetFeatures {
    pub fn new(architecture: impl Into<String>) -> Self {
        let architecture = architecture.into();
        let family = parse_gfx_architecture(&architecture)
            .map(classify_gfx_architecture)
            .unwrap_or(AmdIsaFamily::Unknown);
        Self {
            architecture,
            family,
        }
    }

    pub fn architecture(&self) -> &str {
        &self.architecture
    }

    pub fn family(&self) -> AmdIsaFamily {
        self.family
    }

    pub fn warp_size(&self) -> u32 {
        match self.family {
            AmdIsaFamily::Gcn51
            | AmdIsaFamily::Cdna1
            | AmdIsaFamily::Cdna2
            | AmdIsaFamily::Cdna3
            | AmdIsaFamily::Cdna4 => 64,
            _ => 32,
        }
    }

    pub fn shared_memory_bytes(&self) -> u64 {
        match self.family {
            AmdIsaFamily::Gfx1250 => 320 * 1024,
            AmdIsaFamily::Cdna4 => 160 * 1024,
            _ => 64 * 1024,
        }
    }

    pub fn supports_wave_id(&self) -> bool {
        matches!(self.family, AmdIsaFamily::Rdna4 | AmdIsaFamily::Gfx1250)
    }

    pub fn shared_memory_partition_bytes(&self) -> u64 {
        if self.family == AmdIsaFamily::Gfx1250 {
            64 * 1024
        } else {
            0
        }
    }

    pub fn lds_trans_load(&self, element_bits: u32) -> Option<LdsTransLoadParameters> {
        let (instruction_bits, leading_register_bases, leading_lane_bases) =
            match (self.family, element_bits) {
                (AmdIsaFamily::Cdna4, 16) => (64, 0, 2),
                (AmdIsaFamily::Cdna4, 8) => (64, 0, 1),
                (AmdIsaFamily::Cdna4, 4) => (64, 0, 0),
                (AmdIsaFamily::Gfx1250, 16) => (128, 0, 0),
                (AmdIsaFamily::Gfx1250, 8) => (64, 2, 1),
                (AmdIsaFamily::Gfx1250, 4) => (64, 3, 1),
                _ => return None,
            };
        Some(LdsTransLoadParameters {
            instruction_bits,
            tile_elements: instruction_bits / element_bits,
            leading_register_bases,
            leading_lane_bases,
        })
    }

    pub fn supports_direct_to_lds_scatter(&self) -> bool {
        self.family == AmdIsaFamily::Gfx1250
    }

    pub fn supports_direct_to_lds_load(&self, bit_width: u32) -> bool {
        match self.family {
            AmdIsaFamily::Cdna3 => bit_width == 32,
            AmdIsaFamily::Cdna4 => matches!(bit_width, 32 | 128),
            AmdIsaFamily::Gfx1250 => matches!(bit_width, 32 | 64 | 128),
            _ => false,
        }
    }

    pub fn supports_direct_from_lds_store(&self, bit_width: u32) -> bool {
        self.family == AmdIsaFamily::Gfx1250 && matches!(bit_width, 8 | 32 | 64 | 128)
    }

    pub fn supports_buffer_load_to_local(&self) -> bool {
        matches!(self.family, AmdIsaFamily::Cdna3 | AmdIsaFamily::Cdna4)
    }

    pub fn requires_alias_info_for_async_ops(&self) -> bool {
        matches!(self.family, AmdIsaFamily::Cdna3 | AmdIsaFamily::Cdna4)
    }

    pub fn supports_tdm(&self) -> bool {
        self.family == AmdIsaFamily::Gfx1250
    }

    pub fn supports_multi_cta_launch(&self) -> bool {
        self.family == AmdIsaFamily::Gfx1250
    }

    pub fn maximum_multicast_mask_population(&self) -> u32 {
        if self.family == AmdIsaFamily::Gfx1250 {
            5
        } else {
            1
        }
    }

    pub fn supports_cluster_load(&self, bit_width: u32) -> bool {
        self.family == AmdIsaFamily::Gfx1250 && matches!(bit_width, 32 | 64 | 128)
    }

    pub fn supports_buffer_atomic_read_modify_write(&self) -> bool {
        matches!(
            self.family,
            AmdIsaFamily::Cdna3 | AmdIsaFamily::Cdna4 | AmdIsaFamily::Rdna4 | AmdIsaFamily::Gfx1250
        )
    }

    pub fn buffer_atomic_cache_policy(&self, has_users: bool) -> u32 {
        let return_value = u32::from(has_users);
        let device_scope = if self.family == AmdIsaFamily::Gfx1250 {
            0b1_0000
        } else {
            0
        };
        return_value | device_scope
    }

    pub fn supports_maximum_minimum(&self) -> bool {
        matches!(self.family, AmdIsaFamily::Cdna4 | AmdIsaFamily::Gfx1250)
    }

    pub fn supports_dpp_broadcast(&self) -> bool {
        matches!(
            self.family,
            AmdIsaFamily::Gcn51
                | AmdIsaFamily::Cdna1
                | AmdIsaFamily::Cdna2
                | AmdIsaFamily::Cdna3
                | AmdIsaFamily::Cdna4
        )
    }

    pub fn supports_permlane_swap(&self) -> bool {
        matches!(self.family, AmdIsaFamily::Cdna4 | AmdIsaFamily::Gfx1250)
    }

    pub fn supports_hardware_scaled_conversion(&self) -> bool {
        matches!(self.family, AmdIsaFamily::Cdna4 | AmdIsaFamily::Gfx1250)
    }

    pub fn supports_16_bit_elementwise(&self) -> bool {
        true
    }

    pub fn supports_32_bit_elementwise(&self) -> bool {
        matches!(
            self.family,
            AmdIsaFamily::Cdna2 | AmdIsaFamily::Cdna3 | AmdIsaFamily::Cdna4 | AmdIsaFamily::Gfx1250
        )
    }
}

fn parse_gfx_architecture(architecture: &str) -> Option<(u32, u32, u32)> {
    let digits = architecture.strip_prefix("gfx")?;
    if digits.len() < 3 || !digits.is_ascii() {
        return None;
    }
    let (leading, patch) = digits.split_at(digits.len() - 1);
    let (major, minor) = leading.split_at(leading.len() - 1);
    Some((
        major.parse().ok()?,
        minor.parse().ok()?,
        u32::from_str_radix(patch, 16).ok()?,
    ))
}

fn classify_gfx_architecture((major, minor, patch): (u32, u32, u32)) -> AmdIsaFamily {
    match (major, minor, patch) {
        (12, 5, _) => AmdIsaFamily::Gfx1250,
        (9, 5, 0) => AmdIsaFamily::Cdna4,
        (9, 4, 2) => AmdIsaFamily::Cdna3,
        (9, 0, 10) => AmdIsaFamily::Cdna2,
        (9, 0, 8) => AmdIsaFamily::Cdna1,
        (9, 0, 6) => AmdIsaFamily::Gcn51,
        (12, 0, _) => AmdIsaFamily::Rdna4,
        (11, 7, _) => AmdIsaFamily::Rdna4Mobile,
        (11, _, _) => AmdIsaFamily::Rdna3,
        (10, 3, _) => AmdIsaFamily::Rdna2,
        (10, 1, _) => AmdIsaFamily::Rdna1,
        _ => AmdIsaFamily::Unknown,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn donor_amd_families_and_wave_sizes_are_data_driven() {
        let rdna = AmdTargetFeatures::new("gfx1101");
        assert_eq!(rdna.family(), AmdIsaFamily::Rdna3);
        assert_eq!(rdna.warp_size(), 32);
        assert!(rdna.family().is_rdna());

        let cdna = AmdTargetFeatures::new("gfx942");
        assert_eq!(cdna.family(), AmdIsaFamily::Cdna3);
        assert_eq!(cdna.warp_size(), 64);
        assert!(cdna.family().is_cdna());

        let cdna4 = AmdTargetFeatures::new("gfx950");
        assert_eq!(cdna4.family(), AmdIsaFamily::Cdna4);
        assert_eq!(cdna4.shared_memory_bytes(), 160 * 1024);

        let gfx1250 = AmdTargetFeatures::new("gfx1250");
        assert_eq!(gfx1250.family(), AmdIsaFamily::Gfx1250);
        assert_eq!(gfx1250.warp_size(), 32);
        assert_eq!(gfx1250.shared_memory_bytes(), 320 * 1024);
        assert_eq!(gfx1250.shared_memory_partition_bytes(), 64 * 1024);
        assert!(gfx1250.supports_wave_id());
        assert!(gfx1250.family().is_cdna());
        assert!(gfx1250.supports_tdm());
        assert!(gfx1250.supports_multi_cta_launch());
        assert_eq!(gfx1250.maximum_multicast_mask_population(), 5);
        assert!(gfx1250.supports_cluster_load(128));
        assert_eq!(gfx1250.buffer_atomic_cache_policy(true), 0b1_0001);
        assert_eq!(
            gfx1250.lds_trans_load(8),
            Some(LdsTransLoadParameters {
                instruction_bits: 64,
                tile_elements: 8,
                leading_register_bases: 2,
                leading_lane_bases: 1,
            })
        );
    }

    #[test]
    fn donor_amd_capability_matrix_is_preserved() {
        let cdna3 = AmdTargetFeatures::new("gfx942");
        assert!(cdna3.supports_direct_to_lds_load(32));
        assert!(!cdna3.supports_direct_to_lds_load(16));
        assert!(cdna3.supports_buffer_load_to_local());
        assert!(cdna3.requires_alias_info_for_async_ops());
        assert!(cdna3.supports_buffer_atomic_read_modify_write());
        assert!(cdna3.supports_dpp_broadcast());
        assert!(cdna3.supports_32_bit_elementwise());

        let rdna3 = AmdTargetFeatures::new("gfx1101");
        assert!(!rdna3.supports_buffer_atomic_read_modify_write());
        assert!(!rdna3.supports_dpp_broadcast());
        assert!(!rdna3.supports_32_bit_elementwise());
        assert!(rdna3.supports_16_bit_elementwise());
    }

    #[test]
    fn malformed_architectures_remain_unknown() {
        for architecture in ["", "sm_90", "gfx", "gfx11x1"] {
            assert_eq!(
                AmdTargetFeatures::new(architecture).family(),
                AmdIsaFamily::Unknown
            );
        }
    }
}
