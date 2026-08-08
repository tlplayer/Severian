use super::device::{GpuDevice, GpuVendor};
use std::process::Command;

pub fn validate_architecture(value: &str) -> bool {
    let Some(rest) = value.strip_prefix("gfx") else {
        return false;
    };

    !rest.is_empty()
        && rest
            .chars()
            .all(|character| character.is_ascii_alphanumeric())
}

pub fn normalize_architecture(value: &str) -> Option<String> {
    let value = value.trim().to_ascii_lowercase();

    if validate_architecture(&value) {
        Some(value)
    } else {
        None
    }
}

pub fn detect_devices() -> Vec<GpuDevice> {
    detect_with_rocminfo()
        .or_else(detect_with_rocm_smi)
        .unwrap_or_default()
}

fn detect_with_rocminfo() -> Option<Vec<GpuDevice>> {
    let output = Command::new("rocminfo").output().ok()?;
    if !output.status.success() {
        return None;
    }

    let text = String::from_utf8_lossy(&output.stdout);
    let mut devices = Vec::new();

    for line in text.lines() {
        let line = line.trim();

        // rocminfo commonly emits lines containing "Name: gfx1100" for GPU
        // agents. Restrict to gfx architecture names to avoid CPU agent names.
        let candidate = line
            .split_once(':')
            .map(|(_, value)| value.trim())
            .filter(|value| value.starts_with("gfx"))
            .and_then(normalize_architecture);

        if let Some(architecture) = candidate {
            if devices
                .iter()
                .any(|device: &GpuDevice| device.architecture.as_deref() == Some(&architecture))
            {
                continue;
            }

            devices.push(GpuDevice {
                ordinal: devices.len(),
                vendor: GpuVendor::Amd,
                name: format!("AMD {architecture}"),
                architecture: Some(architecture),
                memory_bytes: None,
                pci_bus_id: None,
                runtime: Some("rocm".into()),
            });
        }
    }

    Some(devices)
}

fn detect_with_rocm_smi() -> Option<Vec<GpuDevice>> {
    let output = Command::new("rocm-smi")
        .args(["--showproductname", "--showuniqueid"])
        .output()
        .ok()?;

    if !output.status.success() {
        return None;
    }

    let text = String::from_utf8_lossy(&output.stdout);
    let names = text
        .lines()
        .filter(|line| line.contains("Card series") || line.contains("Card model"))
        .filter_map(|line| line.split_once(':').map(|(_, value)| value.trim().to_owned()))
        .collect::<Vec<_>>();

    Some(
        names
            .into_iter()
            .enumerate()
            .map(|(ordinal, name)| GpuDevice {
                ordinal,
                vendor: GpuVendor::Amd,
                name,
                architecture: None,
                memory_bytes: None,
                pci_bus_id: None,
                runtime: Some("rocm".into()),
            })
            .collect(),
    )
}

pub fn llvm_triple() -> &'static str {
    "amdgcn-amd-amdhsa"
}
