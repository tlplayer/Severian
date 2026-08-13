use super::device::{GpuDevice, GpuVendor};
use std::process::Command;

pub fn validate_architecture(value: &str) -> bool {
    let Some(digits) = value.strip_prefix("sm_") else {
        return false;
    };

    let digits = digits
        .strip_suffix('a')
        .or_else(|| digits.strip_suffix('f'))
        .unwrap_or(digits);
    !digits.is_empty() && digits.chars().all(|character| character.is_ascii_digit())
}

pub fn normalize_architecture(value: &str) -> Option<String> {
    let value = value.trim().to_ascii_lowercase();

    if validate_architecture(&value) {
        return Some(value);
    }

    // Accept common compute-capability spellings such as "8.9" and "89".
    let digits = value.replace('.', "");
    if !digits.is_empty() && digits.chars().all(|character| character.is_ascii_digit()) {
        let candidate = format!("sm_{digits}");
        if validate_architecture(&candidate) {
            return Some(candidate);
        }
    }

    None
}

pub fn detect_devices() -> Vec<GpuDevice> {
    let output = match Command::new("nvidia-smi")
        .args([
            "--query-gpu=index,name,memory.total,pci.bus_id,compute_cap",
            "--format=csv,noheader,nounits",
        ])
        .output()
    {
        Ok(output) if output.status.success() => output,
        _ => return Vec::new(),
    };

    let text = String::from_utf8_lossy(&output.stdout);

    text.lines().filter_map(parse_device_line).collect()
}

fn parse_device_line(line: &str) -> Option<GpuDevice> {
    let fields = line.split(',').map(str::trim).collect::<Vec<_>>();

    if fields.len() < 5 {
        return None;
    }

    let ordinal = fields[0].parse().ok()?;
    let memory_mib: u64 = fields[2].parse().ok()?;
    let architecture = normalize_architecture(fields[4]);

    Some(GpuDevice {
        ordinal,
        vendor: GpuVendor::Nvidia,
        name: fields[1].to_owned(),
        architecture,
        memory_bytes: memory_mib.checked_mul(1024 * 1024),
        pci_bus_id: Some(fields[3].to_owned()),
        runtime: Some("cuda".into()),
    })
}

pub fn llvm_triple() -> &'static str {
    "nvptx64-nvidia-cuda"
}
