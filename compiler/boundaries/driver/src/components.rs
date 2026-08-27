use serde::Deserialize;
use severian_compile::CompilePlan;
use severian_target::{DeviceKind, TargetSpec};
use severian_universal::{AttrValue, ExecutionPlacement, EXECUTION_PLACEMENT_ATTRIBUTE};
use std::path::Path;
use std::process::Command;

#[derive(Debug, Deserialize)]
struct Catalog {
    component: Vec<Component>,
}

#[derive(Debug, Deserialize)]
struct Component {
    id: String,
    kind: String,
    device: String,
    vendor: String,
    provides: Vec<String>,
    #[serde(rename = "detect-all")]
    detect_all: Vec<String>,
    #[serde(rename = "detect-any", default)]
    detect_any: Vec<String>,
    #[serde(default)]
    install: Vec<Installer>,
}

#[derive(Debug, Deserialize)]
struct Installer {
    os: Vec<String>,
    program: String,
    arguments: Vec<String>,
}

pub(crate) fn ensure_for_plan(
    plan: &CompilePlan,
    target: &TargetSpec,
) -> Result<TargetSpec, String> {
    let placements = requested_placements(plan);
    let catalog = catalog()?;
    let mut resolved = target.clone();
    if placements.contains(&ExecutionPlacement::Simd) {
        ensure_component(&catalog, "compiler", "mlir.vector")?;
        resolved.capabilities.insert("mlir.dialect.vector");
    }
    if !placements.contains(&ExecutionPlacement::Gpu) {
        return Ok(resolved);
    }
    ensure_component(&catalog, "compiler", "mlir.rocdl")?;
    ensure_component(&catalog, "compiler", "mlir.stablehlo")?;
    resolved.capabilities.insert("mlir.dialect.gpu");
    resolved.capabilities.insert("mlir.dialect.rocdl");
    resolved.capabilities.insert("mlir.dialect.stablehlo");
    if resolved.rocm_device().is_some() {
        return Ok(resolved);
    }
    if resolved.amd_gpu().is_none() {
        // Placement is portable. A machine without the requested device keeps
        // the region on the native route; discovering an AMD device is what
        // turns ROCm into a required component.
        return Ok(resolved);
    }
    let Ok(component) = ensure_component(&catalog, "driver", "driver.rocm") else {
        // A portable placement remains executable on the host when component
        // provisioning is unavailable (for example in a restricted build
        // container). The compiler still attempted the manifest-selected
        // installation before taking this route.
        return Ok(resolved);
    };
    if ensure_component(&catalog, "runtime", "runtime.mlir-rocm").is_err() {
        let mut fallback = resolved;
        fallback
            .devices
            .retain(|device| device.kind != DeviceKind::Gpu);
        return Ok(fallback);
    }
    let mut refreshed = resolved.rediscover_devices();
    refreshed.capabilities.insert("mlir.dialect.gpu");
    refreshed.capabilities.insert("mlir.dialect.rocdl");
    if refreshed.rocm_device().is_none() {
        return Err(format!(
            "component `{}` was installed, but ROCm did not expose /dev/kfd; a reboot or device permission repair may be required",
            component.id
        ));
    }
    Ok(refreshed)
}

fn ensure_component<'a>(
    catalog: &'a Catalog,
    kind: &str,
    capability: &str,
) -> Result<&'a Component, String> {
    let component = catalog
        .component
        .iter()
        .find(|component| {
            component.kind == kind
                && component
                    .provides
                    .iter()
                    .any(|provided| provided == capability)
        })
        .ok_or_else(|| {
            format!("the component catalog has no `{kind}` provider for `{capability}`")
        })?;
    if !installed(component) {
        install(component)?;
    }
    Ok(component)
}

fn requested_placements(plan: &CompilePlan) -> Vec<ExecutionPlacement> {
    let compile_operations = plan
        .initializer
        .segments
        .iter()
        .chain(
            plan.functions
                .iter()
                .filter_map(|function| function.body.as_ref())
                .flat_map(|body| &body.segments),
        )
        .filter_map(|segment| match segment {
            severian_compile::PlanSegment::Compiler(region) => Some(region),
            severian_compile::PlanSegment::Standard(_) => None,
        })
        .chain(&plan.nested_regions)
        .flat_map(|region| &region.compile_operations)
        .filter_map(
            |operation| match operation.attributes.get(&EXECUTION_PLACEMENT_ATTRIBUTE) {
                Some(AttrValue::String(value)) => ExecutionPlacement::parse(value),
                _ => None,
            },
        )
        .collect::<Vec<_>>();
    let cfg_blocks = std::iter::once(&plan.source.initializer)
        .chain(
            plan.source
                .functions
                .iter()
                .filter_map(|function| function.body.as_ref()),
        )
        .flat_map(|body| &body.blocks)
        .filter_map(|block| block.execution);
    compile_operations.into_iter().chain(cfg_blocks).collect()
}

fn catalog() -> Result<Catalog, String> {
    let catalog: Catalog = toml::from_str(include_str!(
        "../../../../library/system/driver/components.toml"
    ))
    .map_err(|error| format!("invalid compiler component catalog: {error}"))?;
    for component in &catalog.component {
        if component.id.is_empty()
            || component.device.is_empty()
            || component.vendor.is_empty()
            || component.provides.is_empty()
        {
            return Err(
                "compiler components require an id, device, vendor, and provided capabilities"
                    .into(),
            );
        }
    }
    Ok(catalog)
}

fn installed(component: &Component) -> bool {
    if let Some(path) = local_component(component) {
        if path.is_file() {
            return true;
        }
    }
    let all = component
        .detect_all
        .iter()
        .all(|path| Path::new(path).exists());
    let any = component.detect_any.is_empty()
        || component
            .detect_any
            .iter()
            .any(|path| Path::new(path).exists());
    all && any
}

fn local_component(component: &Component) -> Option<std::path::PathBuf> {
    let components = crate::runtime_paths::component_root();
    match component.id.as_str() {
        "compiler.stablehlo" => Some(components.join("bin/severian-stablehlo-opt")),
        "runtime.mlir-rocm" => Some(components.join("rocm-runtime/libmlir_rocm_runtime.so")),
        _ => None,
    }
}

fn install(component: &Component) -> Result<(), String> {
    let operating_system = os_id();
    let installer = component
        .install
        .iter()
        .find(|installer| installer.os.iter().any(|value| value == &operating_system))
        .ok_or_else(|| {
            format!(
                "component `{}` has no installer for operating system `{operating_system}`",
                component.id
            )
        })?;
    let elevated = !running_as_root();
    let mut command = if elevated {
        let mut command = Command::new("sudo");
        command.args(["-n", installer.program.as_str()]);
        command
    } else {
        Command::new(&installer.program)
    };
    let result = command
        .args(&installer.arguments)
        .output()
        .map_err(|error| format!("could not start installer for `{}`: {error}", component.id))?;
    if result.status.success() {
        return Ok(());
    }
    Err(format!(
        "automatic installation of `{}` failed: {}",
        component.id,
        String::from_utf8_lossy(&result.stderr).trim()
    ))
}

fn running_as_root() -> bool {
    Command::new("id")
        .arg("-u")
        .output()
        .ok()
        .filter(|output| output.status.success())
        .is_some_and(|output| String::from_utf8_lossy(&output.stdout).trim() == "0")
}

fn os_id() -> String {
    std::fs::read_to_string("/etc/os-release")
        .ok()
        .and_then(|contents| {
            contents.lines().find_map(|line| {
                line.strip_prefix("ID=")
                    .map(|value| value.trim_matches('"').to_owned())
            })
        })
        .unwrap_or_else(|| std::env::consts::OS.to_owned())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn catalog_resolves_rocm_driver_and_compiler_components() {
        let catalog = catalog().unwrap();
        assert!(catalog.component.iter().any(|component| {
            component.id == "driver.rocm"
                && component
                    .provides
                    .iter()
                    .any(|value| value == "driver.rocm")
        }));
        assert!(catalog
            .component
            .iter()
            .any(|component| component.id == "compiler.rocdl"));
    }
}
