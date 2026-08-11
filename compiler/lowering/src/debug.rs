use crate::location::MlirLocation;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DebugInformation {
    None,
    LineTables,
    Full,
}

#[derive(Debug, Clone)]
pub struct DebugLoweringOptions {
    pub information: DebugInformation,
    pub producer: String,
    pub language: String,
}

impl Default for DebugLoweringOptions {
    fn default() -> Self {
        Self {
            information: DebugInformation::LineTables,
            producer: format!("Severian {}", env!("CARGO_PKG_VERSION")),
            language: "Severian".into(),
        }
    }
}

impl DebugLoweringOptions {
    pub fn enabled(&self) -> bool {
        self.information != DebugInformation::None
    }
}

pub fn llvm_debug_passes(options: &DebugLoweringOptions) -> Vec<&'static str> {
    if !options.enabled() {
        return Vec::new();
    }
    vec!["--llvm-di-scope-for-llvm-func"]
}

pub fn function_location(name: &str, source: MlirLocation) -> MlirLocation {
    MlirLocation::named(name, source)
}

pub fn inline_location(callee: MlirLocation, caller: MlirLocation) -> String {
    format!(
        "loc(callsite({} at {}))",
        inner_location(&callee),
        inner_location(&caller)
    )
}

fn inner_location(location: &MlirLocation) -> String {
    location
        .render()
        .strip_prefix("loc(")
        .and_then(|value| value.strip_suffix(')'))
        .unwrap_or("?")
        .to_owned()
}
