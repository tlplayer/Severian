use crate::manifest::ProfileSection;
use std::collections::{BTreeMap, BTreeSet};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DebugInfo {
    None,
    LineTables,
    Full,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LtoMode {
    Off,
    Thin,
    Fat,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Sanitizer {
    Address,
    Thread,
    Memory,
    Undefined,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BuildProfile {
    pub name: String,
    pub optimization: u8,
    pub debug: DebugInfo,
    pub lto: LtoMode,
    pub incremental: bool,
    pub overflow_checks: bool,
    pub assertions: bool,
    pub runtime_checks: bool,
    pub coverage: bool,
    pub sanitizer: Option<Sanitizer>,
}

impl BuildProfile {
    pub fn dev() -> Self {
        Self {
            name: "dev".into(),
            optimization: 0,
            debug: DebugInfo::Full,
            lto: LtoMode::Off,
            incremental: true,
            overflow_checks: true,
            assertions: true,
            runtime_checks: true,
            coverage: false,
            sanitizer: None,
        }
    }

    pub fn release() -> Self {
        Self {
            name: "release".into(),
            optimization: 3,
            debug: DebugInfo::LineTables,
            lto: LtoMode::Thin,
            incremental: false,
            overflow_checks: false,
            assertions: false,
            runtime_checks: false,
            coverage: false,
            sanitizer: None,
        }
    }

    pub fn coverage() -> Self {
        Self {
            name: "coverage".into(),
            optimization: 0,
            debug: DebugInfo::Full,
            incremental: false,
            overflow_checks: true,
            assertions: true,
            runtime_checks: true,
            coverage: true,
            sanitizer: None,
        }
    }

    pub fn resolve_all(
        custom: &BTreeMap<String, ProfileSection>,
    ) -> Result<BTreeMap<String, Self>, String> {
        let mut resolved = BTreeMap::from([
            ("dev".into(), Self::dev()),
            ("release".into(), Self::release()),
            ("test".into(), {
                let mut value = Self::dev();
                value.name = "test".into();
                value
            }),
            ("bench".into(), {
                let mut value = Self::release();
                value.name = "bench".into();
                value
            }),
            ("coverage".into(), Self::coverage()),
        ]);

        let names = custom.keys().cloned().collect::<Vec<_>>();
        for name in names {
            resolve_one(&name, custom, &mut resolved, &mut BTreeSet::new())?;
        }
        Ok(resolved)
    }
}

fn resolve_one(
    name: &str,
    custom: &BTreeMap<String, ProfileSection>,
    resolved: &mut BTreeMap<String, BuildProfile>,
    stack: &mut BTreeSet<String>,
) -> Result<BuildProfile, String> {
    let Some(section) = custom.get(name) else {
        return resolved
            .get(name)
            .cloned()
            .ok_or_else(|| format!("unknown profile `{name}`"));
    };

    if !stack.insert(name.to_owned()) {
        return Err(format!("profile inheritance cycle at `{name}`"));
    }

    let base_name = section.inherits.as_deref().unwrap_or("dev");
    let mut profile = if base_name == name {
        return Err(format!("profile `{name}` cannot inherit itself"));
    } else if custom.contains_key(base_name) {
        resolve_one(base_name, custom, resolved, stack)?
    } else {
        resolved
            .get(base_name)
            .cloned()
            .ok_or_else(|| format!("profile `{name}` inherits unknown profile `{base_name}`"))?
    };

    profile.name = name.to_owned();
    apply(section, &mut profile)?;
    stack.remove(name);
    resolved.insert(name.to_owned(), profile.clone());
    Ok(profile)
}

fn apply(section: &ProfileSection, profile: &mut BuildProfile) -> Result<(), String> {
    if let Some(value) = section.optimization {
        profile.optimization = value.min(3);
    }
    if let Some(value) = &section.debug {
        profile.debug = match value {
            toml::Value::Boolean(false) => DebugInfo::None,
            toml::Value::Boolean(true) => DebugInfo::Full,
            toml::Value::String(value) if value == "none" => DebugInfo::None,
            toml::Value::String(value) if value == "line-tables" => DebugInfo::LineTables,
            toml::Value::String(value) if value == "full" => DebugInfo::Full,
            _ => return Err("debug must be false, true, none, line-tables, or full".into()),
        };
    }
    if let Some(value) = &section.lto {
        profile.lto = match value {
            toml::Value::Boolean(false) => LtoMode::Off,
            toml::Value::Boolean(true) => LtoMode::Fat,
            toml::Value::String(value) if value == "off" => LtoMode::Off,
            toml::Value::String(value) if value == "thin" => LtoMode::Thin,
            toml::Value::String(value) if value == "fat" => LtoMode::Fat,
            _ => return Err("lto must be false, true, off, thin, or fat".into()),
        };
    }
    if let Some(value) = section.incremental { profile.incremental = value; }
    if let Some(value) = section.overflow_checks { profile.overflow_checks = value; }
    if let Some(value) = section.assertions { profile.assertions = value; }
    if let Some(value) = section.runtime_checks { profile.runtime_checks = value; }
    if let Some(value) = section.coverage { profile.coverage = value; }
    if let Some(value) = &section.sanitizer {
        profile.sanitizer = Some(match value.as_str() {
            "address" => Sanitizer::Address,
            "thread" => Sanitizer::Thread,
            "memory" => Sanitizer::Memory,
            "undefined" => Sanitizer::Undefined,
            _ => return Err(format!("unknown sanitizer `{value}`")),
        });
    }
    Ok(())
}
