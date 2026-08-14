use super::{policy_error, string_array};
use crate::PackageError;
use std::collections::BTreeSet;
use std::path::Path;

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct LayerPolicy {
    pub order: Vec<String>,
    pub include: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ArchitectureRule {
    pub from: String,
    pub allow: Vec<String>,
    pub deny: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ArchitecturePolicy {
    pub deny_cycles: bool,
    pub deny_unknown_layers: bool,
    pub deny_layer_violations: bool,
    pub layers: LayerPolicy,
    pub rules: Vec<ArchitectureRule>,
}

impl Default for ArchitecturePolicy {
    fn default() -> Self {
        Self {
            deny_cycles: true,
            deny_unknown_layers: false,
            deny_layer_violations: true,
            layers: LayerPolicy::default(),
            rules: Vec::new(),
        }
    }
}

pub(super) fn parse(
    value: &toml::Value,
    manifest: &Path,
) -> Result<ArchitecturePolicy, PackageError> {
    let Some(configured) = value.get("architecture") else {
        return Ok(ArchitecturePolicy::default());
    };
    let table = configured
        .as_table()
        .ok_or_else(|| policy_error(manifest, "`architecture` must be a table"))?;
    for key in table.keys() {
        if !matches!(
            key.as_str(),
            "enforce"
                | "deny_cycles"
                | "deny_unknown_layers"
                | "deny_layer_violations"
                | "files"
                | "layers"
                | "rule"
        ) {
            return Err(policy_error(
                manifest,
                format!("unknown `architecture` setting `{key}`"),
            ));
        }
    }
    if table
        .get("enforce")
        .is_some_and(|value| value.as_bool() != Some(true))
    {
        return Err(policy_error(
            manifest,
            "`architecture.enforce` must be true; the architecture gate is mandatory",
        ));
    }
    let flag = |name: &str, default: bool| -> Result<bool, PackageError> {
        match table.get(name) {
            None => Ok(default),
            Some(toml::Value::Boolean(value)) => Ok(*value),
            Some(_) => Err(policy_error(
                manifest,
                format!("`architecture.{name}` must be a boolean"),
            )),
        }
    };
    let layers = parse_layers(table.get("layers"), manifest)?;
    let deny_unknown_layers = flag("deny_unknown_layers", false)?;
    if deny_unknown_layers && layers.order.is_empty() {
        return Err(policy_error(
            manifest,
            "`architecture.deny_unknown_layers` requires `architecture.layers.order`",
        ));
    }
    Ok(ArchitecturePolicy {
        deny_cycles: flag("deny_cycles", true)?,
        deny_unknown_layers,
        deny_layer_violations: flag("deny_layer_violations", true)?,
        layers,
        rules: parse_rules(table.get("rule"), manifest)?,
    })
}

fn parse_layers(value: Option<&toml::Value>, manifest: &Path) -> Result<LayerPolicy, PackageError> {
    let Some(value) = value else {
        return Ok(LayerPolicy::default());
    };
    let table = value
        .as_table()
        .ok_or_else(|| policy_error(manifest, "`architecture.layers` must be a table"))?;
    for key in table.keys() {
        if !matches!(key.as_str(), "order" | "include") {
            return Err(policy_error(
                manifest,
                format!("unknown `architecture.layers` setting `{key}`"),
            ));
        }
    }
    let order = string_array(table.get("order"), manifest, "architecture.layers.order")?
        .unwrap_or_default();
    let include = string_array(
        table.get("include"),
        manifest,
        "architecture.layers.include",
    )?
    .unwrap_or_default();
    reject_empty_strings(&include, manifest, "architecture.layers.include")?;
    let mut unique = BTreeSet::new();
    for layer in &order {
        if layer.trim().is_empty() {
            return Err(policy_error(
                manifest,
                "architecture layer names cannot be empty",
            ));
        }
        if !unique.insert(layer) {
            return Err(policy_error(
                manifest,
                format!("architecture layer `{layer}` appears more than once"),
            ));
        }
    }
    Ok(LayerPolicy { order, include })
}

fn parse_rules(
    value: Option<&toml::Value>,
    manifest: &Path,
) -> Result<Vec<ArchitectureRule>, PackageError> {
    let Some(value) = value else {
        return Ok(Vec::new());
    };
    value
        .as_array()
        .ok_or_else(|| policy_error(manifest, "`architecture.rule` must be an array of tables"))?
        .iter()
        .map(|value| parse_rule(value, manifest))
        .collect()
}

fn parse_rule(value: &toml::Value, manifest: &Path) -> Result<ArchitectureRule, PackageError> {
    let table = value
        .as_table()
        .ok_or_else(|| policy_error(manifest, "every `architecture.rule` entry must be a table"))?;
    for key in table.keys() {
        if !matches!(key.as_str(), "from" | "allow" | "deny") {
            return Err(policy_error(
                manifest,
                format!("unknown `architecture.rule` setting `{key}`"),
            ));
        }
    }
    let from = table
        .get("from")
        .and_then(toml::Value::as_str)
        .filter(|value| !value.trim().is_empty())
        .map(str::to_owned)
        .ok_or_else(|| policy_error(manifest, "`architecture.rule.from` is required"))?;
    let allow =
        string_array(table.get("allow"), manifest, "architecture.rule.allow")?.unwrap_or_default();
    let deny =
        string_array(table.get("deny"), manifest, "architecture.rule.deny")?.unwrap_or_default();
    reject_empty_strings(&allow, manifest, "architecture.rule.allow")?;
    reject_empty_strings(&deny, manifest, "architecture.rule.deny")?;
    if allow.is_empty() && deny.is_empty() {
        return Err(policy_error(
            manifest,
            format!("architecture rule `{from}` must declare `allow` or `deny`"),
        ));
    }
    Ok(ArchitectureRule { from, allow, deny })
}

fn reject_empty_strings(
    values: &[String],
    manifest: &Path,
    name: &str,
) -> Result<(), PackageError> {
    if values.iter().any(|value| value.trim().is_empty()) {
        Err(policy_error(
            manifest,
            format!("`{name}` entries cannot be empty"),
        ))
    } else {
        Ok(())
    }
}
