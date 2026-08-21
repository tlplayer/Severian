use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::{Path, PathBuf};

const CATALOG_SOURCE: &str = include_str!("../../../../tools/sev/config/options.toml");

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OptionSpec {
    pub path: String,
    pub group: String,
    pub kind: String,
    pub default: String,
    pub values: Vec<String>,
    pub description: String,
}

#[derive(Debug, Clone)]
pub struct Catalog {
    pub options: Vec<OptionSpec>,
}

impl Catalog {
    pub fn load() -> Result<Self, String> {
        let mut options = Vec::new();
        let mut current = BTreeMap::new();
        for line in CATALOG_SOURCE.lines() {
            let line = line.trim();
            if line == "[[option]]" {
                if !current.is_empty() {
                    options.push(option(&current)?);
                    current.clear();
                }
            } else if let Some((key, value)) = assignment(line) {
                current.insert(key.to_owned(), value.to_owned());
            }
        }
        if !current.is_empty() {
            options.push(option(&current)?);
        }
        if options.is_empty() {
            return Err("the compiler configuration catalog is empty".into());
        }
        let catalog = Self { options };
        let mut paths = BTreeSet::new();
        for option in &catalog.options {
            if !paths.insert(&option.path) {
                return Err(format!("duplicate catalog option `{}`", option.path));
            }
            catalog.validate(&option.path, &option.default)?;
        }
        Ok(catalog)
    }

    pub fn get(&self, path: &str) -> Option<&OptionSpec> {
        self.options.iter().find(|option| option.path == path)
    }

    pub fn default(&self, path: &str) -> Result<String, String> {
        self.get(path)
            .map(|option| option.default.clone())
            .ok_or_else(|| format!("compiler read of unregistered configuration `{path}`"))
    }

    pub fn validate(&self, path: &str, value: &str) -> Result<(), String> {
        let option = self
            .get(path)
            .ok_or_else(|| format!("unknown configuration option `{path}`"))?;
        let valid = match option.kind.as_str() {
            "bool" => matches!(value, "true" | "false"),
            "integer" => value.parse::<u64>().is_ok(),
            "enum" => option.values.iter().any(|candidate| candidate == value),
            "string" => !value.is_empty(),
            kind => return Err(format!("unknown catalog type `{kind}` for `{path}`")),
        };
        if valid {
            Ok(())
        } else if option.values.is_empty() {
            Err(format!(
                "invalid value `{value}` for `{path}` ({})",
                option.kind
            ))
        } else {
            Err(format!(
                "invalid value `{value}` for `{path}`; expected one of {}",
                option.values.join(", ")
            ))
        }
    }

    pub fn template(&self, package_name: &str) -> String {
        let mut output = format!(
            "# Severian package manifest.\n# Generated from the compiler-owned configuration catalog.\n\n### PACKAGE ############################################################\n\n[package]\nname = {name}\nversion = \"0.1.0\"\nedition = \"2026\"\nlicense = \"Severian License\"\ndefault-run = {name}\n\n### TARGETS ############################################################\n\n[[bin]]\nname = {name}\npath = \"src/main.sev\"\n\n# [lib]\n# name = {name}\n# path = \"src/lib.sev\"\n\n### DEPENDENCIES #######################################################\n\n[dependencies]\n\n[dev-dependencies]\n",
            name = quote(package_name),
        );
        let mut section = String::new();
        for option in &self.options {
            let (table, key) = option
                .path
                .rsplit_once('.')
                .expect("catalog paths have tables");
            if table != section {
                section = table.to_owned();
                output.push_str(&format!(
                    "\n### {} {}\n\n[{table}]\n",
                    option.group.to_uppercase(),
                    "#".repeat(68usize.saturating_sub(option.group.len()))
                ));
            }
            output.push_str(&format!("{key} = {}\n", render(option, &option.default)));
        }
        output
    }

    pub fn sync(&self, path: &Path) -> Result<usize, String> {
        let original = fs::read_to_string(path)
            .map_err(|error| format!("could not read {}: {error}", path.display()))?;
        let present = parse_values(&original)
            .0
            .into_keys()
            .collect::<BTreeSet<_>>();
        let missing = self
            .options
            .iter()
            .filter(|option| !present.contains(&option.path))
            .collect::<Vec<_>>();
        if missing.is_empty() {
            return Ok(0);
        }
        let mut additions = BTreeMap::<&str, Vec<&OptionSpec>>::new();
        for option in &missing {
            let table = option.path.rsplit_once('.').expect("catalog path").0;
            additions.entry(table).or_default().push(option);
        }
        let mut output = original;
        for (table, options) in additions {
            let header = format!("[{table}]");
            let block = options
                .iter()
                .map(|option| {
                    let key = option.path.rsplit_once('.').expect("catalog path").1;
                    format!(
                        "# {}\n{key} = {}\n",
                        option.description,
                        render(option, &option.default)
                    )
                })
                .collect::<String>();
            if let Some(start) = output.lines().position(|line| line.trim() == header) {
                let offset = output
                    .split_inclusive('\n')
                    .take(start + 1)
                    .map(str::len)
                    .sum::<usize>();
                output.insert_str(offset, &block);
            } else {
                if !output.ends_with('\n') {
                    output.push('\n');
                }
                output.push_str(&format!("\n{header}\n{block}"));
            }
        }
        fs::write(path, output)
            .map_err(|error| format!("could not update {}: {error}", path.display()))?;
        Ok(missing.len())
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BinaryTarget {
    pub name: String,
    pub path: PathBuf,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LibraryTarget {
    pub name: String,
    pub version: String,
    pub path: PathBuf,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DeclaredTarget {
    Binary(BinaryTarget),
    Library(LibraryTarget),
}

impl DeclaredTarget {
    pub fn name(&self) -> &str {
        match self {
            Self::Binary(target) => &target.name,
            Self::Library(target) => &target.name,
        }
    }

    pub fn path(&self) -> &Path {
        match self {
            Self::Binary(target) => &target.path,
            Self::Library(target) => &target.path,
        }
    }
}

#[derive(Debug, Clone)]
pub struct Manifest {
    pub root: PathBuf,
    pub name: String,
    pub default_run: Option<String>,
    pub bins: Vec<BinaryTarget>,
    pub library: Option<LibraryTarget>,
    pub values: BTreeMap<String, String>,
}

impl Manifest {
    pub fn load(path: &Path, catalog: &Catalog) -> Result<Self, String> {
        let source = fs::read_to_string(path)
            .map_err(|error| format!("could not read {}: {error}", path.display()))?;
        let (values, bins) = parse_values(&source);
        for (key, value) in &values {
            if catalog.get(key).is_some() {
                catalog.validate(key, value)?;
            } else if is_configuration_table(key) {
                return Err(format!("unknown configuration option `{key}`"));
            }
        }
        let root = path.parent().unwrap_or_else(|| Path::new(".")).to_owned();
        let name = values
            .get("package.name")
            .cloned()
            .or_else(|| {
                root.file_name()
                    .map(|name| name.to_string_lossy().into_owned())
            })
            .unwrap_or_else(|| "app".into());
        let default_run = values.get("package.default-run").cloned();
        let version = values
            .get("package.version")
            .cloned()
            .unwrap_or_else(|| "0.1.0".into());
        let bins = bins
            .into_iter()
            .map(|bin| {
                let bin_name = bin
                    .get("name")
                    .cloned()
                    .ok_or_else(|| "each `[[bin]]` requires `name`".to_owned())?;
                let bin_path = bin
                    .get("path")
                    .cloned()
                    .ok_or_else(|| format!("binary `{bin_name}` requires `path`"))?;
                Ok(BinaryTarget {
                    name: bin_name,
                    path: root.join(bin_path),
                })
            })
            .collect::<Result<Vec<_>, String>>()?;
        let library = values.keys().any(|key| key.starts_with("lib.")).then(|| {
            let library_name = values
                .get("lib.name")
                .cloned()
                .unwrap_or_else(|| name.clone());
            let library_path = values
                .get("lib.path")
                .cloned()
                .unwrap_or_else(|| "src/lib.sev".into());
            LibraryTarget {
                name: library_name,
                version: version.clone(),
                path: root.join(library_path),
            }
        });
        Ok(Self {
            root,
            name,
            default_run,
            bins,
            library,
            values,
        })
    }
}

fn option(fields: &BTreeMap<String, String>) -> Result<OptionSpec, String> {
    let required = |name: &str| {
        fields
            .get(name)
            .map(|value| unquote(value))
            .ok_or_else(|| format!("catalog option is missing `{name}`"))
    };
    Ok(OptionSpec {
        path: required("path")?,
        group: required("group")?,
        kind: required("type")?,
        default: required("default")?,
        values: fields
            .get("values")
            .map(|value| parse_array(value))
            .unwrap_or_default(),
        description: required("description")?,
    })
}

fn assignment(line: &str) -> Option<(&str, &str)> {
    if line.is_empty() || line.starts_with('#') || line.starts_with('[') {
        return None;
    }
    line.split_once('=')
        .map(|(key, value)| (key.trim(), value.trim()))
}

fn parse_array(value: &str) -> Vec<String> {
    value
        .trim_matches(|character| character == '[' || character == ']')
        .split(',')
        .map(|value| unquote(value.trim()))
        .filter(|value| !value.is_empty())
        .collect()
}

fn unquote(value: &str) -> String {
    value
        .strip_prefix('"')
        .and_then(|value| value.strip_suffix('"'))
        .unwrap_or(value)
        .replace("\\\"", "\"")
        .replace("\\\\", "\\")
}

fn quote(value: &str) -> String {
    format!("\"{}\"", value.replace('\\', "\\\\").replace('"', "\\\""))
}

fn render(option: &OptionSpec, value: &str) -> String {
    match option.kind.as_str() {
        "string" | "enum" => quote(value),
        _ => value.to_owned(),
    }
}

fn is_configuration_table(path: &str) -> bool {
    [
        "language.",
        "build.",
        "diagnostics.",
        "profile.",
        "test.",
        "publish.",
    ]
    .iter()
    .any(|prefix| path.starts_with(prefix))
}

fn parse_values(source: &str) -> (BTreeMap<String, String>, Vec<BTreeMap<String, String>>) {
    let mut values = BTreeMap::new();
    let mut bins = Vec::<BTreeMap<String, String>>::new();
    let mut table = String::new();
    let mut bin = None;
    for raw_line in source.lines() {
        let line = raw_line.split('#').next().unwrap_or("").trim();
        if line == "[[bin]]" {
            bins.push(BTreeMap::new());
            bin = Some(bins.len() - 1);
            table.clear();
            continue;
        }
        if line.starts_with('[') && line.ends_with(']') {
            table = line.trim_matches(&['[', ']'][..]).to_owned();
            bin = None;
            continue;
        }
        let Some((key, raw_value)) = assignment(line) else {
            continue;
        };
        let value = unquote(raw_value);
        if let Some(index) = bin {
            bins[index].insert(key.to_owned(), value);
        } else if !table.is_empty() {
            values.insert(format!("{table}.{key}"), value);
        }
    }
    (values, bins)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn catalog_drives_defaults_validation_and_template() {
        let catalog = Catalog::load().unwrap();
        assert_eq!(catalog.default("build.backend").unwrap(), "auto");
        assert!(catalog.validate("build.backend", "xla").is_ok());
        assert!(catalog.validate("build.backend", "gpu").is_err());
        let template = catalog.template("hello");
        assert!(template.contains("profile = \"dev\""));
        assert!(template.contains("backend = \"auto\""));
    }
}
