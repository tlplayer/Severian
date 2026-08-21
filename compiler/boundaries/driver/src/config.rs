use serde::Deserialize;
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
        let document = original
            .parse::<toml::Value>()
            .map_err(|error| format!("invalid {}: {error}", path.display()))?;
        let present = configuration_values(&document)
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
    pub publish: bool,
    pub default_run: Option<String>,
    pub bins: Vec<BinaryTarget>,
    pub library: Option<LibraryTarget>,
    pub package_graph: ResolvedPackageGraph,
    pub validation: Option<ValidationManifest>,
    pub values: BTreeMap<String, String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ValidationManifest {
    /// The package-local link used for discovery. Keeping this separate from
    /// `canonical_source` makes it possible to reject copied example trees.
    pub source: PathBuf,
    pub canonical_source: PathBuf,
    pub line_coverage: u8,
    pub branch_coverage: u8,
    pub examples: BTreeMap<PathBuf, ExampleRequirement>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExampleRequirement {
    pub required_routes: BTreeSet<String>,
    pub allow_fallback: bool,
    pub expected_exit: Option<i32>,
}

#[derive(Debug, Clone)]
pub struct ResolvedPackageGraph {
    pub root: severian_modules::PackageId,
    pub packages: BTreeMap<severian_modules::PackageId, ResolvedPackage>,
}

#[derive(Debug, Clone)]
pub struct ResolvedPackage {
    pub id: severian_modules::PackageId,
    pub root: PathBuf,
    pub name: String,
    pub manifest: toml::Value,
    pub library: PathBuf,
    pub dependencies: BTreeMap<String, severian_modules::PackageId>,
    pub dev_dependencies: BTreeMap<String, severian_modules::PackageId>,
}

#[derive(Debug, Deserialize)]
struct PackageDocument {
    #[serde(default)]
    package: PackageSection,
    #[serde(default, rename = "bin")]
    bins: Vec<TargetSection>,
    lib: Option<TargetSection>,
    #[serde(default)]
    dependencies: BTreeMap<String, DependencyDeclaration>,
    #[serde(default, rename = "dev-dependencies")]
    dev_dependencies: BTreeMap<String, DependencyDeclaration>,
}

#[derive(Debug, Default, Deserialize)]
struct PackageSection {
    name: Option<String>,
    version: Option<String>,
    publish: Option<bool>,
    #[serde(rename = "default-run")]
    default_run: Option<String>,
}

#[derive(Debug, Deserialize)]
struct TargetSection {
    name: Option<String>,
    path: Option<PathBuf>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
enum DependencyDeclaration {
    Version(String),
    Detailed(DependencyDetail),
}

#[derive(Debug, Clone, Deserialize)]
struct DependencyDetail {
    path: Option<PathBuf>,
    version: Option<String>,
    git: Option<String>,
    registry: Option<String>,
}

#[derive(Debug, Deserialize)]
struct ValidationDocument {
    validation: ValidationSection,
    #[serde(default, rename = "example")]
    examples: Vec<ExampleSection>,
}

#[derive(Debug, Deserialize)]
struct ValidationSection {
    source: PathBuf,
    #[serde(rename = "line-coverage")]
    line_coverage: u8,
    #[serde(rename = "branch-coverage")]
    branch_coverage: u8,
}

#[derive(Debug, Deserialize)]
struct ExampleSection {
    path: PathBuf,
    #[serde(default, rename = "required-routes")]
    required_routes: BTreeSet<String>,
    #[serde(default, rename = "allow-fallback")]
    allow_fallback: bool,
    #[serde(rename = "expected-exit")]
    expected_exit: Option<i32>,
}

impl Manifest {
    pub fn load(path: &Path, catalog: &Catalog) -> Result<Self, String> {
        let mut builder = PackageGraphBuilder::new(catalog);
        let root_id = builder.resolve(path, false)?;
        let package_graph = ResolvedPackageGraph {
            root: root_id,
            packages: builder.packages,
        };
        let root_package = package_graph
            .packages
            .get(&root_id)
            .expect("resolved graph contains its root");
        let document: PackageDocument = root_package
            .manifest
            .clone()
            .try_into()
            .map_err(|error| format!("invalid {}: {error}", path.display()))?;
        let values = configuration_values(&root_package.manifest);
        for (key, value) in &values {
            if catalog.get(key).is_some() {
                catalog.validate(key, value)?;
            } else if is_configuration_table(key) {
                return Err(format!("unknown configuration option `{key}`"));
            }
        }
        let root = root_package.root.clone();
        let name = root_package.name.clone();
        let default_run = document.package.default_run;
        let publish = document.package.publish.unwrap_or(true);
        let version = document.package.version.unwrap_or_else(|| "0.1.0".into());
        let bins = document
            .bins
            .into_iter()
            .map(|bin| {
                let bin_name = bin
                    .name
                    .ok_or_else(|| "each `[[bin]]` requires `name`".to_owned())?;
                let bin_path = bin
                    .path
                    .ok_or_else(|| format!("binary `{bin_name}` requires `path`"))?;
                Ok(BinaryTarget {
                    name: bin_name,
                    path: root.join(bin_path),
                })
            })
            .collect::<Result<Vec<_>, String>>()?;
        let library = document.lib.map(|library| {
            let library_name = library.name.unwrap_or_else(|| name.clone());
            let library_path = library.path.unwrap_or_else(|| "src/lib.sev".into());
            LibraryTarget {
                name: library_name,
                version: version.clone(),
                path: root.join(library_path),
            }
        });
        let validation_path = root.join("validation.toml");
        let validation = validation_path
            .is_file()
            .then(|| ValidationManifest::load(&validation_path, &root))
            .transpose()?;
        if validation.is_some() && publish {
            return Err("a validation package must set `package.publish = false`".into());
        }
        Ok(Self {
            root,
            name,
            publish,
            default_run,
            bins,
            library,
            package_graph,
            validation,
            values,
        })
    }

    pub fn module_graph(&self, include_root_dev: bool) -> severian_modules::PackageGraph {
        let packages = self
            .package_graph
            .packages
            .iter()
            .map(|(id, package)| {
                let mut dependencies = package.dependencies.clone();
                if include_root_dev && *id == self.package_graph.root {
                    dependencies.extend(package.dev_dependencies.clone());
                }
                (
                    *id,
                    severian_modules::ResolvedPackage {
                        id: *id,
                        root: package.root.clone(),
                        library: package.library.clone(),
                        dependencies,
                    },
                )
            })
            .collect();
        severian_modules::PackageGraph {
            root: self.package_graph.root,
            packages,
        }
    }
}

struct PackageGraphBuilder<'a> {
    catalog: &'a Catalog,
    packages: BTreeMap<severian_modules::PackageId, ResolvedPackage>,
    resolved: BTreeMap<PathBuf, severian_modules::PackageId>,
    visiting: BTreeSet<PathBuf>,
    next_id: u32,
}

impl<'a> PackageGraphBuilder<'a> {
    fn new(catalog: &'a Catalog) -> Self {
        Self {
            catalog,
            packages: BTreeMap::new(),
            resolved: BTreeMap::new(),
            visiting: BTreeSet::new(),
            next_id: 0,
        }
    }

    fn resolve(
        &mut self,
        manifest_path: &Path,
        require_library: bool,
    ) -> Result<severian_modules::PackageId, String> {
        let manifest_path = fs::canonicalize(manifest_path)
            .map_err(|error| format!("could not resolve {}: {error}", manifest_path.display()))?;
        if let Some(id) = self.resolved.get(&manifest_path) {
            return Ok(*id);
        }
        if !self.visiting.insert(manifest_path.clone()) {
            return Err(format!(
                "package dependency cycle reaches {}",
                manifest_path.display()
            ));
        }
        let source = fs::read_to_string(&manifest_path)
            .map_err(|error| format!("could not read {}: {error}", manifest_path.display()))?;
        let value = source
            .parse::<toml::Value>()
            .map_err(|error| format!("invalid {}: {error}", manifest_path.display()))?;
        let document: PackageDocument = value
            .clone()
            .try_into()
            .map_err(|error| format!("invalid {}: {error}", manifest_path.display()))?;
        validate_configuration(self.catalog, &value)?;
        let root = manifest_path
            .parent()
            .unwrap_or_else(|| Path::new("."))
            .to_owned();
        let name = document
            .package
            .name
            .clone()
            .or_else(|| {
                root.file_name()
                    .map(|name| name.to_string_lossy().into_owned())
            })
            .unwrap_or_else(|| "package".into());
        let library = document
            .lib
            .as_ref()
            .and_then(|library| library.path.clone())
            .map(|path| root.join(path))
            .or_else(|| {
                document
                    .bins
                    .first()
                    .and_then(|binary| binary.path.clone())
                    .map(|path| root.join(path))
            })
            .unwrap_or_else(|| root.join("src/lib.sev"));
        if require_library && !library.is_file() {
            return Err(format!(
                "dependency `{name}` has no library source at {}",
                library.display()
            ));
        }
        let id = severian_modules::PackageId(self.next_id);
        self.next_id += 1;
        let dependencies = self.resolve_dependencies(&root, &document.dependencies)?;
        let dev_dependencies = self.resolve_dependencies(&root, &document.dev_dependencies)?;
        self.visiting.remove(&manifest_path);
        self.resolved.insert(manifest_path, id);
        self.packages.insert(
            id,
            ResolvedPackage {
                id,
                root,
                name,
                manifest: value,
                library,
                dependencies,
                dev_dependencies,
            },
        );
        Ok(id)
    }

    fn resolve_dependencies(
        &mut self,
        root: &Path,
        declarations: &BTreeMap<String, DependencyDeclaration>,
    ) -> Result<BTreeMap<String, severian_modules::PackageId>, String> {
        declarations
            .iter()
            .map(|(alias, declaration)| {
                let path = dependency_path(alias, declaration)?;
                let manifest = root.join(path).join("package.toml");
                let id = self.resolve(&manifest, true)?;
                Ok((alias.clone(), id))
            })
            .collect()
    }
}

fn dependency_path<'a>(
    alias: &str,
    declaration: &'a DependencyDeclaration,
) -> Result<&'a Path, String> {
    match declaration {
        DependencyDeclaration::Detailed(detail) => {
            if let Some(path) = detail.path.as_deref() {
                return Ok(path);
            }
            let source = if detail.git.is_some() {
                "git"
            } else if detail.registry.is_some() || detail.version.is_some() {
                "registry"
            } else {
                "unspecified"
            };
            Err(format!(
                "dependency `{alias}` uses unsupported {source} resolution; a local `path` is required"
            ))
        }
        DependencyDeclaration::Version(version) => Err(format!(
            "dependency `{alias}` uses unsupported registry version `{version}`; a local `path` is required"
        )),
    }
}

fn validate_configuration(catalog: &Catalog, document: &toml::Value) -> Result<(), String> {
    for (key, value) in configuration_values(document) {
        if catalog.get(&key).is_some() {
            catalog.validate(&key, &value)?;
        } else if is_configuration_table(&key) {
            return Err(format!("unknown configuration option `{key}`"));
        }
    }
    Ok(())
}

fn configuration_values(document: &toml::Value) -> BTreeMap<String, String> {
    fn visit(prefix: &str, value: &toml::Value, output: &mut BTreeMap<String, String>) {
        match value {
            toml::Value::Table(table) => {
                for (key, value) in table {
                    let path = if prefix.is_empty() {
                        key.clone()
                    } else {
                        format!("{prefix}.{key}")
                    };
                    visit(&path, value, output);
                }
            }
            toml::Value::String(value) => {
                output.insert(prefix.into(), value.clone());
            }
            toml::Value::Integer(value) => {
                output.insert(prefix.into(), value.to_string());
            }
            toml::Value::Boolean(value) => {
                output.insert(prefix.into(), value.to_string());
            }
            _ => {}
        }
    }
    let mut output = BTreeMap::new();
    visit("", document, &mut output);
    output
}

impl ValidationManifest {
    fn load(path: &Path, package_root: &Path) -> Result<Self, String> {
        let source = fs::read_to_string(path)
            .map_err(|error| format!("could not read {}: {error}", path.display()))?;
        let document: ValidationDocument = toml::from_str(&source)
            .map_err(|error| format!("invalid {}: {error}", path.display()))?;
        if document.validation.line_coverage > 100 || document.validation.branch_coverage > 100 {
            return Err("validation coverage percentages must be from 0 through 100".into());
        }
        let linked_source = package_root.join(&document.validation.source);
        let metadata = fs::symlink_metadata(&linked_source).map_err(|error| {
            format!(
                "could not inspect validation source {}: {error}",
                linked_source.display()
            )
        })?;
        if !metadata.file_type().is_symlink() {
            return Err(format!(
                "validation source {} must be a relative symlink to the canonical examples",
                linked_source.display()
            ));
        }
        let target = fs::read_link(&linked_source).map_err(|error| {
            format!(
                "could not read validation source link {}: {error}",
                linked_source.display()
            )
        })?;
        if target.is_absolute() {
            return Err(format!(
                "validation source {} must use a relative symlink target",
                linked_source.display()
            ));
        }
        let canonical_source = fs::canonicalize(&linked_source).map_err(|error| {
            format!(
                "could not resolve validation source {}: {error}",
                linked_source.display()
            )
        })?;
        if !canonical_source.is_dir() {
            return Err(format!(
                "validation source {} does not resolve to a directory",
                linked_source.display()
            ));
        }
        let mut examples = BTreeMap::new();
        for example in document.examples {
            let source = package_root.join(&example.path);
            let canonical = fs::canonicalize(&source).map_err(|error| {
                format!(
                    "could not resolve configured example {}: {error}",
                    source.display()
                )
            })?;
            if !canonical.starts_with(&canonical_source) {
                return Err(format!(
                    "configured example {} is outside the canonical example tree",
                    source.display()
                ));
            }
            let requirement = ExampleRequirement {
                required_routes: example.required_routes,
                allow_fallback: example.allow_fallback,
                expected_exit: example.expected_exit,
            };
            if examples.insert(canonical.clone(), requirement).is_some() {
                return Err(format!(
                    "validation.toml configures {} more than once",
                    canonical.display()
                ));
            }
        }
        Ok(Self {
            source: linked_source,
            canonical_source,
            line_coverage: document.validation.line_coverage,
            branch_coverage: document.validation.branch_coverage,
            examples,
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
