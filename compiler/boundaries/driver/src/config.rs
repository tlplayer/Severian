use serde::Deserialize;
use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::{Path, PathBuf};

const CATALOG_SOURCE: &str = include_str!("../config/options.toml");

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
    package: Option<String>,
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

    /// Produces the source manifest stored in a registry release. Runtime
    /// dependencies are frozen to registry package identities; development
    /// paths and root-only targets must not leak into a consumer's graph.
    pub fn published_source_manifest(&self) -> Result<String, String> {
        let root = self
            .package_graph
            .packages
            .get(&self.package_graph.root)
            .expect("resolved package graph contains its root");
        let mut document = root.manifest.clone();
        let table = document
            .as_table_mut()
            .ok_or_else(|| "package manifest root must be a table".to_owned())?;
        table.remove("bin");
        table.remove("dev-dependencies");

        let package = table
            .get_mut("package")
            .and_then(toml::Value::as_table_mut)
            .ok_or_else(|| "published package requires a `[package]` table".to_owned())?;
        package.insert("publish".into(), toml::Value::Boolean(false));
        package.remove("default-run");

        let replacements = root
            .dependencies
            .iter()
            .map(|(alias, id)| {
                let dependency = self.package_graph.packages.get(id).ok_or_else(|| {
                    format!("resolved dependency `{alias}` is missing from the package graph")
                })?;
                let version = manifest_package_version(&dependency.manifest);
                let registry = root
                    .manifest
                    .get("dependencies")
                    .and_then(toml::Value::as_table)
                    .and_then(|dependencies| dependencies.get(alias))
                    .and_then(toml::Value::as_table)
                    .and_then(|detail| detail.get("registry"))
                    .and_then(toml::Value::as_str)
                    .filter(|registry| *registry != "default");
                let value = if alias == &dependency.name && registry.is_none() {
                    toml::Value::String(version)
                } else {
                    let mut detail = toml::map::Map::new();
                    detail.insert(
                        "package".into(),
                        toml::Value::String(dependency.name.clone()),
                    );
                    detail.insert("version".into(), toml::Value::String(version));
                    if let Some(registry) = registry {
                        detail.insert(
                            "registry".into(),
                            toml::Value::String(registry.to_owned()),
                        );
                    }
                    toml::Value::Table(detail)
                };
                Ok((alias.clone(), value))
            })
            .collect::<Result<Vec<_>, String>>()?;
        let dependencies = table
            .entry("dependencies")
            .or_insert_with(|| toml::Value::Table(toml::map::Map::new()))
            .as_table_mut()
            .ok_or_else(|| "`[dependencies]` must be a table".to_owned())?;
        dependencies.clear();
        dependencies.extend(replacements);
        toml::to_string_pretty(&document)
            .map_err(|error| format!("could not serialize published package manifest: {error}"))
    }
}

fn manifest_package_version(manifest: &toml::Value) -> String {
    manifest
        .get("package")
        .and_then(|package| package.get("version"))
        .and_then(toml::Value::as_str)
        .unwrap_or("0.1.0")
        .to_owned()
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
        let dependencies =
            self.resolve_dependencies(&root, &name, &document.dependencies)?;
        let dev_dependencies =
            self.resolve_dependencies(&root, &name, &document.dev_dependencies)?;
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
        owner: &str,
        declarations: &BTreeMap<String, DependencyDeclaration>,
    ) -> Result<BTreeMap<String, severian_modules::PackageId>, String> {
        declarations
            .iter()
            .map(|(alias, declaration)| {
                let path = dependency_path(alias, declaration).map_err(|error| {
                    format!("package `{owner}` dependency `{alias}` could not resolve: {error}")
                })?;
                let manifest = root.join(path).join("package.toml");
                let id = self.resolve(&manifest, true).map_err(|error| {
                    format!("while resolving package `{owner}` dependency `{alias}`: {error}")
                })?;
                let dependency = self
                    .packages
                    .get(&id)
                    .expect("resolved dependency was inserted into the package graph");
                if let Some(expected) = dependency_package_name(alias, declaration) {
                    if dependency.name != expected {
                        return Err(format!(
                            "package `{owner}` dependency `{alias}` requested package `{expected}` but resolved `{}`",
                            dependency.name
                        ));
                    }
                }
                if let Some(expected) = dependency_version(declaration) {
                    let actual = manifest_package_version(&dependency.manifest);
                    if actual != expected {
                        return Err(format!(
                            "package `{owner}` dependency `{alias}` requires version `{expected}` but resolved `{actual}`"
                        ));
                    }
                }
                Ok((alias.clone(), id))
            })
            .collect()
    }
}

fn dependency_package_name<'a>(
    alias: &'a str,
    declaration: &'a DependencyDeclaration,
) -> Option<&'a str> {
    match declaration {
        DependencyDeclaration::Version(_) => Some(alias),
        DependencyDeclaration::Detailed(detail) => detail
            .package
            .as_deref()
            .or_else(|| detail.path.is_none().then_some(alias)),
    }
}

fn dependency_version(declaration: &DependencyDeclaration) -> Option<&str> {
    match declaration {
        DependencyDeclaration::Version(version) => Some(version),
        DependencyDeclaration::Detailed(detail) => detail.version.as_deref(),
    }
}

fn dependency_path(alias: &str, declaration: &DependencyDeclaration) -> Result<PathBuf, String> {
    match declaration {
        DependencyDeclaration::Detailed(detail) => {
            if let Some(path) = detail.path.as_deref() {
                return Ok(path.to_owned());
            }
            if detail.git.is_some() {
                return Err(format!(
                    "dependency `{alias}` uses unsupported git resolution"
                ));
            }
            let version = detail.version.as_deref().ok_or_else(|| {
                format!("registry dependency `{alias}` requires an exact version")
            })?;
            registry_package(
                detail.package.as_deref().unwrap_or(alias),
                version,
                detail.registry.as_deref(),
            )
        }
        DependencyDeclaration::Version(version) => registry_package(alias, version, None),
    }
}

fn registry_package(name: &str, version: &str, registry: Option<&str>) -> Result<PathBuf, String> {
    let root = registry_root(registry)?;
    let source = registry_release_path(&root, name, version)?.join("source");
    if source.join("package.toml").is_file() {
        Ok(source)
    } else {
        Err(format!(
            "package `{name}` version `{version}` is not present in registry `{}`; publish or pull it first",
            root.display()
        ))
    }
}

pub fn registry_release_path(root: &Path, name: &str, version: &str) -> Result<PathBuf, String> {
    for (label, value) in [("package", name), ("version", version)] {
        if value.is_empty()
            || matches!(value, "." | "..")
            || value.contains('/')
            || value.contains('\\')
        {
            return Err(format!("invalid registry {label} component `{value}`"));
        }
    }
    Ok(root.join("packages").join(name).join(version))
}

pub fn registry_root(registry: Option<&str>) -> Result<PathBuf, String> {
    if let Some(registry) = registry.filter(|registry| *registry != "default") {
        if registry.contains("://") {
            return Err(format!(
                "remote registry transport `{registry}` is not implemented; use a filesystem registry"
            ));
        }
        return Ok(PathBuf::from(registry));
    }
    if let Some(path) = std::env::var_os("SEVERIAN_REGISTRY") {
        return Ok(PathBuf::from(path));
    }
    if let Some(path) = std::env::var_os("XDG_DATA_HOME") {
        return Ok(PathBuf::from(path).join("severian/registry"));
    }
    std::env::var_os("HOME")
        .map(PathBuf::from)
        .map(|path| path.join(".local/share/severian/registry"))
        .ok_or_else(|| "could not locate the default registry; set SEVERIAN_REGISTRY".to_owned())
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

    fn temporary_package(name: &str) -> PathBuf {
        let root = std::env::temp_dir().join(format!(
            "severian-package-config-{name}-{}",
            std::process::id()
        ));
        if root.exists() {
            std::fs::remove_dir_all(&root).unwrap();
        }
        std::fs::create_dir_all(&root).unwrap();
        root
    }

    #[test]
    fn catalog_drives_defaults_validation_and_template() {
        let catalog = Catalog::load().unwrap();
        let template = catalog.template("hello");
        assert!(template.contains("profile = \"dev\""));
        assert!(template.contains("target = \"host\""));
        assert!(!template.contains("backend ="));
    }

    #[test]
    fn registry_release_paths_cannot_escape_the_registry() {
        let root = Path::new("/tmp/severian-registry-test");
        assert_eq!(
            registry_release_path(root, "example", "1.2.3").unwrap(),
            root.join("packages/example/1.2.3")
        );
        assert!(registry_release_path(root, "../example", "1.2.3").is_err());
        assert!(registry_release_path(root, "example", "../1.2.3").is_err());
    }

    #[test]
    fn published_source_manifest_freezes_path_dependencies_to_registry_identities() {
        let root = temporary_package("published-manifest");
        let dependency = root.join("dependency");
        let package = root.join("package");
        std::fs::create_dir_all(dependency.join("src")).unwrap();
        std::fs::create_dir_all(package.join("src")).unwrap();
        std::fs::write(
            dependency.join("package.toml"),
            "[package]\nname = \"actual-dependency\"\nversion = \"2.3.4\"\n\n[lib]\npath = \"src/lib.sev\"\n",
        )
        .unwrap();
        std::fs::write(dependency.join("src/lib.sev"), "def value() -> int:\n    return 1\n")
            .unwrap();
        std::fs::write(
            package.join("package.toml"),
            "[package]\nname = \"root\"\nversion = \"1.0.0\"\ndefault-run = \"root\"\n\n[[bin]]\nname = \"root\"\npath = \"src/main.sev\"\n\n[lib]\npath = \"src/lib.sev\"\n\n[dependencies]\nhelper = { path = \"../dependency\" }\n\n[dev-dependencies]\ntest_helper = { path = \"../dependency\" }\n",
        )
        .unwrap();
        std::fs::write(package.join("src/lib.sev"), "import helper\n").unwrap();
        std::fs::write(package.join("src/main.sev"), "print(\"root\")\n").unwrap();

        let manifest = Manifest::load(&package.join("package.toml"), &Catalog::load().unwrap())
            .unwrap();
        let published = manifest.published_source_manifest().unwrap();
        let value = published.parse::<toml::Value>().unwrap();
        let dependency = &value["dependencies"]["helper"];
        assert_eq!(dependency["package"].as_str(), Some("actual-dependency"));
        assert_eq!(dependency["version"].as_str(), Some("2.3.4"));
        assert!(dependency.get("path").is_none());
        assert!(value.get("dev-dependencies").is_none());
        assert!(value.get("bin").is_none());
        assert!(value["package"].get("default-run").is_none());
        assert_eq!(value["package"]["publish"].as_bool(), Some(false));
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn mirrored_driver_manifest_retains_its_dependency_aliases() {
        let catalog = Catalog::load().unwrap();
        let manifest = Manifest::load(
            &Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("../../../sev_compiler/boundaries/driver/package.toml"),
            &catalog,
        )
        .unwrap();
        let graph = manifest.module_graph(false);
        let root = graph.packages.get(&graph.root).unwrap();
        assert!(root.dependencies.contains_key("abi"));
        assert!(!root.dependencies.contains_key("runtime"));
        severian_modules::resolve_with_packages(
            manifest.library.as_ref().unwrap().path.as_path(),
            &graph,
        )
        .unwrap();
        let compiler = crate::Compiler::new(severian_target::TargetSpec::host())
            .unwrap()
            .with_packages(graph);
        let output = std::env::temp_dir().join(format!(
            "severian-bootstrap-driver-test-{}",
            std::process::id()
        ));
        compiler
            .compile_file(&manifest.bins[0].path, &output)
            .unwrap();
        compiler
            .check_file(&manifest.library.as_ref().unwrap().path)
            .unwrap();
        std::fs::remove_file(output).unwrap();
    }
}
