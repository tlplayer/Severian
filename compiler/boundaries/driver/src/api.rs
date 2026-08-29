use std::collections::{BTreeMap, BTreeSet};
use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

const REQUIRED_FIELDS: &[&str] = &[
    "id",
    "kind",
    "syntax",
    "type_params",
    "parameters",
    "constraints",
    "returns",
    "effects",
    "errors",
    "ownership",
    "universal",
    "lowering",
    "status",
    "since",
    "tests",
    "examples",
    "limitations",
    "snippets",
];

const REQUIRED_TOPOLOGY: &[&str] = &[
    "language/syntax",
    "language/literals",
    "language/primitives",
    "language/operators",
    "language/conversions",
    "language/expressions",
    "language/statements",
    "language/declarations",
    "language/types",
    "language/generics",
    "language/constraints",
    "language/ownership",
    "language/errors",
    "language/effects",
    "language/control",
    "language/concurrency",
    "language/decorators",
    "language/patterns",
    "language/introspection",
    "language/unsafe",
    "language/testing",
    "prelude/types",
    "prelude/functions",
    "prelude/traits",
    "prelude/constants",
    "library/collections",
    "library/file",
    "library/network",
    "library/tensor",
    "library/process",
    "compiler/compile_types",
    "compiler/hooks",
    "compiler/directives",
    "compiler/backends",
];

#[derive(Clone)]
struct Feature {
    id: String,
    parent_id: Option<String>,
    file: PathBuf,
    value: toml::Value,
}

pub fn run(arguments: Vec<String>) -> Result<(), String> {
    let (root, arguments) = parse_root(arguments)?;
    let command = arguments.first().map(String::as_str).unwrap_or("list");
    match command {
        "list" => {
            if arguments.len() != 1 {
                return Err("usage: sev api list [--root PATH]".into());
            }
            let features = load_features(&root)?;
            for feature in features.values() {
                println!(
                    "{}\t{}\t{}",
                    feature.id,
                    string_field(&feature.value, "status")?,
                    string_field(&feature.value, "kind")?
                );
            }
            Ok(())
        }
        "show" => {
            if arguments.len() != 2 {
                return Err("usage: sev api show ID [--root PATH]".into());
            }
            let features = load_features(&root)?;
            let feature = features
                .get(&arguments[1])
                .ok_or_else(|| format!("unknown API feature `{}`", arguments[1]))?;
            println!(
                "# {}\n# source: {}",
                feature.id,
                display_relative(&root, &feature.file)
            );
            println!(
                "{}",
                toml::to_string_pretty(&feature.value).map_err(|error| error.to_string())?
            );
            Ok(())
        }
        "check" => {
            if arguments.len() != 1 {
                return Err("usage: sev api check [--root PATH]".into());
            }
            let report = check(&root)?;
            println!(
                "API check passed: {} features, {} snippets, {} topology sections; coverage: {} Rust enums, {} primitive registries, {} structural IDs, {} library export surfaces (100% source agreement)",
                report.features,
                report.snippets,
                report.sections,
                report.rust_enums,
                report.rust_registries,
                report.source_tokens,
                report.export_surfaces,
            );
            Ok(())
        }
        "diff" => diff(&root, &arguments[1..]),
        _ => Err("usage: sev api <list|show ID|check|diff BASE [CURRENT]> [--root PATH]".into()),
    }
}

fn parse_root(arguments: Vec<String>) -> Result<(PathBuf, Vec<String>), String> {
    let mut root = None;
    let mut remaining = Vec::new();
    let mut iter = arguments.into_iter();
    while let Some(argument) = iter.next() {
        if argument == "--root" {
            root = Some(PathBuf::from(iter.next().ok_or("--root requires a path")?));
        } else {
            remaining.push(argument);
        }
    }
    let root = match root {
        Some(root) => normalize_root(root)?,
        None => discover_root()?,
    };
    Ok((root, remaining))
}

fn normalize_root(path: PathBuf) -> Result<PathBuf, String> {
    let candidate = if path.join("index.toml").is_file() {
        path
    } else {
        path.join("api")
    };
    if candidate.join("index.toml").is_file() {
        fs::canonicalize(&candidate)
            .map_err(|error| format!("could not resolve {}: {error}", candidate.display()))
    } else {
        Err(format!(
            "{} is not an API root (missing index.toml)",
            candidate.display()
        ))
    }
}

fn discover_root() -> Result<PathBuf, String> {
    let cwd = env::current_dir().map_err(|error| error.to_string())?;
    for directory in cwd.ancestors() {
        let candidate = directory.join("api");
        if candidate.join("index.toml").is_file() {
            return Ok(candidate);
        }
    }
    Err("could not find api/index.toml; run inside a Severian checkout or pass --root".into())
}

struct CheckReport {
    features: usize,
    snippets: usize,
    sections: usize,
    rust_enums: usize,
    rust_registries: usize,
    source_tokens: usize,
    export_surfaces: usize,
}

fn check(root: &Path) -> Result<CheckReport, String> {
    let index: toml::Value = parse_toml(&root.join("index.toml"))?;
    for field in [
        "specification_version",
        "root",
        "edition",
        "schema",
        "record_globs",
    ] {
        if index.get(field).is_none() {
            return Err(format!(
                "{}: missing `{field}`",
                root.join("index.toml").display()
            ));
        }
    }
    for section in REQUIRED_TOPOLOGY {
        let path = root.join(section);
        if !path.is_dir() {
            return Err(format!("missing required API section {}", path.display()));
        }
        if !contains_contract_file(&path)? {
            return Err(format!(
                "API section {} has no README.md or TOML contract",
                path.display()
            ));
        }
    }

    let mut files = Vec::new();
    for section in ["language", "prelude", "library", "compiler"] {
        collect_toml(&root.join(section), &mut files)?;
    }
    files.sort();
    let mut features = BTreeMap::<String, Feature>::new();
    let mut snippets = BTreeMap::<String, (PathBuf, toml::Value)>::new();
    let mut export_surfaces = 0;
    for file in files {
        let document = parse_toml(&file)?;
        for value in array(&document, "snippet", &file)? {
            let id = string_field(value, "id")?.to_string();
            if let Some((previous, _)) = snippets.insert(id.clone(), (file.clone(), value.clone()))
            {
                return Err(format!(
                    "duplicate snippet `{id}` in {} and {}",
                    previous.display(),
                    file.display()
                ));
            }
        }
        for value in array(&document, "feature", &file)? {
            validate_feature(value, &file)?;
            validate_source_exports(root, value, &file)?;
            if value.get("export_sources").is_some() {
                export_surfaces += 1;
            }
            for feature in expand_feature(value, &file)? {
                let id = feature.id.clone();
                if let Some(previous) = features.insert(id.clone(), feature) {
                    return Err(format!(
                        "duplicate feature `{id}` in {} and {}",
                        previous.file.display(),
                        file.display()
                    ));
                }
            }
        }
    }
    if features.is_empty() {
        return Err("the API contains no feature records".into());
    }
    let statuses = index
        .get("statuses")
        .and_then(toml::Value::as_array)
        .ok_or("api/index.toml: `statuses` must be an array")?
        .iter()
        .filter_map(toml::Value::as_str)
        .collect::<BTreeSet<_>>();
    for feature in features.values() {
        let status = string_field(&feature.value, "status")?;
        if !statuses.contains(status) {
            return Err(format!(
                "{}: feature `{}` has unknown status `{status}`",
                feature.file.display(),
                feature.id
            ));
        }
        for snippet in string_array(&feature.value, "snippets")? {
            let Some((file, value)) = snippets.get(snippet) else {
                return Err(format!(
                    "{}: feature `{}` references unknown snippet `{snippet}`",
                    feature.file.display(),
                    feature.id
                ));
            };
            let covers = string_array(value, "covers")?;
            if !covers.contains(&feature.id.as_str())
                && !feature
                    .parent_id
                    .as_deref()
                    .is_some_and(|parent| covers.contains(&parent))
            {
                return Err(format!(
                    "{}: snippet `{snippet}` does not cover `{}`",
                    file.display(),
                    feature.id
                ));
            }
        }
        if string_field(&feature.value, "status")? == "implemented"
            && string_array(&feature.value, "tests")?.is_empty()
        {
            return Err(format!(
                "{}: implemented feature `{}` has no tests",
                feature.file.display(),
                feature.id
            ));
        }
    }
    for (id, (file, value)) in &snippets {
        for covered in string_array(value, "covers")? {
            if !features.contains_key(covered) {
                return Err(format!(
                    "{}: snippet `{id}` covers unknown feature `{covered}`",
                    file.display()
                ));
            }
        }
        if string_field(value, "source")?.trim().is_empty() {
            return Err(format!("{}: snippet `{id}` is empty", file.display()));
        }
    }
    for required in index
        .get("coverage")
        .and_then(|value| value.get("required_feature_ids"))
        .and_then(toml::Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(toml::Value::as_str)
    {
        if !features.contains_key(required) {
            return Err(format!(
                "api/index.toml requires missing feature `{required}`"
            ));
        }
    }
    validate_compiler_coverage(root, &index, &features)?;
    let coverage = index.get("coverage").expect("coverage was validated");
    Ok(CheckReport {
        features: features.len(),
        snippets: snippets.len(),
        sections: REQUIRED_TOPOLOGY.len(),
        rust_enums: coverage
            .get("rust_enum")
            .and_then(toml::Value::as_array)
            .map_or(0, Vec::len),
        rust_registries: coverage
            .get("rust_string_registry")
            .and_then(toml::Value::as_array)
            .map_or(0, Vec::len),
        source_tokens: coverage
            .get("source_token")
            .and_then(toml::Value::as_array)
            .map_or(0, Vec::len),
        export_surfaces,
    })
}

fn validate_compiler_coverage(
    root: &Path,
    index: &toml::Value,
    features: &BTreeMap<String, Feature>,
) -> Result<(), String> {
    let repository = root
        .parent()
        .ok_or_else(|| format!("{} has no repository parent", root.display()))?;
    let coverage = index
        .get("coverage")
        .ok_or("api/index.toml: missing [coverage]")?;
    if let Some(enums) = coverage.get("rust_enum").and_then(toml::Value::as_array) {
        for contract in enums {
            let source = string_field(contract, "source")?;
            let enumeration = string_field(contract, "enumeration")?;
            let prefix = string_field(contract, "feature_prefix")?;
            let family_id = contract.get("family_id").and_then(toml::Value::as_str);
            let path = repository.join(source);
            let text = fs::read_to_string(&path)
                .map_err(|error| format!("could not read {}: {error}", path.display()))?;
            let variants = rust_enum_variants(&text, enumeration).ok_or_else(|| {
                format!(
                    "{}: could not find public enum `{enumeration}`",
                    path.display()
                )
            })?;
            let mapped = variants
                .iter()
                .map(|variant| format!("{prefix}.{}", normalize_member_id(variant)))
                .collect::<BTreeSet<_>>();
            let documented = features
                .values()
                .filter(|feature| feature.id.starts_with(&format!("{prefix}.")))
                .filter(|feature| !coverage_family_id(&feature.id))
                .filter(|feature| {
                    family_id.is_none_or(|family| feature.parent_id.as_deref() == Some(family))
                })
                .map(|feature| feature.id.clone())
                .collect::<BTreeSet<_>>();
            let missing = mapped.difference(&documented).cloned().collect::<Vec<_>>();
            let stale = documented.difference(&mapped).cloned().collect::<Vec<_>>();
            if !missing.is_empty() || !stale.is_empty() {
                return Err(format!(
                    "{}::{enumeration} API coverage mismatch; missing={missing:?}, stale={stale:?}",
                    path.display()
                ));
            }
        }
    }
    if let Some(registries) = coverage
        .get("rust_string_registry")
        .and_then(toml::Value::as_array)
    {
        for contract in registries {
            let source = string_field(contract, "source")?;
            let marker = string_field(contract, "marker")?;
            let prefix = string_field(contract, "feature_prefix")?;
            let path = repository.join(source);
            let text = fs::read_to_string(&path)
                .map_err(|error| format!("could not read {}: {error}", path.display()))?;
            let mut names = BTreeSet::new();
            for line in text.lines() {
                let Some((_, suffix)) = line.split_once(marker) else {
                    continue;
                };
                if let Some(name) = suffix.split('"').next().filter(|name| !name.is_empty()) {
                    names.insert(format!("{prefix}.{}", normalize_member_id(name)));
                }
            }
            for id in names {
                if !features.contains_key(&id) {
                    return Err(format!(
                        "{} registry entry has no API feature `{id}`",
                        path.display()
                    ));
                }
            }
        }
    }
    if let Some(tokens) = coverage.get("source_token").and_then(toml::Value::as_array) {
        for contract in tokens {
            let id = string_field(contract, "id")?;
            let source = string_field(contract, "source")?;
            let token = string_field(contract, "token")?;
            if !features.contains_key(id) {
                return Err(format!(
                    "source-token contract references missing feature `{id}`"
                ));
            }
            let path = repository.join(source);
            let text = fs::read_to_string(&path)
                .map_err(|error| format!("could not read {}: {error}", path.display()))?;
            if !text.contains(token) {
                return Err(format!(
                    "{}: feature `{id}` requires absent compiler token `{token}`",
                    path.display()
                ));
            }
        }
    }
    Ok(())
}

fn coverage_family_id(id: &str) -> bool {
    matches!(
        id,
        "literal.kind"
            | "expression.kind"
            | "statement.kind"
            | "declaration.item"
            | "conversion.kind"
            | "operator.unary"
            | "operator.binary"
    )
}

fn rust_enum_variants(source: &str, enumeration: &str) -> Option<BTreeSet<String>> {
    let declaration = format!("pub enum {enumeration}");
    let start = source.find(&declaration)?;
    let source = &source[start..];
    let mut depth = 0_i32;
    let mut entered = false;
    let mut variants = BTreeSet::new();
    for line in source.lines() {
        let depth_at_start = depth;
        if entered && depth_at_start == 1 {
            let candidate = line.trim_start();
            if candidate
                .chars()
                .next()
                .is_some_and(|character| character.is_ascii_uppercase())
            {
                let name = candidate
                    .split(|character: char| {
                        character.is_whitespace() || matches!(character, '(' | '{' | ',' | '=')
                    })
                    .next()
                    .unwrap_or_default();
                if !name.is_empty() {
                    variants.insert(name.to_string());
                }
            }
        }
        for character in line.chars() {
            if character == '{' {
                depth += 1;
                entered = true;
            } else if character == '}' {
                depth -= 1;
            }
        }
        if entered && depth == 0 {
            break;
        }
    }
    Some(variants)
}

fn load_features(root: &Path) -> Result<BTreeMap<String, Feature>, String> {
    check(root)?;
    let mut files = Vec::new();
    for section in ["language", "prelude", "library", "compiler"] {
        collect_toml(&root.join(section), &mut files)?;
    }
    let mut features = BTreeMap::new();
    for file in files {
        let document = parse_toml(&file)?;
        for value in array(&document, "feature", &file)? {
            for feature in expand_feature(value, &file)? {
                features.insert(feature.id.clone(), feature);
            }
        }
    }
    Ok(features)
}

fn diff(root: &Path, arguments: &[String]) -> Result<(), String> {
    if arguments.is_empty() {
        let repository = root
            .parent()
            .ok_or_else(|| format!("{} has no repository parent", root.display()))?;
        let output = Command::new("git")
            .args(["status", "--short", "--", "api"])
            .current_dir(repository)
            .output()
            .map_err(|error| format!("could not inspect API workspace changes: {error}"))?;
        if !output.status.success() {
            return Err(String::from_utf8_lossy(&output.stderr).trim().to_string());
        }
        let changes = String::from_utf8_lossy(&output.stdout);
        if changes.trim().is_empty() {
            println!("no API workspace changes");
        } else {
            print!("{changes}");
        }
        return Ok(());
    }
    if arguments.len() > 2 {
        return Err("usage: sev api diff BASE [CURRENT] [--root PATH]".into());
    }
    let base = normalize_root(PathBuf::from(&arguments[0]))?;
    let current = if arguments.len() == 2 {
        normalize_root(PathBuf::from(&arguments[1]))?
    } else {
        root.to_path_buf()
    };
    let before = load_features(&base)?;
    let after = load_features(&current)?;
    let mut changes = 0;
    for id in before.keys().filter(|id| !after.contains_key(*id)) {
        println!("- {id}");
        changes += 1;
    }
    for id in after.keys().filter(|id| !before.contains_key(*id)) {
        println!("+ {id}");
        changes += 1;
    }
    for (id, old) in &before {
        if let Some(new) = after.get(id) {
            let old_status = string_field(&old.value, "status")?;
            let new_status = string_field(&new.value, "status")?;
            if old_status != new_status {
                println!("~ {id}: {old_status} -> {new_status}");
                changes += 1;
            }
        }
    }
    if changes == 0 {
        println!("no API surface changes");
    }
    Ok(())
}

fn validate_feature(value: &toml::Value, file: &Path) -> Result<(), String> {
    let table = value
        .as_table()
        .ok_or_else(|| format!("{}: feature must be a table", file.display()))?;
    for field in REQUIRED_FIELDS {
        if !table.contains_key(*field) {
            return Err(format!("{}: feature is missing `{field}`", file.display()));
        }
    }
    let id = string_field(value, "id")?;
    if !valid_id(id) {
        return Err(format!(
            "{}: invalid stable feature id `{id}`",
            file.display()
        ));
    }
    for field in [
        "type_params",
        "parameters",
        "constraints",
        "effects",
        "errors",
        "tests",
        "examples",
        "limitations",
        "snippets",
    ] {
        string_array(value, field)?;
    }
    if string_array(value, "snippets")?.is_empty() {
        return Err(format!(
            "{}: feature `{id}` has no conformance snippet",
            file.display()
        ));
    }
    for field in [
        "kind",
        "syntax",
        "returns",
        "ownership",
        "universal",
        "lowering",
        "status",
        "since",
    ] {
        string_field(value, field)?;
    }
    Ok(())
}

fn validate_source_exports(root: &Path, value: &toml::Value, file: &Path) -> Result<(), String> {
    let Some(sources) = value.get("export_sources") else {
        return Ok(());
    };
    let sources = sources
        .as_array()
        .ok_or_else(|| format!("{}: `export_sources` must be an array", file.display()))?;
    let declared = if value.get("export_symbols").is_some() {
        string_array(value, "export_symbols")?
            .into_iter()
            .map(str::to_string)
            .collect::<BTreeSet<_>>()
    } else {
        value
            .get("members")
            .and_then(toml::Value::as_array)
            .into_iter()
            .flatten()
            .filter_map(|member| {
                member.as_str().or_else(|| {
                    member
                        .as_table()
                        .and_then(|table| table.get("syntax"))
                        .and_then(toml::Value::as_str)
                })
            })
            .map(source_symbol_from_syntax)
            .collect()
    };
    let repository = root
        .parent()
        .ok_or_else(|| format!("{} has no repository parent", root.display()))?;
    let mut actual = BTreeSet::new();
    for source in sources {
        let relative = source
            .as_str()
            .ok_or_else(|| format!("{}: export source must be a string", file.display()))?;
        let path = repository.join(relative);
        let text = fs::read_to_string(&path)
            .map_err(|error| format!("could not read export source {}: {error}", path.display()))?;
        actual.extend(public_source_symbols(&text));
    }
    if actual != declared {
        let missing = actual.difference(&declared).cloned().collect::<Vec<_>>();
        let stale = declared.difference(&actual).cloned().collect::<Vec<_>>();
        return Err(format!(
            "{}: exported symbol inventory disagrees with source; missing={missing:?}, stale={stale:?}",
            file.display()
        ));
    }
    let member_ids = value
        .get("members")
        .and_then(toml::Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(|member| {
            member.as_str().or_else(|| {
                member
                    .as_table()
                    .and_then(|table| table.get("syntax"))
                    .and_then(toml::Value::as_str)
            })
        })
        .map(source_symbol_from_syntax)
        .collect::<BTreeSet<_>>();
    let undocumented = declared
        .difference(&member_ids)
        .cloned()
        .collect::<Vec<_>>();
    if !undocumented.is_empty() {
        return Err(format!(
            "{}: exported symbols have no API member records: {undocumented:?}",
            file.display()
        ));
    }
    Ok(())
}

fn public_source_symbols(source: &str) -> BTreeSet<String> {
    source
        .lines()
        .filter_map(|line| {
            if line.starts_with(|character: char| character.is_whitespace()) {
                return None;
            }
            ["def ", "class ", "trait ", "enum ", "type "]
                .into_iter()
                .find_map(|prefix| line.strip_prefix(prefix))
                .map(source_symbol_from_syntax)
        })
        .collect()
}

fn source_symbol_from_syntax(source: &str) -> String {
    source
        .split(|character: char| {
            character.is_whitespace()
                || matches!(character, '[' | '(' | ':' | '=' | '<' | '|' | ',')
        })
        .next()
        .unwrap_or(source)
        .to_string()
}

fn expand_feature(value: &toml::Value, file: &Path) -> Result<Vec<Feature>, String> {
    let parent_id = string_field(value, "id")?.to_string();
    let mut output = vec![Feature {
        id: parent_id.clone(),
        parent_id: None,
        file: file.to_path_buf(),
        value: value.clone(),
    }];
    let Some(members) = value.get("members") else {
        return Ok(output);
    };
    let members = members
        .as_array()
        .ok_or_else(|| format!("{}: `members` must be an array", file.display()))?;
    let namespace = parent_id
        .split_once('.')
        .map(|(namespace, _)| namespace)
        .ok_or_else(|| format!("{}: feature `{parent_id}` has no namespace", file.display()))?;
    let namespace = value
        .get("member_namespace")
        .and_then(toml::Value::as_str)
        .unwrap_or(namespace);
    for member in members {
        let (id, syntax, universal) = if let Some(name) = member.as_str() {
            (
                format!("{namespace}.{}", normalize_member_id(name)),
                name.to_string(),
                format!("{} member `{name}`", string_field(value, "universal")?),
            )
        } else {
            let table = member.as_table().ok_or_else(|| {
                format!(
                    "{}: members of `{parent_id}` must be strings or tables",
                    file.display()
                )
            })?;
            (
                table
                    .get("id")
                    .and_then(toml::Value::as_str)
                    .ok_or_else(|| format!("{}: member is missing string `id`", file.display()))?
                    .to_string(),
                table
                    .get("syntax")
                    .and_then(toml::Value::as_str)
                    .ok_or_else(|| {
                        format!("{}: member is missing string `syntax`", file.display())
                    })?
                    .to_string(),
                table
                    .get("universal")
                    .and_then(toml::Value::as_str)
                    .ok_or_else(|| {
                        format!("{}: member is missing string `universal`", file.display())
                    })?
                    .to_string(),
            )
        };
        if !valid_id(&id) {
            return Err(format!(
                "{}: invalid member feature id `{id}`",
                file.display()
            ));
        }
        let mut expanded = value.clone();
        let table = expanded.as_table_mut().expect("validated feature table");
        table.insert("id".into(), toml::Value::String(id.clone()));
        table.insert("syntax".into(), toml::Value::String(syntax));
        table.insert("universal".into(), toml::Value::String(universal));
        table.remove("members");
        output.push(Feature {
            id,
            parent_id: Some(parent_id.clone()),
            file: file.to_path_buf(),
            value: expanded,
        });
    }
    Ok(output)
}

fn normalize_member_id(name: &str) -> String {
    let mut output = String::new();
    for (index, character) in name.chars().enumerate() {
        if character.is_ascii_uppercase() {
            if index > 0 {
                output.push('_');
            }
            output.push(character.to_ascii_lowercase());
        } else if character.is_ascii_alphanumeric() || character == '_' {
            output.push(character.to_ascii_lowercase());
        } else if !output.ends_with('_') {
            output.push('_');
        }
    }
    output.trim_matches('_').to_string()
}

fn valid_id(id: &str) -> bool {
    let mut parts = id.split('.');
    let mut count = 0;
    for part in &mut parts {
        count += 1;
        let mut chars = part.chars();
        if !chars.next().is_some_and(|ch| ch.is_ascii_lowercase())
            || !chars.all(|ch| ch.is_ascii_lowercase() || ch.is_ascii_digit() || ch == '_')
        {
            return false;
        }
    }
    count >= 2
}

fn contains_contract_file(path: &Path) -> Result<bool, String> {
    for entry in
        fs::read_dir(path).map_err(|error| format!("could not read {}: {error}", path.display()))?
    {
        let path = entry.map_err(|error| error.to_string())?.path();
        if path.is_file()
            && matches!(
                path.extension().and_then(|value| value.to_str()),
                Some("md" | "toml")
            )
        {
            return Ok(true);
        }
    }
    Ok(false)
}

fn collect_toml(path: &Path, output: &mut Vec<PathBuf>) -> Result<(), String> {
    if !path.exists() {
        return Ok(());
    }
    for entry in
        fs::read_dir(path).map_err(|error| format!("could not read {}: {error}", path.display()))?
    {
        let path = entry.map_err(|error| error.to_string())?.path();
        if path.is_dir() {
            collect_toml(&path, output)?;
        } else if path.extension().and_then(|value| value.to_str()) == Some("toml") {
            output.push(path);
        }
    }
    Ok(())
}

fn parse_toml(path: &Path) -> Result<toml::Value, String> {
    let source = fs::read_to_string(path)
        .map_err(|error| format!("could not read {}: {error}", path.display()))?;
    toml::from_str(&source).map_err(|error| format!("{}: {error}", path.display()))
}

fn array<'a>(
    document: &'a toml::Value,
    field: &str,
    file: &Path,
) -> Result<&'a [toml::Value], String> {
    match document.get(field) {
        Some(value) => value
            .as_array()
            .map(Vec::as_slice)
            .ok_or_else(|| format!("{}: `{field}` must be an array", file.display())),
        None => Ok(&[]),
    }
}

fn string_field<'a>(value: &'a toml::Value, field: &str) -> Result<&'a str, String> {
    value
        .get(field)
        .and_then(toml::Value::as_str)
        .ok_or_else(|| format!("`{field}` must be a string"))
}

fn string_array<'a>(value: &'a toml::Value, field: &str) -> Result<Vec<&'a str>, String> {
    value
        .get(field)
        .and_then(toml::Value::as_array)
        .ok_or_else(|| format!("`{field}` must be an array"))?
        .iter()
        .map(|item| {
            item.as_str()
                .ok_or_else(|| format!("`{field}` must contain only strings"))
        })
        .collect()
}

fn display_relative(root: &Path, path: &Path) -> String {
    path.strip_prefix(root)
        .unwrap_or(path)
        .display()
        .to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stable_ids_require_two_lower_snake_case_segments() {
        assert!(valid_id("operator.add"));
        assert!(valid_id("tensor.reshape_view"));
        assert!(!valid_id("operator"));
        assert!(!valid_id("Operator.Add"));
        assert!(!valid_id("operator.add-value"));
    }

    #[test]
    fn member_names_become_stable_ids() {
        assert_eq!(normalize_member_id("TypeApplication"), "type_application");
        assert_eq!(normalize_member_id("f8e4m3fn"), "f8e4m3fn");
        assert_eq!(normalize_member_id("None"), "none");
    }

    #[test]
    fn source_symbols_are_collected_once_across_overloads() {
        let source = "trait File:\nclass TextFile: File\ndef write(path: string)\ndef write(path: string, bytes: list[u8])\n";
        assert_eq!(
            public_source_symbols(source),
            ["File", "TextFile", "write"]
                .into_iter()
                .map(str::to_string)
                .collect()
        );
    }

    #[test]
    fn rust_enum_variants_ignore_struct_fields() {
        let source = "pub enum Value {\n    Unit,\n    Named {\n        name: String,\n    },\n    Tuple(u8),\n}\n";
        assert_eq!(
            rust_enum_variants(source, "Value").unwrap(),
            ["Named", "Tuple", "Unit"]
                .into_iter()
                .map(str::to_string)
                .collect()
        );
    }
}
