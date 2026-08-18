use crate::{PackageError, ResolvedDependency};
use std::fs;
use std::path::Path;

pub const LOCK_VERSION: i64 = 1;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LockedPackage {
    pub name: String,
    pub version: String,
    pub source: String,
    pub checksum: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LockedExternal {
    pub name: String,
    pub version: String,
    pub publisher: String,
    pub source: String,
    pub sha256: String,
    pub signature: String,
    pub trusted_from: String,
    pub trusted_until: String,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Lockfile {
    pub packages: Vec<LockedPackage>,
    pub external: Vec<LockedExternal>,
}

pub fn read(path: &Path) -> Result<Lockfile, PackageError> {
    let source = match fs::read_to_string(path) {
        Ok(source) => source,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
            return Ok(Lockfile::default())
        }
        Err(error) => return Err(error.into()),
    };
    let value = toml::from_str::<toml::Value>(&source).map_err(|error| {
        PackageError::Manifest(format!("invalid lockfile {}: {error}", path.display()))
    })?;
    if value
        .get("version")
        .and_then(toml::Value::as_integer)
        .unwrap_or(LOCK_VERSION)
        != LOCK_VERSION
    {
        return Err(PackageError::Manifest(format!(
            "unsupported lockfile version in {}",
            path.display()
        )));
    }
    Ok(Lockfile {
        packages: read_packages(&value, path)?,
        external: read_external(&value, path)?,
    })
}

pub fn write(path: &Path, lock: &Lockfile) -> Result<(), PackageError> {
    let mut packages = lock.packages.clone();
    packages.sort_by(|left, right| {
        (&left.name, &left.version, &left.source).cmp(&(&right.name, &right.version, &right.source))
    });
    packages.dedup_by(|left, right| {
        left.name == right.name && left.version == right.version && left.source == right.source
    });
    let mut external = lock.external.clone();
    external.sort_by(|left, right| (&left.name, &left.version).cmp(&(&right.name, &right.version)));
    external.dedup_by(|left, right| left.name == right.name && left.version == right.version);

    let mut output = format!("version = {LOCK_VERSION}\n");
    for package in packages {
        output.push_str("\n[[packages]]\n");
        push_string(&mut output, "name", &package.name);
        push_string(&mut output, "version", &package.version);
        push_string(&mut output, "source", &package.source);
        if let Some(checksum) = package.checksum {
            push_string(&mut output, "checksum", &checksum);
        }
    }
    for item in external {
        output.push_str("\n[[external]]\n");
        push_string(&mut output, "name", &item.name);
        push_string(&mut output, "version", &item.version);
        push_string(&mut output, "publisher", &item.publisher);
        push_string(&mut output, "source", &item.source);
        push_string(&mut output, "sha256", &item.sha256);
        push_string(&mut output, "signature", &item.signature);
        push_string(&mut output, "trusted_from", &item.trusted_from);
        push_string(&mut output, "trusted_until", &item.trusted_until);
    }
    let temporary = path.with_extension("lock.tmp");
    fs::write(&temporary, output)?;
    fs::rename(temporary, path)?;
    Ok(())
}

pub fn packages_from_resolution(dependencies: &[ResolvedDependency]) -> Vec<LockedPackage> {
    dependencies
        .iter()
        .map(|dependency| LockedPackage {
            name: dependency.package_name.clone(),
            version: dependency.version.clone(),
            source: dependency.source.clone(),
            checksum: dependency.checksum.clone(),
        })
        .collect()
}

fn read_packages(value: &toml::Value, path: &Path) -> Result<Vec<LockedPackage>, PackageError> {
    let Some(entries) = value.get("packages") else {
        return Ok(Vec::new());
    };
    entries
        .as_array()
        .ok_or_else(|| field_error(path, "`packages` must be an array"))?
        .iter()
        .map(|entry| {
            let table = entry
                .as_table()
                .ok_or_else(|| field_error(path, "package entry must be a table"))?;
            Ok(LockedPackage {
                name: string_field(table, "name", path)?,
                version: string_field(table, "version", path)?,
                source: string_field(table, "source", path)?,
                checksum: table
                    .get("checksum")
                    .and_then(toml::Value::as_str)
                    .map(str::to_owned),
            })
        })
        .collect()
}

fn read_external(value: &toml::Value, path: &Path) -> Result<Vec<LockedExternal>, PackageError> {
    let Some(entries) = value.get("external") else {
        return Ok(Vec::new());
    };
    entries
        .as_array()
        .ok_or_else(|| field_error(path, "`external` must be an array"))?
        .iter()
        .map(|entry| {
            let table = entry
                .as_table()
                .ok_or_else(|| field_error(path, "external entry must be a table"))?;
            Ok(LockedExternal {
                name: string_field(table, "name", path)?,
                version: string_field(table, "version", path)?,
                publisher: string_field(table, "publisher", path)?,
                source: string_field(table, "source", path)?,
                sha256: string_field(table, "sha256", path)?,
                signature: string_field(table, "signature", path)?,
                trusted_from: string_field(table, "trusted_from", path)?,
                trusted_until: string_field(table, "trusted_until", path)?,
            })
        })
        .collect()
}

fn string_field(table: &toml::Table, name: &str, path: &Path) -> Result<String, PackageError> {
    table
        .get(name)
        .and_then(toml::Value::as_str)
        .map(str::to_owned)
        .ok_or_else(|| field_error(path, format!("entry has no string `{name}`")))
}

fn field_error(path: &Path, message: impl std::fmt::Display) -> PackageError {
    PackageError::Manifest(format!("invalid lockfile {}: {message}", path.display()))
}

fn push_string(output: &mut String, key: &str, value: &str) {
    output.push_str(&format!("{key} = {value:?}\n"));
}
