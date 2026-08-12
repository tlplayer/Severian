use crate::signature::validate_ed25519_public_key;
use crate::PackageError;
use std::cmp::Ordering;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Date(String);

impl Date {
    pub fn parse(value: &str) -> Result<Self, PackageError> {
        let bytes = value.as_bytes();
        if bytes.len() != 10
            || bytes[4] != b'-'
            || bytes[7] != b'-'
            || bytes
                .iter()
                .enumerate()
                .any(|(index, byte)| !matches!(index, 4 | 7) && !byte.is_ascii_digit())
        {
            return Err(PackageError::Manifest(format!(
                "invalid trust date `{value}`; expected YYYY-MM-DD"
            )));
        }
        let year = value[0..4].parse::<i32>().unwrap_or(0);
        let month = value[5..7].parse::<u32>().unwrap_or(0);
        let day = value[8..10].parse::<u32>().unwrap_or(0);
        let days = days_from_civil(year, month, day);
        let (round_year, round_month, round_day) = civil_from_days(days);
        if (year, month, day) != (round_year, round_month, round_day) {
            return Err(PackageError::Manifest(format!(
                "invalid trust date `{value}`"
            )));
        }
        Ok(Self(value.to_owned()))
    }

    pub fn today() -> Self {
        let days = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs()
            / 86_400;
        let (year, month, day) = civil_from_days(days as i64);
        Self(format!("{year:04}-{month:02}-{day:02}"))
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl Ord for Date {
    fn cmp(&self, other: &Self) -> Ordering {
        self.0.cmp(&other.0)
    }
}

impl PartialOrd for Date {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TrustedPublisher {
    pub name: String,
    pub allowed_domains: Vec<String>,
    pub signing_keys: Vec<String>,
    pub package_namespaces: Vec<String>,
    pub trusted_from: Date,
    pub trusted_until: Date,
    pub allow_system_install: bool,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct TrustRegistry {
    pub publishers: Vec<TrustedPublisher>,
}

impl TrustRegistry {
    pub fn load_default() -> Result<Self, PackageError> {
        let path = severian_home().join("trust/publishers.toml");
        if !path.is_file() {
            return Ok(Self::default());
        }
        Self::load(&path)
    }

    pub fn load(path: &Path) -> Result<Self, PackageError> {
        let source = fs::read_to_string(path).map_err(|error| {
            PackageError::Manifest(format!(
                "could not read compiler trust registry {}: {error}",
                path.display()
            ))
        })?;
        let value = toml::from_str::<toml::Value>(&source).map_err(|error| {
            PackageError::Manifest(format!(
                "invalid trust registry {}: {error}",
                path.display()
            ))
        })?;
        let entries = value
            .get("publisher")
            .and_then(toml::Value::as_array)
            .ok_or_else(|| {
                PackageError::Manifest(format!("{} has no `[[publisher]]` entries", path.display()))
            })?;
        let publishers = entries
            .iter()
            .map(|entry| parse_publisher(entry, path))
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Self { publishers })
    }

    pub fn publisher(&self, name: &str) -> Result<&TrustedPublisher, PackageError> {
        self.publishers
            .iter()
            .find(|publisher| publisher.name == name)
            .ok_or_else(|| PackageError::Manifest(format!("publisher `{name}` is not trusted")))
    }
}

pub fn severian_home() -> PathBuf {
    std::env::var_os("SEVERIAN_HOME")
        .map(PathBuf::from)
        .or_else(|| std::env::var_os("HOME").map(|home| PathBuf::from(home).join(".sev")))
        .unwrap_or_else(|| PathBuf::from(".sev"))
}

pub fn validate_publisher(
    publisher: &TrustedPublisher,
    package: &str,
    source: &str,
    today: &Date,
) -> Result<(), PackageError> {
    if today < &publisher.trusted_from || today > &publisher.trusted_until {
        return Err(PackageError::Manifest(format!(
            "publisher `{}` is outside its trust period {} through {}",
            publisher.name,
            publisher.trusted_from.as_str(),
            publisher.trusted_until.as_str()
        )));
    }
    if !publisher.allow_system_install {
        return Err(PackageError::Manifest(format!(
            "publisher `{}` is not authorized for system installation",
            publisher.name
        )));
    }
    if !publisher.package_namespaces.iter().any(|namespace| {
        package == namespace
            || package.starts_with(&format!("{namespace}."))
            || package.starts_with(&format!("{namespace}-"))
    }) {
        return Err(PackageError::Manifest(format!(
            "package `{package}` is outside publisher `{}` namespaces",
            publisher.name
        )));
    }
    let domain = https_domain(source)?;
    if !publisher
        .allowed_domains
        .iter()
        .any(|allowed| domain == allowed || domain.ends_with(&format!(".{allowed}")))
    {
        return Err(PackageError::Manifest(format!(
            "source domain `{domain}` is not allowed for publisher `{}`",
            publisher.name
        )));
    }
    Ok(())
}

pub fn https_domain(source: &str) -> Result<&str, PackageError> {
    let remainder = source.strip_prefix("https://").ok_or_else(|| {
        PackageError::Manifest(format!("external source `{source}` must use HTTPS"))
    })?;
    let authority = remainder.split('/').next().unwrap_or_default();
    if authority.is_empty() || authority.contains('@') {
        return Err(PackageError::Manifest(format!(
            "external source `{source}` has an invalid domain"
        )));
    }
    let domain = authority.split(':').next().unwrap_or_default();
    if domain.is_empty()
        || !domain
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'.' | b'-'))
    {
        return Err(PackageError::Manifest(format!(
            "external source `{source}` has an invalid domain"
        )));
    }
    Ok(domain)
}

fn parse_publisher(value: &toml::Value, path: &Path) -> Result<TrustedPublisher, PackageError> {
    let table = value.as_table().ok_or_else(|| {
        PackageError::Manifest(format!(
            "publisher entry in {} must be a table",
            path.display()
        ))
    })?;
    let signing_keys = strings(table, "signing_keys", path)?;
    if signing_keys.is_empty() {
        return Err(PackageError::Manifest(format!(
            "publisher in {} must contain at least one signing key",
            path.display()
        )));
    }
    for key in &signing_keys {
        validate_ed25519_public_key(key)?;
    }
    Ok(TrustedPublisher {
        name: string(table, "name", path)?,
        allowed_domains: strings(table, "allowed_domains", path)?,
        signing_keys,
        package_namespaces: strings(table, "package_namespaces", path)?,
        trusted_from: Date::parse(&string(table, "trusted_from", path)?)?,
        trusted_until: Date::parse(&string(table, "trusted_until", path)?)?,
        allow_system_install: table
            .get("allow_system_install")
            .and_then(toml::Value::as_bool)
            .unwrap_or(false),
    })
}

fn string(table: &toml::Table, key: &str, path: &Path) -> Result<String, PackageError> {
    table
        .get(key)
        .and_then(toml::Value::as_str)
        .map(str::to_owned)
        .ok_or_else(|| {
            PackageError::Manifest(format!(
                "publisher in {} has no string `{key}`",
                path.display()
            ))
        })
}

fn strings(table: &toml::Table, key: &str, path: &Path) -> Result<Vec<String>, PackageError> {
    table
        .get(key)
        .and_then(toml::Value::as_array)
        .ok_or_else(|| {
            PackageError::Manifest(format!(
                "publisher in {} has no `{key}` array",
                path.display()
            ))
        })?
        .iter()
        .map(|value| {
            value.as_str().map(str::to_owned).ok_or_else(|| {
                PackageError::Manifest(format!(
                    "publisher `{key}` in {} must contain strings",
                    path.display()
                ))
            })
        })
        .collect()
}

// Howard Hinnant's civil calendar conversion, with the Unix epoch as day zero.
fn days_from_civil(year: i32, month: u32, day: u32) -> i64 {
    let year = year - i32::from(month <= 2);
    let era = if year >= 0 { year } else { year - 399 } / 400;
    let year_of_era = year - era * 400;
    let adjusted_month = month as i32 + if month > 2 { -3 } else { 9 };
    let day_of_year = (153 * adjusted_month + 2) / 5 + day as i32 - 1;
    let day_of_era = year_of_era * 365 + year_of_era / 4 - year_of_era / 100 + day_of_year;
    (era * 146_097 + day_of_era - 719_468) as i64
}

fn civil_from_days(days: i64) -> (i32, u32, u32) {
    let days = days + 719_468;
    let era = if days >= 0 { days } else { days - 146_096 } / 146_097;
    let day_of_era = days - era * 146_097;
    let year_of_era =
        (day_of_era - day_of_era / 1460 + day_of_era / 36_524 - day_of_era / 146_096) / 365;
    let mut year = year_of_era as i32 + era as i32 * 400;
    let day_of_year = day_of_era - (365 * year_of_era + year_of_era / 4 - year_of_era / 100);
    let month_prime = (5 * day_of_year + 2) / 153;
    let day = day_of_year - (153 * month_prime + 2) / 5 + 1;
    let month = month_prime + if month_prime < 10 { 3 } else { -9 };
    year += i32::from(month <= 2);
    (year, month as u32, day as u32)
}
