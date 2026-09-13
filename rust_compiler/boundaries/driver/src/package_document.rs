//! Package JSON5 boundary. TOML remains readable during migration.
use std::path::{Path, PathBuf};

pub fn parse(text: &str) -> Result<toml::Value, String> {
    let trimmed = text.trim_start();
    if trimmed.starts_with('{') || trimmed.starts_with("//") || trimmed.starts_with("/*") {
        json5::from_str(text).map_err(|error| format!("invalid JSON5: {error}"))
    } else {
        text.parse()
            .map_err(|error| format!("invalid TOML: {error}"))
    }
}

pub fn render(value: &toml::Value) -> Result<String, String> {
    serde_json::to_string_pretty(value)
        .map(|text| format!("{text}\n"))
        .map_err(|error| format!("cannot render package JSON: {error}"))
}

pub fn path(root: &Path) -> PathBuf {
    let current = root.join("package.json");
    if current.is_file() || !root.join("package.toml").is_file() {
        current
    } else {
        root.join("package.toml")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn comments_strings_trailing_commas_and_legacy_documents() {
        let document = parse("/* manifest */ {package: {name: 'demo', version: '0.1.0',}, bin: [{name:'demo',path:'main.sev'}]}").unwrap();
        assert_eq!(document["package"]["name"].as_str(), Some("demo"));
        assert_eq!(parse(&render(&document).unwrap()).unwrap(), document);
        assert_eq!(
            parse("[package]\nname='demo'").unwrap()["package"]["name"].as_str(),
            Some("demo")
        );
        assert!(parse("{package: {name: }}").is_err());
    }
}
