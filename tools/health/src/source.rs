use crate::repository;
use std::collections::BTreeSet;
use std::fs;
use std::path::{Path, PathBuf};

const MAX_SOURCE_FILE_BYTES: u64 = 16 * 1024 * 1024;
const MAX_REPOSITORY_SOURCE_BYTES: u64 = 256 * 1024 * 1024;

#[derive(Clone)]
pub struct SourceUnit {
    pub path: PathBuf,
    pub text: String,
    pub lines: usize,
}

#[derive(Clone)]
pub struct FunctionBody {
    pub path: PathBuf,
    pub line: usize,
    pub name: String,
    pub source: String,
}

#[derive(Clone)]
pub struct EnumCatalog {
    pub path: PathBuf,
    pub line: usize,
    pub name: String,
    pub variants: BTreeSet<String>,
}

pub fn load(root: &Path) -> Result<Vec<SourceUnit>, String> {
    let mut units = Vec::new();
    let mut total = 0_u64;
    for path in repository::rust_sources(root)? {
        let absolute = root.join(&path);
        let bytes = fs::metadata(&absolute)
            .map_err(|error| format!("could not inspect {}: {error}", path.display()))?
            .len();
        if bytes > MAX_SOURCE_FILE_BYTES {
            return Err(format!(
                "refusing to load {}: {bytes} bytes exceeds the 16 MiB source limit",
                path.display()
            ));
        }
        total = total.saturating_add(bytes);
        if total > MAX_REPOSITORY_SOURCE_BYTES {
            return Err(
                "refusing to load Rust sources: repository input exceeds the 256 MiB limit".into(),
            );
        }
        let text = fs::read_to_string(&absolute)
            .map_err(|error| format!("could not read {}: {error}", path.display()))?;
        units.push(SourceUnit {
            lines: text.lines().count(),
            path,
            text,
        });
    }
    Ok(units)
}

pub fn extract_functions(unit: &SourceUnit) -> Vec<FunctionBody> {
    let lines = unit.text.lines().collect::<Vec<_>>();
    let mut output = Vec::new();
    let mut index = 0;
    while index < lines.len() {
        let trimmed = lines[index].trim_start();
        let Some(offset) = trimmed.find("fn ") else {
            index += 1;
            continue;
        };
        if offset > 12 || trimmed.starts_with("//") {
            index += 1;
            continue;
        }
        let name = trimmed[offset + 3..]
            .split(|character: char| !character.is_ascii_alphanumeric() && character != '_')
            .next()
            .unwrap_or("anonymous")
            .to_string();
        let start = index;
        let mut source = String::new();
        let mut depth = 0_i32;
        let mut entered = false;
        while index < lines.len() {
            source.push_str(lines[index]);
            source.push('\n');
            for character in lines[index].chars() {
                if character == '{' {
                    depth += 1;
                    entered = true;
                } else if character == '}' {
                    depth -= 1;
                }
            }
            index += 1;
            if entered && depth == 0 {
                break;
            }
            if !entered && source.ends_with(";\n") {
                break;
            }
        }
        if entered {
            output.push(FunctionBody {
                path: unit.path.clone(),
                line: start + 1,
                name,
                source,
            });
        }
    }
    output
}

pub fn extract_enums(unit: &SourceUnit) -> Vec<EnumCatalog> {
    let lines = unit.text.lines().collect::<Vec<_>>();
    let mut output = Vec::new();
    let mut index = 0;
    while index < lines.len() {
        let trimmed = lines[index].trim_start();
        let Some(suffix) = trimmed.strip_prefix("pub enum ") else {
            index += 1;
            continue;
        };
        let name = suffix
            .split(|character: char| character.is_whitespace() || character == '{')
            .next()
            .unwrap_or_default()
            .to_string();
        let start = index;
        let mut depth = 0_i32;
        let mut entered = false;
        let mut variants = BTreeSet::new();
        while index < lines.len() {
            let candidate = lines[index].trim_start();
            if entered
                && depth == 1
                && candidate
                    .chars()
                    .next()
                    .is_some_and(|character| character.is_ascii_uppercase())
            {
                let variant = candidate
                    .split(|character: char| {
                        character.is_whitespace() || matches!(character, '(' | '{' | ',' | '=')
                    })
                    .next()
                    .unwrap_or_default();
                if !variant.is_empty() {
                    variants.insert(variant.into());
                }
            }
            for character in lines[index].chars() {
                if character == '{' {
                    depth += 1;
                    entered = true;
                } else if character == '}' {
                    depth -= 1;
                }
            }
            index += 1;
            if entered && depth == 0 {
                break;
            }
        }
        output.push(EnumCatalog {
            path: unit.path.clone(),
            line: start + 1,
            name,
            variants,
        });
    }
    output
}

pub fn normalize_exact(source: &str) -> String {
    source
        .lines()
        .map(|line| line.split("//").next().unwrap_or(line))
        .flat_map(str::chars)
        .filter(|character| !character.is_whitespace())
        .collect()
}

pub fn normalize_renamed(source: &str) -> String {
    let mut output = String::new();
    let mut token = String::new();
    for character in source.chars().chain([' ']) {
        if character.is_ascii_alphanumeric() || character == '_' {
            token.push(character);
            continue;
        }
        if !token.is_empty() {
            if is_rust_keyword(&token) || token.chars().all(|character| character.is_ascii_digit())
            {
                output.push_str(&token);
            } else {
                output.push('_');
            }
            token.clear();
        }
        if !character.is_whitespace() {
            output.push(character);
        }
    }
    output
}

fn is_rust_keyword(token: &str) -> bool {
    matches!(
        token,
        "fn" | "if"
            | "else"
            | "match"
            | "for"
            | "while"
            | "loop"
            | "return"
            | "let"
            | "mut"
            | "pub"
            | "impl"
            | "Self"
            | "self"
            | "Some"
            | "None"
            | "Ok"
            | "Err"
            | "true"
            | "false"
    )
}

pub fn maximum_nesting(source: &str) -> usize {
    let mut depth = 0_usize;
    let mut maximum = 0;
    for character in source.chars() {
        if character == '{' {
            depth += 1;
            maximum = maximum.max(depth);
        } else if character == '}' {
            depth = depth.saturating_sub(1);
        }
    }
    maximum
}

pub fn call_names(source: &str) -> BTreeSet<String> {
    let mut output = BTreeSet::new();
    let bytes = source.as_bytes();
    let mut start = 0;
    while start < bytes.len() {
        if bytes[start].is_ascii_alphabetic() || bytes[start] == b'_' {
            let mut end = start + 1;
            while end < bytes.len() && (bytes[end].is_ascii_alphanumeric() || bytes[end] == b'_') {
                end += 1;
            }
            let mut next = end;
            while next < bytes.len() && bytes[next].is_ascii_whitespace() {
                next += 1;
            }
            if next < bytes.len() && bytes[next] == b'(' {
                output.insert(source[start..end].to_string());
            }
            start = end;
        } else {
            start += 1;
        }
    }
    output
}

pub fn contains_code(line: &str, needle: &str) -> bool {
    line.split("//").next().unwrap_or(line).contains(needle)
}

pub fn normalized_line(line: &str) -> String {
    line.split("//")
        .next()
        .unwrap_or(line)
        .split_whitespace()
        .collect()
}
