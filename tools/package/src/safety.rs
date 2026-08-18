use super::*;

/// Enforces capability-scoped unsafe exceptions from `package.toml`. Unsafe is
/// denied by default. Native ABI declarations remain library-only; selected
/// examples may opt individual source files into language capabilities such as
/// `pointers` or `runtime-owned-tasks`. Tests are rejected by the parser even
/// when their package has an exception.
pub fn enforce_unsafe_policy(
    manifest_path: Option<&Path>,
    source_path: &Path,
    source: &str,
) -> Result<(), PackageError> {
    let tokens = severian_lexer::lex(source).map_err(|error| PackageError::Frontend {
        package: package_label(manifest_path),
        stage: "lexer",
        span: error.span,
        message: error.message,
        source_path: Some(source_path.to_path_buf()),
        source: Some(source.to_owned()),
    })?;
    let Some(token) = tokens
        .iter()
        .find(|token| token.kind == severian_lexer::TokenKind::Unsafe)
    else {
        return Ok(());
    };
    let Some(manifest_path) = manifest_path else {
        return Err(with_frontend_source(
            unsafe_policy_error("source without package.toml", token.span, "source-file"),
            source_path,
            source,
        ));
    };
    let manifest = parse_manifest(manifest_path)?;
    enforce_manifest_unsafe_policy(
        &manifest,
        manifest_path,
        source_path,
        &tokens,
        token.span,
        false,
    )
    .map_err(|error| with_frontend_source(error, source_path, source))
}

fn package_label(manifest_path: Option<&Path>) -> String {
    manifest_path
        .and_then(|path| parse_manifest(path).ok().map(|manifest| (path, manifest)))
        .and_then(|(path, manifest)| package_name(&manifest, path).ok().map(str::to_owned))
        .unwrap_or_else(|| "application".into())
}

pub(super) fn enforce_manifest_unsafe_policy(
    manifest: &toml::Value,
    manifest_path: &Path,
    source_path: &Path,
    tokens: &[severian_lexer::Token],
    span: Span,
    interface_library: bool,
) -> Result<(), PackageError> {
    let package = package_name(manifest, manifest_path).unwrap_or("workspace");
    let configuration = manifest
        .get("package")
        .and_then(toml::Value::as_table)
        .and_then(|package| package.get("unsafe"))
        .and_then(toml::Value::as_table);
    let capabilities = unsafe_string_array(configuration, "capabilities", manifest_path)?;
    let sources = unsafe_string_array(configuration, "sources", manifest_path)?;
    let requested = unsafe_capabilities(tokens);
    let relative_source = manifest_relative_source(manifest_path, source_path);
    let source_allowed = sources
        .iter()
        .any(|source| Path::new(source) == relative_source);
    let is_library = interface_library
        || source_is_library(manifest, manifest_path, source_path)
        || (source_allowed && manifest.get("lib").is_some());
    if requested.contains("native-abi") && !is_library {
        return Err(unsafe_policy_error(
            &format!("non-library target `{}`", relative_source.display()),
            span,
            "native-abi",
        ));
    }
    let missing = requested
        .iter()
        .find(|capability| !capabilities.iter().any(|allowed| allowed == *capability));
    if source_allowed && missing.is_none() {
        return Ok(());
    }
    let capability = missing.copied().unwrap_or("source-file");
    Err(unsafe_policy_error(
        &format!("package `{package}` source `{}`", relative_source.display()),
        span,
        capability,
    ))
}

fn unsafe_string_array(
    configuration: Option<&toml::Table>,
    key: &str,
    manifest_path: &Path,
) -> Result<Vec<String>, PackageError> {
    let Some(value) = configuration.and_then(|configuration| configuration.get(key)) else {
        return Ok(Vec::new());
    };
    value
        .as_array()
        .ok_or_else(|| {
            PackageError::Manifest(format!(
                "{}.package.unsafe.{key} must be an array of strings",
                manifest_path.display()
            ))
        })?
        .iter()
        .map(|value| {
            value.as_str().map(str::to_owned).ok_or_else(|| {
                PackageError::Manifest(format!(
                    "{}.package.unsafe.{key} must contain only strings",
                    manifest_path.display()
                ))
            })
        })
        .collect()
}

fn unsafe_capabilities(tokens: &[severian_lexer::Token]) -> BTreeSet<&'static str> {
    use severian_lexer::TokenKind;
    let mut capabilities = BTreeSet::new();
    if tokens.iter().any(|token| token.kind == TokenKind::Extern) {
        capabilities.insert("native-abi");
    }
    if tokens
        .iter()
        .any(|token| token.kind == TokenKind::Ampersand)
    {
        capabilities.insert("pointers");
    }
    if tokens.iter().any(|token| token.kind == TokenKind::Async)
        && tokens
            .iter()
            .any(|token| matches!(&token.kind, TokenKind::Identifier(name) if name == "runtime"))
    {
        capabilities.insert("runtime-owned-tasks");
    }
    if capabilities.is_empty() {
        capabilities.insert("unsafe-blocks");
    }
    capabilities
}

fn manifest_relative_source(manifest_path: &Path, source_path: &Path) -> PathBuf {
    if let (Some(root), Ok(source)) = (manifest_path.parent(), source_path.canonicalize()) {
        if let Ok(root) = root.canonicalize() {
            if let Ok(relative) = source.strip_prefix(root) {
                return relative.to_path_buf();
            }
        }
    }
    manifest_path
        .parent()
        .and_then(|root| source_path.strip_prefix(root).ok())
        .unwrap_or(source_path)
        .to_path_buf()
}

fn source_is_library(manifest: &toml::Value, manifest_path: &Path, source_path: &Path) -> bool {
    let Some(root) = manifest_path.parent() else {
        return false;
    };
    let configured = root.join(library_path(manifest));
    match (configured.canonicalize(), source_path.canonicalize()) {
        (Ok(configured), Ok(source)) => configured == source,
        _ => configured == source_path,
    }
}

fn unsafe_policy_error(reason: &str, span: Span, capability: &str) -> PackageError {
    PackageError::Frontend {
        package: reason.into(),
        stage: "unsafe policy",
        span,
        message: format!(
            "E000701: unsafe capability `{capability}` is not allowed; add it and this source path to `[package.unsafe]`, while native ABI remains library-only and tests remain safe-only"
        ),
        source_path: None,
        source: None,
    }
}

pub(super) fn with_frontend_source(error: PackageError, path: &Path, source: &str) -> PackageError {
    match error {
        PackageError::Frontend {
            package,
            stage,
            span,
            message,
            ..
        } => PackageError::Frontend {
            package,
            stage,
            span,
            message,
            source_path: Some(path.to_path_buf()),
            source: Some(source.to_owned()),
        },
        error => error,
    }
}
