use crate::CompileError;
use severian_ast::Module as AstModule;
use severian_hir::Program;
use severian_mlir::Module;
use severian_package::PackageInterface;
use std::collections::HashSet;
use std::path::{Path, PathBuf};

include!(concat!(env!("OUT_DIR"), "/official_libraries.rs"));

#[derive(Debug, Clone)]
pub struct Compilation {
    pub hir: Program,
    pub optimized_hir: Program,
    pub mir: severian_mir::Program,
    pub mlir: Module,
}

impl Drop for Compilation {
    fn drop(&mut self) {
        // Linked standard-library HIR contains deeply nested expression trees.
        // Rust test workers and embedding hosts commonly use a 2 MiB stack,
        // which is too small for their recursive derived drop glue. Move the
        // recursive payloads to the same bounded compiler stack used to build
        // them, while leaving the flat emitted module to drop normally.
        let hir = std::mem::take(&mut self.hir);
        let optimized_hir = std::mem::take(&mut self.optimized_hir);
        let mir = std::mem::take(&mut self.mir);
        std::thread::Builder::new()
            .name("severian-compiler-drop".into())
            .stack_size(16 * 1024 * 1024)
            .spawn(move || drop((hir, optimized_hir, mir)))
            .expect("creating the compiler cleanup thread must succeed")
            .join()
            .unwrap_or_else(|panic| std::panic::resume_unwind(panic));
    }
}
pub fn compile_source(source: &str) -> Result<Compilation, CompileError> {
    let ast = parse_source(source, Path::new("<memory>"))?;
    compile_ast(&ast, &[], Path::new("<memory>"), source)
}

fn parse_source(source: &str, source_path: &Path) -> Result<AstModule, CompileError> {
    let tokens = severian_lexer::lex(source).map_err(|error| CompileError::Frontend {
        stage: "lexer",
        span: error.span,
        message: error.message,
        source_path: source_path.to_path_buf(),
        source: source.to_owned(),
    })?;
    let ast = severian_parser::parse(&tokens).map_err(|error| CompileError::Frontend {
        stage: "parser",
        span: error.span,
        message: error.message,
        source_path: source_path.to_path_buf(),
        source: source.to_owned(),
    })?;
    Ok(ast)
}

fn compile_ast(
    ast: &AstModule,
    interfaces: &[PackageInterface],
    source_path: &Path,
    source: &str,
) -> Result<Compilation, CompileError> {
    let hir = check_ast(ast, interfaces, source_path, source)?;
    let mut optimized_hir = hir.clone();
    link_package_hir(&mut optimized_hir, interfaces)?;
    let fusion_rules = interfaces
        .iter()
        .flat_map(|interface| interface.compiler.fusion_rules.iter().cloned());
    let fusion_aliases = interfaces
        .iter()
        .flat_map(|interface| interface.compiler.fusion_aliases.iter().cloned());
    let graph_rules = interfaces
        .iter()
        .flat_map(|interface| interface.compiler.graph_rules.iter().cloned());
    verify_hir(&optimized_hir, "linked HIR")?;
    severian_passes::standard_pipeline_with_graph(fusion_rules, fusion_aliases, graph_rules)
        .run_verified(&mut optimized_hir, |program, pass| {
            verify_hir(program, &format!("HIR after `{pass}`")).map_err(|error| error.to_string())
        })
        .map_err(|error| CompileError::Optimization(error.to_string()))?;
    let mir = severian_mir::lower(&optimized_hir);
    verify_mir(&mir)?;
    let mlir = severian_lowering::lower(&mir);

    Ok(Compilation {
        hir,
        optimized_hir,
        mir,
        mlir,
    })
}

fn link_package_hir(
    program: &mut Program,
    interfaces: &[PackageInterface],
) -> Result<(), CompileError> {
    for interface in interfaces {
        let mut dependency =
            severian_semantic::analyze_with_packages(&interface.module, interfaces).map_err(
                |error| CompileError::Frontend {
                    stage: "semantic",
                    span: error.span,
                    message: format!("package `{}`: {}", interface.name, error.message),
                    source_path: interface.source_path.clone(),
                    source: interface.source.clone(),
                },
            )?;
        severian_ownership::check(&dependency).map_err(|error| {
            ownership_compile_error(
                format!("package `{}`: {}", interface.name, error.message),
                &interface.source_path,
                &interface.source,
            )
        })?;
        qualify_package_functions(&mut dependency, &interface.name);
        let mut metadata = std::mem::take(&mut program.metadata);
        severian_semantic::attach_module_metadata_to_with_packages(
            &interface.module,
            &mut dependency,
            &mut metadata,
            interface.source_path.clone(),
            interface.source.clone(),
            Some(&interface.name),
            interfaces,
        );
        program.metadata = metadata;

        for global in dependency.globals {
            if !program
                .globals
                .iter()
                .any(|existing| existing.name.id == global.name.id)
            {
                program.globals.push(global);
            }
        }
        for class in dependency.classes {
            if !program
                .classes
                .iter()
                .any(|existing| existing.id == class.id)
            {
                program.classes.push(class);
            }
        }
        for function in dependency.functions {
            if !program
                .functions
                .iter()
                .any(|existing| existing.id == function.id)
            {
                program.functions.push(function);
            }
        }
    }
    Ok(())
}

fn qualify_package_functions(program: &mut Program, package: &str) {
    program.namespace_bindings(package);
    let local_names = program
        .functions
        .iter()
        .map(|function| function.name.clone())
        .collect::<HashSet<_>>();
    for function in &mut program.functions {
        function.name = format!("{package}.{}", function.name);
        function.id = severian_hir::FunctionId::from_name(&function.name);
    }
    for class in &mut program.classes {
        class.id = severian_hir::TypeDefinitionId::from_name(&format!("{package}.{}", class.name));
        for function in &mut class.constructors {
            function.id = function.id.in_namespace(package);
        }
        for function in &mut class.methods {
            function.id = severian_hir::FunctionId::from_name(&format!(
                "{package}.{}.{}",
                class.name, function.name
            ));
        }
    }
    program.visit_expressions_mut(&mut |expression| match expression {
        severian_hir::Expression::Call { target, .. }
            if local_names.contains(&target.name) && !is_intrinsic_call(&target.name) =>
        {
            target.name = format!("{package}.{}", target.name);
            target.id = severian_hir::FunctionId::from_name(&target.name);
        }
        severian_hir::Expression::Function(target) if local_names.contains(&target.name) => {
            target.name = format!("{package}.{}", target.name);
            target.id = severian_hir::FunctionId::from_name(&target.name);
        }
        severian_hir::Expression::ChaosRule { function, .. }
            if local_names.contains(&function.name) =>
        {
            function.name = format!("{package}.{}", function.name);
            function.id = severian_hir::FunctionId::from_name(&function.name);
        }
        _ => {}
    });
}

fn is_intrinsic_call(name: &str) -> bool {
    matches!(
        name,
        "print"
            | "panic"
            | "int"
            | "float"
            | "string"
            | "range"
            | "indices"
            | "enumerate"
            | "zip"
            | "any"
            | "all"
            | "abs"
            | "min"
            | "max"
            | "divmod"
            | "len"
            | "size"
            | "bytes"
            | "bits"
            | "capacity"
    )
}

fn check_ast(
    ast: &AstModule,
    interfaces: &[PackageInterface],
    source_path: &Path,
    source: &str,
) -> Result<Program, CompileError> {
    let mut hir = severian_semantic::analyze_with_packages(ast, interfaces).map_err(|error| {
        CompileError::Frontend {
            stage: "semantic",
            span: error.span,
            message: error.message,
            source_path: source_path.to_path_buf(),
            source: source.to_owned(),
        }
    })?;
    severian_semantic::attach_module_metadata_with_packages(
        ast,
        &mut hir,
        source_path.to_path_buf(),
        source.to_owned(),
        None,
        interfaces,
    );
    verify_hir(&hir, "resolved HIR")?;
    severian_ownership::check(&hir)
        .map_err(|error| ownership_compile_error(error.message, source_path, source))?;
    Ok(hir)
}

fn verify_hir(program: &Program, boundary: &str) -> Result<(), CompileError> {
    let diagnostics = severian_diagnostics::verify::verify(program);
    let errors = diagnostics
        .diagnostics()
        .iter()
        .filter(|diagnostic| diagnostic.severity >= severian_diagnostics::Severity::Error)
        .map(|diagnostic| format!("{}: {}", diagnostic.code.0, diagnostic.message))
        .collect::<Vec<_>>();
    if errors.is_empty() {
        Ok(())
    } else {
        Err(CompileError::Verification(format!(
            "{boundary}: {}",
            errors.join("; ")
        )))
    }
}

fn verify_mir(program: &severian_mir::Program) -> Result<(), CompileError> {
    severian_mir::verify(program).map_err(|errors| {
        CompileError::Verification(format!(
            "MIR: {}",
            errors
                .iter()
                .map(ToString::to_string)
                .collect::<Vec<_>>()
                .join("; ")
        ))
    })
}

fn ownership_compile_error(message: String, source_path: &Path, source: &str) -> CompileError {
    let name = message.split('`').nth(1);
    let start = name
        .and_then(|name| {
            if message.contains("cannot be mutated") {
                source.rfind(&format!("{name}."))
            } else {
                identifier_occurrences(source, name).last()
            }
        })
        .unwrap_or(0);
    let end = start + name.map_or(1, str::len);
    CompileError::Frontend {
        stage: "ownership",
        span: severian_ast::Span::new(start, end),
        message,
        source_path: source_path.to_path_buf(),
        source: source.to_owned(),
    }
}

fn identifier_occurrences<'a>(source: &'a str, name: &'a str) -> impl Iterator<Item = usize> + 'a {
    source.match_indices(name).filter_map(move |(index, _)| {
        let before = source[..index].chars().next_back();
        let after = source[index + name.len()..].chars().next();
        let boundary = |character: Option<char>| {
            character.is_none_or(|character| !(character == '_' || character.is_alphanumeric()))
        };
        (boundary(before) && boundary(after)).then_some(index)
    })
}

pub fn compile_path(path: &Path) -> Result<Compilation, CompileError> {
    let path = path.to_path_buf();
    std::thread::Builder::new()
        .name("severian-compiler".into())
        .stack_size(16 * 1024 * 1024)
        .spawn(move || {
            let (ast, interfaces, source) = frontend_path(&path, true, None)?;
            compile_ast(&ast, &interfaces, &path, &source)
        })
        .map_err(CompileError::Io)?
        .join()
        .unwrap_or_else(|panic| std::panic::resume_unwind(panic))
}

pub fn compile_dependency_path(
    path: &Path,
    manifest_path: &Path,
) -> Result<Compilation, CompileError> {
    let (ast, interfaces, source) = frontend_path(path, false, Some(manifest_path))?;
    compile_ast(&ast, &interfaces, path, &source)
}

pub fn check_path(path: &Path) -> Result<Program, CompileError> {
    let (ast, interfaces, source) = frontend_path(path, true, None)?;
    check_ast(&ast, &interfaces, path, &source)
}

fn frontend_path(
    path: &Path,
    write_lock: bool,
    manifest_override: Option<&Path>,
) -> Result<(AstModule, Vec<PackageInterface>, String), CompileError> {
    let source = std::fs::read_to_string(path)?;
    let ast = parse_source(&source, path)?;
    let manifest_path = manifest_override
        .map(Path::to_path_buf)
        .or_else(|| severian_package::find_manifest(path));
    severian_package::enforce_unsafe_policy(manifest_path.as_deref(), path, &source)
        .map_err(|error| CompileError::Package(error.to_string()))?;
    severian_package::enforce_type_safe_policy(manifest_path.as_deref(), path, &ast, &source)
        .map_err(|error| CompileError::Package(error.to_string()))?;
    let project_root = manifest_path
        .as_deref()
        .and_then(Path::parent)
        .or_else(|| path.parent())
        .unwrap_or_else(|| Path::new("."));
    let project_root = if project_root.as_os_str().is_empty() {
        Path::new(".")
    } else {
        project_root
    };
    let mut interfaces = if let Some(manifest_path) = &manifest_path {
        let loaded = if write_lock {
            severian_package::load_path_dependency_interfaces(manifest_path)
        } else {
            severian_package::load_transient_dependency_interfaces(manifest_path)
        };
        loaded.map_err(|error| CompileError::Package(error.to_string()))?
    } else {
        Vec::new()
    };
    let dependency_names = interfaces
        .iter()
        .map(|interface| interface.name.clone())
        .collect::<HashSet<_>>();
    let local_interfaces = severian_package::load_local_interfaces(&ast, project_root)
        .map_err(|error| CompileError::Package(error.to_string()))?;
    for interface in &local_interfaces {
        severian_package::enforce_type_safe_policy(
            manifest_path.as_deref(),
            &interface.source_path,
            &interface.module,
            &interface.source,
        )
        .map_err(|error| {
            CompileError::Package(format!("{}: {error}", interface.source_path.display()))
        })?;
        for official in load_official_interfaces(&interface.module)? {
            insert_official_interface(&mut interfaces, official, &dependency_names)?;
        }
    }
    for interface in local_interfaces {
        insert_interface(&mut interfaces, interface)?;
    }
    for interface in load_official_interfaces(&ast)? {
        insert_official_interface(&mut interfaces, interface, &dependency_names)?;
    }
    Ok((ast, interfaces, source))
}

fn insert_official_interface(
    interfaces: &mut Vec<PackageInterface>,
    interface: PackageInterface,
    dependency_names: &HashSet<String>,
) -> Result<(), CompileError> {
    if let Some(existing) = interfaces
        .iter()
        .find(|existing| existing.name == interface.name)
    {
        let same_source = existing.source_path == interface.source_path
            || matches!(
                (
                    existing.source_path.canonicalize(),
                    interface.source_path.canonicalize(),
                ),
                (Ok(existing), Ok(official)) if existing == official
            );
        if dependency_names.contains(&interface.name) && !same_source {
            return Err(CompileError::Package(format!(
                "dependency `{}` cannot shadow the reserved Severian standard-library package at {}",
                interface.name,
                interface.source_path.display()
            )));
        }
        return Ok(());
    }
    insert_interface(interfaces, interface)
}

fn insert_interface(
    interfaces: &mut Vec<PackageInterface>,
    interface: PackageInterface,
) -> Result<(), CompileError> {
    if let Some(existing) = interfaces
        .iter()
        .find(|existing| existing.name == interface.name)
    {
        let same_source = existing.source_path == interface.source_path
            || matches!(
                (
                    existing.source_path.canonicalize(),
                    interface.source_path.canonicalize(),
                ),
                (Ok(existing), Ok(incoming)) if existing == incoming
            );
        if !same_source {
            return Err(CompileError::Package(format!(
                "module `{}` resolves to both {} and {}",
                interface.name,
                existing.source_path.display(),
                interface.source_path.display()
            )));
        }
    } else {
        interfaces.push(interface);
    }
    Ok(())
}

fn load_official_interfaces(module: &AstModule) -> Result<Vec<PackageInterface>, CompileError> {
    if let Some(library_root) = std::env::var_os("SEVERIAN_LIBRARY_PATH").map(PathBuf::from) {
        return severian_package::load_official_interfaces(module, &library_root)
            .map_err(|error| CompileError::Package(error.to_string()));
    }

    if let Some(library_root) = installed_library_root() {
        return severian_package::load_official_interfaces(module, &library_root)
            .map_err(|error| CompileError::Package(error.to_string()));
    }

    let checkout = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../library");
    if checkout.is_dir() {
        severian_package::load_official_interfaces(module, &checkout)
            .map_err(|error| CompileError::Package(error.to_string()))
    } else {
        severian_package::load_embedded_official_interfaces(module, EMBEDDED_OFFICIAL_PACKAGES)
            .map_err(|error| CompileError::Package(error.to_string()))
    }
}

fn installed_library_root() -> Option<PathBuf> {
    let mut candidates = Vec::new();
    if let Some(home) = std::env::var_os("SEVERIAN_HOME").map(PathBuf::from) {
        candidates.push(home.join("lib/severian/2026"));
    }
    if let Ok(executable) = std::env::current_exe() {
        if let Some(binary_directory) = executable.parent() {
            candidates.push(binary_directory.join("../lib/severian/2026"));
        }
    }
    candidates.into_iter().find(|candidate| candidate.is_dir())
}

#[cfg(test)]
mod official_library_tests {
    use super::{parse_source, EMBEDDED_OFFICIAL_PACKAGES};
    use std::path::Path;

    #[test]
    fn embedded_distribution_contains_nested_official_packages() {
        assert!(EMBEDDED_OFFICIAL_PACKAGES
            .iter()
            .any(|package| package.name == "model.speech"));
        let module = parse_source(
            "import model.speech as speech\n",
            Path::new("embedded-consumer.sev"),
        )
        .unwrap();
        let interfaces = severian_package::load_embedded_official_interfaces(
            &module,
            EMBEDDED_OFFICIAL_PACKAGES,
        )
        .unwrap();
        let names = interfaces
            .iter()
            .map(|interface| interface.name.as_str())
            .collect::<Vec<_>>();
        assert!(names.contains(&"model.speech"));
        assert!(names.contains(&"math"));
        assert!(names.contains(&"random"));
    }
}

pub fn compile_native(compilation: &Compilation, output: &Path) -> Result<(), CompileError> {
    let has_xla_regions = compilation.optimized_hir.functions.iter().any(|function| {
        function
            .decorators
            .iter()
            .any(|decorator| decorator.package == "tensor")
    });
    let result = if has_xla_regions {
        let directory = output.parent().unwrap_or_else(|| Path::new("."));
        let runtime = crate::runtime_asset::materialize_xla_runtime(directory)?;
        severian_backend::compile_native_with_xla_runtime(
            &compilation.optimized_hir,
            &compilation.mlir,
            output,
            &runtime,
        )
    } else {
        severian_backend::compile_native(&compilation.optimized_hir, &compilation.mlir, output)
    };
    result.map_err(|error| CompileError::Io(std::io::Error::other(error.to_string())))
}

pub fn compile_native_with_options(
    compilation: &Compilation,
    output: &Path,
    options: &severian_backend::NativeCompileOptions,
) -> Result<(), CompileError> {
    let has_xla_regions = compilation.optimized_hir.functions.iter().any(|function| {
        function
            .decorators
            .iter()
            .any(|decorator| decorator.package == "tensor")
    });
    let result = if has_xla_regions {
        let directory = output.parent().unwrap_or_else(|| Path::new("."));
        let runtime = crate::runtime_asset::materialize_xla_runtime(directory)?;
        severian_backend::compile_native_with_xla_runtime_and_options(
            &compilation.optimized_hir,
            &compilation.mlir,
            output,
            &runtime,
            options,
        )
    } else {
        severian_backend::compile_native_with_options(
            &compilation.optimized_hir,
            &compilation.mlir,
            output,
            options,
        )
    };
    result.map_err(|error| CompileError::Io(std::io::Error::other(error.to_string())))
}

pub fn inspect_toolchain() -> severian_backend::ToolchainReport {
    severian_backend::inspect_toolchain()
}
