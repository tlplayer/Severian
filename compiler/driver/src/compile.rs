use crate::CompileError;
use severian_ast::Module as AstModule;
use severian_hir::Program;
use severian_mlir::Module;
use severian_package::PackageInterface;
use std::collections::HashSet;
use std::path::{Path, PathBuf};

#[derive(Debug, Clone)]
pub struct Compilation {
    pub hir: Program,
    pub optimized_hir: Program,
    pub mir: severian_mir::Program,
    pub mlir: Module,
}
pub fn compile_source(source: &str) -> Result<Compilation, CompileError> {
    let ast = parse_source(source)?;
    compile_ast(&ast, &[], Path::new("<memory>"), source)
}

fn parse_source(source: &str) -> Result<AstModule, CompileError> {
    let tokens = severian_lexer::lex(source).map_err(|error| CompileError::Frontend {
        stage: "lexer",
        span: error.span,
        message: error.message,
    })?;
    let ast = severian_parser::parse(&tokens).map_err(|error| CompileError::Frontend {
        stage: "parser",
        span: error.span,
        message: error.message,
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
    severian_passes::standard_pipeline_with_graph(fusion_rules, fusion_aliases, graph_rules)
        .run(&mut optimized_hir)
        .map_err(|error| CompileError::Optimization(error.to_string()))?;
    let mir = severian_mir::lower(&optimized_hir);
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
                },
            )?;
        severian_ownership::check(&dependency).map_err(|error| {
            CompileError::Ownership(format!("package `{}`: {}", interface.name, error.message))
        })?;
        qualify_package_functions(&mut dependency, &interface.name);
        let mut metadata = std::mem::take(&mut program.metadata);
        severian_semantic::attach_module_metadata_to(
            &interface.module,
            &mut dependency,
            &mut metadata,
            interface.source_path.clone(),
            interface.source.clone(),
            Some(&interface.name),
        );
        program.metadata = metadata;

        for global in dependency.globals {
            if !program
                .globals
                .iter()
                .any(|existing| existing.name == global.name)
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
    let local_names = program
        .functions
        .iter()
        .map(|function| function.name.clone())
        .collect::<HashSet<_>>();
    for function in &mut program.functions {
        function.name = format!("{package}.{}", function.name);
        function.id = severian_hir::FunctionId::from_name(&function.name);
    }
    program.visit_expressions_mut(&mut |expression| match expression {
        severian_hir::Expression::Call { target, .. } if local_names.contains(&target.name) => {
            target.name = format!("{package}.{}", target.name);
            target.id = severian_hir::FunctionId::from_name(&target.name);
        }
        severian_hir::Expression::Function(name) if local_names.contains(name) => {
            *name = format!("{package}.{name}");
        }
        severian_hir::Expression::ChaosRule { function, .. } if local_names.contains(function) => {
            *function = format!("{package}.{function}");
        }
        _ => {}
    });
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
        }
    })?;
    severian_semantic::attach_module_metadata(
        ast,
        &mut hir,
        source_path.to_path_buf(),
        source.to_owned(),
        None,
    );
    severian_ownership::check(&hir).map_err(|error| CompileError::Ownership(error.message))?;
    Ok(hir)
}

pub fn compile_path(path: &Path) -> Result<Compilation, CompileError> {
    let (ast, interfaces, source) = frontend_path(path)?;
    compile_ast(&ast, &interfaces, path, &source)
}

pub fn check_path(path: &Path) -> Result<Program, CompileError> {
    let (ast, interfaces, source) = frontend_path(path)?;
    check_ast(&ast, &interfaces, path, &source)
}

fn frontend_path(path: &Path) -> Result<(AstModule, Vec<PackageInterface>, String), CompileError> {
    let source = std::fs::read_to_string(path)?;
    let Some(manifest_path) = severian_package::find_manifest(path) else {
        let ast = parse_source(&source)?;
        let interfaces = load_official_interfaces(&ast)?;
        return Ok((ast, interfaces, source));
    };
    let mut interfaces = severian_package::load_path_dependency_interfaces(&manifest_path)
        .map_err(|error| CompileError::Package(error.to_string()))?;
    let ast = parse_source(&source)?;
    for interface in load_official_interfaces(&ast)? {
        if !interfaces
            .iter()
            .any(|existing| existing.name == interface.name)
        {
            interfaces.push(interface);
        }
    }
    Ok((ast, interfaces, source))
}

fn load_official_interfaces(module: &AstModule) -> Result<Vec<PackageInterface>, CompileError> {
    let library_root = std::env::var_os("SEVERIAN_LIBRARY_PATH")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../library"));
    severian_package::load_official_interfaces(module, &library_root)
        .map_err(|error| CompileError::Package(error.to_string()))
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

pub fn inspect_toolchain() -> severian_backend::ToolchainReport {
    severian_backend::inspect_toolchain()
}
