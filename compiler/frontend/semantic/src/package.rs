use crate::{analyze_with_package_functions, AnalysisContext, AnalysisMode, PackageFunction};
use severian_ast::{ImportSubject, Item, TypeAnnotation, TypeAnnotationKind};
use severian_diagnostics::Diagnostic;
use severian_hir::{Expression, ExpressionKind, FunctionId, Program, Statement};
use severian_modules::{ModuleGraph, ModuleId, PackageId};
use severian_universal::{DeclarationId, DefId, UniversalContext};
use std::collections::BTreeMap;

mod generic;
#[cfg(test)]
mod tests;

use generic::{collect_generic_specializations, specialize_function, specialize_signature};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Visibility {
    Public,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FunctionDecl {
    pub type_parameters: Vec<String>,
    pub parameters: Vec<TypeAnnotation>,
    pub result: TypeAnnotation,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DefKind {
    Function(FunctionDecl),
    Type,
    Trait,
    Constant,
    Import,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Definition {
    pub id: DefId,
    pub name: String,
    pub module: ModuleId,
    pub visibility: Visibility,
    pub kind: DefKind,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Resolution {
    Def(DefId),
    Module(ModuleId),
    OverloadSet(Vec<DefId>),
    Ambiguous(Vec<DefId>),
}

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct Scope {
    pub bindings: BTreeMap<String, Resolution>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModuleScope {
    pub id: ModuleId,
    pub package: PackageId,
    pub items: Vec<DefId>,
    pub scope: Scope,
}

pub type ExportMap = BTreeMap<String, Resolution>;

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct ProgramIndex {
    pub packages: BTreeMap<PackageId, Vec<ModuleId>>,
    pub modules: BTreeMap<ModuleId, ModuleScope>,
    pub definitions: BTreeMap<DefId, Definition>,
    pub exports: BTreeMap<ModuleId, ExportMap>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TypedProgram {
    pub index: ProgramIndex,
    pub hir: Program,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct PackageAnalysisContext {
    /// Tests are materialized only for modules in this package. `None` is a
    /// normal build of every package in the graph.
    pub test_package: Option<PackageId>,
}

pub fn analyze_package(
    module_graph: &ModuleGraph,
    universal: &UniversalContext,
) -> Result<TypedProgram, Diagnostic> {
    analyze_package_with_context(module_graph, universal, PackageAnalysisContext::default())
}

pub fn analyze_package_with_context(
    module_graph: &ModuleGraph,
    universal: &UniversalContext,
    context: PackageAnalysisContext,
) -> Result<TypedProgram, Diagnostic> {
    let mut index = collect_declarations(module_graph)?;
    resolve_imports(module_graph, &mut index);
    let specializations = collect_generic_specializations(module_graph, &index)?;

    // Lowering retains a stable hash of DefId. It is compact enough for the
    // existing HIR/MIR handle while remaining independent of collection order.
    let function_ids = index
        .definitions
        .iter()
        .filter(|(_, definition)| matches!(definition.kind, DefKind::Function(_)))
        .map(|(id, _)| (*id, stable_function_id(*id)))
        .collect::<BTreeMap<_, _>>();
    let mut next_binding = 0u32;
    let mut hir = Program::default();

    for source_module in &module_graph.modules {
        let mut own_definitions = Vec::new();
        let mut ast = severian_ast::Module::default();
        for item in &source_module.ast.items {
            match item {
                Item::Function(function) if !function.type_parameters.is_empty() => {
                    let id = function_def_id(source_module.package, source_module.id, function);
                    if let Some(substitution) = specializations.get(&id) {
                        own_definitions.push(id);
                        ast.items
                            .push(Item::Function(specialize_function(function, substitution)));
                    }
                }
                Item::Function(function) => {
                    own_definitions.push(function_def_id(
                        source_module.package,
                        source_module.id,
                        function,
                    ));
                    ast.items.push(item.clone());
                }
                _ => ast.items.push(item.clone()),
            }
        }
        let mut visible = imported_function_bindings(source_module.id, &index, &specializations);
        visible.extend(own_definitions.iter().map(|definition| FunctionBinding {
            lookup: index.definitions[definition].name.clone(),
            definition: *definition,
        }));
        let visible = visible
            .into_iter()
            .map(|binding| {
                let definition = &index.definitions[&binding.definition];
                let DefKind::Function(original) = &definition.kind else {
                    unreachable!("only functions enter the callable environment")
                };
                let signature = specializations
                    .get(&binding.definition)
                    .map(|substitution| specialize_signature(original, substitution))
                    .unwrap_or_else(|| original.clone());
                Ok(PackageFunction {
                    lookup: binding.lookup,
                    id: function_ids[&binding.definition],
                    definition: binding.definition,
                    type_parameters: Vec::new(),
                    parameters: signature
                        .parameters
                        .iter()
                        .map(|annotation| {
                            crate::resolve_type_annotation(&universal.types, annotation)
                        })
                        .collect::<Result<Vec<_>, _>>()?,
                    result: crate::resolve_type_annotation(&universal.types, &signature.result)?,
                })
            })
            .collect::<Result<Vec<_>, Diagnostic>>()?;

        let mode = if context.test_package == Some(source_module.package) {
            AnalysisMode::Test
        } else {
            AnalysisMode::Build
        };
        let module_name = module_name(&source_module.path);
        let own_function_ids = own_definitions
            .iter()
            .map(|definition| function_ids[definition])
            .collect::<Vec<_>>();
        let test_function_ids = ast
            .items
            .iter()
            .filter(|item| matches!(item, Item::Test(_)))
            .enumerate()
            .map(|(ordinal, _)| {
                FunctionId(stable_hash(&format!(
                    "test:{:032x}:{ordinal}",
                    source_module.id.0
                )))
            })
            .collect::<Vec<_>>();
        let mut analyzed = analyze_with_package_functions(
            &ast,
            &universal.types,
            AnalysisContext {
                mode,
                module_name: &module_name,
            },
            &visible,
            &own_function_ids,
            &test_function_ids,
        )?
        .modules
        .pop()
        .expect("single-module analysis returns one HIR module");

        remap_module_bindings(&mut analyzed, next_binding);
        next_binding = analyzed
            .bindings
            .iter()
            .map(|binding| binding.id.0)
            .chain(
                analyzed
                    .functions
                    .iter()
                    .flat_map(|function| function.parameters.iter())
                    .map(|parameter| parameter.binding.0),
            )
            .max()
            .map_or(next_binding, |id| id + 1);
        hir.modules.push(analyzed);
    }

    Ok(TypedProgram { index, hir })
}

#[derive(Debug)]
struct FunctionBinding {
    lookup: String,
    definition: DefId,
}

fn imported_function_bindings(
    module: ModuleId,
    index: &ProgramIndex,
    specializations: &BTreeMap<DefId, BTreeMap<String, String>>,
) -> Vec<FunctionBinding> {
    let mut stubs = Vec::new();
    let scope = &index.modules[&module].scope;
    for (name, resolution) in &scope.bindings {
        match resolution {
            Resolution::Module(target) => {
                if let Some(exports) = index.exports.get(target) {
                    for (export, resolution) in exports {
                        for definition in resolution_definitions(resolution) {
                            if emit_function(definition, index, specializations) {
                                stubs.push(FunctionBinding {
                                    lookup: format!("{name}.{export}"),
                                    definition,
                                });
                            }
                        }
                    }
                }
            }
            resolution => {
                for definition in resolution_definitions(resolution) {
                    if definition.module != module.0
                        && emit_function(definition, index, specializations)
                    {
                        stubs.push(FunctionBinding {
                            lookup: name.clone(),
                            definition,
                        });
                    }
                }
            }
        }
    }
    stubs.sort_by_key(|stub| (stub.lookup.clone(), stub.definition));
    stubs.dedup_by_key(|stub| (stub.lookup.clone(), stub.definition));
    stubs
}

fn emit_function(
    definition: DefId,
    index: &ProgramIndex,
    specializations: &BTreeMap<DefId, BTreeMap<String, String>>,
) -> bool {
    match &index.definitions[&definition].kind {
        DefKind::Function(function) => {
            function.type_parameters.is_empty() || specializations.contains_key(&definition)
        }
        _ => false,
    }
}

fn resolution_definitions(resolution: &Resolution) -> Vec<DefId> {
    match resolution {
        Resolution::Def(id) => vec![*id],
        Resolution::OverloadSet(ids) | Resolution::Ambiguous(ids) => ids.clone(),
        Resolution::Module(_) => Vec::new(),
    }
}

fn collect_declarations(module_graph: &ModuleGraph) -> Result<ProgramIndex, Diagnostic> {
    let mut index = ProgramIndex::default();
    for module in &module_graph.modules {
        index
            .packages
            .entry(module.package)
            .or_default()
            .push(module.id);
        let mut scope = Scope::default();
        let mut exports = ExportMap::new();
        let mut items = Vec::new();
        for item in &module.ast.items {
            if let Item::Import(import) = item {
                let subject = match &import.subject {
                    ImportSubject::Name(name) | ImportSubject::Locator(name) => name,
                };
                let name = import.alias.clone().unwrap_or_else(|| subject.clone());
                let key = format!(
                    "import:{subject}:{}:{name}",
                    import.source.as_deref().unwrap_or("")
                );
                let id = DefId {
                    package: u128::from(module.package.0),
                    module: module.id.0,
                    declaration: DeclarationId(stable_hash(&key)),
                };
                if index.definitions.contains_key(&id) {
                    return Err(Diagnostic::new(
                        "E000203",
                        format!("import `{name}` is declared more than once"),
                        Some(import.span),
                    ));
                }
                index.definitions.insert(
                    id,
                    Definition {
                        id,
                        name,
                        module: module.id,
                        visibility: Visibility::Public,
                        kind: DefKind::Import,
                    },
                );
                items.push(id);
                continue;
            }
            let (name, kind, id) = match item {
                Item::Function(function) => {
                    let id = function_def_id(module.package, module.id, function);
                    (
                        function.name.clone(),
                        DefKind::Function(FunctionDecl {
                            type_parameters: function.type_parameters.clone(),
                            parameters: function
                                .parameters
                                .iter()
                                .map(|parameter| parameter.annotation.clone())
                                .collect(),
                            result: function.result.clone(),
                        }),
                        id,
                    )
                }
                Item::Type(declaration) => item_identity(
                    module.package,
                    module.id,
                    "type",
                    &declaration.name,
                    DefKind::Type,
                ),
                Item::Trait(declaration) => item_identity(
                    module.package,
                    module.id,
                    "trait",
                    &declaration.name,
                    DefKind::Trait,
                ),
                Item::Binding(binding) if !binding.update => item_identity(
                    module.package,
                    module.id,
                    "constant",
                    &binding.name,
                    DefKind::Constant,
                ),
                Item::Import(_) => unreachable!("imports are collected above"),
                _ => continue,
            };
            if let Some(existing) = index.definitions.get(&id) {
                return Err(Diagnostic::new(
                    "E000203",
                    format!(
                        "declaration `{name}` has the same canonical identity as `{}`",
                        existing.name
                    ),
                    None,
                ));
            }
            let definition = Definition {
                id,
                name: name.clone(),
                module: module.id,
                visibility: Visibility::Public,
                kind,
            };
            index.definitions.insert(id, definition);
            items.push(id);
            insert_binding(
                &mut scope.bindings,
                name.clone(),
                Resolution::Def(id),
                &index.definitions,
            );
            insert_binding(&mut exports, name, Resolution::Def(id), &index.definitions);
        }
        index.modules.insert(
            module.id,
            ModuleScope {
                id: module.id,
                package: module.package,
                items,
                scope,
            },
        );
        index.exports.insert(module.id, exports);
    }
    Ok(index)
}

fn resolve_imports(module_graph: &ModuleGraph, index: &mut ProgramIndex) {
    for module in &module_graph.modules {
        for import in module.ast.items.iter().filter_map(|item| match item {
            Item::Import(import) => Some(import),
            _ => None,
        }) {
            let Some(edge) = module.imports.iter().find(|edge| edge.span == import.span) else {
                continue;
            };
            let (name, resolution) = if import.source.is_some() {
                let imported_name = match &import.subject {
                    ImportSubject::Name(name) | ImportSubject::Locator(name) => name,
                };
                let resolution = index
                    .exports
                    .get(&edge.module)
                    .and_then(|exports| exports.get(imported_name))
                    .cloned()
                    .unwrap_or_else(|| Resolution::Ambiguous(Vec::new()));
                (
                    import
                        .alias
                        .clone()
                        .unwrap_or_else(|| imported_name.clone()),
                    resolution,
                )
            } else {
                let default = match &import.subject {
                    ImportSubject::Name(name) => name.clone(),
                    ImportSubject::Locator(locator) => std::path::Path::new(locator)
                        .file_stem()
                        .and_then(|name| name.to_str())
                        .unwrap_or(locator)
                        .to_owned(),
                };
                (
                    import.alias.clone().unwrap_or(default),
                    Resolution::Module(edge.module),
                )
            };
            insert_binding(
                &mut index
                    .modules
                    .get_mut(&module.id)
                    .expect("every graph module has a scope")
                    .scope
                    .bindings,
                name,
                resolution,
                &index.definitions,
            );
        }
    }
}

fn insert_binding(
    bindings: &mut BTreeMap<String, Resolution>,
    name: String,
    new: Resolution,
    definitions: &BTreeMap<DefId, Definition>,
) {
    let Some(old) = bindings.remove(&name) else {
        bindings.insert(name, new);
        return;
    };
    let mut ids = resolution_definitions(&old);
    ids.extend(resolution_definitions(&new));
    ids.sort();
    ids.dedup();
    let only_functions = !ids.is_empty()
        && ids.iter().all(|id| {
            definitions
                .get(id)
                .is_some_and(|definition| matches!(definition.kind, DefKind::Function(_)))
        });
    bindings.insert(
        name,
        if only_functions {
            Resolution::OverloadSet(ids)
        } else {
            Resolution::Ambiguous(ids)
        },
    );
}

fn item_identity(
    package: PackageId,
    module: ModuleId,
    tag: &str,
    name: &str,
    kind: DefKind,
) -> (String, DefKind, DefId) {
    (
        name.to_owned(),
        kind,
        DefId {
            package: u128::from(package.0),
            module: module.0,
            declaration: DeclarationId(stable_hash(&format!("{tag}:{name}"))),
        },
    )
}

fn function_def_id(
    package: PackageId,
    module: ModuleId,
    function: &severian_ast::FunctionDeclaration,
) -> DefId {
    let parameters = function
        .parameters
        .iter()
        .map(|parameter| type_key(&parameter.annotation))
        .collect::<Vec<_>>()
        .join(",");
    let generics = function.type_parameters.join(",");
    let key = format!(
        "function:{}[{generics}]({parameters})->{}",
        function.name,
        type_key(&function.result)
    );
    DefId {
        package: u128::from(package.0),
        module: module.0,
        declaration: DeclarationId(stable_hash(&key)),
    }
}

fn type_key(annotation: &TypeAnnotation) -> String {
    match &annotation.kind {
        TypeAnnotationKind::Named { name, arguments } if arguments.is_empty() => name.clone(),
        TypeAnnotationKind::Named { name, arguments } => format!(
            "{name}[{}]",
            arguments.iter().map(type_key).collect::<Vec<_>>().join(",")
        ),
        TypeAnnotationKind::Union(types) => {
            format!(
                "({})",
                types.iter().map(type_key).collect::<Vec<_>>().join("|")
            )
        }
    }
}

fn stable_hash(value: &str) -> u128 {
    const OFFSET: u128 = 0x6c62_272e_07bb_0142_62b8_2175_6295_c58d;
    const PRIME: u128 = 0x0000_0000_0100_0000_0000_0000_0000_013b;
    value.as_bytes().iter().fold(OFFSET, |hash, byte| {
        (hash ^ u128::from(*byte)).wrapping_mul(PRIME)
    })
}

fn stable_function_id(definition: DefId) -> FunctionId {
    FunctionId(stable_hash(&format!(
        "function:{}:{:032x}:{:032x}",
        definition.package, definition.module, definition.declaration.0
    )))
}

fn module_name(path: &std::path::Path) -> String {
    path.file_stem()
        .and_then(|name| name.to_str())
        .unwrap_or("module")
        .chars()
        .map(|character| {
            if character.is_ascii_alphanumeric() {
                character
            } else {
                '_'
            }
        })
        .collect()
}

fn remap_module_bindings(module: &mut severian_hir::Module, offset: u32) {
    if offset == 0 {
        return;
    }
    for binding in &mut module.bindings {
        binding.id.0 += offset;
        remap_expression_bindings(&mut binding.value, offset);
    }
    remap_block_bindings(&mut module.initializer, offset);
    for function in &mut module.functions {
        for parameter in &mut function.parameters {
            parameter.binding.0 += offset;
        }
        if let Some(body) = &mut function.body {
            remap_block_bindings(body, offset);
        }
    }
}

fn remap_block_bindings(block: &mut severian_hir::Block, offset: u32) {
    for statement in &mut block.statements {
        match statement {
            Statement::Binding(binding) => binding.0 += offset,
            Statement::Expression(expression) | Statement::Return(Some(expression)) => {
                remap_expression_bindings(expression, offset)
            }
            Statement::Return(None) => {}
            Statement::Assert {
                condition, message, ..
            } => {
                remap_expression_bindings(condition, offset);
                if let Some(message) = message {
                    remap_expression_bindings(message, offset);
                }
            }
            Statement::If {
                condition,
                then_block,
                else_block,
            } => {
                remap_expression_bindings(condition, offset);
                remap_block_bindings(then_block, offset);
                remap_block_bindings(else_block, offset);
            }
            Statement::Match { subject, arms } => {
                remap_expression_bindings(subject, offset);
                for arm in arms {
                    if let Some(binding) = &mut arm.binding {
                        binding.0 += offset;
                    }
                    remap_block_bindings(&mut arm.body, offset);
                }
            }
        }
    }
}

fn remap_expression_bindings(expression: &mut Expression, offset: u32) {
    match &mut expression.kind {
        ExpressionKind::Binding(binding) => binding.0 += offset,
        ExpressionKind::Convert { operand, .. } => {
            remap_expression_bindings(operand, offset);
        }
        ExpressionKind::Call { arguments, .. } => {
            for argument in arguments {
                remap_expression_bindings(argument, offset);
            }
        }
        ExpressionKind::Unary { operand, .. } => remap_expression_bindings(operand, offset),
        ExpressionKind::Binary { left, right, .. } => {
            remap_expression_bindings(left, offset);
            remap_expression_bindings(right, offset);
        }
        ExpressionKind::Literal(_) | ExpressionKind::Function(_) => {}
    }
}
