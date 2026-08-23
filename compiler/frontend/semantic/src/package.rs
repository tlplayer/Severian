use crate::{
    analyze_with_package_functions, AnalysisContext, AnalysisMode, PackageClass, PackageFunction,
    PackageList,
};
use severian_ast::{GenericConstraint, ImportSubject, Item, TypeAnnotation, TypeAnnotationKind};
use severian_diagnostics::Diagnostic;
use severian_hir::{Expression, ExpressionKind, FunctionId, Program, Statement};
use severian_modules::{ModuleGraph, ModuleId, PackageId};
use severian_universal::{DeclarationId, DefId, TypeId, UniversalContext};
use std::collections::BTreeMap;

mod generic;
#[cfg(test)]
mod tests;

use generic::{
    collect_generic_specializations, specialize_function, specialize_signature,
    validate_generic_bodies, Specializations, Substitution as GenericSubstitution,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Visibility {
    Public,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct SignatureId(pub u128);

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FunctionDecl {
    pub signature: SignatureId,
    pub type_parameters: Vec<String>,
    pub parameters: Vec<TypeAnnotation>,
    pub result: TypeAnnotation,
    pub constraints: Vec<GenericConstraint>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TraitDecl {
    pub type_parameters: Vec<String>,
    pub constraints: Vec<GenericConstraint>,
    pub bases: Vec<TypeAnnotation>,
    pub properties: Vec<severian_ast::PropertyDeclaration>,
    pub methods: Vec<severian_ast::FunctionDeclaration>,
    pub operators: Vec<severian_ast::OperatorDeclaration>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DefKind {
    Function(FunctionDecl),
    Type,
    Trait(TraitDecl),
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

impl ProgramIndex {
    pub fn function_definition(
        &self,
        module: ModuleId,
        name: &str,
        overload_ordinal: usize,
    ) -> Option<DefId> {
        let scope = self.modules.get(&module)?;
        let id = DefId {
            package: u128::from(scope.package.0),
            module: module.0,
            declaration: DeclarationId(stable_hash(&format!("function:{name}:{overload_ordinal}"))),
        };
        self.definitions.contains_key(&id).then_some(id)
    }
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
    validate_generic_bodies(module_graph, &index, &universal.types)?;
    let specializations = collect_generic_specializations(module_graph, &index, &universal.types)?;
    let package_classes = collect_package_classes(module_graph);
    let package_lists = collect_package_lists(
        module_graph,
        &universal.types,
        u32::try_from(package_classes.len()).unwrap_or(u32::MAX),
    );

    let mut next_binding = 0u32;
    let mut hir = Program::default();

    for source_module in &module_graph.modules {
        let mut own_instances = Vec::new();
        let mut ast = severian_ast::Module::default();
        for item in &source_module.ast.items {
            match item {
                Item::Function(function) if !function.type_parameters.is_empty() => {
                    let id = function_def_id(
                        source_module.package,
                        source_module.id,
                        &source_module.ast,
                        function,
                    );
                    if let Some(instances) = specializations.get(&id) {
                        for substitution in instances {
                            own_instances.push((id, substitution.clone()));
                            ast.items
                                .push(Item::Function(specialize_function(function, substitution)));
                        }
                    }
                }
                Item::Function(function) => {
                    own_instances.push((
                        function_def_id(
                            source_module.package,
                            source_module.id,
                            &source_module.ast,
                            function,
                        ),
                        GenericSubstitution::default(),
                    ));
                    ast.items.push(item.clone());
                }
                _ => ast.items.push(item.clone()),
            }
        }
        let mut visible = imported_function_bindings(source_module.id, &index, &specializations);
        visible.extend(
            own_instances
                .iter()
                .map(|(definition, substitution)| FunctionBinding {
                    lookup: index.definitions[definition].name.clone(),
                    definition: *definition,
                    substitution: substitution.clone(),
                }),
        );
        let visible = visible
            .into_iter()
            .map(|binding| {
                let definition = &index.definitions[&binding.definition];
                let DefKind::Function(original) = &definition.kind else {
                    unreachable!("only functions enter the callable environment")
                };
                let signature = if binding.substitution.is_empty() {
                    original.clone()
                } else {
                    specialize_signature(original, &binding.substitution)
                };
                let substitution =
                    universal_substitution(original, &binding.substitution, &universal.types)?;
                Ok(PackageFunction {
                    lookup: binding.lookup,
                    id: stable_instance_function_id(binding.definition, &binding.substitution),
                    definition: binding.definition,
                    substitution,
                    type_parameters: Vec::new(),
                    parameters: signature
                        .parameters
                        .iter()
                        .map(|annotation| {
                            resolve_package_type(
                                &universal.types,
                                annotation,
                                definition.module,
                                &package_classes,
                                &package_lists,
                            )
                        })
                        .collect::<Result<Vec<_>, _>>()?,
                    result: resolve_package_type(
                        &universal.types,
                        &signature.result,
                        definition.module,
                        &package_classes,
                        &package_lists,
                    )?,
                    specificity: if original.type_parameters.is_empty() {
                        0
                    } else if original.constraints.is_empty() {
                        2
                    } else {
                        1
                    },
                })
            })
            .collect::<Result<Vec<_>, Diagnostic>>()?;

        let mode = if context.test_package == Some(source_module.package) {
            AnalysisMode::Test
        } else {
            AnalysisMode::Build
        };
        let module_name = module_name(&source_module.path);
        let own_function_ids = own_instances
            .iter()
            .map(|(definition, substitution)| {
                stable_instance_function_id(*definition, substitution)
            })
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
            &package_classes,
            &package_lists,
            Some(source_module.id),
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

fn collect_package_classes(module_graph: &ModuleGraph) -> Vec<PackageClass> {
    module_graph
        .modules
        .iter()
        .flat_map(|module| {
            module.ast.items.iter().filter_map(move |item| match item {
                Item::Class(declaration) => Some((module.id, declaration.clone())),
                Item::Enum(declaration) => {
                    let mut fields = vec![severian_ast::PropertyDeclaration {
                        name: "__tag".into(),
                        annotation: TypeAnnotation::named("int", Vec::new(), declaration.span),
                        default: None,
                        span: declaration.span,
                    }];
                    for variant in &declaration.variants {
                        for field in &variant.fields {
                            if !fields.iter().any(|known| known.name == field.name) {
                                fields.push(field.clone());
                            }
                        }
                    }
                    Some((
                        module.id,
                        severian_ast::ClassDeclaration {
                            decorators: Vec::new(),
                            name: declaration.name.clone(),
                            type_parameters: Vec::new(),
                            constraints: Vec::new(),
                            traits: Vec::new(),
                            fields,
                            constructors: Vec::new(),
                            methods: Vec::new(),
                            span: declaration.span,
                        },
                    ))
                }
                _ => None,
            })
        })
        .enumerate()
        .map(|(ordinal, (module, declaration))| PackageClass {
            module,
            ty: TypeId(u32::MAX.saturating_sub(ordinal as u32)),
            declaration,
        })
        .collect()
}

fn resolve_package_type(
    types: &severian_universal::TypeContext,
    annotation: &TypeAnnotation,
    module: ModuleId,
    classes: &[PackageClass],
    lists: &[PackageList],
) -> Result<TypeId, Diagnostic> {
    if let Some(("list", [element])) = annotation.named_parts() {
        let element = crate::resolve_type_annotation(types, element)?;
        if let Some(list) = lists.iter().find(|list| list.element == element) {
            return Ok(list.ty);
        }
    }
    if let Some(name) = annotation.simple_name() {
        if let Some(class) = classes
            .iter()
            .find(|class| class.module == module && class.declaration.name == name)
        {
            return Ok(class.ty);
        }
    }
    crate::resolve_type_annotation(types, annotation)
}

fn collect_package_lists(
    module_graph: &ModuleGraph,
    types: &severian_universal::TypeContext,
    class_count: u32,
) -> Vec<PackageList> {
    let mut uses = BTreeMap::<TypeId, ModuleId>::new();
    for module in &module_graph.modules {
        for function in module.ast.items.iter().filter_map(|item| match item {
            Item::Function(function) => Some(function),
            _ => None,
        }) {
            for annotation in function
                .parameters
                .iter()
                .map(|parameter| &parameter.annotation)
                .chain(std::iter::once(&function.result))
            {
                collect_list_elements(annotation, types, module.id, &mut uses);
            }
        }
    }
    uses.into_iter()
        .enumerate()
        .map(|(ordinal, (element, module))| PackageList {
            module,
            ty: TypeId(
                u32::MAX
                    .saturating_sub(class_count)
                    .saturating_sub(ordinal as u32),
            ),
            element,
        })
        .collect()
}

fn collect_list_elements(
    annotation: &TypeAnnotation,
    types: &severian_universal::TypeContext,
    module: ModuleId,
    output: &mut BTreeMap<TypeId, ModuleId>,
) {
    let Some((name, arguments)) = annotation.named_parts() else {
        return;
    };
    if name == "list" && arguments.len() == 1 {
        if let Ok(element) = crate::resolve_type_annotation(types, &arguments[0]) {
            output.entry(element).or_insert(module);
        }
    }
    for argument in arguments {
        collect_list_elements(argument, types, module, output);
    }
}

#[derive(Debug)]
struct FunctionBinding {
    lookup: String,
    definition: DefId,
    substitution: GenericSubstitution,
}

fn imported_function_bindings(
    module: ModuleId,
    index: &ProgramIndex,
    specializations: &Specializations,
) -> Vec<FunctionBinding> {
    let mut stubs = Vec::new();
    let scope = &index.modules[&module].scope;
    for (name, resolution) in &scope.bindings {
        match resolution {
            Resolution::Module(target) => {
                if let Some(exports) = index.exports.get(target) {
                    for (export, resolution) in exports {
                        for definition in resolution_definitions(resolution) {
                            for substitution in
                                function_instances(definition, index, specializations)
                            {
                                stubs.push(FunctionBinding {
                                    lookup: format!("{name}.{export}"),
                                    definition,
                                    substitution,
                                });
                            }
                        }
                    }
                }
            }
            resolution => {
                for definition in resolution_definitions(resolution) {
                    if definition.module != module.0 {
                        for substitution in function_instances(definition, index, specializations) {
                            stubs.push(FunctionBinding {
                                lookup: name.clone(),
                                definition,
                                substitution,
                            });
                        }
                    }
                }
            }
        }
    }
    stubs.sort_by_key(|stub| {
        (
            stub.lookup.clone(),
            stub.definition,
            stub.substitution.clone(),
        )
    });
    stubs.dedup_by_key(|stub| {
        (
            stub.lookup.clone(),
            stub.definition,
            stub.substitution.clone(),
        )
    });
    stubs
}

fn function_instances(
    definition: DefId,
    index: &ProgramIndex,
    specializations: &Specializations,
) -> Vec<GenericSubstitution> {
    match &index.definitions[&definition].kind {
        DefKind::Function(function) if function.type_parameters.is_empty() => {
            vec![GenericSubstitution::default()]
        }
        DefKind::Function(_) => specializations
            .get(&definition)
            .map(|instances| instances.iter().cloned().collect())
            .unwrap_or_default(),
        _ => Vec::new(),
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
                    let id = function_def_id(module.package, module.id, &module.ast, function);
                    (
                        function.name.clone(),
                        DefKind::Function(FunctionDecl {
                            signature: function_signature_id(function),
                            type_parameters: function.type_parameters.clone(),
                            parameters: function
                                .parameters
                                .iter()
                                .map(|parameter| parameter.annotation.clone())
                                .collect(),
                            result: function.result.clone(),
                            constraints: function.constraints.clone(),
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
                    DefKind::Trait(TraitDecl {
                        type_parameters: declaration.type_parameters.clone(),
                        constraints: declaration.constraints.clone(),
                        bases: declaration.bases.clone(),
                        properties: declaration.properties.clone(),
                        methods: declaration.methods.clone(),
                        operators: declaration.operators.clone(),
                    }),
                ),
                Item::Class(declaration) => item_identity(
                    module.package,
                    module.id,
                    "class",
                    &declaration.name,
                    DefKind::Type,
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
    ast: &severian_ast::Module,
    function: &severian_ast::FunctionDeclaration,
) -> DefId {
    let overload_ordinal = ast
        .items
        .iter()
        .filter_map(|item| match item {
            Item::Function(candidate)
                if candidate.name == function.name
                    && candidate.span.start < function.span.start =>
            {
                Some(())
            }
            _ => None,
        })
        .count();
    let key = format!("function:{}:{overload_ordinal}", function.name);
    DefId {
        package: u128::from(package.0),
        module: module.0,
        declaration: DeclarationId(stable_hash(&key)),
    }
}

fn function_signature_id(function: &severian_ast::FunctionDeclaration) -> SignatureId {
    let parameters = function
        .parameters
        .iter()
        .map(|parameter| type_key(&parameter.annotation))
        .collect::<Vec<_>>()
        .join(",");
    let generics = function.type_parameters.join(",");
    let constraints = function
        .constraints
        .iter()
        .map(constraint_key)
        .collect::<Vec<_>>()
        .join(",");
    SignatureId(stable_hash(&format!(
        "function:{}[{generics}]({parameters})->{} with [{constraints}]",
        function.name,
        type_key(&function.result)
    )))
}

fn constraint_key(constraint: &GenericConstraint) -> String {
    match constraint {
        GenericConstraint::Parameter {
            parameter, bound, ..
        } => format!("{parameter}:{}", type_key(bound)),
        GenericConstraint::Predicate(expression) => format!("predicate:{:?}", expression.kind),
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

fn stable_instance_function_id(
    definition: DefId,
    substitution: &GenericSubstitution,
) -> FunctionId {
    let arguments = substitution
        .iter()
        .map(|(parameter, ty)| format!("{parameter}={ty}"))
        .collect::<Vec<_>>()
        .join(",");
    FunctionId(stable_hash(&format!(
        "function:{}:{:032x}:{:032x}[{arguments}]",
        definition.package, definition.module, definition.declaration.0,
    )))
}

fn universal_substitution(
    function: &FunctionDecl,
    substitution: &GenericSubstitution,
    types: &severian_universal::TypeContext,
) -> Result<severian_universal::Substitution, Diagnostic> {
    let arguments = function
        .type_parameters
        .iter()
        .enumerate()
        .filter_map(|(index, parameter)| {
            substitution
                .get(parameter)
                .map(|name| (severian_universal::GenericParamId(index as u32), name))
        })
        .map(|(parameter, name)| {
            types
                .resolve_name(name)
                .map(|ty| (parameter, ty))
                .ok_or_else(|| Diagnostic::new("E000204", format!("unknown type `{name}`"), None))
        })
        .collect::<Result<Vec<_>, _>>()?;
    Ok(severian_universal::Substitution::new(arguments))
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
            Statement::Sequence(block) => remap_block_bindings(block, offset),
            Statement::Binding(binding) => binding.0 += offset,
            Statement::FieldUpdate { binding, value, .. } => {
                binding.0 += offset;
                remap_expression_bindings(value, offset);
            }
            Statement::FieldSet { binding, value, .. } => {
                binding.0 += offset;
                remap_expression_bindings(value, offset);
            }
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
        ExpressionKind::Aggregate { fields, .. } => {
            for field in fields {
                remap_expression_bindings(field, offset);
            }
        }
        ExpressionKind::Field { object, .. } => remap_expression_bindings(object, offset),
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
