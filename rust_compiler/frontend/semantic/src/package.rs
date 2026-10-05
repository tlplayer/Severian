use crate::{
    analyze_with_package_functions, AnalysisContext, AnalysisMode, PackageClass, PackageConstant,
    PackageEnum, PackageFunction, PackageList, PackageCollection,
};
use severian_ast::{GenericConstraint, ImportSubject, Item, TypeAnnotation, TypeAnnotationKind};
use severian_diagnostics::Diagnostic;
use severian_hir::{Expression, ExpressionKind, FunctionId, Program, Statement};
use severian_modules::{ModuleGraph, ModuleId, PackageId};
use severian_universal::{
    DeclarationId, DefId, GenericParamId, GenericParamKind, GenericParameter, OperatorSignature,
    TypeId, TypePattern, UniversalContext,
};
use std::collections::{BTreeMap, BTreeSet};

pub(crate) mod generic;
mod grammar;
pub mod imports;
#[cfg(test)]
mod tests;

use generic::{
    collect_generic_specializations, specialize_function, specialize_signature,
    validate_generic_bodies, Specializations, Substitution as GenericSubstitution,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Visibility {
    Public,
    Explicit,
    File,
}

impl Visibility {
    pub fn for_name(name: &str) -> Self {
        let name = name.rsplit('.').next().unwrap_or(name);
        if name.starts_with("__") { Self::File }
        else if name.starts_with('_') { Self::Explicit }
        else { Self::Public }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct SignatureId(pub u128);

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FunctionDecl {
    pub signature: SignatureId,
    pub type_parameters: Vec<String>,
    pub parameter_names: Vec<String>,
    pub parameters: Vec<TypeAnnotation>,
    pub parameter_variadics: Vec<bool>,
    /// Immutable reference contracts retained by package declaration metadata.
    pub parameter_views: Vec<bool>,
    pub parameter_defaults: Vec<Option<severian_ast::Expression>>,
    pub result: TypeAnnotation,
    pub constraints: Vec<GenericConstraint>,
    /// Source body retained by the package declaration interface so a
    /// downstream package can instantiate a generic definition. `None`
    /// continues to mean a declaration-only/foreign interface.
    pub generic_body: Option<Vec<severian_ast::Statement>>,
}

/// Classifies source generics without turning dimension or shape parameters
/// into ordinary types. Parameter IDs remain declaration-local and stable by
/// source order, matching the IDs used by HIR substitutions.
pub(crate) fn generic_parameters(
    names: &[String],
    constraints: &[GenericConstraint],
) -> Vec<GenericParameter> {
    names
        .iter()
        .enumerate()
        .map(|(index, name)| {
            let variadic = constraints.iter().any(|constraint| {
                matches!(constraint, GenericConstraint::VariadicPack { parameter, .. } if parameter == name)
            });
            let bound_kind = constraints.iter().find_map(|constraint| {
                let GenericConstraint::Parameter {
                    parameter, bound, ..
                } = constraint
                else {
                    return None;
                };
                if parameter != name {
                    return None;
                }
                match bound.simple_name().and_then(|name| name.rsplit('.').next()) {
                    Some("Dim" | "usize") => Some(GenericParamKind::Dimension),
                    Some("Shape") => Some(GenericParamKind::Shape),
                    _ => None,
                }
            });
            GenericParameter {
                id: GenericParamId(index as u32),
                name: name.clone(),
                kind: if variadic {
                    GenericParamKind::Shape
                } else {
                    bound_kind.unwrap_or(GenericParamKind::Type)
                },
                variadic,
            }
        })
        .collect()
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TraitDecl {
    pub type_parameters: Vec<String>,
    pub type_parameter_defaults: Vec<Option<TypeAnnotation>>,
    pub constraints: Vec<GenericConstraint>,
    pub bases: Vec<TypeAnnotation>,
    pub properties: Vec<severian_ast::PropertyDeclaration>,
    pub methods: Vec<severian_ast::FunctionDeclaration>,
    pub sentences: Vec<severian_ast::SentenceDeclaration>,
    pub operators: Vec<severian_ast::OperatorDeclaration>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ClassDecl {
    pub primitive: bool,
    pub type_parameters: Vec<String>,
    pub fields: Vec<severian_ast::PropertyDeclaration>,
    pub constructors: Vec<severian_ast::FunctionDeclaration>,
    pub methods: Vec<severian_ast::FunctionDeclaration>,
    pub sentences: Vec<severian_ast::SentenceDeclaration>,
}

/// A grammar's declaration identity and source contract, collected without
/// constructing its owner or evaluating its construction body.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GrammarRegistration {
    pub label: DefId,
    pub owner: DefId,
    pub declaration: severian_ast::SentenceDeclaration,
    /// Complete lexical owner contract, including generic defaults, receiver
    /// fields, inherited contracts, and owner constraints.
    pub owner_declaration: GrammarOwnerDeclaration,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GrammarOwnerDeclaration {
    Class(severian_ast::ClassDeclaration),
    Trait(severian_ast::TraitDeclaration),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DefKind {
    Function(FunctionDecl),
    Type,
    Class(ClassDecl),
    Trait(TraitDecl),
    Constant,
    Import,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Definition {
    pub id: DefId,
    pub name: String,
    pub module: ModuleId,
    pub span: severian_source::Span,
    pub visibility: Visibility,
    pub kind: DefKind,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MethodDecl {
    pub owner: String,
    pub owner_type_parameters: Vec<String>,
    pub type_parameters: Vec<String>,
    pub parameters: Vec<TypeAnnotation>,
    pub result: TypeAnnotation,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FieldDecl {
    pub owner: String,
    pub owner_type_parameters: Vec<String>,
    pub annotation: TypeAnnotation,
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
    pub import_requirements: Option<imports::ImportRequirements>,
    pub packages: BTreeMap<PackageId, Vec<ModuleId>>,
    pub modules: BTreeMap<ModuleId, ModuleScope>,
    pub definitions: BTreeMap<DefId, Definition>,
    pub type_aliases: BTreeMap<DefId, severian_ast::TypeDeclaration>,
    pub exports: BTreeMap<ModuleId, ExportMap>,
    pub methods: BTreeMap<String, Vec<MethodDecl>>,
    pub fields: BTreeMap<String, Vec<FieldDecl>>,
    /// Dependency-first module order, then owner and grammar source order.
    pub grammars: Vec<GrammarRegistration>,
}

impl ProgramIndex {
    /// Declaration-level registry; querying it never executes grammar bodies.
    pub fn grammar_registry<'a>(
        &'a self,
        result: Option<&'a TypeAnnotation>,
    ) -> impl Iterator<Item = &'a GrammarRegistration> + 'a {
        self.grammars.iter().filter(move |entry| {
            result.is_none_or(|expected| annotation_matches(&entry.declaration.function.result, expected))
        })
    }

    /// Resolve the callable/owner reference carried by the source registry.
    /// This returns declaration AST; it never invokes an executable adapter.
    pub fn grammar_declaration(&self, label: DefId) -> Option<&GrammarRegistration> {
        self.grammars.iter().find(|entry| entry.label == label)
    }

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
    /// Program-local structural applications (including Tensor element and
    /// shape refinements) used by every later lowering stage.
    pub types: severian_universal::TypeContext,
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

/// Resolve declaration identities and import visibility without checking bodies.
/// Refactoring tools need this even while a package is being migrated.
pub fn import_index(module_graph: &ModuleGraph) -> Result<ProgramIndex, Diagnostic> {
    let mut graph = module_graph.clone();
    graph.policies.clear();
    let mut index = collect_declarations(&graph)
        .map_err(|diagnostic| module_graph.contextualize(diagnostic, "import resolution"))?;
    resolve_imports(&graph, &mut index);
    resolve_declaration_aliases(&graph, &mut index)?;
    Ok(index)
}

/// The same demand plan used by compilation, exposed for source tooling.
pub fn import_plan(graph: &ModuleGraph) -> Result<imports::ImportPlan, Diagnostic> {
    let index = collect_declarations(graph)
        .map_err(|diagnostic| graph.contextualize(diagnostic, "import planning"))?;
    Ok(imports::resolve_required_imports(graph, &index, imports::collect_import_requirements(graph)))
}

pub fn analyze_package_with_context(
    module_graph: &ModuleGraph,
    universal: &UniversalContext,
    context: PackageAnalysisContext,
) -> Result<TypedProgram, Diagnostic> {
    analyze_package_impl(module_graph, universal, context)
        .map_err(|diagnostic| module_graph.contextualize(diagnostic, "semantic analysis"))
}

fn analyze_package_impl(
    module_graph: &ModuleGraph,
    universal: &UniversalContext,
    context: PackageAnalysisContext,
) -> Result<TypedProgram, Diagnostic> {
    severian_modules::validate_import_policy(module_graph)?;
    let lowered_module_graph = lower_extensions(module_graph)?;
    let lowered_module_graph = grammar::lower_registrations(&lowered_module_graph)?;
    let lowered_module_graph = lower_trait_typed_parameters(&lowered_module_graph);
    let module_graph = &lowered_module_graph;
    let mut types = universal.types.clone();
    let mut index = collect_declarations(module_graph)?;
    let plan = imports::resolve_required_imports(module_graph, &index, imports::collect_import_requirements(module_graph));
    imports::apply_import_plan(&mut index, &plan);
    resolve_declaration_aliases(module_graph, &mut index)?;
    for module in index.modules.values() {
        for (name, resolution) in &module.scope.bindings {
            if let Resolution::Ambiguous(ids) = resolution {
                let declarations: Vec<_> = ids.iter().filter_map(|id| index.definitions.get(id)).collect();
                let mut diagnostic = Diagnostic::new("E000203",
                    format!("name `{name}` has conflicting declarations in the same scope"),
                    declarations.last().map(|definition| definition.span))
                    .with_help(format!("import `{name}` from one provider, or use distinct `as` aliases"));
                for declaration in declarations {
                    diagnostic = diagnostic.with_label(declaration.span, format!("conflicting declaration of `{}`", declaration.name));
                }
                return Err(diagnostic);
            }
        }
    }
    if std::env::var("SEVERIAN_PROFILE_ACTIVE").as_deref() == Ok("1") {
        eprintln!("  Imports: {} name requests, {} imported bindings, {} export entries", plan.requested_names, plan.imported_bindings, index.exports.values().map(|e|e.len()).sum::<usize>());
    }
    let package_classes = collect_package_classes(module_graph, &index, &mut types)?;
    let package_enums = collect_package_enums(module_graph, &package_classes);
    install_primitive_class_operators(&mut types, &package_classes, &index)?;
    validate_generic_bodies(module_graph, &index, &types)?;
    let specializations = collect_generic_specializations(module_graph, &index, &types)?;
    let package_lists = collect_package_lists(module_graph, &types);
    let package_trait_names = index
        .modules
        .keys()
        .copied()
        .map(|module| (module, visible_trait_names(module, &index)))
        .collect::<BTreeMap<_, _>>();
    let registry_traits = module_graph
        .modules
        .iter()
        .flat_map(|module| &module.ast.items)
        .filter_map(|item| match item {
            Item::Trait(declaration)
                if !declaration.namespaces.is_empty()
                    || declaration
                        .methods
                        .iter()
                        .any(|method| !method.decorators.is_empty()) =>
            {
                Some(declaration.name.clone())
            }
            _ => None,
        })
        .collect::<BTreeSet<_>>();
    let registry_modules = module_graph
        .modules
        .iter()
        .filter(|module| {
            module.ast.items.iter().any(|item| {
                matches!(item, Item::Class(class) if class.traits.iter().any(|implemented| {
                    implemented
                        .simple_name()
                        .is_some_and(|name| registry_traits.contains(name))
                }))
            })
        })
        .map(|module| module.id)
        .collect::<BTreeSet<_>>();
    // Trait namespaces are extension registries. Their declarations and
    // implementations intentionally cross module/package boundaries, while
    // ordinary declarations remain scoped through the package index.
    let registry_ast = severian_ast::Module {
        items: module_graph
            .modules
            .iter()
            .flat_map(|module| module.ast.items.iter())
            .filter(|item| matches!(item, Item::Trait(_) | Item::Class(_)))
            .cloned()
            .collect(),
    };
    crate::validate_trait_implementations(&registry_ast)?;

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
                        let mut retained = function.clone();
                        if let DefKind::Function(interface) = &index.definitions[&id].kind {
                            retained.body.clone_from(&interface.generic_body);
                            for (parameter, default) in retained
                                .parameters
                                .iter_mut()
                                .zip(&interface.parameter_defaults)
                            {
                                parameter.default.clone_from(default);
                            }
                        }
                        for substitution in instances.keys() {
                            if std::env::var_os("SEVERIAN_TRACE_SPECIALIZATIONS").is_some() {
                                eprintln!("  Specialization {}::{} {:?}", source_module.path.display(), function.name, substitution.bindings());
                            }
                            own_instances.push((id, substitution.clone()));
                            ast.items
                                .push(Item::Function(specialize_function(&retained, substitution)));
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
        if context.test_package == Some(source_module.package) {
            ast.items.extend(
                source_module
                    .ast
                    .items
                    .iter()
                    .filter_map(|item| match item {
                        Item::Class(class) => Some(class.tests.iter()),
                        _ => None,
                    })
                    .flatten()
                    .cloned()
                    .map(Item::Test),
            );
        }
        let mut visible = imported_function_bindings(source_module.id, &index, &specializations);
        let local_callables = visible
            .iter()
            .map(|binding| binding.lookup.clone())
            .chain(own_instances.iter().map(|(definition, _)| {
                index.definitions[definition].name.clone()
            }))
            .collect::<BTreeSet<_>>();
        // Class methods retain the lexical module in which they were declared,
        // even when a downstream package is the first caller that makes the
        // method body reachable. Install those origin-module callables before
        // analyzing the downstream body; otherwise `codec.write()` can retain
        // its source body while losing bindings such as `audio.write_wav`.
        let lexical_class_modules = class_lexical_modules(source_module.id, &package_classes);
        for origin in &lexical_class_modules {
            if *origin != source_module.id {
                // Origin helpers supplement the caller's scope; they must not
                // turn an existing local or imported callable into an overload
                // of an unrelated declaration from another module.
                visible.extend(
                    module_function_bindings(*origin, &index, &specializations)
                        .into_iter()
                        .filter(|binding| !local_callables.contains(&binding.lookup)),
                );
            }
        }
        visible.extend(registry_function_bindings(
            &registry_modules,
            &index,
            &specializations,
        ));
        visible.sort_by_key(|binding| {
            (
                binding.lookup.clone(),
                binding.definition,
                binding.substitution.clone(),
            )
        });
        visible.dedup_by(|left, right| {
            left.lookup == right.lookup
                && left.definition == right.definition
                && left.substitution == right.substitution
        });
        let mut package_constants =
            imported_constant_bindings(source_module.id, module_graph, &index);
        for origin in lexical_class_modules {
            if origin != source_module.id {
                package_constants.extend(module_constant_bindings(origin, module_graph, &index));
            }
        }
        order_package_constants(&mut package_constants, module_graph);
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
                let substitution = universal_substitution(
                    &definition.name,
                    original,
                    &binding.substitution,
                    &mut types,
                    definition.module,
                    &package_classes,
                    &package_lists,
                    &index,
                )?;
                let mut callable_types = Vec::new();
                let mut callable_unions = Vec::new();
                let mut collections = Vec::new();
                for annotation in signature.parameters.iter().chain(std::iter::once(&signature.result)) {
                    collect_signature_layouts(&mut types, annotation, definition.module, &package_classes, &package_lists, &index, &mut callable_types, &mut callable_unions, &mut collections)?;
                }
                Ok(PackageFunction {
                    collections,
                    callable_types,
                    callable_unions,
                    lookup: binding.lookup,
                    id: stable_instance_function_id(binding.definition, &binding.substitution),
                    definition: binding.definition,
                    substitution,
                    generic_parameters: generic_parameters(
                        &original.type_parameters,
                        &original.constraints,
                    ),
                    type_parameters: Vec::new(),
                    parameter_names: signature.parameter_names.clone(),
                    parameter_variadics: signature.parameter_variadics.clone(),
                    parameters: signature
                        .parameters
                        .iter()
                        .map(|annotation| {
                            resolve_package_type(
                                &mut types,
                                annotation,
                                definition.module,
                                &package_classes,
                                &package_lists,
                                &index,
                            )
                        })
                        .collect::<Result<Vec<_>, _>>()?,
                    parameter_defaults: signature.parameter_defaults.clone(),
                    parameter_unions: signature
                        .parameters
                        .iter()
                        .map(|annotation| {
                            resolve_package_union_members(
                                &mut types,
                                annotation,
                                definition.module,
                                &package_classes,
                                &package_lists,
                                &index,
                            )
                        })
                        .collect::<Result<Vec<_>, _>>()?,
                    result: resolve_package_type(
                        &mut types,
                        &signature.result,
                        definition.module,
                        &package_classes,
                        &package_lists,
                        &index,
                    )?,
                    result_union: resolve_package_union_members(
                        &mut types,
                        &signature.result,
                        definition.module,
                        &package_classes,
                        &package_lists,
                        &index,
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
            &mut types,
            AnalysisContext {
                mode,
                module_name: &module_name,
            },
            &visible,
            &own_function_ids,
            &test_function_ids,
            &package_classes,
            &package_enums,
            Some(&index),
            &package_trait_names,
            &package_lists,
            &package_constants,
            Some(source_module.id),
            Some(&registry_ast),
            &index.modules[&source_module.id].scope.bindings.iter()
                .filter_map(|(name,resolution)|matches!(resolution,Resolution::Module(_)).then_some(name.clone()))
                .collect(),
        )?
        .modules
        .pop()
        .expect("single-module analysis returns one HIR module");

        remap_module_bindings(&mut analyzed, next_binding);
        let mut scoped_bindings = Vec::new();
        collect_scoped_binding_ids(&analyzed.initializer, &mut scoped_bindings);
        for function in &analyzed.functions {
            if let Some(body) = &function.body {
                collect_scoped_binding_ids(body, &mut scoped_bindings);
            }
        }
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
            .chain(scoped_bindings)
            .max()
            .map_or(next_binding, |id| id + 1);
        hir.modules.push(analyzed);
    }

    Ok(TypedProgram { index, hir, types })
}

fn lower_extensions(module_graph: &ModuleGraph) -> Result<ModuleGraph, Diagnostic> {
    let mut lowered = module_graph.clone();
    let imported_extensions = lowered
        .modules
        .iter()
        .flat_map(|module| {
            module.ast.items.iter().filter_map(move |item| {
                let Item::Extension(extension) = item else {
                    return None;
                };
                let (target, _) = extension.target.named_parts()?;
                let defined_here = module
                    .ast
                    .items
                    .iter()
                    .any(|item| matches!(item, Item::Class(class) if class.name == target));
                (!defined_here && target != "set" && extension.decorators.is_empty())
                    .then(|| (module.id, extension.clone(), target.to_owned()))
            })
        })
        .collect::<Vec<_>>();
    for (_, extension, target) in &imported_extensions {
        let Some(class) = lowered.modules.iter_mut().find_map(|module| {
            module.ast.items.iter_mut().find_map(|item| match item {
                Item::Class(class) if class.name == *target => Some(class),
                _ => None,
            })
        }) else {
            return Err(Diagnostic::new(
                "E000204",
                format!("cannot extend unknown type `{target}`"),
                Some(extension.target.span),
            ));
        };
        for method in &extension.methods {
            if class.fields.iter().any(|known| known.name == method.name)
                || class.methods.iter().any(|known| known.name == method.name)
                || class
                    .constructors
                    .iter()
                    .any(|known| known.name == method.name)
            {
                return Err(Diagnostic::new(
                    "E000203",
                    format!(
                        "extension cannot replace behavior `{target}.{}` defined directly on `{target}`",
                        method.name
                    ),
                    Some(method.span),
                ));
            }
        }
        for operator in &extension.operators {
            if class.operators.iter().any(|known| {
                known.operator == operator.operator
                    && annotations_match(
                        &known
                            .parameters
                            .iter()
                            .map(|parameter| parameter.annotation.clone())
                            .collect::<Vec<_>>(),
                        &operator
                            .parameters
                            .iter()
                            .map(|parameter| parameter.annotation.clone())
                            .collect::<Vec<_>>(),
                    )
            }) {
                return Err(Diagnostic::new(
                    "E000203",
                    format!("extension cannot replace an operator defined directly on `{target}`"),
                    Some(operator.span),
                ));
            }
        }
        class.methods.extend(extension.methods.clone());
        class.operators.extend(extension.operators.clone());
    }
    for module in &mut lowered.modules {
        module.ast.items.retain(|item| {
            let Item::Extension(extension) = item else {
                return true;
            };
            !imported_extensions
                .iter()
                .any(|(source, moved, _)| *source == module.id && moved.span == extension.span)
        });
    }
    for module in &mut lowered.modules {
        module.ast = crate::normalize_extensions(&module.ast)?;
    }
    Ok(lowered)
}

fn lower_trait_typed_parameters(module_graph: &ModuleGraph) -> ModuleGraph {
    let trait_names = module_graph
        .modules
        .iter()
        .flat_map(|module| &module.ast.items)
        .filter_map(|item| match item {
            Item::Trait(declaration) => Some(declaration.name.clone()),
            _ => None,
        })
        .collect::<BTreeSet<_>>();
    let mut lowered = module_graph.clone();
    for module in &mut lowered.modules {
        for function in module.ast.items.iter_mut().filter_map(|item| match item {
            Item::Function(function) => Some(function),
            _ => None,
        }) {
            let mut used = function
                .type_parameters
                .iter()
                .cloned()
                .collect::<BTreeSet<_>>();
            for ordinal in 0..function.parameters.len() {
                let bound = function.parameters[ordinal].annotation.clone();
                let Some((declared_bound, arguments)) = bound.named_parts() else {
                    continue;
                };
                if !arguments.is_empty() {
                    continue;
                }
                let bound_name = declared_bound.rsplit('.').next().unwrap_or(declared_bound);
                if !trait_names.contains(bound_name) {
                    continue;
                }
                let base = format!(
                    "__sev_trait_{}_{}",
                    function.parameters[ordinal].name, ordinal
                );
                let mut parameter = base.clone();
                let mut suffix = 0usize;
                while !used.insert(parameter.clone()) {
                    suffix += 1;
                    parameter = format!("{base}_{suffix}");
                }
                function.parameters[ordinal].annotation =
                    TypeAnnotation::named(parameter.clone(), Vec::new(), bound.span);
                function.type_parameters.push(parameter.clone());
                function.constraints.push(GenericConstraint::Parameter {
                    parameter,
                    bound: TypeAnnotation::named(bound_name, Vec::new(), bound.span),
                    span: function.parameters[ordinal].span,
                });
            }
        }
    }
    lowered
}

fn collect_package_classes(
    module_graph: &ModuleGraph,
    index: &ProgramIndex,
    types: &mut severian_universal::TypeContext,
) -> Result<Vec<PackageClass>, Diagnostic> {
    let mut classes = module_graph
        .modules
        .iter()
        .flat_map(|module| {
            module.ast.items.iter().filter_map(move |item| match item {
                Item::Class(declaration) => Some((module.id, declaration.clone())),
                Item::Enum(declaration) => Some((module.id, declaration.storage_class())),
                _ => None,
            })
        })
        .map(|(module, declaration)| {
            let ty = if declaration.primitive {
                if !declaration.type_parameters.is_empty() {
                    let supported = matches!(
                        declaration.name.as_str(),
                        "pointer" | "array" | "char" | "slice"
                    );
                    if !supported {
                        return Err(Diagnostic::new(
                            "E000204",
                            format!("the bootstrap does not support generic primitive completion for `{}`", declaration.name),
                            Some(declaration.span),
                        ).with_help("if this is a primitive, implement its generic structural type contract in the bootstrap; only for an ordinary source class, remove the trailing `:` after the implemented traits"));
                    }
                    if declaration.name == "pointer" && declaration.aliases.is_empty() {
                        return Err(Diagnostic::new(
                            "E000204",
                            "generic pointer primitive completion requires a source alias",
                            Some(declaration.span),
                        ).with_help("declare the intended source alias using `self as Alias` in the pointer completion body; for an ordinary source class, remove the trailing `:` after the implemented traits"));
                    }
                    let path = format!("source.{:032x}.{}", module.0, declaration.name);
                    types
                        .register_source_declaration(
                            path,
                            declaration.name.clone(),
                            declaration.type_parameters.len(),
                        )
                        .map_err(|error| {
                            Diagnostic::new("E000204", error.to_string(), Some(declaration.span))
                        })?
                } else {
                    let ty = types.resolve_name(&declaration.name).ok_or_else(|| {
                        Diagnostic::new(
                            "E000204",
                            format!(
                                "primitive declaration `{}` has no compiler-owned type to complete",
                                declaration.name
                            ),
                            Some(declaration.span),
                        )
                    })?;
                    if types.primitive(ty).is_none() {
                        return Err(Diagnostic::new(
                            "E000204",
                            format!("`{}` is not a compiler-owned primitive", declaration.name),
                            Some(declaration.span),
                        ));
                    }
                    ty
                }
            } else {
                let path = format!("source.{:032x}.{}", module.0, declaration.name);
                types
                    .register_source_declaration(
                        path,
                        declaration.name.clone(),
                        declaration.type_parameters.len(),
                    )
                    .map_err(|error| {
                        Diagnostic::new("E000204", error.to_string(), Some(declaration.span))
                    })?
            };
            if declaration.name == "Tensor" {
                types.mark_tensor_constructor(ty).map_err(|error| {
                    Diagnostic::new("E000204", error.to_string(), Some(declaration.span))
                })?;
            }
            Ok(PackageClass {
                module,
                ty,
                declaration,
                lookups: BTreeMap::new(),
            })
        })
        .collect::<Result<Vec<_>, Diagnostic>>()?;
    for class in &mut classes {
        for source in index.modules.keys().copied() {
            let names = visible_class_names(source, class, index);
            if !names.is_empty() {
                class.lookups.insert(source, names);
            }
        }
    }
    Ok(classes)
}

fn collect_package_enums(module_graph: &ModuleGraph, classes: &[PackageClass]) -> Vec<PackageEnum> {
    module_graph
        .modules
        .iter()
        .flat_map(|module| {
            module.ast.items.iter().filter_map(move |item| {
                let Item::Enum(declaration) = item else {
                    return None;
                };
                let lookups = classes
                    .iter()
                    .find(|class| {
                        class.module == module.id && class.declaration.name == declaration.name
                    })
                    .map(|class| class.lookups.clone())
                    .unwrap_or_default();
                Some(PackageEnum {
                    module: module.id,
                    declaration: declaration.clone(),
                    lookups,
                })
            })
        })
        .collect()
}

fn install_primitive_class_operators(
    types: &mut severian_universal::TypeContext,
    classes: &[PackageClass],
    index: &ProgramIndex,
) -> Result<(), Diagnostic> {
    for class in classes
        .iter()
        .filter(|class| class.declaration.primitive && class.declaration.type_parameters.is_empty())
    {
        for implementation in &class.declaration.operators {
            // Generic source operators are resolved at their concrete use
            // sites; they cannot be installed as an exact universal
            // signature before their type parameters are substituted.
            if !implementation.type_parameters.is_empty() {
                continue;
            }
            let result = resolve_package_type(
                types,
                &implementation.result,
                class.module,
                classes,
                &[],
                index,
            )?;
            // Compound-assignment declarations use the base operator syntax
            // in the current AST but return `unit`; they describe mutation,
            // not an additional value-producing binary overload.
            if result == types.resolve_name("unit").expect("bootstrap defines unit") {
                continue;
            }
            match implementation.parameters.as_slice() {
                [] => {
                    if let Some(operator) = crate::universal_unary_syntax(implementation.operator) {
                        types.add_source_unary(operator, class.ty, result);
                    }
                }
                [right] => {
                    let Some(operator) = crate::universal_binary_syntax(implementation.operator)
                    else {
                        continue;
                    };
                    let right = resolve_package_type(
                        types,
                        &right.annotation,
                        class.module,
                        classes,
                        &[],
                        index,
                    )?;
                    types.add_source_binary(OperatorSignature {
                        operator,
                        left: TypePattern::Exact(class.ty),
                        right: TypePattern::Exact(right),
                        result: TypePattern::Exact(result),
                    });
                }
                _ => {}
            }
        }
    }
    Ok(())
}

/// A declaration alias adds a spelling, not a second nominal type or callable.
/// Resolve before collecting type layouts and function specializations so every
/// consumer sees the original DefId, including facade and namespace imports.
fn resolve_declaration_aliases(graph: &ModuleGraph, index: &mut ProgramIndex) -> Result<(), Diagnostic> {
    let aliases = graph.modules.iter().flat_map(|module| {
        module.ast.items.iter().filter_map(move |item| {
            let Item::Type(alias) = item else { return None; };
            if !alias.type_parameters.is_empty() { return None; }
            let target = alias.definition.as_ref()?.simple_name()?;
            Some((module.id, alias.name.clone(), target.to_owned(), alias.span))
        })
    }).filter_map(|(module, name, target, span)| {
        index.definitions.values().find(|definition| {
            definition.module == module && definition.name == name && matches!(definition.kind, DefKind::Type)
        }).map(|definition| (definition.id, (module, target, span)))
    }).collect::<BTreeMap<_, _>>();

    fn canonical(
        resolution: &Resolution, index: &ProgramIndex,
        aliases: &BTreeMap<DefId, (ModuleId, String, severian_source::Span)>,
        active: &mut BTreeSet<DefId>, cache: &mut BTreeMap<DefId, Resolution>,
    ) -> Result<Resolution, Diagnostic> {
        let Resolution::Def(id) = resolution else { return Ok(resolution.clone()); };
        let Some((module, target, span)) = aliases.get(id) else { return Ok(resolution.clone()); };
        if let Some(resolved) = cache.get(id) { return Ok(resolved.clone()); }
        if !active.insert(*id) {
            return Err(Diagnostic::new("E000204", format!("cyclic declaration alias `{}`", index.definitions[id].name), Some(*span)));
        }
        let mut parts = target.split('.');
        let head = parts.next().expect("a named alias has a target");
        let mut resolved = index.modules.get(module).and_then(|module| module.scope.bindings.get(head)).cloned();
        for member in parts {
            resolved = match resolved {
                Some(ref resolution) => match canonical(resolution, index, aliases, active, cache)? {
                    Resolution::Module(module) => index.exports.get(&module).and_then(|exports| exports.get(member)).cloned(),
                    _ => None,
                },
                None => None,
            };
        }
        let result = match resolved {
            Some(resolution) => canonical(&resolution, index, aliases, active, cache)?,
            // Structural and builtin type aliases are not declaration bindings.
            None => resolution.clone(),
        };
        active.remove(id);
        cache.insert(*id, result.clone());
        Ok(result)
    }

    let mut cache = BTreeMap::new();
    for id in aliases.keys() {
        canonical(&Resolution::Def(*id), index, &aliases, &mut BTreeSet::new(), &mut cache)?;
    }
    for resolution in index.modules.values_mut().flat_map(|module| module.scope.bindings.values_mut())
        .chain(index.exports.values_mut().flat_map(|exports| exports.values_mut()))
    {
        if let Resolution::Def(id) = resolution {
            if let Some(target) = cache.get(id) { *resolution = target.clone(); }
        }
    }
    Ok(())
}

pub(crate) fn internal_type_name(module: ModuleId, name: &str) -> String {
    format!("source.{:032x}.{name}", module.0)
}

/// Expand in the alias's lexical scope, preserving nominal declaration names.
/// Type arguments supplied by a caller retain the caller's lexical scope.
pub(crate) fn expand_type_alias(annotation: &TypeAnnotation, module: ModuleId, index: &ProgramIndex) -> Result<TypeAnnotation, Diagnostic> {
    fn lookup<'a>(name: &str, module: ModuleId, index: &'a ProgramIndex) -> Option<&'a Resolution> {
        let mut parts = name.split('.');
        let mut result = index.modules.get(&module)?.scope.bindings.get(parts.next()?)?;
        for part in parts {
            let Resolution::Module(module) = result else { return None; };
            result = index.exports.get(module)?.get(part)?;
        }
        Some(result)
    }
    fn expand(annotation: &TypeAnnotation, module: ModuleId, index: &ProgramIndex, qualify: bool, active: &mut BTreeSet<DefId>) -> Result<TypeAnnotation, Diagnostic> {
        let kind = match &annotation.kind {
            TypeAnnotationKind::Named { name, arguments } => {
                let resolved = lookup(name, module, index);
                if let Some(Resolution::Def(id)) = resolved {
                    if let Some(alias) = index.type_aliases.get(id).filter(|alias| alias.definition.is_some()) {
                        if alias.type_parameters.len() != arguments.len() {
                            return Err(Diagnostic::new("E000204", format!("alias `{name}` expects {} type argument(s), received {}", alias.type_parameters.len(), arguments.len()), Some(annotation.span)));
                        }
                        let mut substitution = BTreeMap::new();
                        for (parameter, argument) in alias.type_parameters.iter().zip(arguments) {
                            substitution.insert(parameter.clone(), expand(argument, module, index, true, active)?);
                        }
                        if !active.insert(*id) {
                            return Err(Diagnostic::new("E000204", format!("cyclic type alias `{name}`"), Some(annotation.span)));
                        }
                        let target = crate::substitute_type_annotation(alias.definition.as_ref().unwrap(), &substitution);
                        let result = expand(&target, index.definitions[id].module, index, true, active)?;
                        active.remove(id);
                        return Ok(result);
                    }
                }
                let name = if qualify {
                    match resolved {
                        Some(Resolution::Def(id)) if matches!(index.definitions[id].kind, DefKind::Class(_) | DefKind::Type) => {
                            let definition = &index.definitions[id];
                            if matches!(&definition.kind, DefKind::Class(class) if class.primitive && class.type_parameters.is_empty()) {
                                definition.name.clone()
                            } else {
                                internal_type_name(definition.module, &definition.name)
                            }
                        }
                        _ => name.clone(),
                    }
                } else { name.clone() };
                TypeAnnotationKind::Named { name, arguments: arguments.iter().map(|argument| expand(argument, module, index, qualify, active)).collect::<Result<_, _>>()? }
            }
            TypeAnnotationKind::Union(members) => {
                let mut flattened = Vec::new();
                for member in members {
                    let expanded = expand(member, module, index, qualify, active)?;
                    let members = match expanded.kind {
                        TypeAnnotationKind::Union(members) => members,
                        _ => vec![expanded],
                    };
                    for member in members {
                        if !flattened.iter().any(|known| annotation_matches(known, &member)) {
                            flattened.push(member);
                        }
                    }
                }
                TypeAnnotationKind::Union(flattened)
            },
            TypeAnnotationKind::Function { parameters, result } => TypeAnnotationKind::Function {
                parameters: parameters.iter().map(|parameter| expand(parameter, module, index, qualify, active)).collect::<Result<_, _>>()?,
                result: Box::new(expand(result, module, index, qualify, active)?),
            },
            kind => kind.clone(),
        };
        Ok(TypeAnnotation { kind, span: annotation.span })
    }
    expand(annotation, module, index, false, &mut BTreeSet::new())
}

/// Named fundamental families retain the bootstrap's concrete default storage.
/// Their expanded member set is used for generic bounds, never trait admission.
pub(crate) fn primitive_family_storage(original: &TypeAnnotation, expanded: &TypeAnnotation, module: ModuleId, index: &ProgramIndex, types: &severian_universal::TypeContext) -> Option<TypeId> {
    let TypeAnnotationKind::Union(members) = &expanded.kind else { return None; };
    let name = original.simple_name()?;
    let definitions = generic::resolve_path(module, name, index);
    let canonical = match definitions.as_slice() {
        [id] => index.definitions[id].name.as_str(),
        _ => name,
    };
    let ty = types.resolve_name(canonical)?;
    types.primitive(ty)?;
    members.iter().all(|member| member.simple_name().and_then(|name| types.resolve_name(name)).is_some_and(|member| types.primitive(member).is_some())).then_some(ty)
}

fn visible_class_names(
    source: ModuleId,
    class: &PackageClass,
    index: &ProgramIndex,
) -> Vec<String> {
    let Some(scope) = index.modules.get(&source) else {
        return Vec::new();
    };
    let matches_class = |resolution: &Resolution| {
        resolution_definitions(resolution)
            .into_iter()
            .any(|definition| {
                index
                    .definitions
                    .get(&definition)
                    .is_some_and(|definition| {
                        definition.module == class.module
                            && definition.name == class.declaration.name
                            && matches!(definition.kind, DefKind::Type | DefKind::Class(_))
                    })
            })
    };
    let mut names = Vec::new();
    for (binding, resolution) in &scope.scope.bindings {
        match resolution {
            Resolution::Module(target) => {
                for (exported, resolution) in index.exports.get(target).into_iter().flatten() {
                    if !matches_class(resolution) {
                        continue;
                    }
                    names.push(format!("{binding}.{exported}"));
                    // Tensor is the language-facing generic value type. Keep
                    // its annotation available beside the `tensor(...)`
                    // constructor after an ordinary `import tensor`.
                    if class.declaration.name == "Tensor" {
                        names.push("Tensor".into());
                    }
                }
            }
            resolution if matches_class(resolution) => names.push(binding.clone()),
            _ => {}
        }
    }
    names.sort();
    names.dedup();
    names
}

fn class_lexical_modules(source: ModuleId, classes: &[PackageClass]) -> BTreeSet<ModuleId> {
    let mut modules = BTreeSet::from([source]);
    let mut selected = classes
        .iter()
        .enumerate()
        .filter(|(_, class)| {
            class
                .lookups
                .get(&source)
                .is_some_and(|names| !names.is_empty())
        })
        .map(|(index, _)| index)
        .collect::<BTreeSet<_>>();
    loop {
        let previous = selected.len();
        let referenced = selected
            .iter()
            .flat_map(|index| {
                let owner = &classes[*index];
                owner.declaration.fields.iter().filter_map(move |field| {
                    let name = field.annotation.named_parts()?.0;
                    classes
                        .iter()
                        .enumerate()
                        .find_map(|(candidate_index, candidate)| {
                            candidate
                                .lookups
                                .get(&owner.module)
                                .is_some_and(|lookups| lookups.iter().any(|lookup| lookup == name))
                                .then_some(candidate_index)
                        })
                })
            })
            .collect::<Vec<_>>();
        selected.extend(referenced);
        if selected.len() == previous {
            // Field-only records need their type layouts, not the defining
            // module's callable namespace. Importing Metadata must not make
            // unrelated os functions compete with the caller's declarations.
            modules.extend(selected.iter().filter_map(|index| {
                let class = &classes[*index];
                (!class.declaration.methods.is_empty()).then_some(class.module)
            }));
            return modules;
        }
    }
}

fn collect_scoped_binding_ids(block: &severian_hir::Block, ids: &mut Vec<u32>) {
    use severian_hir::Statement;
    for statement in &block.statements {
        match statement {
            Statement::Try { body, catch_binding, catch_body, .. } => {
                ids.push(catch_binding.0);
                collect_scoped_binding_ids(body, ids);
                collect_scoped_binding_ids(catch_body, ids);
            }
            Statement::Sequence(body) | Statement::Placement { body, .. }
            | Statement::While { body, .. } | Statement::ExpectThrow { body, .. } =>
                collect_scoped_binding_ids(body, ids),
            Statement::If { then_block, else_block, .. } => {
                collect_scoped_binding_ids(then_block, ids);
                collect_scoped_binding_ids(else_block, ids);
            }
            Statement::Match { arms, .. } => for arm in arms {
                if let Some(binding) = arm.binding { ids.push(binding.0); }
                collect_scoped_binding_ids(&arm.body, ids);
            },
            _ => {}
        }
    }
}

fn collect_signature_layouts(
    types: &mut severian_universal::TypeContext, annotation: &TypeAnnotation, module: ModuleId,
    classes: &[PackageClass], lists: &[PackageList], index: &ProgramIndex,
    callables: &mut Vec<(Vec<TypeId>, TypeId)>, unions: &mut Vec<Vec<TypeId>>,
    collections: &mut Vec<PackageCollection>,
) -> Result<(), Diagnostic> {
    let expanded = expand_type_alias(annotation, module, index)?;
    match &expanded.kind {
        TypeAnnotationKind::Function { parameters, result } => {
            for annotation in parameters.iter().chain(std::iter::once(result.as_ref())) {
                collect_signature_layouts(types, annotation, module, classes, lists, index, callables, unions, collections)?;
            }
            let parameters = parameters.iter().map(|annotation| resolve_package_type(types, annotation, module, classes, lists, index)).collect::<Result<Vec<_>, _>>()?;
            let result = resolve_package_type(types, result, module, classes, lists, index)?;
            callables.push((parameters, result));
        }
        TypeAnnotationKind::Union(members) => {
            for member in members { collect_signature_layouts(types, member, module, classes, lists, index, callables, unions, collections)?; }
            unions.push(members.iter().map(|member| resolve_package_type(types, member, module, classes, lists, index)).collect::<Result<Vec<_>, _>>()?);
        }
        _ => {
            if let Some((name, arguments)) = expanded.named_parts() {
                for argument in arguments {
                    collect_signature_layouts(types, argument, module, classes, lists, index, callables, unions, collections)?;
                }
                if matches!(name, "list" | "tuple" | "map") {
                    let elements = arguments.iter().map(|argument| resolve_package_type(types, argument, module, classes, lists, index)).collect::<Result<Vec<_>, _>>()?;
                    match (name, elements.as_slice()) {
                        ("list", [element]) => collections.push(PackageCollection::List(*element)),
                        ("tuple", _) => collections.push(PackageCollection::Tuple(elements)),
                        ("map", [key, value]) => collections.push(PackageCollection::Map(*key, *value)),
                        _ => {}
                    }
                }
            }
        }
    }
    Ok(())
}

fn resolve_package_type(
    types: &mut severian_universal::TypeContext,
    annotation: &TypeAnnotation,
    module: ModuleId,
    classes: &[PackageClass],
    lists: &[PackageList],
    index: &ProgramIndex,
) -> Result<TypeId, Diagnostic> {
    let expanded = expand_type_alias(annotation, module, index)?;
    if expanded != *annotation {
        if let Some(ty) = primitive_family_storage(annotation, &expanded, module, index, types) { return Ok(ty); }
        return resolve_package_type(types, &expanded, module, classes, lists, index);
    }
    if let Some(("array", [element])) = annotation.named_parts() {
        let element = resolve_package_type(types, element, module, classes, lists, index)?;
        return types.instantiate_memory_buffer(element).map_err(|error| {
            Diagnostic::new("E000204", error.to_string(), Some(annotation.span))
        });
    }
    if annotation.simple_name() == Some("pointer") {
        return super::resolve_type_annotation(types, annotation);
    }
    if let Some(("pointer", [element])) = annotation.named_parts() {
        let element = resolve_package_type(types, element, module, classes, lists, index)?;
        return Ok(super::pointer_type_id(element));
    }
    if let Some(("borrowed" | "owned" | "transferred" | "out" | "inout" | "nullable", [inner])) =
        annotation.named_parts()
    {
        return resolve_package_type(types, inner, module, classes, lists, index);
    }
    if annotation.simple_name() == Some("Any") {
        return Ok(crate::any_type_id());
    }
    if let TypeAnnotationKind::Function { parameters, result } = &annotation.kind {
        let parameters = parameters
            .iter()
            .map(|parameter| resolve_package_type(types, parameter, module, classes, lists, index))
            .collect::<Result<Vec<_>, _>>()?;
        let result = resolve_package_type(types, result, module, classes, lists, index)?;
        return Ok(crate::function_type_id(&parameters, result));
    }
    if let severian_ast::TypeAnnotationKind::Union(members) = &annotation.kind {
        let mut success = Vec::new();
        let mut errors = Vec::new();
        for member in members {
            if matches!(member.simple_name(), Some("None" | "absent")) {
                success.push(types.resolve_name("None").expect("bootstrap defines None"));
                continue;
            }
            let ty = resolve_package_type(types, member, module, classes, lists, index)?;
            let source_error = member.simple_name().is_some_and(|name| {
                name.ends_with("Error")
                    || package_class_for_lookup(classes, module, name).is_some_and(|class| {
                        class
                            .declaration
                            .traits
                            .iter()
                            .any(|implemented| implemented.simple_name() == Some("Error"))
                    })
            });
            if types.resolve_name("Error") == Some(ty) || source_error {
                errors.push(ty);
            } else {
                success.push(ty);
            }
        }
        success.sort();
        success.dedup();
        errors.sort();
        errors.dedup();
        if let ([success], [error]) = (success.as_slice(), errors.as_slice()) {
            return Ok(crate::fallible_type_id(*success, *error));
        }
        if errors.is_empty() {
            return match success.as_slice() {
                [success] => Ok(*success),
                [_, _, ..] => Ok(crate::union_type_id(&success)),
                [] => Err(Diagnostic::new(
                    "E000204",
                    "a union must contain at least one concrete type",
                    Some(annotation.span),
                )),
            };
        }
        return Err(Diagnostic::new(
            "E000204",
            "fallible unions currently require one success type and one error type",
            Some(annotation.span),
        ));
    }
    if let Some(("Result", [success, error])) = annotation.named_parts() {
        let success = resolve_package_type(types, success, module, classes, lists, index)?;
        let error_ty = resolve_package_type(types, error, module, classes, lists, index)?;
        let source_error = error.simple_name().is_some_and(|name| {
            name.ends_with("Error")
                || package_class_for_lookup(classes, module, name).is_some_and(|class| {
                    class
                        .declaration
                        .traits
                        .iter()
                        .any(|implemented| implemented.simple_name() == Some("Error"))
                })
        });
        if types.resolve_name("Error") != Some(error_ty) && !source_error {
            return Err(Diagnostic::new(
                "E000204",
                "the second Result type must be an error type",
                Some(annotation.span),
            ));
        }
        return Ok(crate::fallible_type_id(success, error_ty));
    }
    if let Some(("tuple", elements)) = annotation.named_parts() {
        let elements = elements
            .iter()
            .map(|element| resolve_package_type(types, element, module, classes, lists, index))
            .collect::<Result<Vec<_>, _>>()?;
        return Ok(crate::tuple_type_id(&elements));
    }
    if let Some(("list", [element])) = annotation.named_parts() {
        let element = resolve_package_type(types, element, module, classes, lists, index)?;
        if let Some(list) = lists.iter().find(|list| list.element == element) {
            return Ok(list.ty);
        }
        return Ok(crate::list_type_id(element));
    }
    if let Some(("set", [element])) = annotation.named_parts() {
        // Resolve the argument here so unknown element types still fail at the
        // package boundary. Sets currently share one representation identity.
        resolve_package_type(types, element, module, classes, lists, index)?;
        return Ok(crate::set_type_id());
    }
    if let Some(("map", [key, value])) = annotation.named_parts() {
        let key = resolve_package_type(types, key, module, classes, lists, index)?;
        let value = resolve_package_type(types, value, module, classes, lists, index)?;
        return Ok(crate::map_type_id(key, value));
    }
    if let Some((name, arguments)) = annotation.named_parts() {
        if let Some((minimum, maximum)) = trait_type_arity(index, module, name) {
            if arguments.len() < minimum || arguments.len() > maximum {
                return Err(Diagnostic::new("E000204", format!("trait `{name}` expects {minimum}..={maximum} type argument(s), received {}", arguments.len()), Some(annotation.span)));
            }
            let arguments = arguments.iter().map(|argument|
                resolve_package_type(types, argument, module, classes, lists, index)
            ).collect::<Result<Vec<_>, _>>()?;
            let identity = trait_identity(index, module, name).expect("resolved trait identity");
            return super::trait_object::register_type(types, &identity, name, &arguments, annotation.span);
        }
        if !arguments.is_empty() {
            if let Some(class) = package_class_for_lookup(classes, module, name) {
                if class.declaration.name == "Tensor" {
                    let element =
                        resolve_package_type(types, &arguments[0], module, classes, lists, index)?;
                    if arguments.len() == 1 {
                        return types
                            .instantiate_tensor(
                                class.ty,
                                element,
                                severian_universal::TensorShape::Unranked,
                            )
                            .map_err(|error| {
                                Diagnostic::new("E000204", error.to_string(), Some(annotation.span))
                            });
                    }
                    let dimensions = arguments[1..]
                        .iter()
                        .map(|argument| match &argument.kind {
                            TypeAnnotationKind::DimensionConstant(value) => {
                                Ok(severian_universal::DimExpr::Constant(*value))
                            }
                            TypeAnnotationKind::DimensionRuntime(runtime) => Ok(
                                severian_universal::DimExpr::Runtime(
                                    severian_universal::RuntimeDimId(*runtime),
                                ),
                            ),
                            TypeAnnotationKind::ShapeSpread(name) => Err(Diagnostic::new(
                                "E000204",
                                format!(
                                    "shape pack `*{name}` must be inferred or specialized before tensor type resolution"
                                ),
                                Some(argument.span),
                            )),
                            TypeAnnotationKind::Named { name, arguments }
                                if arguments.is_empty() => Err(Diagnostic::new(
                                    "E000204",
                                    format!(
                                        "dimension `{name}` must be bound by generic shape specialization before tensor type resolution"
                                    ),
                                    Some(argument.span),
                                )),
                            _ => Err(Diagnostic::new(
                                "E000204",
                                "tensor shape arguments must be dimension values or shape packs",
                                Some(argument.span),
                            )),
                        })
                        .collect::<Result<Vec<_>, _>>()?;
                    return types
                        .instantiate_symbolic_tensor(
                            class.ty,
                            element,
                            severian_universal::ShapeTerm::Ranked(dimensions),
                        )
                        .map_err(|error| {
                            Diagnostic::new("E000204", error.to_string(), Some(annotation.span))
                        });
                }
                if class.declaration.type_parameters.len() != arguments.len() {
                    return Err(Diagnostic::new(
                        "E000204",
                        format!(
                            "class `{name}` expects {} type argument(s), received {}",
                            class.declaration.type_parameters.len(),
                            arguments.len()
                        ),
                        Some(annotation.span),
                    ));
                }
                let arguments = arguments
                    .iter()
                    .map(|argument| {
                        resolve_package_type(types, argument, module, classes, lists, index)
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                return types
                    .instantiate_applied(class.ty, arguments)
                    .map_err(|error| {
                        Diagnostic::new("E000204", error.to_string(), Some(annotation.span))
                    });
            }
        }
    }
    if let Some(name) = annotation.simple_name() {
        if let Some(class) = package_class_for_lookup(classes, module, name) {
            return Ok(class.ty);
        }
        if package_trait_for_lookup(index, module, name) {
            let identity = trait_identity(index, module, name).expect("resolved trait identity");
            return super::trait_object::register_type(types, &identity, name, &[], annotation.span);
        }
    }
    crate::resolve_type_annotation(types, annotation)
}

fn resolve_package_union_members(
    types: &mut severian_universal::TypeContext,
    annotation: &TypeAnnotation,
    module: ModuleId,
    classes: &[PackageClass],
    lists: &[PackageList],
    index: &ProgramIndex,
) -> Result<Option<Vec<TypeId>>, Diagnostic> {
    let severian_ast::TypeAnnotationKind::Union(members) = &annotation.kind else {
        return Ok(None);
    };
    let mut resolved = Vec::new();
    for member in members {
        if matches!(member.simple_name(), Some("None" | "absent")) {
            resolved.push(types.resolve_name("None").expect("bootstrap defines None"));
            continue;
        }
        let ty = resolve_package_type(types, member, module, classes, lists, index)?;
        // Callers need both success and error members to reconstruct the
        // imported result representation and propagate failures correctly.
        resolved.push(ty);
    }
    resolved.sort();
    resolved.dedup();
    Ok((resolved.len() > 1).then_some(resolved))
}

#[cfg(test)]
mod memory_boundary_tests {
    use super::*;

    #[test]
    fn generic_memory_boundary_resolves_a_view_without_erasing_its_buffer() {
        let source = severian_source::SourceFile::virtual_source(
            "memory-boundary.sev",
            "@mlir(\"memref.extract_aligned_pointer_as_index\")\ndef address[T](value: view array[T]) -> index\n",
        );
        let ast = severian_parser::parse(&severian_lexer::scan(&source).unwrap()).unwrap();
        let Item::Function(function) = &ast.items[0] else { panic!("expected boundary") };
        let mut substitution = GenericSubstitution::new();
        substitution.insert_type("T".into(), "u8".into());
        let concrete = specialize_function(function, &substitution);
        assert!(concrete.parameters[0].immutable_reference);
        let mut context = severian_bootstrap::load().unwrap();
        let ty = resolve_package_type(
            &mut context.types, &concrete.parameters[0].annotation, ModuleId(0),
            &[], &[], &ProgramIndex::default(),
        ).unwrap();
        let byte = context.types.resolve_name("u8").unwrap();
        assert_eq!(context.types.memory_buffer_element(ty), Some(byte));
        assert_ne!(ty, crate::list_type_id(byte));
        assert_ne!(ty, crate::pointer_type_id(byte));
        let module = severian_ast::Module { items: vec![Item::Function(concrete)] };
        let (program, types) = crate::analyze_with_context_and_types(
            &module, &context.types, crate::AnalysisContext {
                mode: crate::AnalysisMode::Build, module_name: "memory_boundary",
            },
        ).unwrap();
        let parameter = &program.modules[0].functions[0].parameters[0];
        assert_eq!(types.memory_buffer_element(parameter.contract.ty), Some(byte));
        assert!(parameter.contract.modifiers.iter().any(|modifier| modifier.name == "view"));
        severian_mir::build(&program).unwrap();
    }
}

fn package_class_for_lookup<'a>(
    classes: &'a [PackageClass],
    module: ModuleId,
    name: &str,
) -> Option<&'a PackageClass> {
    classes.iter().find(|class| {
        name == internal_type_name(class.module, &class.declaration.name) || class
            .lookups
            .get(&module)
            .is_some_and(|lookups| lookups.iter().any(|lookup| lookup == name))
    })
}

fn package_trait_for_lookup(index: &ProgramIndex, module: ModuleId, name: &str) -> bool {
    let Some(scope) = index.modules.get(&module) else {
        return false;
    };
    let resolution = if let Some((namespace, member)) = name.split_once('.') {
        let Some(Resolution::Module(target)) = scope.scope.bindings.get(namespace) else {
            return false;
        };
        index
            .exports
            .get(target)
            .and_then(|exports| exports.get(member))
    } else {
        scope.scope.bindings.get(name)
    };
    resolution.is_some_and(|resolution| {
        resolution_definitions(resolution)
            .into_iter()
            .any(|definition| {
                index
                    .definitions
                    .get(&definition)
                    .is_some_and(|definition| matches!(definition.kind, DefKind::Trait(_)))
            })
    })
}

pub(super) fn trait_identity(index: &ProgramIndex, module: ModuleId, name: &str) -> Option<String> {
    generic::resolve_path(module, name, index).into_iter().find_map(|id| {
        matches!(index.definitions.get(&id)?.kind, DefKind::Trait(_))
            .then(|| format!("{:x}:{:x}:{:x}", id.package, id.module, id.declaration.0))
    })
}

pub(super) fn resolve_trait_definitions(index: &ProgramIndex, module: ModuleId, name: &str) -> Vec<DefId> {
    generic::resolve_path(module, name, index)
}

pub(super) fn trait_type_arity(index: &ProgramIndex, module: ModuleId, name: &str) -> Option<(usize, usize)> {
    generic::resolve_path(module, name, index).into_iter().find_map(|id| {
        let DefKind::Trait(declaration) = &index.definitions.get(&id)?.kind else { return None; };
        let maximum = declaration.type_parameters.len();
        let minimum = declaration.type_parameter_defaults.iter().rposition(Option::is_none)
            .map_or(0, |position| position + 1);
        Some((minimum, maximum))
    })
}

pub(super) fn trait_identities(index: &ProgramIndex, module: ModuleId, name: &str) -> BTreeSet<String> {
    let mut pending = generic::resolve_path(module, name, index);
    let mut seen = BTreeSet::new();
    let mut identities = BTreeSet::new();
    while let Some(id) = pending.pop() {
        if !seen.insert(id) { continue; }
        let Some(definition) = index.definitions.get(&id) else { continue; };
        let DefKind::Trait(declaration) = &definition.kind else { continue; };
        identities.insert(format!("{:x}:{:x}:{:x}", id.package, id.module, id.declaration.0));
        for base in &declaration.bases {
            if let Some((name, _)) = base.named_parts() {
                pending.extend(generic::resolve_path(definition.module, name, index));
            }
        }
    }
    identities
}

fn visible_trait_names(module: ModuleId, index: &ProgramIndex) -> Vec<String> {
    let Some(scope) = index.modules.get(&module) else {
        return Vec::new();
    };
    let is_trait = |resolution: &Resolution| {
        resolution_definitions(resolution)
            .into_iter()
            .any(|definition| {
                index
                    .definitions
                    .get(&definition)
                    .is_some_and(|definition| matches!(definition.kind, DefKind::Trait(_)))
            })
    };
    let mut names = Vec::new();
    for (binding, resolution) in &scope.scope.bindings {
        match resolution {
            Resolution::Module(target) => {
                if let Some(exports) = index.exports.get(target) {
                    names.extend(exports.iter().filter_map(|(name, resolution)| {
                        is_trait(resolution).then(|| format!("{binding}.{name}"))
                    }));
                }
            }
            resolution if is_trait(resolution) => names.push(binding.clone()),
            _ => {}
        }
    }
    names.sort();
    names.dedup();
    names
}

fn collect_package_lists(
    module_graph: &ModuleGraph,
    types: &severian_universal::TypeContext,
) -> Vec<PackageList> {
    let mut uses = BTreeMap::<TypeId, ModuleId>::new();
    for module in &module_graph.modules {
        for item in &module.ast.items {
            match item {
                Item::Function(function) => {
                    collect_function_lists(function, types, module.id, &mut uses);
                }
                Item::Class(class) => {
                    for field in &class.fields {
                        collect_list_elements(&field.annotation, types, module.id, &mut uses);
                    }
                    for function in class.constructors.iter().chain(&class.methods) {
                        collect_function_lists(function, types, module.id, &mut uses);
                    }
                }
                Item::Enum(declaration) => {
                    for field in declaration
                        .variants
                        .iter()
                        .flat_map(|variant| &variant.fields)
                    {
                        collect_list_elements(&field.annotation, types, module.id, &mut uses);
                    }
                }
                Item::Binding(binding) => {
                    if let Some(annotation) = &binding.annotation {
                        collect_list_elements(annotation, types, module.id, &mut uses);
                    }
                }
                _ => {}
            }
        }
    }
    uses.into_iter()
        .map(|(element, module)| PackageList {
            module,
            ty: crate::list_type_id(element),
            element,
        })
        .collect()
}

fn collect_function_lists(
    function: &severian_ast::FunctionDeclaration,
    types: &severian_universal::TypeContext,
    module: ModuleId,
    output: &mut BTreeMap<TypeId, ModuleId>,
) {
    for annotation in function
        .parameters
        .iter()
        .map(|parameter| &parameter.annotation)
        .chain(std::iter::once(&function.result))
    {
        collect_list_elements(annotation, types, module, output);
    }
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
    for argument in arguments {
        collect_list_elements(argument, types, module, output);
    }
    if name == "list" && arguments.len() == 1 {
        if let Ok(element) = resolve_collection_type(types, &arguments[0]) {
            output.entry(element).or_insert(module);
        }
    }
}

fn resolve_collection_type(
    types: &severian_universal::TypeContext,
    annotation: &TypeAnnotation,
) -> Result<TypeId, Diagnostic> {
    if let Some(("list", [element])) = annotation.named_parts() {
        return Ok(crate::list_type_id(resolve_collection_type(
            types, element,
        )?));
    }
    if let Some(("tuple", elements)) = annotation.named_parts() {
        let elements = elements
            .iter()
            .map(|element| resolve_collection_type(types, element))
            .collect::<Result<Vec<_>, _>>()?;
        return Ok(crate::tuple_type_id(&elements));
    }
    crate::resolve_type_annotation(types, annotation)
}

#[derive(Debug)]
struct FunctionBinding {
    lookup: String,
    definition: DefId,
    substitution: GenericSubstitution,
}

fn namespace_member_needed(index: &ProgramIndex, module: ModuleId, namespace: &str, member: &str) -> bool {
    let Some(requirements) = &index.import_requirements else { return true };
    let Some(names) = requirements.get(&module) else { return false };
    let path = format!("{namespace}.{member}");
    names.contains("*") || names.contains(&path)
        || names.iter().any(|name|name.starts_with(&(path.clone()+".")))
        || (names.contains(namespace) && namespace == member)
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
                        if !namespace_member_needed(index, module, name, export) { continue; }
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
                    if definition.module != module.0 || index.definitions[&definition].name != *name {
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

fn module_function_bindings(
    module: ModuleId,
    index: &ProgramIndex,
    specializations: &Specializations,
) -> Vec<FunctionBinding> {
    let mut bindings = imported_function_bindings(module, index, specializations);
    for definition in &index.modules[&module].items {
        let Some(item) = index.definitions.get(definition) else {
            continue;
        };
        if !matches!(item.kind, DefKind::Function(_)) {
            continue;
        }
        if item.name == "print" {
            continue;
        }
        for substitution in function_instances(*definition, index, specializations) {
            bindings.push(FunctionBinding {
                lookup: item.name.clone(),
                definition: *definition,
                substitution,
            });
        }
    }
    bindings.retain(|binding| binding.lookup != "print");
    bindings
}

fn registry_function_bindings(
    modules: &BTreeSet<ModuleId>,
    index: &ProgramIndex,
    specializations: &Specializations,
) -> Vec<FunctionBinding> {
    let mut bindings = Vec::new();
    for module in modules {
        let Some(scope) = index.modules.get(module) else {
            continue;
        };
        for definition in &scope.items {
            let Some(item) = index.definitions.get(definition) else {
                continue;
            };
            let DefKind::Function(_) = &item.kind else {
                continue;
            };
            for substitution in function_instances(*definition, index, specializations) {
                bindings.push(FunctionBinding {
                    lookup: format!("__sev_registry_{:032x}.{}", module.0, item.name),
                    definition: *definition,
                    substitution,
                });
            }
        }
    }
    bindings
}

// Modules retain the graph's dependency order; each global block retains its
// source order. Lookup names (including import aliases) never order execution.
fn order_package_constants(constants: &mut Vec<PackageConstant>, graph: &ModuleGraph) {
    let mut seen = BTreeSet::new();
    constants.retain(|constant| seen.insert(constant.lookup.clone()));
    let modules = graph.modules.iter().enumerate()
        .map(|(ordinal, module)| (module.id, ordinal))
        .collect::<BTreeMap<_, _>>();
    constants.sort_by_key(|constant| (modules[&constant.module], constant.ordinal));
}

fn imported_constant_bindings(
    module: ModuleId,
    module_graph: &ModuleGraph,
    index: &ProgramIndex,
) -> Vec<PackageConstant> {
    let scope = &index.modules[&module].scope;
    let mut constants = Vec::new();
    let mut add = |lookup: String, definition: DefId| {
        let Some(item) = index.definitions.get(&definition) else {
            return;
        };
        if !matches!(item.kind, DefKind::Constant) || item.module == module {
            return;
        }
        let Some(source) = module_graph
            .modules
            .iter()
            .find(|source| source.id == item.module)
        else {
            return;
        };
        let Some((ordinal, binding)) = source
            .ast
            .items
            .iter()
            .enumerate()
            .find_map(|(ordinal, candidate)| match candidate {
                Item::Binding(binding) if binding.name == item.name => Some((ordinal, binding)),
                _ => None,
            })
        else {
            return;
        };
        constants.push(PackageConstant {
            lookup,
            value: binding.value.clone(),
            module: source.id,
            ordinal,
        });
    };
    for (name, resolution) in &scope.bindings {
        match resolution {
            Resolution::Module(target) => {
                if let Some(exports) = index.exports.get(target) {
                    for (export, resolution) in exports {
                        if !namespace_member_needed(index, module, name, export) { continue; }
                        for definition in resolution_definitions(resolution) {
                            add(format!("{name}.{export}"), definition);
                        }
                    }
                }
            }
            resolution => {
                for definition in resolution_definitions(resolution) {
                    add(name.clone(), definition);
                }
            }
        }
    }
    order_package_constants(&mut constants, module_graph);
    constants
}

fn module_constant_bindings(
    module: ModuleId,
    module_graph: &ModuleGraph,
    index: &ProgramIndex,
) -> Vec<PackageConstant> {
    let mut constants = imported_constant_bindings(module, module_graph, index);
    let Some(source) = module_graph
        .modules
        .iter()
        .find(|source| source.id == module)
    else {
        return constants;
    };
    constants.extend(source.ast.items.iter().enumerate().filter_map(|(ordinal, item)| {
        let Item::Binding(binding) = item else {
            return None;
        };
        Some(PackageConstant {
            lookup: binding.name.clone(),
            value: binding.value.clone(),
            module,
            ordinal,
        })
    }));
    constants
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
            .map(|instances| instances.keys().cloned().collect())
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

/// The driver temporarily injects bootstrap prelude declarations into every
/// module's local AST. They belong in that module's lexical scope, but they are
/// not declarations owned by the module and therefore must never be re-exported
/// through an unqualified source import.
fn is_injected_prelude_item(item: &Item) -> bool {
    item_span(item).source >= u32::MAX - 4
}

fn item_span(item: &Item) -> severian_source::Span {
    match item {
        Item::Trait(value) => value.span,
        Item::Class(value) => value.span,
        Item::Enum(value) => value.span,
        Item::Binding(value) => value.span,
        Item::Expression(value) => value.span,
        Item::Function(value) => value.span,
        Item::Type(value) => value.span,
        Item::Test(value) => value.span,
        Item::Import(value) => value.span,
        Item::Extension(value) => value.span,
    }
}

// Explicit imports introduce names even when no expression uses them. Diagnose
// conflicts before demand-driven import discovery can discard either binding.
fn validate_explicit_import_names(module_graph: &ModuleGraph) -> Result<(), Diagnostic> {
    for module in &module_graph.modules {
        let mut names = BTreeMap::new();
        for item in &module.ast.items {
            let Item::Import(import) = item else { continue; };
            if import.selected_name().is_some_and(|name| name.starts_with("__")) {
                return Err(Diagnostic::new("E000124", "file-local declarations cannot be imported", Some(import.span))
                    .with_help("use a public declaration, or rename the provider's declaration with a single leading underscore to allow explicit imports"));
            }
            if import.is_wildcard() && import.alias.is_none() { continue; }
            let name = import.alias.clone().or_else(|| import.selected_name().map(str::to_owned))
                .unwrap_or_else(|| match &import.subject {
                    ImportSubject::Name(name) => name.clone(),
                    ImportSubject::Locator(path) => std::path::Path::new(path).file_stem()
                        .and_then(|name| name.to_str()).unwrap_or(path).to_owned(),
                });
            if let Some(previous) = names.insert(name.clone(), import.span) {
                return Err(Diagnostic::new("E000203", format!("name `{name}` is already defined in this scope"), Some(import.span))
                    .with_label(import.span, "this import introduces the name again")
                    .with_label(previous, "previous import introduces this name")
                    .with_help(format!("remove one import of `{name}`, or give it a distinct `as` alias")));
            }
        }
        for item in &module.ast.items {
            if is_injected_prelude_item(item) { continue; }
            let (name, span) = match item {
                Item::Class(value) => (&value.name, value.span),
                Item::Enum(value) => (&value.name, value.span),
                Item::Function(value) => (&value.name, value.span),
                Item::Trait(value) => (&value.name, value.span),
                Item::Binding(value) if value.annotation.is_some() || value.mutable => (&value.name, value.span),
                _ => continue,
            };
            if let Some(previous) = names.get(name) {
                return Err(Diagnostic::new("E000203", format!("name `{name}` is already defined in this scope"), Some(span))
                    .with_label(span, "this declaration conflicts with the import")
                    .with_label(*previous, "previous import introduces this name")
                    .with_help(format!("rename this declaration or import `{name}` with a distinct `as` alias")));
            }
        }
    }
    Ok(())
}

fn collect_declarations(module_graph: &ModuleGraph) -> Result<ProgramIndex, Diagnostic> {
    validate_explicit_import_names(module_graph)?;
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
        let mut module_bindings = BTreeSet::new();
        for item in &module.ast.items {
            let injected_prelude = is_injected_prelude_item(item);
            if let Item::Class(class) = item {
                for field in &class.fields {
                    index
                        .fields
                        .entry(field.name.clone())
                        .or_default()
                        .push(FieldDecl {
                            owner: class.name.clone(),
                            owner_type_parameters: class.type_parameters.clone(),
                            annotation: field.annotation.clone(),
                        });
                }
                for method in &class.methods {
                    index
                        .methods
                        .entry(method.name.clone())
                        .or_default()
                        .push(MethodDecl {
                            owner: class.name.clone(),
                            owner_type_parameters: class.type_parameters.clone(),
                            type_parameters: method.type_parameters.clone(),
                            parameters: method
                                .parameters
                                .iter()
                                .map(|parameter| parameter.annotation.clone())
                                .collect(),
                            result: method.result.clone(),
                        });
                }
            }
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
                if let Some(previous) = index.definitions.get(&id) {
                    return Err(Diagnostic::new(
                        "E000203",
                        format!("import `{name}` is declared more than once"),
                        Some(import.span),
                    ).with_label(previous.span, "previous import")
                        .with_help(format!("remove the repeated import of `{name}`")));
                }
                index.definitions.insert(
                    id,
                    Definition {
                        id,
                        name,
                        module: module.id,
                        span: import.span,
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
                            parameter_names: function
                                .parameters
                                .iter()
                                .map(|parameter| parameter.name.clone())
                                .collect(),
                            parameters: function
                                .parameters
                                .iter()
                                .map(|parameter| parameter.annotation.clone())
                                .collect(),
                            parameter_defaults: function
                                .parameters
                                .iter()
                                .map(|parameter| parameter.default.clone())
                                .collect(),
                            parameter_variadics: function
                                .parameters
                                .iter()
                                .map(|parameter| parameter.variadic)
                                .collect(),
                            parameter_views: function.parameters.iter().map(|parameter| parameter.immutable_reference).collect(),
                            result: function.result.clone(),
                            constraints: function.constraints.clone(),
                            generic_body: function.body.clone(),
                        }),
                        id,
                    )
                }
                Item::Type(declaration) => {
                    let identity = item_identity(module.package, module.id, "type", &declaration.name, DefKind::Type);
                    index.type_aliases.insert(identity.2, declaration.clone());
                    identity
                },
                Item::Trait(declaration) => item_identity(
                    module.package,
                    module.id,
                    "trait",
                    &declaration.name,
                    DefKind::Trait(TraitDecl {
                        type_parameters: declaration.type_parameters.clone(),
                        type_parameter_defaults: declaration.type_parameter_defaults.clone(),
                        constraints: declaration.constraints.clone(),
                        bases: declaration.bases.clone(),
                        properties: declaration.properties.clone(),
                        methods: declaration.methods.clone(),
                        sentences: declaration.sentences.clone(),
                        operators: declaration.operators.clone(),
                    }),
                ),
                Item::Class(declaration) => item_identity(
                    module.package,
                    module.id,
                    "class",
                    &declaration.name,
                    DefKind::Class(ClassDecl {
                        primitive: declaration.primitive,
                        type_parameters: declaration.type_parameters.clone(),
                        fields: declaration.fields.clone(),
                        constructors: declaration.constructors.clone(),
                        methods: declaration.methods.clone(),
                        sentences: declaration.sentences.clone(),
                    }),
                ),
                Item::Enum(declaration) => item_identity(
                    module.package,
                    module.id,
                    "enum",
                    &declaration.name,
                    DefKind::Type,
                ),
                Item::Binding(binding)
                    if !binding.update && module_bindings.insert(binding.name.clone()) =>
                {
                    item_identity(
                        module.package,
                        module.id,
                        "constant",
                        &binding.name,
                        DefKind::Constant,
                    )
                }
                Item::Import(_) => unreachable!("imports are collected above"),
                _ => continue,
            };
            if let Some(existing) = index.definitions.get(&id) {
                if let (DefKind::Trait(existing), DefKind::Trait(candidate)) =
                    (&existing.kind, &kind)
                {
                    if compatible_trait_redeclaration(existing, candidate) {
                        continue;
                    }
                }
                return Err(Diagnostic::new(
                    "E000203",
                    format!(
                        "declaration `{name}` has the same canonical identity as `{}`",
                        existing.name
                    ),
                    Some(item_span(item)),
                ).with_label(existing.span, "previous declaration has this identity")
                    .with_help(format!("remove the duplicate declaration of `{name}` or rename it")));
            }
            let visibility = Visibility::for_name(&name);
            let definition = Definition {
                id,
                name: name.clone(),
                module: module.id,
                span: item_span(item),
                visibility,
                kind,
            };
            let grammar_owner = match item {
                Item::Class(class) => Some((&class.sentences, GrammarOwnerDeclaration::Class(class.clone()))),
                Item::Trait(trait_) => Some((&trait_.sentences, GrammarOwnerDeclaration::Trait(trait_.clone()))),
                _ => None,
            };
            if let Some((sentences, owner_declaration)) = grammar_owner {
                for (ordinal, declaration) in sentences.iter().enumerate() {
                    let label = DefId {
                        package: id.package,
                        module: id.module,
                        declaration: DeclarationId(stable_hash(&format!(
                            "grammar:{}:{}:{ordinal}", name, declaration.function.name,
                        ))),
                    };
                    index.grammars.push(GrammarRegistration {
                        label,
                        owner: id,
                        declaration: declaration.clone(),
                        owner_declaration: owner_declaration.clone(),
                    });
                }
            }
            index.definitions.insert(id, definition);
            items.push(id);
            insert_binding(
                &mut scope.bindings,
                name.clone(),
                Resolution::Def(id),
                &index.definitions,
            );
            if !injected_prelude && visibility != Visibility::File {
                insert_binding(&mut exports, name, Resolution::Def(id), &index.definitions);
            }
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

#[cfg(test)]
mod grammar_registration_tests {
    use super::*;

    fn graph() -> ModuleGraph {
        let source = severian_source::SourceFile::virtual_source(
            "grammar-registration.sev",
            "class first:\n    grammar literal[\"first\"]() -> F:\n        return must_not_execute()\nclass second:\n    grammar literal[\"second\", value: Lexeme]() -> B with accepted(value):\n        return must_not_execute(value)\n",
        );
        let ast = severian_parser::parse(&severian_lexer::scan(&source).unwrap()).unwrap();
        ModuleGraph {
            policies: BTreeMap::new(),
            modules: vec![severian_modules::ResolvedModule {
                id: ModuleId(1),
                package: PackageId(0),
                path: "grammar-registration.sev".into(),
                source,
                ast,
                imports: Vec::new(),
            }],
        }
    }

    #[test]
    fn class_grammars_register_automatically_during_declaration_collection() {
        let graph = graph();
        let index = collect_declarations(&graph).unwrap();
        let entries = index.grammar_registry(None).collect::<Vec<_>>();
        assert_eq!(entries.len(), 2);
        assert_ne!(entries[0].label, entries[1].label);
        assert_ne!(entries[0].owner, entries[1].owner);
        assert_eq!(index.definitions[&entries[0].owner].name, "first");
        assert_eq!(index.definitions[&entries[1].owner].name, "second");
        assert_eq!(entries[1].declaration.function.constraints.len(), 1);
        assert_eq!(entries[1].declaration.function.parameters[0].annotation.simple_name(), Some("Lexeme"));
        assert!(entries[1].declaration.function.body.is_some());
        let DefKind::Class(owner) = &index.definitions[&entries[1].owner].kind else {
            panic!("expected grammar owner");
        };
        assert_eq!(owner.sentences[0], entries[1].declaration);
        let rebuilt = collect_declarations(&graph).unwrap();
        assert_eq!(index.grammars, rebuilt.grammars);
    }

    #[test]
    fn grammar_registry_filters_result_contracts_independently_of_query_span() {
        let index = collect_declarations(&graph()).unwrap();
        let location = severian_source::Span::new(99, 20, 21);
        for result in ["F", "B"] {
            let expected = TypeAnnotation::named(result, Vec::new(), location);
            let entries = index.grammar_registry(Some(&expected)).collect::<Vec<_>>();
            assert_eq!(entries.len(), 1);
            assert_eq!(entries[0].declaration.function.result.simple_name(), Some(result));
        }
        let missing = TypeAnnotation::named("Unknown", Vec::new(), location);
        assert_eq!(index.grammar_registry(Some(&missing)).count(), 0);
    }
}

fn compatible_trait_redeclaration(left: &TraitDecl, right: &TraitDecl) -> bool {
    left.type_parameters == right.type_parameters
        && left.type_parameter_defaults == right.type_parameter_defaults
        && left.sentences == right.sentences
        && left.constraints.len() == right.constraints.len()
        && annotations_match(&left.bases, &right.bases)
        && left.properties.len() == right.properties.len()
        && left.methods.len() == right.methods.len()
        && left.operators.len() == right.operators.len()
        && left
            .methods
            .iter()
            .zip(&right.methods)
            .all(|(left, right)| {
                left.name == right.name
                    && annotations_match(
                        &left
                            .parameters
                            .iter()
                            .map(|parameter| parameter.annotation.clone())
                            .collect::<Vec<_>>(),
                        &right
                            .parameters
                            .iter()
                            .map(|parameter| parameter.annotation.clone())
                            .collect::<Vec<_>>(),
                    )
                    && annotation_matches(&left.result, &right.result)
            })
        && left
            .operators
            .iter()
            .zip(&right.operators)
            .all(|(left, right)| {
                left.operator == right.operator
                    && left.type_parameters == right.type_parameters
                    && annotations_match(
                        &left
                            .parameters
                            .iter()
                            .map(|parameter| parameter.annotation.clone())
                            .collect::<Vec<_>>(),
                        &right
                            .parameters
                            .iter()
                            .map(|parameter| parameter.annotation.clone())
                            .collect::<Vec<_>>(),
                    )
                    && annotation_matches(&left.result, &right.result)
            })
}

fn annotations_match(left: &[TypeAnnotation], right: &[TypeAnnotation]) -> bool {
    left.len() == right.len()
        && left
            .iter()
            .zip(right)
            .all(|(left, right)| annotation_matches(left, right))
}

fn annotation_matches(left: &TypeAnnotation, right: &TypeAnnotation) -> bool {
    use severian_ast::TypeAnnotationKind as Kind;
    match (&left.kind, &right.kind) {
        (
            Kind::Named {
                name: left_name,
                arguments: left_arguments,
            },
            Kind::Named {
                name: right_name,
                arguments: right_arguments,
            },
        ) => left_name == right_name && annotations_match(left_arguments, right_arguments),
        (Kind::DimensionConstant(left), Kind::DimensionConstant(right)) => left == right,
        (Kind::DimensionRuntime(left), Kind::DimensionRuntime(right)) => left == right,
        (Kind::ShapeSpread(left), Kind::ShapeSpread(right)) => left == right,
        (
            Kind::Function {
                parameters: left_parameters,
                result: left_result,
            },
            Kind::Function {
                parameters: right_parameters,
                result: right_result,
            },
        ) => {
            annotations_match(left_parameters, right_parameters)
                && annotation_matches(left_result, right_result)
        }
        (Kind::Union(left), Kind::Union(right)) => annotations_match(left, right),
        _ => false,
    }
}

fn resolve_imports(module_graph: &ModuleGraph, index: &mut ProgramIndex) {
    // Source modules may form declaration-only import cycles and package
    // facades commonly re-export declarations from files that appear later in
    // graph order. Resolve exports to a fixed point so visibility never
    // depends on filesystem traversal order.
    for _ in 0..=module_graph.modules.len() {
        let previous_exports = index.exports.clone();
        for module in &module_graph.modules {
            for import in module.ast.items.iter().filter_map(|item| match item {
                Item::Import(import) => Some(import),
                _ => None,
            }) {
                let Some(edge) = module.imports.iter().find(|edge| edge.span == import.span) else {
                    continue;
                };
                if import.is_wildcard() && import.alias.is_none() {
                    let members = index.exports.get(&edge.module).cloned().unwrap_or_default();
                    for (name, resolution) in members {
                        if name.starts_with('_') || resolution_definitions(&resolution).iter().any(|id| index.definitions[id].visibility != Visibility::Public) { continue; }
                        insert_binding(
                            &mut index
                                .modules
                                .get_mut(&module.id)
                                .expect("every graph module has a scope")
                                .scope
                                .bindings,
                            name.clone(),
                            resolution.clone(),
                            &index.definitions,
                        );
                        insert_binding(
                            index
                                .exports
                                .get_mut(&module.id)
                                .expect("every graph module has exports"),
                            name,
                            resolution,
                            &index.definitions,
                        );
                    }
                    continue;
                }
                let (name, resolution) = if let Some(imported_name) = import.selected_name() {
                    let Some(resolution) = index
                        .exports
                        .get(&edge.module)
                        .and_then(|exports| exports.get(imported_name))
                        .cloned()
                    else {
                        // The facade can be visited before its own imports.
                        // Defer this binding until the export fixed point.
                        continue;
                    };
                    (
                        import
                            .alias
                            .clone()
                            .unwrap_or_else(|| imported_name.to_owned()),
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
                // Selective imports are facade declarations too. Keep their
                // original DefIds when re-exporting so downstream packages see
                // the same nominal type and callable, not a copied definition.
                if !name.starts_with("__") && (import.selected_name().is_some() || import.alias.is_some()) {
                    insert_binding(
                        index
                            .exports
                            .get_mut(&module.id)
                            .expect("every graph module has exports"),
                        name.clone(),
                        resolution.clone(),
                        &index.definitions,
                    );
                }
                let scope = &mut index
                    .modules
                    .get_mut(&module.id)
                    .expect("every graph module has a scope")
                    .scope
                    .bindings;
                if scope.get(&name) == Some(&resolution) {
                    continue;
                }
                insert_binding(scope, name, resolution, &index.definitions);
            }
        }
        if index.exports == previous_exports {
            break;
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
    if old == new {
        bindings.insert(name, old);
        return;
    }
    let has_module = matches!(&old, Resolution::Module(_)) || matches!(&new, Resolution::Module(_));
    let mut ids = resolution_definitions(&old);
    ids.extend(resolution_definitions(&new));
    ids.sort();
    ids.dedup();
    if has_module {
        bindings.insert(name, Resolution::Ambiguous(ids));
        return;
    }
    if let [id] = ids.as_slice() {
        bindings.insert(name, Resolution::Def(*id));
        return;
    }
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
        .map(|parameter| {
            format!(
                "{}{}{}",
                if parameter.immutable_reference { "view " } else { "" },
                type_key(&parameter.annotation),
                if parameter.variadic { "..." } else { "" }
            )
        })
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
        GenericConstraint::VariadicPack { parameter, .. } => format!("*{parameter}"),
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
        TypeAnnotationKind::DimensionConstant(value) => value.to_string(),
        TypeAnnotationKind::DimensionRuntime(runtime) => format!("?{runtime}"),
        TypeAnnotationKind::ShapeSpread(name) => format!("*{name}"),
        TypeAnnotationKind::Union(types) => {
            format!(
                "({})",
                types.iter().map(type_key).collect::<Vec<_>>().join("|")
            )
        }
        TypeAnnotationKind::Function { parameters, result } => format!(
            "({})->{}",
            parameters
                .iter()
                .map(type_key)
                .collect::<Vec<_>>()
                .join(","),
            type_key(result)
        ),
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
        .bindings()
        .into_iter()
        .map(|(parameter, ty)| format!("{parameter}={ty}"))
        .collect::<Vec<_>>()
        .join(",");
    FunctionId(stable_hash(&format!(
        "function:{:032x}:{:032x}[{arguments}]",
        definition.module, definition.declaration.0,
    )))
}

fn universal_substitution(
    function_name: &str,
    function: &FunctionDecl,
    substitution: &GenericSubstitution,
    types: &mut severian_universal::TypeContext,
    module: ModuleId,
    classes: &[PackageClass],
    lists: &[PackageList],
    index: &ProgramIndex,
) -> Result<severian_universal::Substitution, Diagnostic> {
    let arguments = function
        .type_parameters
        .iter()
        .enumerate()
        .filter(|(index, _)| {
            generic_parameters(&function.type_parameters, &function.constraints)
                .get(*index)
                .is_some_and(|parameter| parameter.kind == GenericParamKind::Type)
        })
        .filter_map(|(index, parameter)| {
            substitution
                .get(parameter)
                .map(|name| (severian_universal::GenericParamId(index as u32), name))
        })
        .map(|(parameter, name)| {
            let annotation = TypeAnnotation {
                kind: generic::applied_type_spelling(name, function.result.span),
                span: function.result.span,
            };
            resolve_package_type(types, &annotation, module, classes, lists, index)
                .map(|ty| (parameter, ty))
                .map_err(|diagnostic| {
                    diagnostic.with_note(format!("while specializing `{function_name}` with `{name}`"))
                })
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
        binding.variable.0 += offset;
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
            Statement::Sequence(block) | Statement::Placement { body: block, .. } => {
                remap_block_bindings(block, offset)
            }
            Statement::Binding(binding) => binding.0 += offset,
            Statement::FieldUpdate { binding, value, .. }
            | Statement::FieldSet { binding, value, .. } => {
                binding.0 += offset;
                remap_expression_bindings(value, offset);
            }
            Statement::Expression(expression) | Statement::Destroy(expression) | Statement::Return(Some(expression)) => {
                remap_expression_bindings(expression, offset)
            }
            Statement::Return(None) | Statement::Break { .. } | Statement::Continue { .. } => {}
            Statement::Assert {
                condition, message, ..
            } => {
                remap_expression_bindings(condition, offset);
                if let Some(message) = message {
                    remap_expression_bindings(message, offset);
                }
            }
            Statement::ExpectThrow { body, .. } => {
                remap_block_bindings(body, offset);
            }
            Statement::Try {
                body,
                catch_binding,
                catch_body,
                ..
            } => {
                remap_block_bindings(body, offset);
                catch_binding.0 += offset;
                remap_block_bindings(catch_body, offset);
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
            Statement::While {
                condition, body, ..
            } => {
                remap_expression_bindings(condition, offset);
                remap_block_bindings(body, offset);
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
        ExpressionKind::Binding(binding) | ExpressionKind::AddressOf(binding) => {
            binding.0 += offset
        }
        ExpressionKind::Aggregate { fields, .. } | ExpressionKind::Variant { fields, .. } => {
            for field in fields {
                remap_expression_bindings(field, offset);
            }
        }
        ExpressionKind::Field { object, .. } => remap_expression_bindings(object, offset),
        ExpressionKind::Convert { operand, .. } => {
            remap_expression_bindings(operand, offset);
        }
        ExpressionKind::Call { callee, arguments, .. } => {
            if let severian_hir::Callee::FunctionValue(value) = callee { remap_expression_bindings(value, offset); }
            for argument in arguments {
                remap_expression_bindings(argument, offset);
            }
        }
        ExpressionKind::Async { expression, .. } | ExpressionKind::Await(expression) => {
            remap_expression_bindings(expression, offset)
        }
        ExpressionKind::AsyncFieldUpdate { binding, value, .. } => {
            binding.0 += offset;
            remap_expression_bindings(value, offset);
        }
        ExpressionKind::Fallback {
            condition,
            value,
            fallback,
        } => {
            remap_expression_bindings(condition, offset);
            remap_expression_bindings(value, offset);
            remap_expression_bindings(fallback, offset);
        }
        ExpressionKind::Throw(error) => remap_expression_bindings(error, offset),
        ExpressionKind::Unary { operand, .. }
        | ExpressionKind::Borrow { operand, .. }
        | ExpressionKind::Move(operand) => remap_expression_bindings(operand, offset),
        ExpressionKind::Binary { left, right, .. } => {
            remap_expression_bindings(left, offset);
            remap_expression_bindings(right, offset);
        }
        ExpressionKind::Literal(_) | ExpressionKind::Function(_) => {}
    }
}

#[cfg(test)]
mod scope_resolution_tests {
    use super::*;

    #[test]
    fn conflicting_explicit_imports_are_rejected_even_when_unused() {
        let root = std::env::temp_dir().join(format!("sev-import-conflict-{}-{}", std::process::id(),
            std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos()));
        std::fs::create_dir_all(&root).unwrap();
        std::fs::write(root.join("foo.sev"), "x = 1\ny = 2\n").unwrap();
        std::fs::write(root.join("main.sev"), "import x from \"foo.sev\"\nfrom \"foo.sev\" import y as x\n").unwrap();
        let graph = severian_modules::resolve(&root.join("main.sev")).unwrap();
        let result = import_index(&graph);
        let planning_error = import_plan(&graph).unwrap_err();
        std::fs::remove_dir_all(&root).unwrap();
        let error = result.unwrap_err();
        assert_eq!(error.code, "E000203");
        assert!(error.message.contains("`x`"));
        assert_eq!(error.labels.len(), 2);
        assert!(error.validate().is_ok());
        let rendered = error.to_string();
        assert!(rendered.contains("main.sev:2:"));
        assert!(rendered.contains("main.sev:1:"));
        assert!(rendered.contains("previous import introduces this name"));
        assert!(rendered.contains("stage: import resolution"));
        assert!(rendered.contains("help: remove one import of `x`"));
        assert!(planning_error.validate().is_ok());
        assert!(planning_error.to_string().contains("main.sev:2:"));
        assert!(planning_error.to_string().contains("stage: import planning"));
    }
}

#[cfg(test)]
mod export_visibility_tests {
    use super::*;

    #[test]
    fn leading_underscores_define_import_access() {
        assert_eq!(Visibility::for_name("foo"), Visibility::Public);
        assert_eq!(Visibility::for_name("_foo"), Visibility::Explicit);
        assert_eq!(Visibility::for_name("__foo"), Visibility::File);
    }

    #[test]
    fn star_skips_explicit_exports_and_private_imports_are_rejected() {
        let root = std::env::temp_dir().join(format!("sev-export-{}-{}", std::process::id(), std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos()));
        std::fs::create_dir_all(&root).unwrap();
        std::fs::write(root.join("foo.sev"), "def foo() -> int:\n    return 1\ndef _foo() -> int:\n    return 2\ndef __foo() -> int:\n    return 3\n").unwrap();
        let entry = root.join("main.sev");
        std::fs::write(&entry, "import * from \"foo.sev\"\n").unwrap();
        let graph = severian_modules::resolve(&entry).unwrap();
        let index = import_index(&graph).unwrap();
        let scope = &index.modules[&graph.modules.last().unwrap().id].scope.bindings;
        assert!(scope.contains_key("foo"));
        assert!(!scope.contains_key("_foo"));
        assert!(!scope.contains_key("__foo"));
        std::fs::write(&entry, "import _foo from \"foo.sev\"\n").unwrap();
        let graph = severian_modules::resolve(&entry).unwrap();
        let index = import_index(&graph).unwrap();
        assert!(index.modules[&graph.modules.last().unwrap().id].scope.bindings.contains_key("_foo"));
        std::fs::write(&entry, "import __foo from \"foo.sev\"\n").unwrap();
        let graph = severian_modules::resolve(&entry).unwrap();
        assert_eq!(import_index(&graph).unwrap_err().code, "E000124");
        std::fs::remove_dir_all(root).unwrap();
    }
}

#[cfg(test)]
mod generic_enum_tests {
    use super::*;

    #[test]
    fn generic_enums_keep_imported_alias_identity_and_payload_types() {
        let root = std::env::temp_dir().join(format!("sev-generic-enum-{}", std::process::id()));
        std::fs::create_dir_all(&root).unwrap();
        std::fs::write(root.join("types.sev"), "enum Choice[T, B]:\n    Value(value: T)\n    Block(block: B)\ndef make() -> Choice[int, string]:\n    return Choice[int, string].Value(3)\n").unwrap();
        std::fs::write(root.join("main.sev"), "import Choice as Pick from \"types.sev\"\ndef choose() -> Pick[int, string]:\n    return Pick[int, string].Block(\"body\")\n").unwrap();
        let graph = severian_modules::resolve(&root.join("main.sev")).unwrap();
        let typed = analyze_package(&graph, &severian_bootstrap::load().unwrap()).unwrap();
        let results = typed.hir.modules.iter().flat_map(|module| &module.functions)
            .filter(|function| matches!(function.name.as_str(), "choose" | "make"))
            .map(|function| function.result.ty).collect::<BTreeSet<_>>();
        assert_eq!(results.len(), 1);
        severian_mir::build(&typed.hir).unwrap();
        std::fs::remove_dir_all(root).unwrap();
    }
}

#[cfg(test)]
mod declaration_alias_tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};
    static NEXT: AtomicUsize = AtomicUsize::new(0);

    fn graph(files: &[(&str, &str)]) -> (std::path::PathBuf, ModuleGraph) {
        let root = std::env::temp_dir().join(format!("sev-alias-{}-{}", std::process::id(), NEXT.fetch_add(1, Ordering::Relaxed)));
        std::fs::create_dir_all(&root).unwrap();
        for (name, text) in files { std::fs::write(root.join(name), text).unwrap(); }
        let graph = severian_modules::resolve(&root.join(files[0].0)).unwrap();
        (root, graph)
    }

    #[test]
    fn declaration_alias_preserves_local_type_constructor_and_method() {
        let (root, graph) = graph(&[("main.sev", "class DefinitionId:\n    value: int\n    def read() -> int:\n        return value\nDefinitionId as DefinitionID\nDefinitionID as Identity\nclass Holder:\n    definition: Identity\ndef selected() -> int:\n    identity = DefinitionID(7)\n    holder = Holder(identity)\n    selected = holder.definition\n    return selected.read()\n")]);
        let index = import_index(&graph).unwrap();
        let names = &index.modules[&graph.modules[0].id].scope.bindings;
        assert_eq!(names["DefinitionId"], names["DefinitionID"]);
        assert_eq!(names["DefinitionId"], names["Identity"]);
        let typed = analyze_package(&graph, &severian_bootstrap::load().unwrap()).unwrap();
        assert_eq!(typed.hir.modules.iter().flat_map(|module| &module.classes).filter(|class| class.name == "DefinitionId").count(), 1);
        severian_mir::build(&typed.hir).unwrap();
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn declaration_alias_reexports_functions_and_generic_constructors() {
        let (root, graph) = graph(&[
            ("main.sev", "import * from \"facade.sev\" as api\ndef selected() -> int:\n    container = api.Container[int](7)\n    value = container.value\n    return api.apply(value)\n"),
            ("facade.sev", "import * from \"origin.sev\"\nBox as Container\noperation as apply\n"),
            ("origin.sev", "class Box[T]:\n    value: T\ndef operation(value: int) -> int:\n    return value + 1\n"),
        ]);
        let typed = analyze_package(&graph, &severian_bootstrap::load().unwrap()).unwrap();
        severian_mir::build(&typed.hir).unwrap();
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn declaration_alias_preserves_local_function_identity() {
        let (root, graph) = graph(&[("main.sev", "def operation(value: int) -> int:\n    return value + 1\noperation as apply\ndef selected() -> int:\n    return apply(7)\n")]);
        let typed = analyze_package(&graph, &severian_bootstrap::load().unwrap()).unwrap();
        assert_eq!(typed.hir.modules.iter().flat_map(|module| &module.functions).filter(|function| function.name == "operation").count(), 1);
        severian_mir::build(&typed.hir).unwrap();
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn applied_alias_resolves_imported_cfg_field_and_constructor() {
        let (root, graph) = graph(&[
            ("main.sev", "import CfgTerminator from \"aliases.sev\"\nclass Block:\n    terminator: CfgTerminator | None = None\ndef make() -> CfgTerminator:\n    return CfgTerminator(7, \"grammar\", true)\ndef read(value: CfgTerminator) -> int:\n    return value.exit\n"),
            ("aliases.sev", "import \"cfg.sev\" as cfg\ncfg.Terminator[int, string, bool] as CfgTerminator\n"),
            ("cfg.sev", "class Terminator[V, G, Span]:\n    exit: V\n    grammar: G\n    span: Span\n"),
        ]);
        let typed = analyze_package(&graph, &severian_bootstrap::load().unwrap()).unwrap();
        let make = typed.hir.modules.iter().flat_map(|module| &module.functions).find(|function| function.name == "make").unwrap();
        let read = typed.hir.modules.iter().flat_map(|module| &module.functions).find(|function| function.name == "read").unwrap();
        assert_eq!(make.result.ty, read.parameters[0].contract.ty);
        severian_mir::build(&typed.hir).unwrap();
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn applied_alias_arguments_keep_their_declaring_scope_through_reexports() {
        let (root, graph) = graph(&[
            ("main.sev", "import Again from \"facade.sev\"\nclass ValueId:\n    value: string\ndef read(value: Again) -> int:\n    field = value.item\n    return field.value\n"),
            ("facade.sev", "import Wrapped from \"aliases.sev\"\nWrapped as Again\n"),
            ("aliases.sev", "import \"provider.sev\" as cfg\nimport ValueId from \"provider.sev\"\ncfg.Box[ValueId] as Wrapped\n"),
            ("provider.sev", "class ValueId:\n    value: int\nclass Box[T]:\n    item: T\n"),
        ]);
        let typed = analyze_package(&graph, &severian_bootstrap::load().unwrap()).unwrap();
        severian_mir::build(&typed.hir).unwrap();
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn applied_alias_preserves_cfg_enum_variants_in_construction_and_matching() {
        let (root, graph) = graph(&[
            ("main.sev", "import CfgExit, CfgTerminator from \"aliases.sev\"\ndef make() -> CfgTerminator:\n    exit = CfgExit.Finish(7)\n    return CfgTerminator(exit, \"grammar\", true)\ndef read(exit: CfgExit) -> int:\n    match exit:\n        case Finish:\n            return value\n        case Dead:\n            return 0\n"),
            ("aliases.sev", "import \"cfg.sev\" as cfg\ncfg.Exit[int] as CfgExit\ncfg.Terminator[int, string, bool] as CfgTerminator\n"),
            ("cfg.sev", "enum Exit[V]:\n    Finish(value: V)\n    Dead\nclass Terminator[V, G, Span]:\n    exit: Exit[V]\n    grammar: G\n    span: Span\n"),
        ]);
        let typed = analyze_package(&graph, &severian_bootstrap::load().unwrap()).unwrap();
        severian_mir::build(&typed.hir).unwrap();
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn applied_alias_parameters_and_cycle_diagnostics() {
        let (root, graph) = graph(&[("main.sev", "class Box[T]:\n    value: T\nBox[T] as Wrapped[T]\ndef make() -> Wrapped[int]:\n    return Wrapped[int](7)\n")]);
        let typed = analyze_package(&graph, &severian_bootstrap::load().unwrap()).unwrap();
        severian_mir::build(&typed.hir).unwrap();
        std::fs::remove_dir_all(root).unwrap();
        let (root, graph) = self::graph(&[("main.sev", "class Box[T]:\n    value: T\nBox[Cycle] as Cycle\ndef bad(value: Cycle):\n    return\n")]);
        let error = analyze_package(&graph, &severian_bootstrap::load().unwrap()).unwrap_err();
        assert!(error.message.contains("cyclic type alias"));
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn primitive_union_keeps_primitive_identity_and_flattens_families() {
        let (root, graph) = graph(&[("main.sev", "trait Integer:\n    bits: u16\nclass i8: Integer :\n    bits: u16 = 8\nclass u8: Integer :\n    bits: u16 = 8\nunion signed_int:\n    i8\n    i16\nunion unsigned_int:\n    u8\nunion int:\n    signed_int\n    unsigned_int\n    i8\nint as Number\ndef keep(value: Number) -> int:\n    return value\n")]);
        let index = import_index(&graph).unwrap();
        let module = graph.modules[0].id;
        let span = item_span(&graph.modules[0].ast.items[0]);
        let annotation = TypeAnnotation::named("int", Vec::new(), span);
        let expanded = expand_type_alias(&annotation, module, &index).unwrap();
        let TypeAnnotationKind::Union(members) = expanded.kind else { panic!("expected primitive union") };
        assert_eq!(members.iter().map(|member| member.simple_name().unwrap()).collect::<Vec<_>>(), ["i8", "i16", "u8"]);
        let universal = severian_bootstrap::load().unwrap();
        let typed = analyze_package(&graph, &universal).unwrap();
        let function = typed.hir.modules.iter().flat_map(|module| &module.functions).find(|function| function.name == "keep").unwrap();
        assert_eq!(Some(function.parameters[0].contract.ty), universal.types.resolve_name("int"));
        assert_eq!(function.result.ty, function.parameters[0].contract.ty);
        severian_mir::build(&typed.hir).unwrap();
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn primitive_union_bound_accepts_only_listed_types() {
        let family = "trait Integer:\n    bits: u16\nclass i8: Integer :\n    bits: u16 = 8\nclass u8: Integer :\n    bits: u16 = 8\nclass Custom: Integer\n    bits: u16\nunion signed_int:\n    i8\n    i16\ndef keep[T: signed_int](value: T) -> T:\n    return value\n";
        for (name, accepted) in [("i8", true), ("u8", false), ("Custom", false)] {
            let text = format!("{family}def selected(value: {name}) -> {name}:\n    return keep(value)\n");
            let (root, graph) = graph(&[("main.sev", &text)]);
            let result = analyze_package(&graph, &severian_bootstrap::load().unwrap());
            if accepted {
                let typed = result.unwrap();
                severian_mir::build(&typed.hir).unwrap();
            } else {
                let error = result.unwrap_err();
                assert_eq!(error.code, "E000217", "{error:?}");
                assert!(error.message.contains("does not satisfy `signed_int`"));
            }
            std::fs::remove_dir_all(root).unwrap();
        }
    }

    #[test]
    fn declaration_alias_cycles_report_the_alias() {
        let (root, graph) = graph(&[("main.sev", "B as A\nA as B\ndef main():\n    return\n")]);
        let error = import_index(&graph).unwrap_err();
        assert!(error.message.contains("cyclic declaration alias"));
        std::fs::remove_dir_all(root).unwrap();
    }
}
