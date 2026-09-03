use crate::{
    analyze_with_package_functions, AnalysisContext, AnalysisMode, PackageClass, PackageConstant,
    PackageFunction, PackageList,
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
    pub parameter_names: Vec<String>,
    pub parameters: Vec<TypeAnnotation>,
    pub parameter_variadics: Vec<bool>,
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
                    Some("Dim") => Some(GenericParamKind::Dimension),
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
    pub constraints: Vec<GenericConstraint>,
    pub bases: Vec<TypeAnnotation>,
    pub properties: Vec<severian_ast::PropertyDeclaration>,
    pub methods: Vec<severian_ast::FunctionDeclaration>,
    pub operators: Vec<severian_ast::OperatorDeclaration>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ClassDecl {
    pub type_parameters: Vec<String>,
    pub fields: Vec<severian_ast::PropertyDeclaration>,
    pub constructors: Vec<severian_ast::FunctionDeclaration>,
    pub methods: Vec<severian_ast::FunctionDeclaration>,
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
    pub packages: BTreeMap<PackageId, Vec<ModuleId>>,
    pub modules: BTreeMap<ModuleId, ModuleScope>,
    pub definitions: BTreeMap<DefId, Definition>,
    pub exports: BTreeMap<ModuleId, ExportMap>,
    pub methods: BTreeMap<String, Vec<MethodDecl>>,
    pub fields: BTreeMap<String, Vec<FieldDecl>>,
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

pub fn analyze_package_with_context(
    module_graph: &ModuleGraph,
    universal: &UniversalContext,
    context: PackageAnalysisContext,
) -> Result<TypedProgram, Diagnostic> {
    let lowered_module_graph = lower_extensions(module_graph)?;
    let lowered_module_graph = lower_trait_typed_parameters(&lowered_module_graph);
    let module_graph = &lowered_module_graph;
    let mut types = universal.types.clone();
    let mut index = collect_declarations(module_graph)?;
    resolve_imports(module_graph, &mut index);
    let package_classes = collect_package_classes(module_graph, &index, &mut types)?;
    install_primitive_class_operators(&mut types, &package_classes)?;
    validate_generic_bodies(module_graph, &index, &types)?;
    let specializations = collect_generic_specializations(module_graph, &index, &types)?;
    let package_lists = collect_package_lists(module_graph, &types);
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
        // Class methods retain the lexical module in which they were declared,
        // even when a downstream package is the first caller that makes the
        // method body reachable. Install those origin-module callables before
        // analyzing the downstream body; otherwise `codec.write()` can retain
        // its source body while losing bindings such as `audio.write_wav`.
        let lexical_class_modules = class_lexical_modules(source_module.id, &package_classes);
        for origin in &lexical_class_modules {
            if *origin != source_module.id {
                visible.extend(module_function_bindings(*origin, &index, &specializations));
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
        package_constants.sort_by(|left, right| left.lookup.cmp(&right.lookup));
        package_constants.dedup_by(|left, right| left.lookup == right.lookup);
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
                    &types,
                    definition.module,
                    &package_classes,
                )?;
                Ok(PackageFunction {
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
                            )
                        })
                        .collect::<Result<Vec<_>, _>>()?,
                    result: resolve_package_type(
                        &mut types,
                        &signature.result,
                        definition.module,
                        &package_classes,
                        &package_lists,
                    )?,
                    result_union: resolve_package_union_members(
                        &mut types,
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
            &mut types,
            AnalysisContext {
                mode,
                module_name: &module_name,
            },
            &visible,
            &own_function_ids,
            &test_function_ids,
            &package_classes,
            &package_lists,
            &package_constants,
            Some(source_module.id),
            Some(&registry_ast),
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

    Ok(TypedProgram { index, hir, types })
}

fn lower_extensions(module_graph: &ModuleGraph) -> Result<ModuleGraph, Diagnostic> {
    let mut lowered = module_graph.clone();
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
                Item::Enum(declaration) => {
                    let mut fields = vec![severian_ast::PropertyDeclaration {
                        name: "__tag".into(),
                        annotation: TypeAnnotation::named("int", Vec::new(), declaration.span),
                        default: None,
                        constraints: Vec::new(),
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
                            primitive: false,
                            type_parameters: Vec::new(),
                            constraints: Vec::new(),
                            traits: Vec::new(),
                            fields,
                            constructors: Vec::new(),
                            methods: Vec::new(),
                            operators: Vec::new(),
                            tests: Vec::new(),
                            span: declaration.span,
                        },
                    ))
                }
                _ => None,
            })
        })
        .map(|(module, declaration)| {
            let ty = if declaration.primitive {
                if !declaration.type_parameters.is_empty() {
                    return Err(Diagnostic::new(
                        "E000204",
                        "a primitive declaration cannot have type parameters",
                        Some(declaration.span),
                    ));
                }
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

fn install_primitive_class_operators(
    types: &mut severian_universal::TypeContext,
    classes: &[PackageClass],
) -> Result<(), Diagnostic> {
    for class in classes.iter().filter(|class| class.declaration.primitive) {
        for implementation in &class.declaration.operators {
            // Generic source operators are resolved at their concrete use
            // sites; they cannot be installed as an exact universal
            // signature before their type parameters are substituted.
            if !implementation.type_parameters.is_empty() {
                continue;
            }
            let result =
                resolve_package_type(types, &implementation.result, class.module, classes, &[])?;
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
                    let right =
                        resolve_package_type(types, &right.annotation, class.module, classes, &[])?;
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
            Resolution::Module(target) if *target == class.module => {
                if index
                    .exports
                    .get(target)
                    .and_then(|exports| exports.get(&class.declaration.name))
                    .is_some_and(&matches_class)
                {
                    names.push(format!("{binding}.{}", class.declaration.name));
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
            modules.extend(selected.iter().map(|index| classes[*index].module));
            return modules;
        }
    }
}

fn resolve_package_type(
    types: &mut severian_universal::TypeContext,
    annotation: &TypeAnnotation,
    module: ModuleId,
    classes: &[PackageClass],
    lists: &[PackageList],
) -> Result<TypeId, Diagnostic> {
    if let Some(("borrowed" | "owned" | "transferred" | "out" | "inout" | "nullable", [inner])) =
        annotation.named_parts()
    {
        return resolve_package_type(types, inner, module, classes, lists);
    }
    if annotation.simple_name() == Some("Any") {
        return Ok(crate::any_type_id());
    }
    if let TypeAnnotationKind::Function { parameters, result } = &annotation.kind {
        let parameters = parameters
            .iter()
            .map(|parameter| resolve_package_type(types, parameter, module, classes, lists))
            .collect::<Result<Vec<_>, _>>()?;
        let result = resolve_package_type(types, result, module, classes, lists)?;
        return Ok(crate::function_type_id(&parameters, result));
    }
    if let severian_ast::TypeAnnotationKind::Union(members) = &annotation.kind {
        let mut success = Vec::new();
        let mut errors = Vec::new();
        for member in members {
            if matches!(member.simple_name(), Some("None" | "absent")) {
                continue;
            }
            let ty = resolve_package_type(types, member, module, classes, lists)?;
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
        let success = resolve_package_type(types, success, module, classes, lists)?;
        let error_ty = resolve_package_type(types, error, module, classes, lists)?;
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
            .map(|element| resolve_package_type(types, element, module, classes, lists))
            .collect::<Result<Vec<_>, _>>()?;
        return Ok(crate::tuple_type_id(&elements));
    }
    if let Some(("list", [element])) = annotation.named_parts() {
        let element = resolve_package_type(types, element, module, classes, lists)?;
        if let Some(list) = lists.iter().find(|list| list.element == element) {
            return Ok(list.ty);
        }
        return Ok(crate::list_type_id(element));
    }
    if let Some(("set", [element])) = annotation.named_parts() {
        // Resolve the argument here so unknown element types still fail at the
        // package boundary. Sets currently share one representation identity.
        resolve_package_type(types, element, module, classes, lists)?;
        return Ok(crate::set_type_id());
    }
    if let Some(("map", [key, value])) = annotation.named_parts() {
        let key = resolve_package_type(types, key, module, classes, lists)?;
        let value = resolve_package_type(types, value, module, classes, lists)?;
        return Ok(crate::map_type_id(key, value));
    }
    if let Some((name, arguments)) = annotation.named_parts() {
        if !arguments.is_empty() {
            if let Some(class) = package_class_for_lookup(classes, module, name) {
                if class.declaration.name == "Tensor" {
                    let element =
                        resolve_package_type(types, &arguments[0], module, classes, lists)?;
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
                    .map(|argument| resolve_package_type(types, argument, module, classes, lists))
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
    }
    crate::resolve_type_annotation(types, annotation)
}

fn resolve_package_union_members(
    types: &mut severian_universal::TypeContext,
    annotation: &TypeAnnotation,
    module: ModuleId,
    classes: &[PackageClass],
    lists: &[PackageList],
) -> Result<Option<Vec<TypeId>>, Diagnostic> {
    let severian_ast::TypeAnnotationKind::Union(members) = &annotation.kind else {
        return Ok(None);
    };
    let mut resolved = Vec::new();
    for member in members {
        if matches!(member.simple_name(), Some("None" | "absent")) {
            continue;
        }
        let ty = resolve_package_type(types, member, module, classes, lists)?;
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
            return Ok(None);
        }
        resolved.push(ty);
    }
    resolved.sort();
    resolved.dedup();
    Ok((resolved.len() > 1).then_some(resolved))
}

fn package_class_for_lookup<'a>(
    classes: &'a [PackageClass],
    module: ModuleId,
    name: &str,
) -> Option<&'a PackageClass> {
    classes.iter().find(|class| {
        class
            .lookups
            .get(&module)
            .is_some_and(|lookups| lookups.iter().any(|lookup| lookup == name))
    })
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
        let Some(binding) = source
            .ast
            .items
            .iter()
            .find_map(|candidate| match candidate {
                Item::Binding(binding) if binding.name == item.name => Some(binding),
                _ => None,
            })
        else {
            return;
        };
        constants.push(PackageConstant {
            lookup,
            value: binding.value.clone(),
        });
    };
    for (name, resolution) in &scope.bindings {
        match resolution {
            Resolution::Module(target) => {
                if let Some(exports) = index.exports.get(target) {
                    for (export, resolution) in exports {
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
    constants.sort_by(|left, right| left.lookup.cmp(&right.lookup));
    constants.dedup_by(|left, right| left.lookup == right.lookup);
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
    constants.extend(source.ast.items.iter().filter_map(|item| {
        let Item::Binding(binding) = item else {
            return None;
        };
        Some(PackageConstant {
            lookup: binding.name.clone(),
            value: binding.value.clone(),
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
    let source = match item {
        Item::Trait(declaration) => declaration.span.source,
        Item::Class(declaration) => declaration.span.source,
        Item::Enum(declaration) => declaration.span.source,
        Item::Binding(binding) => binding.span.source,
        Item::Expression(expression) => expression.span.source,
        Item::Function(function) => function.span.source,
        Item::Type(declaration) => declaration.span.source,
        Item::Test(declaration) => declaration.span.source,
        Item::Import(import) => import.span.source,
        Item::Extension(extension) => extension.span.source,
    };
    source.0 >= u32::MAX - 3
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
                            result: function.result.clone(),
                            constraints: function.constraints.clone(),
                            generic_body: function.body.clone(),
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
                    DefKind::Class(ClassDecl {
                        type_parameters: declaration.type_parameters.clone(),
                        fields: declaration.fields.clone(),
                        constructors: declaration.constructors.clone(),
                        methods: declaration.methods.clone(),
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
            if !injected_prelude {
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

fn compatible_trait_redeclaration(left: &TraitDecl, right: &TraitDecl) -> bool {
    left.type_parameters == right.type_parameters
        && left.constraints.len() == right.constraints.len()
        && annotations_match(&left.bases, &right.bases)
        && left.properties.len() == right.properties.len()
        && left.methods.len() == right.methods.len()
        && left.operators.len() == right.operators.len()
        && left.methods.iter().zip(&right.methods).all(|(left, right)| {
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
        && left.operators.iter().zip(&right.operators).all(|(left, right)| {
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
    for module in &module_graph.modules {
        for import in module.ast.items.iter().filter_map(|item| match item {
            Item::Import(import) => Some(import),
            _ => None,
        }) {
            let Some(edge) = module.imports.iter().find(|edge| edge.span == import.span) else {
                continue;
            };
            if matches!(import.subject, ImportSubject::Locator(_)) && import.alias.is_none() {
                let members = index.exports.get(&edge.module).cloned().unwrap_or_default();
                for (name, resolution) in members {
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
                "{}{}",
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
        "function:{}:{:032x}:{:032x}[{arguments}]",
        definition.package, definition.module, definition.declaration.0,
    )))
}

fn universal_substitution(
    function_name: &str,
    function: &FunctionDecl,
    substitution: &GenericSubstitution,
    types: &severian_universal::TypeContext,
    module: ModuleId,
    classes: &[PackageClass],
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
            types
                .resolve_name(name)
                .or_else(|| (name == "Any").then_some(crate::any_type_id()))
                .or_else(|| {
                    classes
                        .iter()
                        .find(|class| {
                            class.module == module
                                && class.declaration.name == (*name).as_str()
                        })
                        .or_else(|| {
                            classes
                                .iter()
                                .find(|class| class.declaration.name == (*name).as_str())
                        })
                        .map(|class| class.ty)
                })
                .map(|ty| (parameter, ty))
                .ok_or_else(|| {
                    Diagnostic::new(
                        "E000204",
                        format!(
                            "cannot specialize `{function_name}` because inferred type `{name}` is unresolved"
                        ),
                        None,
                    )
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
            Statement::Expression(expression) | Statement::Return(Some(expression)) => {
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
