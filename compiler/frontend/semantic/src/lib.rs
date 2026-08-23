#![forbid(unsafe_code)]

mod package;
mod queries;

pub use package::{
    analyze_package, analyze_package_with_context, DefKind, Definition, ExportMap, FunctionDecl,
    ModuleScope, PackageAnalysisContext, ProgramIndex, Resolution, Scope, SignatureId, TraitDecl,
    TypedProgram, Visibility,
};
pub use queries::{QueryError, ScopeId, SemanticQueries};
pub use severian_universal::{DeclarationId, DefId};

use severian_ast::{
    BinaryOperator as AstBinaryOperator, Expression as AstExpression,
    ExpressionKind as AstExpressionKind, Literal as AstLiteral, Statement as AstStatement,
    TypeAnnotation, UnaryOperator as AstUnaryOperator,
};
use severian_diagnostics::Diagnostic;
use severian_hir::{
    Binding, BindingId, Block, BoundaryType, CallType, ClassDeclaration as HirClassDeclaration,
    ClassFieldDeclaration as HirClassFieldDeclaration, Expression, ExpressionKind,
    FunctionDeclaration, FunctionId, FunctionParameter, HirId, Module, Program, Statement,
    TraitDeclaration as HirTraitDeclaration, TraitMethodDeclaration as HirTraitMethodDeclaration,
    TraitType as HirTraitType, TypeId,
};
use severian_universal::{
    BinaryOperator, LiteralValue, TypeConstraint, TypeContext, UnaryOperator,
};
use std::collections::{BTreeMap, BTreeSet};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AnalysisMode {
    Build,
    Test,
}

#[derive(Debug, Clone, Copy)]
pub struct AnalysisContext<'a> {
    pub mode: AnalysisMode,
    pub module_name: &'a str,
}

pub fn analyze(ast: &severian_ast::Module, types: &TypeContext) -> Result<Program, Diagnostic> {
    analyze_with_context(
        ast,
        types,
        AnalysisContext {
            mode: AnalysisMode::Build,
            module_name: "memory",
        },
    )
}

pub fn analyze_with_context(
    ast: &severian_ast::Module,
    types: &TypeContext,
    context: AnalysisContext<'_>,
) -> Result<Program, Diagnostic> {
    analyze_with_package_functions(ast, types, context, &[], &[], &[], &[], &[], None)
}

#[derive(Debug, Clone)]
pub(crate) struct PackageFunction {
    pub lookup: String,
    pub id: FunctionId,
    pub definition: DefId,
    pub substitution: severian_universal::Substitution,
    pub type_parameters: Vec<severian_universal::GenericParamId>,
    pub parameters: Vec<TypeId>,
    pub result: TypeId,
    pub specificity: u8,
}

#[derive(Debug, Clone)]
pub(crate) struct PackageClass {
    pub module: severian_modules::ModuleId,
    pub ty: TypeId,
    pub declaration: severian_ast::ClassDeclaration,
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct PackageList {
    pub module: severian_modules::ModuleId,
    pub ty: TypeId,
    pub element: TypeId,
}

pub(crate) fn analyze_with_package_functions(
    ast: &severian_ast::Module,
    types: &TypeContext,
    context: AnalysisContext<'_>,
    visible_functions: &[PackageFunction],
    own_function_ids: &[FunctionId],
    test_function_ids: &[FunctionId],
    package_classes: &[PackageClass],
    package_lists: &[PackageList],
    source_module: Option<severian_modules::ModuleId>,
) -> Result<Program, Diagnostic> {
    validate_trait_implementations(ast)?;
    validate_class_declarations(ast)?;
    let mut analyzer = Analyzer {
        types,
        names: BTreeMap::new(),
        value_substitutions: BTreeMap::new(),
        declarations: BTreeSet::new(),
        functions: BTreeMap::new(),
        function_definitions: BTreeMap::new(),
        function_substitutions: BTreeMap::new(),
        function_specificity: BTreeMap::new(),
        signatures: BTreeMap::new(),
        classes: ast
            .items
            .iter()
            .filter_map(|item| match item {
                severian_ast::Item::Class(declaration) => {
                    Some((declaration.name.clone(), declaration.clone()))
                }
                _ => None,
            })
            .collect(),
        enums: BTreeMap::new(),
        enum_variants: BTreeMap::new(),
        class_instances: BTreeMap::new(),
        class_instances_by_type: BTreeMap::new(),
        list_types: BTreeMap::new(),
        list_elements: BTreeMap::new(),
        tuple_types: BTreeMap::new(),
        tuple_elements: BTreeMap::new(),
        lowered_classes: Vec::new(),
        runtime_functions: Vec::new(),
        runtime_definitions: BTreeMap::new(),
        next_hir: 0,
        next_binding: 0,
        next_class_type: u32::MAX,
    };
    analyzer.install_package_types(package_classes, package_lists, source_module)?;
    analyzer.install_enums(ast)?;
    for function in visible_functions {
        analyzer
            .functions
            .entry(function.lookup.clone())
            .or_default()
            .push(function.id);
        analyzer.signatures.insert(
            function.id,
            FunctionSignature {
                parameters: function
                    .parameters
                    .iter()
                    .copied()
                    .map(|type_id| SignatureParameter {
                        name: String::new(),
                        type_id,
                        default: None,
                    })
                    .collect(),
                result: function.result,
            },
        );
        analyzer
            .function_definitions
            .insert(function.id, function.definition);
        analyzer
            .function_substitutions
            .insert(function.id, function.substitution.clone());
        analyzer
            .function_specificity
            .insert(function.id, function.specificity);
    }
    let mut module = Module::default();

    for declaration in ast.items.iter().filter_map(|item| match item {
        severian_ast::Item::Trait(declaration) => Some(declaration),
        _ => None,
    }) {
        let methods = declaration
            .methods
            .iter()
            .map(|method| {
                Ok(HirTraitMethodDeclaration {
                    name: method.name.clone(),
                    parameters: method
                        .parameters
                        .iter()
                        .map(|parameter| resolve_trait_type(types, &parameter.annotation))
                        .collect::<Result<Vec<_>, Diagnostic>>()?,
                    result: resolve_trait_type(types, &method.result)?,
                })
            })
            .collect::<Result<Vec<_>, Diagnostic>>()?;
        module.traits.push(HirTraitDeclaration {
            definition: synthetic_trait_definition(context.module_name, &declaration.name),
            name: declaration.name.clone(),
            methods,
        });
    }

    // Function identities and signatures are registered before executable
    // statements are analyzed. Bodies remain ordinary analyzed blocks.
    for (ordinal, ast_function) in ast
        .items
        .iter()
        .filter_map(|item| match item {
            severian_ast::Item::Function(function) => Some(function),
            _ => None,
        })
        .enumerate()
    {
        let id = own_function_ids
            .get(ordinal)
            .copied()
            .unwrap_or(FunctionId(module.functions.len() as u128));
        let package_function = visible_functions.iter().find(|function| function.id == id);
        let overload_ordinal = ast
            .items
            .iter()
            .filter_map(|item| match item {
                severian_ast::Item::Function(function)
                    if function.name == ast_function.name
                        && function.span.start < ast_function.span.start =>
                {
                    Some(())
                }
                _ => None,
            })
            .count();
        let definition = package_function.map_or_else(
            || synthetic_definition(context.module_name, ast_function, overload_ordinal),
            |function| function.definition,
        );
        let mut parameters = Vec::new();
        let mut parameter_types = Vec::new();
        for parameter in &ast_function.parameters {
            let type_id = analyzer.resolve_source_type(&parameter.annotation)?;
            let binding = analyzer.new_binding_id();
            parameter_types.push(type_id);
            parameters.push(FunctionParameter {
                binding,
                name: parameter.name.clone(),
                contract: universal_boundary(type_id),
            });
        }
        let result = analyzer.resolve_source_type(&ast_function.result)?;
        let compile_route = if types.definition(result).is_some() {
            types
                .compile_route(result)
                .map_err(|error| semantic_error(error.to_string(), ast_function.result.span))?
        } else {
            severian_universal::CompileRoute::Standard
        };
        if own_function_ids.is_empty() {
            analyzer
                .functions
                .entry(ast_function.name.clone())
                .or_default()
                .push(id);
            analyzer.signatures.insert(
                id,
                FunctionSignature {
                    parameters: ast_function
                        .parameters
                        .iter()
                        .zip(parameter_types.iter().copied())
                        .map(|(parameter, type_id)| SignatureParameter {
                            name: parameter.name.clone(),
                            type_id,
                            default: parameter.default.clone(),
                        })
                        .collect(),
                    result,
                },
            );
            analyzer.function_definitions.insert(id, definition);
            analyzer
                .function_substitutions
                .insert(id, severian_universal::Substitution::default());
            analyzer.function_specificity.insert(id, 0);
        }
        module.functions.push(FunctionDeclaration {
            id,
            definition,
            substitution: package_function
                .map(|function| function.substitution.clone())
                .unwrap_or_default(),
            name: ast_function.name.clone(),
            type_parameters: package_function
                .map(|function| function.type_parameters.clone())
                .unwrap_or_default(),
            parameters,
            result: universal_boundary(result),
            compile_route,
            call_type: CallType::Severian,
            body: None,
        });
    }
    let source_function_count = module.functions.len();
    if context.mode == AnalysisMode::Test {
        for (index, test) in ast
            .items
            .iter()
            .filter_map(|item| match item {
                severian_ast::Item::Test(test) => Some(test),
                _ => None,
            })
            .enumerate()
        {
            let id = test_function_ids
                .get(index)
                .copied()
                .unwrap_or(FunctionId(module.functions.len() as u128));
            let unit = types.resolve_name("unit").expect("bootstrap defines unit");
            let mode_suffix = if test.modes.is_empty() {
                String::new()
            } else {
                format!("_with_{}", test.modes.join("-"))
            };
            let name_suffix = test
                .name
                .as_deref()
                .map(internal_name_part)
                .filter(|name| !name.is_empty())
                .map(|name| format!("_{name}"))
                .unwrap_or_default();
            module.functions.push(FunctionDeclaration {
                id,
                definition: synthetic_test_definition(context.module_name, index),
                substitution: severian_universal::Substitution::default(),
                name: format!(
                    "__sev_{}_test{mode_suffix}{name_suffix}_{index}",
                    context.module_name
                ),
                parameters: Vec::new(),
                type_parameters: Vec::new(),
                result: universal_boundary(unit),
                compile_route: severian_universal::CompileRoute::Standard,
                call_type: CallType::Severian,
                body: None,
            });
            let mut modes = test
                .modes
                .iter()
                .map(|mode| test_mode(mode, test.span))
                .collect::<Result<Vec<_>, _>>()?;
            if modes.is_empty()
                && test
                    .body
                    .iter()
                    .any(|statement| integration_expectation(statement).is_some())
            {
                modes.push(severian_hir::TestMode::Integration);
            }
            module.tests.push(severian_hir::TestDeclaration {
                name: test
                    .name
                    .clone()
                    .unwrap_or_else(|| format!("test {}", index + 1)),
                modes,
                function: id,
                expectations: Vec::new(),
            });
        }
    }

    for item in &ast.items {
        match item {
            severian_ast::Item::Binding(binding) => {
                let id = analyzer.binding(binding, &mut module.bindings)?;
                module.initializer.statements.push(Statement::Binding(id));
            }
            severian_ast::Item::Expression(expression) => {
                module.initializer.statements.push(Statement::Expression(
                    analyzer.expression(expression, None)?,
                ));
            }
            _ => {}
        }
    }

    let globals = analyzer.names.clone();
    let ast_functions = ast.items.iter().filter_map(|item| match item {
        severian_ast::Item::Function(function) => Some(function),
        _ => None,
    });
    for (ast_function, function) in ast_functions.zip(module.functions.iter_mut()) {
        let Some(ast_body) = &ast_function.body else {
            continue;
        };
        analyzer.names = globals.clone();
        analyzer.declarations.clear();
        for parameter in &function.parameters {
            let type_id = parameter.contract.ty;
            if !analyzer.declarations.insert(parameter.name.clone()) {
                return Err(Diagnostic::new(
                    "E000203",
                    format!("parameter `{}` is declared more than once", parameter.name),
                    Some(ast_function.span),
                ));
            }
            analyzer
                .names
                .insert(parameter.name.clone(), (parameter.binding, type_id));
        }
        let mut body = Block::default();
        let result_type = function.result.ty;
        for statement in ast_body {
            body.statements.push(analyzer.statement(
                statement,
                &mut module.bindings,
                result_type,
            )?);
        }
        if result_type != types.resolve_name("unit").expect("bootstrap defines unit")
            && block_flow(ast_body) == ControlFlow::FallsThrough
        {
            return Err(Diagnostic::new(
                "E000209",
                "not every path in this function returns its declared result",
                Some(ast_function.span),
            ));
        }
        function.body = Some(body);
        if function.name == "main" {
            let arguments_type = types.resolve_name("args").expect("bootstrap defines args");
            let valid_parameters = match function.parameters.as_slice() {
                [] => true,
                [parameter] => parameter.contract.ty == arguments_type,
                _ => false,
            };
            if !valid_parameters {
                return Err(Diagnostic::new(
                    "E000209",
                    "entry must be `main()` or `main(args: args)`",
                    Some(ast_function.span),
                ));
            }
            if module.entry.replace(function.id).is_some() {
                return Err(Diagnostic::new(
                    "E000208",
                    "module defines more than one `main` function",
                    Some(ast_function.span),
                ));
            }
        }
    }
    if context.mode == AnalysisMode::Test {
        for (offset, test) in ast
            .items
            .iter()
            .filter_map(|item| match item {
                severian_ast::Item::Test(test) => Some(test),
                _ => None,
            })
            .enumerate()
        {
            analyzer.names = globals.clone();
            analyzer.declarations.clear();
            let unit = types.resolve_name("unit").expect("bootstrap defines unit");
            let mut body = Block::default();
            if module.tests[offset]
                .modes
                .contains(&severian_hir::TestMode::Compiler)
            {
                if !test.body.is_empty() {
                    return Err(Diagnostic::new(
                        "E000217",
                        "compiler tests currently allow only `accept:` and `reject:` cases; diagnostic assertions are not implemented",
                        Some(test.span),
                    ));
                }
                if test.compiler_cases.is_empty() {
                    return Err(Diagnostic::new(
                        "E000217",
                        "a compiler test requires at least one `accept:` or `reject:` case",
                        Some(test.span),
                    ));
                }
                if test
                    .compiler_cases
                    .iter()
                    .any(|case| case.diagnostic_name.is_some())
                {
                    return Err(Diagnostic::new(
                        "E000217",
                        "named compiler diagnostics are not implemented; use `reject:` without a binding",
                        Some(test.span),
                    ));
                }
                module.functions[source_function_count + offset].body = Some(body);
                continue;
            }
            for statement in &test.body {
                if module.tests[offset]
                    .modes
                    .contains(&severian_hir::TestMode::Integration)
                {
                    if let Some(expectation) = integration_expectation(statement) {
                        module.tests[offset].expectations.push(expectation);
                        continue;
                    }
                }
                body.statements
                    .push(analyzer.statement(statement, &mut module.bindings, unit)?);
            }
            module.functions[source_function_count + offset].body = Some(body);
        }
    }
    module.functions.extend(analyzer.runtime_functions.clone());
    module.classes = analyzer.lowered_classes.clone();
    Ok(Program {
        modules: vec![module],
    })
}

struct Analyzer<'a> {
    types: &'a TypeContext,
    names: BTreeMap<String, (BindingId, TypeId)>,
    value_substitutions: BTreeMap<String, Expression>,
    /// Names declared in the current lexical scope. `names` also contains
    /// readable parent bindings, which may be shadowed by this set.
    declarations: BTreeSet<String>,
    next_hir: u32,
    next_binding: u32,
    functions: BTreeMap<String, Vec<FunctionId>>,
    function_definitions: BTreeMap<FunctionId, DefId>,
    function_substitutions: BTreeMap<FunctionId, severian_universal::Substitution>,
    function_specificity: BTreeMap<FunctionId, u8>,
    signatures: BTreeMap<FunctionId, FunctionSignature>,
    classes: BTreeMap<String, severian_ast::ClassDeclaration>,
    enums: BTreeMap<String, EnumInstance>,
    enum_variants: BTreeMap<String, (String, usize)>,
    class_instances: BTreeMap<(String, Vec<TypeId>), ClassInstance>,
    class_instances_by_type: BTreeMap<TypeId, ClassInstance>,
    list_types: BTreeMap<TypeId, TypeId>,
    list_elements: BTreeMap<TypeId, TypeId>,
    tuple_types: BTreeMap<Vec<TypeId>, TypeId>,
    tuple_elements: BTreeMap<TypeId, Vec<TypeId>>,
    lowered_classes: Vec<HirClassDeclaration>,
    runtime_functions: Vec<FunctionDeclaration>,
    runtime_definitions: BTreeMap<String, DefId>,
    next_class_type: u32,
}

#[derive(Debug, Clone)]
struct ClassInstance {
    ty: TypeId,
    name: String,
    fields: Vec<HirClassFieldDeclaration>,
    constructors: Vec<severian_ast::FunctionDeclaration>,
    methods: Vec<severian_ast::FunctionDeclaration>,
}

#[derive(Debug, Clone)]
struct EnumInstance {
    ty: TypeId,
    name: String,
    fields: Vec<HirClassFieldDeclaration>,
    variants: Vec<severian_ast::EnumVariant>,
}

#[derive(Debug, Clone)]
struct FunctionSignature {
    parameters: Vec<SignatureParameter>,
    result: TypeId,
}

#[derive(Debug, Clone)]
struct SignatureParameter {
    name: String,
    type_id: TypeId,
    default: Option<AstExpression>,
}

enum Prepared {
    Literal(severian_universal::LiteralValue, severian_source::Span),
    Resolved(Expression),
}

impl Prepared {
    fn constraint(&self) -> TypeConstraint {
        match self {
            Self::Literal(value, _) => TypeConstraint::Literal(value.kind()),
            Self::Resolved(expression) => TypeConstraint::Known(expression.type_id),
        }
    }
}

impl Analyzer<'_> {
    fn install_enums(&mut self, ast: &severian_ast::Module) -> Result<(), Diagnostic> {
        let declarations = ast
            .items
            .iter()
            .filter_map(|item| match item {
                severian_ast::Item::Enum(declaration) => Some(declaration),
                _ => None,
            })
            .cloned()
            .collect::<Vec<_>>();
        for declaration in &declarations {
            let ty = self
                .class_instances
                .get(&(declaration.name.clone(), Vec::new()))
                .map(|instance| instance.ty)
                .unwrap_or_else(|| {
                    let ty = TypeId(self.next_class_type);
                    self.next_class_type = self.next_class_type.saturating_sub(1);
                    ty
                });
            let placeholder = ClassInstance {
                ty,
                name: declaration.name.clone(),
                fields: Vec::new(),
                constructors: Vec::new(),
                methods: Vec::new(),
            };
            self.class_instances
                .insert((declaration.name.clone(), Vec::new()), placeholder.clone());
            self.class_instances_by_type.insert(ty, placeholder);
            self.enums.insert(
                declaration.name.clone(),
                EnumInstance {
                    ty,
                    name: declaration.name.clone(),
                    fields: Vec::new(),
                    variants: declaration.variants.clone(),
                },
            );
        }
        let integer = self
            .types
            .resolve_name("int")
            .expect("bootstrap defines int");
        for declaration in declarations {
            let ty = self.enums[&declaration.name].ty;
            let mut fields = vec![HirClassFieldDeclaration {
                name: "__tag".into(),
                ty: integer,
            }];
            for (ordinal, variant) in declaration.variants.iter().enumerate() {
                self.enum_variants
                    .insert(variant.name.clone(), (declaration.name.clone(), ordinal));
                self.enum_variants.insert(
                    format!("{}.{}", declaration.name, variant.name),
                    (declaration.name.clone(), ordinal),
                );
                for field in &variant.fields {
                    let field_type = self.resolve_source_type(&field.annotation)?;
                    if let Some(existing) = fields.iter().find(|known| known.name == field.name) {
                        if existing.ty != field_type {
                            return Err(Diagnostic::new(
                                "E000204",
                                format!(
                                    "enum payload field `{}` has conflicting types",
                                    field.name
                                ),
                                Some(field.span),
                            ));
                        }
                    } else {
                        fields.push(HirClassFieldDeclaration {
                            name: field.name.clone(),
                            ty: field_type,
                        });
                    }
                }
            }
            let instance = self.enums.get_mut(&declaration.name).unwrap();
            instance.fields.clone_from(&fields);
            self.class_instances.insert(
                (declaration.name.clone(), Vec::new()),
                ClassInstance {
                    ty,
                    name: declaration.name.clone(),
                    fields: fields.clone(),
                    constructors: Vec::new(),
                    methods: Vec::new(),
                },
            );
            self.class_instances_by_type.insert(
                ty,
                ClassInstance {
                    ty,
                    name: declaration.name.clone(),
                    fields: fields.clone(),
                    constructors: Vec::new(),
                    methods: Vec::new(),
                },
            );
            self.lowered_classes.retain(|class| class.id != ty);
            self.lowered_classes.push(HirClassDeclaration {
                id: ty,
                name: declaration.name,
                fields,
            });
        }
        Ok(())
    }

    fn install_package_types(
        &mut self,
        classes: &[PackageClass],
        lists: &[PackageList],
        source_module: Option<severian_modules::ModuleId>,
    ) -> Result<(), Diagnostic> {
        let storage = self
            .types
            .resolve_name("string")
            .expect("bootstrap defines pointer-backed string");
        for list in lists {
            self.list_types.insert(list.element, list.ty);
            self.list_elements.insert(list.ty, list.element);
            if source_module == Some(list.module) {
                self.lowered_classes.push(HirClassDeclaration {
                    id: list.ty,
                    name: format!("list[type#{}]", list.element.0),
                    fields: vec![HirClassFieldDeclaration {
                        name: "storage".into(),
                        ty: storage,
                    }],
                });
            }
            self.next_class_type = self.next_class_type.min(list.ty.0.saturating_sub(1));
        }
        for package_class in classes {
            if !package_class.declaration.type_parameters.is_empty() {
                continue;
            }
            let fields = package_class
                .declaration
                .fields
                .iter()
                .map(|field| {
                    Ok(HirClassFieldDeclaration {
                        name: field.name.clone(),
                        ty: self.resolve_source_type(&field.annotation)?,
                    })
                })
                .collect::<Result<Vec<_>, Diagnostic>>()?;
            let instance = ClassInstance {
                ty: package_class.ty,
                name: package_class.declaration.name.clone(),
                fields: fields.clone(),
                constructors: package_class.declaration.constructors.clone(),
                methods: package_class.declaration.methods.clone(),
            };
            self.class_instances_by_type
                .insert(package_class.ty, instance.clone());
            if source_module == Some(package_class.module) {
                self.class_instances.insert(
                    (package_class.declaration.name.clone(), Vec::new()),
                    instance,
                );
                self.lowered_classes.push(HirClassDeclaration {
                    id: package_class.ty,
                    name: package_class.declaration.name.clone(),
                    fields,
                });
            }
            self.next_class_type = self
                .next_class_type
                .min(package_class.ty.0.saturating_sub(1));
        }
        Ok(())
    }

    fn resolve_source_type(&self, annotation: &TypeAnnotation) -> Result<TypeId, Diagnostic> {
        if let Some(("list", [element])) = annotation.named_parts() {
            let element = resolve_type_annotation(self.types, element)?;
            if let Some(list) = self.list_types.get(&element) {
                return Ok(*list);
            }
        }
        if let Some(name) = annotation.simple_name() {
            if let Some(instance) = self.class_instances.get(&(name.to_owned(), Vec::new())) {
                return Ok(instance.ty);
            }
        }
        resolve_type_annotation(self.types, annotation)
    }

    fn new_binding_id(&mut self) -> BindingId {
        let id = BindingId(self.next_binding);
        self.next_binding += 1;
        id
    }

    fn binding(
        &mut self,
        ast_binding: &severian_ast::Binding,
        bindings: &mut Vec<Binding>,
    ) -> Result<BindingId, Diagnostic> {
        let inferred_update = !ast_binding.update
            && ast_binding.annotation.is_none()
            && self.declarations.contains(&ast_binding.name);
        let is_update = ast_binding.update || inferred_update;
        let update_type = if is_update {
            Some(
                self.names
                    .get(&ast_binding.name)
                    .map(|(_, type_id)| *type_id)
                    .ok_or_else(|| {
                        Diagnostic::new(
                            "E000201",
                            format!("cannot update unknown binding `{}`", ast_binding.name),
                            Some(ast_binding.span),
                        )
                    })?,
            )
        } else {
            None
        };
        if !is_update && !self.declarations.insert(ast_binding.name.clone()) {
            return Err(Diagnostic::new(
                "E000203",
                format!("binding `{}` is already defined", ast_binding.name),
                Some(ast_binding.span),
            ));
        }
        let expected = ast_binding
            .annotation
            .as_ref()
            .map(|annotation| resolve_type_annotation(self.types, annotation))
            .transpose()?
            .or(update_type);
        let value = self.expression(&ast_binding.value, expected)?;
        let type_id = expected.unwrap_or(value.type_id);
        if !self.types.assignable(value.type_id, type_id) {
            return Err(Diagnostic::new(
                "E000205",
                "binding value is not assignable to its declared type",
                Some(ast_binding.value.span),
            ));
        }
        let id = self.new_binding_id();
        self.names.insert(ast_binding.name.clone(), (id, type_id));
        bindings.push(Binding {
            id,
            type_id,
            value,
            preserve_error: ast_binding.preserve_error,
            span: ast_binding.span,
        });
        Ok(id)
    }

    fn statement(
        &mut self,
        statement: &AstStatement,
        bindings: &mut Vec<Binding>,
        result_type: TypeId,
    ) -> Result<Statement, Diagnostic> {
        match statement {
            AstStatement::While {
                condition,
                initializer,
                body,
                span,
            } => {
                let Some(initializer) = initializer else {
                    return Err(Diagnostic::new(
                        "E000211",
                        "runtime while loops without an initializer are not lowered yet",
                        Some(*span),
                    ));
                };
                let initial = static_integer(&initializer.value).ok_or_else(|| {
                    Diagnostic::new(
                        "E000211",
                        "while lowering currently requires a literal initializer",
                        Some(*span),
                    )
                })?;
                let mut sequence = Block::default();
                sequence
                    .statements
                    .push(Statement::Binding(self.binding(initializer, bindings)?));
                let mut environment = BTreeMap::from([(initializer.name.clone(), initial)]);
                for _ in 0..10_000 {
                    if static_boolean(condition, &environment) != Some(true) {
                        break;
                    }
                    let (specialized, control) = specialize_loop_body(body, &mut environment);
                    let lowered = self.block(&specialized, bindings, result_type)?;
                    sequence.statements.extend(lowered.statements);
                    if control == StaticLoopControl::Break {
                        break;
                    }
                }
                Ok(Statement::Sequence(sequence))
            }
            AstStatement::For {
                binding,
                iterable,
                body,
                span,
            } => {
                let values = static_range_values(iterable).ok_or_else(|| {
                    Diagnostic::new(
                        "E000211",
                        "for lowering currently requires `range` with literal bounds",
                        Some(*span),
                    )
                })?;
                let mut sequence = Block::default();
                for value in values {
                    let mut environment = BTreeMap::from([(binding.clone(), value)]);
                    let (specialized, control) = specialize_loop_body(body, &mut environment);
                    let value = AstExpression {
                        kind: AstExpressionKind::Literal(AstLiteral::Integer(value.to_string())),
                        span: iterable.span,
                    };
                    let loop_binding = severian_ast::Binding {
                        name: binding.clone(),
                        annotation: None,
                        value,
                        update: self.declarations.contains(binding),
                        preserve_error: false,
                        span: iterable.span,
                    };
                    sequence
                        .statements
                        .push(Statement::Binding(self.binding(&loop_binding, bindings)?));
                    let lowered = self.block(&specialized, bindings, result_type)?;
                    sequence.statements.extend(lowered.statements);
                    if control == StaticLoopControl::Break {
                        break;
                    }
                }
                Ok(Statement::Sequence(sequence))
            }
            AstStatement::Break { span } | AstStatement::Continue { span } => Err(Diagnostic::new(
                "E000211",
                "loop control must be lowered by its enclosing finite loop",
                Some(*span),
            )),
            AstStatement::Binding(binding) => {
                Ok(Statement::Binding(self.binding(binding, bindings)?))
            }
            AstStatement::FieldAssignment {
                object,
                field,
                value,
                span,
            } => {
                let AstExpressionKind::Name(object_name) = &object.kind else {
                    return Err(Diagnostic::new(
                        "E000211",
                        "field assignment requires a named object",
                        Some(*span),
                    ));
                };
                let existing = self.names.get(object_name).copied();
                if existing.is_none() {
                    let candidates = self
                        .class_instances_by_type
                        .values()
                        .filter(|instance| instance.fields.iter().any(|known| known.name == *field))
                        .cloned()
                        .collect::<Vec<_>>();
                    let [instance] = candidates.as_slice() else {
                        return Err(Diagnostic::new(
                            "E000201",
                            format!("unknown binding `{object_name}`"),
                            Some(object.span),
                        ));
                    };
                    let mut fields = Vec::with_capacity(instance.fields.len());
                    for declaration in &instance.fields {
                        if declaration.name == *field {
                            fields.push(self.expression(value, Some(declaration.ty))?);
                        } else {
                            fields.push(self.default_expression(declaration.ty, *span)?);
                        }
                    }
                    let id = self.new_binding_id();
                    self.names.insert(object_name.clone(), (id, instance.ty));
                    self.declarations.insert(object_name.clone());
                    bindings.push(Binding {
                        id,
                        type_id: instance.ty,
                        value: Expression {
                            id: self.next_id(),
                            type_id: instance.ty,
                            kind: ExpressionKind::Aggregate {
                                class: instance.ty,
                                fields,
                            },
                            span: *span,
                        },
                        preserve_error: false,
                        span: *span,
                    });
                    return Ok(Statement::Binding(id));
                }
                let (binding, object_type) = existing.unwrap();
                let instance = self
                    .class_instances_by_type
                    .get(&object_type)
                    .cloned()
                    .ok_or_else(|| {
                        Diagnostic::new(
                            "E000211",
                            "field assignment requires a class value",
                            Some(object.span),
                        )
                    })?;
                let Some((index, declaration)) = instance
                    .fields
                    .iter()
                    .enumerate()
                    .find(|(_, declaration)| declaration.name == *field)
                else {
                    return Err(Diagnostic::new(
                        "E000211",
                        format!("class `{}` has no field `{field}`", instance.name),
                        Some(*span),
                    ));
                };
                Ok(Statement::FieldSet {
                    binding,
                    field: index as u32,
                    value: self.expression(value, Some(declaration.ty))?,
                })
            }
            AstStatement::Expression(expression) => match self.class_method_update(expression)? {
                Some(update) => Ok(update),
                None => Ok(Statement::Expression(self.expression(expression, None)?)),
            },
            AstStatement::Return { value, span } => {
                let unit = self
                    .types
                    .resolve_name("unit")
                    .expect("bootstrap defines unit");
                let value = match value {
                    Some(value) if result_type == unit => {
                        return Err(Diagnostic::new(
                            "E000210",
                            "a unit function cannot return a value",
                            Some(*span),
                        ))
                    }
                    Some(value) => Some(self.expression(value, Some(result_type))?),
                    None if result_type != unit => {
                        return Err(Diagnostic::new(
                            "E000210",
                            "this function must return its declared result",
                            Some(*span),
                        ))
                    }
                    None => None,
                };
                Ok(Statement::Return(value))
            }
            AstStatement::Assert {
                condition,
                message,
                span,
            } => {
                let boolean = self
                    .types
                    .resolve_name("bool")
                    .expect("bootstrap defines bool");
                let string = self
                    .types
                    .resolve_name("string")
                    .expect("bootstrap defines string");
                Ok(Statement::Assert {
                    condition: self.expression(condition, Some(boolean))?,
                    message: message
                        .as_ref()
                        .map(|message| self.expression(message, Some(string)))
                        .transpose()?,
                    span: *span,
                    condition_span: condition.span,
                })
            }
            AstStatement::If {
                condition,
                then_block,
                else_block,
                ..
            } => {
                let boolean = self
                    .types
                    .resolve_name("bool")
                    .expect("bootstrap defines bool");
                let condition = self.expression(condition, Some(boolean))?;
                let outer_names = self.names.clone();
                let outer_declarations = self.declarations.clone();
                let then_block = self.block(then_block, bindings, result_type)?;
                self.names.clone_from(&outer_names);
                self.declarations.clone_from(&outer_declarations);
                let else_block = self.block(else_block, bindings, result_type)?;
                self.names = outer_names;
                self.declarations = outer_declarations;
                Ok(Statement::If {
                    condition,
                    then_block,
                    else_block,
                })
            }
            AstStatement::Match {
                subject,
                cases,
                span,
            } => self.match_statement(subject, cases, *span, bindings, result_type),
        }
    }

    fn match_statement(
        &mut self,
        subject: &AstExpression,
        cases: &[severian_ast::MatchCase],
        span: severian_source::Span,
        bindings: &mut Vec<Binding>,
        result_type: TypeId,
    ) -> Result<Statement, Diagnostic> {
        let subject = self.expression(subject, None)?;
        let subject_type = subject.type_id;
        if self
            .enums
            .values()
            .any(|instance| instance.ty == subject_type)
        {
            return self.enum_match_statement(subject, cases, span, bindings, result_type);
        }
        let outer_names = self.names.clone();
        let outer_declarations = self.declarations.clone();
        let mut seen_types = BTreeSet::new();
        let mut catch_all = false;
        let mut arms = Vec::new();
        for case in cases {
            if catch_all {
                return Err(Diagnostic::new(
                    "E000214",
                    "this case is unreachable because a previous case matches every remaining value",
                    Some(case.span),
                ));
            }
            let type_id = case
                .annotation
                .as_ref()
                .map(|annotation| resolve_type_annotation(self.types, annotation))
                .transpose()?;
            if let Some(type_id) = type_id {
                if !seen_types.insert(type_id) {
                    return Err(Diagnostic::new(
                        "E000214",
                        "this match type is handled more than once",
                        Some(case.span),
                    ));
                }
                if type_id != subject_type {
                    return Err(Diagnostic::new(
                        "E000215",
                        "case type cannot occur in this non-union matched expression",
                        Some(case.span),
                    ));
                }
                catch_all = true;
            } else {
                catch_all = true;
            }

            self.names.clone_from(&outer_names);
            self.declarations.clone_from(&outer_declarations);
            let binding = if let Some(name) = &case.binding {
                let id = self.new_binding_id();
                let binding_type = type_id.unwrap_or(subject_type);
                self.names.insert(name.clone(), (id, binding_type));
                self.declarations.insert(name.clone());
                bindings.push(Binding {
                    id,
                    type_id: binding_type,
                    value: subject.clone(),
                    preserve_error: true,
                    span: case.span,
                });
                Some(id)
            } else {
                None
            };
            arms.push(severian_hir::MatchArm {
                binding,
                type_id,
                body: self.block(&case.body, bindings, result_type)?,
            });
        }
        self.names = outer_names;
        self.declarations = outer_declarations;
        if !catch_all && !seen_types.contains(&subject_type) {
            return Err(Diagnostic::new(
                "E000216",
                "match is not exhaustive; add a compatible typed case or `case _:`",
                Some(span),
            ));
        }
        Ok(Statement::Match { subject, arms })
    }

    fn enum_match_statement(
        &mut self,
        subject: Expression,
        cases: &[severian_ast::MatchCase],
        span: severian_source::Span,
        bindings: &mut Vec<Binding>,
        result_type: TypeId,
    ) -> Result<Statement, Diagnostic> {
        let instance = self
            .enums
            .values()
            .find(|instance| instance.ty == subject.type_id)
            .cloned()
            .expect("caller selected an enum subject");
        let integer = self
            .types
            .resolve_name("int")
            .expect("bootstrap defines int");
        let boolean = self
            .types
            .resolve_name("bool")
            .expect("bootstrap defines bool");
        let outer_names = self.names.clone();
        let outer_declarations = self.declarations.clone();
        let outer_substitutions = self.value_substitutions.clone();
        let mut handled = BTreeSet::new();
        let mut lowered = Vec::new();
        let mut returned = Some(Vec::new());
        for case in cases {
            self.names.clone_from(&outer_names);
            self.declarations.clone_from(&outer_declarations);
            self.value_substitutions.clone_from(&outer_substitutions);
            let pattern = case.binding.as_deref().unwrap_or("_");
            let variant = instance.variants.iter().enumerate().find(|(_, variant)| {
                pattern == variant.name || pattern == format!("{}.{}", instance.name, variant.name)
            });
            let (condition, selected) = if let Some((ordinal, variant)) = variant {
                if !handled.insert(ordinal) {
                    return Err(Diagnostic::new(
                        "E000214",
                        format!("enum variant `{}` is handled more than once", variant.name),
                        Some(case.span),
                    ));
                }
                for payload in &variant.fields {
                    let index = instance
                        .fields
                        .iter()
                        .position(|field| field.name == payload.name)
                        .expect("enum payload has a lowered field");
                    let field = &instance.fields[index];
                    let id = self.next_id();
                    self.value_substitutions.insert(
                        payload.name.clone(),
                        Expression {
                            id,
                            type_id: field.ty,
                            kind: ExpressionKind::Field {
                                object: Box::new(subject.clone()),
                                index: index as u32,
                            },
                            span: case.span,
                        },
                    );
                }
                let tag = Expression {
                    id: self.next_id(),
                    type_id: integer,
                    kind: ExpressionKind::Field {
                        object: Box::new(subject.clone()),
                        index: 0,
                    },
                    span: case.span,
                };
                let ordinal = self.integer_expression(&ordinal.to_string(), integer, case.span);
                (
                    Some(Expression {
                        id: self.next_id(),
                        type_id: boolean,
                        kind: ExpressionKind::Binary {
                            operator: BinaryOperator::Equal,
                            left: Box::new(tag),
                            right: Box::new(ordinal),
                        },
                        span: case.span,
                    }),
                    true,
                )
            } else if pattern == "_" {
                (None, true)
            } else {
                return Err(Diagnostic::new(
                    "E000215",
                    format!("`{pattern}` is not a variant of `{}`", instance.name),
                    Some(case.span),
                ));
            };
            debug_assert!(selected);
            if let Some(values) = &mut returned {
                if let [AstStatement::Return {
                    value: Some(value), ..
                }] = case.body.as_slice()
                {
                    values.push((
                        condition.clone(),
                        self.expression(value, Some(result_type))?,
                    ));
                } else {
                    returned = None;
                }
            }
            lowered.push((condition, self.block(&case.body, bindings, result_type)?));
        }
        self.names = outer_names;
        self.declarations = outer_declarations;
        self.value_substitutions = outer_substitutions;
        if handled.len() != instance.variants.len()
            && !lowered.iter().any(|(condition, _)| condition.is_none())
        {
            return Err(Diagnostic::new(
                "E000216",
                format!("match does not cover every `{}` variant", instance.name),
                Some(span),
            ));
        }
        if let Some(mut values) = returned {
            let Some((_, mut selected)) = values.pop() else {
                return Err(Diagnostic::new(
                    "E000216",
                    "an enum match requires at least one arm",
                    Some(span),
                ));
            };
            let suffix = self.select_runtime_suffix(result_type, span)?;
            while let Some((condition, value)) = values.pop() {
                let Some(condition) = condition else {
                    selected = value;
                    continue;
                };
                selected = self.runtime_call(
                    &format!("__sev_select_{suffix}"),
                    &[boolean, result_type, result_type],
                    result_type,
                    vec![condition, value, selected],
                    span,
                );
            }
            return Ok(Statement::Return(Some(selected)));
        }
        let mut lowered = lowered.into_iter().rev();
        let Some((_last_condition, last_body)) = lowered.next() else {
            return Err(Diagnostic::new(
                "E000216",
                "an enum match requires at least one arm",
                Some(span),
            ));
        };
        let mut tail = last_body;
        let mut nested = false;
        for (condition, body) in lowered {
            if let Some(condition) = condition {
                nested = true;
                tail = Block {
                    statements: vec![Statement::If {
                        condition,
                        then_block: body,
                        else_block: tail,
                    }],
                };
            } else {
                tail = body;
            }
        }
        if nested && tail.statements.len() == 1 {
            return Ok(tail.statements.into_iter().next().unwrap());
        }
        let condition = Expression {
            id: self.next_id(),
            type_id: boolean,
            kind: ExpressionKind::Literal(LiteralValue::Boolean(true)),
            span,
        };
        Ok(Statement::If {
            condition,
            then_block: tail.clone(),
            else_block: tail,
        })
    }

    fn block(
        &mut self,
        statements: &[AstStatement],
        bindings: &mut Vec<Binding>,
        result_type: TypeId,
    ) -> Result<Block, Diagnostic> {
        let mut block = Block::default();
        for statement in statements {
            block
                .statements
                .push(self.statement(statement, bindings, result_type)?);
        }
        Ok(block)
    }

    fn next_id(&mut self) -> HirId {
        let id = HirId(self.next_hir);
        self.next_hir += 1;
        id
    }

    fn prepare(&mut self, ast: &AstExpression) -> Result<Prepared, Diagnostic> {
        match &ast.kind {
            AstExpressionKind::Literal(value) => {
                Ok(Prepared::Literal(universal_literal(value), ast.span))
            }
            _ => self.expression(ast, None).map(Prepared::Resolved),
        }
    }

    fn finish(&mut self, prepared: Prepared, expected: TypeId) -> Result<Expression, Diagnostic> {
        match prepared {
            Prepared::Literal(value, span) => {
                let type_id = self
                    .types
                    .resolve_literal(&value, Some(expected))
                    .map_err(|error| semantic_error(error.to_string(), span))?;
                Ok(Expression {
                    id: self.next_id(),
                    type_id,
                    kind: ExpressionKind::Literal(value),
                    span,
                })
            }
            Prepared::Resolved(expression)
                if self.types.assignable(expression.type_id, expected) =>
            {
                Ok(expression)
            }
            Prepared::Resolved(expression) => Err(semantic_error(
                "operator operand does not satisfy the selected signature".into(),
                expression.span,
            )),
        }
    }

    fn expression(
        &mut self,
        ast: &AstExpression,
        expected: Option<TypeId>,
    ) -> Result<Expression, Diagnostic> {
        match &ast.kind {
            AstExpressionKind::Literal(value) => {
                let value = if matches!(value, AstLiteral::None)
                    && expected.is_some_and(|expected| {
                        self.types
                            .definition(expected)
                            .is_some_and(|definition| definition.name == "string")
                    }) {
                    LiteralValue::String(String::new())
                } else {
                    universal_literal(value)
                };
                let type_id = self
                    .types
                    .resolve_literal(&value, expected)
                    .map_err(|error| semantic_error(error.to_string(), ast.span))?;
                Ok(Expression {
                    id: self.next_id(),
                    type_id,
                    kind: ExpressionKind::Literal(value),
                    span: ast.span,
                })
            }
            AstExpressionKind::List(values) => {
                if values.is_empty()
                    && expected.is_none_or(|ty| !self.list_elements.contains_key(&ty))
                {
                    return Err(Diagnostic::new(
                        "E000204",
                        "a list literal requires an expected `list[T]` type",
                        Some(ast.span),
                    ));
                }
                let (list_type, element) = if let Some(list_type) =
                    expected.filter(|ty| self.list_elements.contains_key(ty))
                {
                    (list_type, self.list_elements[&list_type])
                } else {
                    let first = self.expression(&values[0], None)?;
                    let element = first.type_id;
                    (self.instantiate_list_type(element), element)
                };
                let list = self.empty_list_expression(list_type, ast.span)?;
                if values.is_empty() {
                    return Ok(list);
                }
                let storage_type = self
                    .types
                    .resolve_name("string")
                    .expect("bootstrap defines pointer-backed string");
                let suffix = self.list_runtime_suffix(element, ast.span)?;
                let symbol = format!("__sev_list_append_{suffix}");
                let mut storage = self.list_storage_expression(list, ast.span);
                for value in values {
                    let value = self.expression(value, Some(element))?;
                    storage = self.runtime_call(
                        &symbol,
                        &[storage_type, element],
                        storage_type,
                        vec![storage, value],
                        ast.span,
                    );
                }
                Ok(Expression {
                    id: self.next_id(),
                    type_id: list_type,
                    kind: ExpressionKind::Aggregate {
                        class: list_type,
                        fields: vec![storage],
                    },
                    span: ast.span,
                })
            }
            AstExpressionKind::Tuple(values) => {
                if values.is_empty() {
                    return self.expression(
                        &AstExpression {
                            kind: AstExpressionKind::Literal(AstLiteral::Unit),
                            span: ast.span,
                        },
                        expected,
                    );
                }
                let mut fields = Vec::with_capacity(values.len());
                for value in values {
                    fields.push(self.expression(value, None)?);
                }
                let element_types = fields.iter().map(|field| field.type_id).collect::<Vec<_>>();
                let tuple_type = self.instantiate_tuple_type(&element_types);
                if expected.is_some_and(|expected| expected != tuple_type) {
                    return Err(semantic_error(
                        "tuple does not satisfy the expected type".into(),
                        ast.span,
                    ));
                }
                Ok(Expression {
                    id: self.next_id(),
                    type_id: tuple_type,
                    kind: ExpressionKind::Aggregate {
                        class: tuple_type,
                        fields,
                    },
                    span: ast.span,
                })
            }
            AstExpressionKind::Name(name) => {
                if name == "absent" {
                    let Some(expected) = expected else {
                        return Err(Diagnostic::new(
                            "E000204",
                            "`absent` requires an optional expected type",
                            Some(ast.span),
                        ));
                    };
                    if self
                        .types
                        .definition(expected)
                        .is_some_and(|definition| definition.name == "string")
                    {
                        return Ok(Expression {
                            id: self.next_id(),
                            type_id: expected,
                            kind: ExpressionKind::Literal(LiteralValue::String(String::new())),
                            span: ast.span,
                        });
                    }
                }
                if self.enum_variants.contains_key(name) {
                    return self.enum_constructor(name, &[], expected, ast.span);
                }
                if let Some(value) = self.value_substitutions.get(name).cloned() {
                    if expected
                        .is_some_and(|expected| !self.types.assignable(value.type_id, expected))
                    {
                        return Err(semantic_error(
                            "substituted value does not satisfy the expected type".into(),
                            ast.span,
                        ));
                    }
                    return Ok(value);
                }
                let Some((binding, type_id)) = self.names.get(name).copied() else {
                    return Err(Diagnostic::new(
                        "E000201",
                        format!("unknown binding `{name}`"),
                        Some(ast.span),
                    ));
                };
                if expected.is_some_and(|expected| !self.types.assignable(type_id, expected)) {
                    return Err(semantic_error(
                        "binding type does not satisfy the expected type".into(),
                        ast.span,
                    ));
                }
                Ok(Expression {
                    id: self.next_id(),
                    type_id,
                    kind: ExpressionKind::Binding(binding),
                    span: ast.span,
                })
            }
            AstExpressionKind::Member { object, name } => {
                if let Some(path) = callable_path(ast) {
                    if self.enum_variants.contains_key(&path) {
                        return self.enum_constructor(&path, &[], expected, ast.span);
                    }
                }
                let object = self.expression(object, None)?;
                let Some(instance) = self.class_instances_by_type.get(&object.type_id) else {
                    return Err(Diagnostic::new(
                        "E000211",
                        format!("type has no field `{name}`"),
                        Some(ast.span),
                    ));
                };
                let Some((index, field)) = instance
                    .fields
                    .iter()
                    .enumerate()
                    .find(|(_, field)| field.name == *name)
                else {
                    return Err(Diagnostic::new(
                        "E000211",
                        format!("class `{}` has no field `{name}`", instance.name),
                        Some(ast.span),
                    ));
                };
                let field_type = field.ty;
                if expected.is_some_and(|expected| !self.types.assignable(field_type, expected)) {
                    return Err(semantic_error(
                        "field type does not satisfy the expected type".into(),
                        ast.span,
                    ));
                }
                Ok(Expression {
                    id: self.next_id(),
                    type_id: field_type,
                    kind: ExpressionKind::Field {
                        object: Box::new(object),
                        index: index as u32,
                    },
                    span: ast.span,
                })
            }
            AstExpressionKind::Index { object, index } => {
                let object = self.expression(object, None)?;
                if let Some(elements) = self.tuple_elements.get(&object.type_id).cloned() {
                    let index_span = index.span;
                    let AstExpressionKind::Literal(AstLiteral::Integer(index)) = &index.kind else {
                        return Err(Diagnostic::new(
                            "E000211",
                            "tuple indices must be integer literals",
                            Some(index.span),
                        ));
                    };
                    let index = index.parse::<usize>().map_err(|_| {
                        Diagnostic::new("E000211", "invalid tuple index", Some(index_span))
                    })?;
                    let Some(element) = elements.get(index).copied() else {
                        return Err(Diagnostic::new(
                            "E000211",
                            "tuple index is out of bounds",
                            Some(ast.span),
                        ));
                    };
                    return Ok(Expression {
                        id: self.next_id(),
                        type_id: element,
                        kind: ExpressionKind::Field {
                            object: Box::new(object),
                            index: index as u32,
                        },
                        span: ast.span,
                    });
                }
                let Some(element) = self.list_elements.get(&object.type_id).copied() else {
                    return Err(Diagnostic::new(
                        "E000211",
                        "indexing is not implemented for this type",
                        Some(ast.span),
                    ));
                };
                if expected.is_some_and(|expected| !self.types.assignable(element, expected)) {
                    return Err(semantic_error(
                        "indexed value does not satisfy the expected type".into(),
                        ast.span,
                    ));
                }
                let index = self.expression(index, None)?;
                let index_name = self
                    .types
                    .definition(index.type_id)
                    .map(|definition| definition.name.as_str());
                if !matches!(index_name, Some("int" | "i64" | "usize")) {
                    return Err(Diagnostic::new(
                        "E000211",
                        "list indices must be integers",
                        Some(index.span),
                    ));
                }
                let storage = self.list_storage_expression(object, ast.span);
                let storage_type = storage.type_id;
                let suffix = self.list_runtime_suffix(element, ast.span)?;
                Ok(self.runtime_call(
                    &format!("__sev_list_get_{suffix}"),
                    &[storage_type, index.type_id],
                    element,
                    vec![storage, index],
                    ast.span,
                ))
            }
            AstExpressionKind::Slice {
                object,
                start,
                end,
                step,
            } => {
                let object_value = self.expression(object, None)?;
                if let Some(elements) = self.tuple_elements.get(&object_value.type_id).cloned() {
                    let reverse = start.is_none()
                        && end.is_none()
                        && step.as_deref().is_some_and(|step| {
                            matches!(
                                &step.kind,
                                AstExpressionKind::Unary {
                                    operator: AstUnaryOperator::Negative,
                                    operand
                                } if matches!(operand.kind, AstExpressionKind::Literal(AstLiteral::Integer(ref value)) if value == "1")
                            )
                        });
                    if !reverse {
                        return Err(Diagnostic::new(
                            "E000211",
                            "tuple slicing currently supports `[::-1]`",
                            Some(ast.span),
                        ));
                    }
                    let reversed_types = elements.iter().rev().copied().collect::<Vec<_>>();
                    let result_type = self.instantiate_tuple_type(&reversed_types);
                    let mut fields = Vec::with_capacity(elements.len());
                    for (index, element) in elements.iter().enumerate().rev() {
                        fields.push(Expression {
                            id: self.next_id(),
                            type_id: *element,
                            kind: ExpressionKind::Field {
                                object: Box::new(object_value.clone()),
                                index: index as u32,
                            },
                            span: ast.span,
                        });
                    }
                    return Ok(Expression {
                        id: self.next_id(),
                        type_id: result_type,
                        kind: ExpressionKind::Aggregate {
                            class: result_type,
                            fields,
                        },
                        span: ast.span,
                    });
                }
                let string = self
                    .types
                    .resolve_name("string")
                    .expect("bootstrap defines string");
                let integer = self
                    .types
                    .resolve_name("int")
                    .expect("bootstrap defines int");
                if object_value.type_id != string {
                    return Err(Diagnostic::new(
                        "E000211",
                        "slicing is not implemented for this type",
                        Some(ast.span),
                    ));
                }
                let object = object_value;
                if expected.is_some_and(|expected| !self.types.assignable(string, expected)) {
                    return Err(semantic_error(
                        "slice result does not satisfy the expected type".into(),
                        ast.span,
                    ));
                }
                let start = match start {
                    Some(start) => self.expression(start, Some(integer))?,
                    None => self.integer_expression("0", integer, ast.span),
                };
                let end = match end {
                    Some(end) => self.expression(end, Some(integer))?,
                    None => self.integer_expression("9223372036854775807", integer, ast.span),
                };
                let step = match step {
                    Some(step) => self.expression(step, Some(integer))?,
                    None => self.integer_expression("1", integer, ast.span),
                };
                Ok(self.runtime_call(
                    "__sev_string_slice",
                    &[string, integer, integer, integer],
                    string,
                    vec![object, start, end, step],
                    ast.span,
                ))
            }
            AstExpressionKind::TypeApplication { .. } => Err(Diagnostic::new(
                "E000211",
                "a generic type application must be constructed",
                Some(ast.span),
            )),
            AstExpressionKind::Call { callee, arguments } => {
                if let Some(path) = callable_path(callee) {
                    if self.enum_variants.contains_key(&path) {
                        return self.enum_constructor(&path, arguments, expected, ast.span);
                    }
                }
                if let Some((class, type_arguments)) = class_application(callee) {
                    return self.class_constructor(
                        class,
                        type_arguments,
                        arguments,
                        expected,
                        ast.span,
                    );
                }
                if let AstExpressionKind::Name(class) = &callee.kind {
                    if self
                        .classes
                        .get(class)
                        .is_some_and(|declaration| !declaration.type_parameters.is_empty())
                    {
                        return self
                            .inferred_class_constructor(class, arguments, expected, ast.span);
                    }
                    if self.classes.contains_key(class) {
                        return self.class_constructor(class, &[], arguments, expected, ast.span);
                    }
                }
                if callable_path(callee).as_deref() == Some("size") && arguments.len() == 1 {
                    let value = self.expression(&arguments[0].value, None)?;
                    if self.list_elements.contains_key(&value.type_id) {
                        let storage = self.list_storage_expression(value, ast.span);
                        let storage_type = storage.type_id;
                        let result = self
                            .types
                            .resolve_name("usize")
                            .expect("bootstrap defines usize");
                        return Ok(self.runtime_call(
                            "__sev_list_len",
                            &[storage_type],
                            result,
                            vec![storage],
                            ast.span,
                        ));
                    }
                }
                if callable_path(callee).as_deref() == Some("print") && arguments.len() == 1 {
                    let value = self.expression(&arguments[0].value, None)?;
                    if self.tuple_elements.contains_key(&value.type_id) {
                        let rendered = self.tuple_string(value, ast.span)?;
                        let string = rendered.type_id;
                        let result = self
                            .types
                            .resolve_name("i32")
                            .expect("bootstrap defines i32");
                        return Ok(self.runtime_call(
                            "__sev_print_string",
                            &[string],
                            result,
                            vec![rendered],
                            ast.span,
                        ));
                    }
                }
                if let Some(method) =
                    self.class_method_call(callee, arguments, expected, ast.span)?
                {
                    return Ok(method);
                }
                if matches!(callee.kind, AstExpressionKind::Member { .. })
                    && !callable_path(callee).is_some_and(|path| self.functions.contains_key(&path))
                {
                    return Err(Diagnostic::new(
                        "E000211",
                        "class method is unknown or cannot be used as a value",
                        Some(callee.span),
                    ));
                }
                let Some(name) = callable_path(callee) else {
                    return Err(Diagnostic::new(
                        "E000206",
                        "call target must resolve to a function declaration",
                        Some(callee.span),
                    ));
                };
                let candidates = self.functions.get(&name).cloned().unwrap_or_default();
                let mut matches = Vec::new();
                for function in candidates {
                    let signature = self.signatures[&function].clone();
                    if expected
                        .is_some_and(|expected| !self.types.assignable(signature.result, expected))
                    {
                        continue;
                    }
                    let mut ordered = vec![None; signature.parameters.len()];
                    let mut positional = 0usize;
                    let mut named = false;
                    let mut valid = true;
                    for argument in arguments {
                        let index = if let Some(argument_name) = &argument.name {
                            named = true;
                            signature
                                .parameters
                                .iter()
                                .position(|parameter| parameter.name == *argument_name)
                        } else if named {
                            None
                        } else {
                            let index = positional;
                            positional += 1;
                            (index < ordered.len()).then_some(index)
                        };
                        let Some(index) = index else {
                            valid = false;
                            break;
                        };
                        if ordered[index].replace(&argument.value).is_some() {
                            valid = false;
                            break;
                        }
                    }
                    if !valid {
                        continue;
                    }
                    let ordered = ordered
                        .into_iter()
                        .zip(&signature.parameters)
                        .map(|(argument, parameter)| argument.or(parameter.default.as_ref()))
                        .collect::<Option<Vec<_>>>();
                    let Some(ordered) = ordered else {
                        continue;
                    };
                    let resolved = ordered
                        .iter()
                        .zip(&signature.parameters)
                        .map(|(argument, parameter)| {
                            self.expression(argument, Some(parameter.type_id))
                        })
                        .collect::<Result<Vec<_>, _>>();
                    if let Ok(arguments) = resolved {
                        let conversions = arguments
                            .iter()
                            .zip(&signature.parameters)
                            .map(|(argument, parameter)| {
                                conversion_rank(self.types, argument.type_id, parameter.type_id)
                            })
                            .collect::<Option<Vec<_>>>();
                        let Some(conversions) = conversions else {
                            continue;
                        };
                        matches.push((
                            conversions,
                            self.function_specificity[&function],
                            function,
                            signature.result,
                            arguments,
                        ));
                    }
                }
                let best = matches
                    .iter()
                    .enumerate()
                    .filter(|(index, candidate)| {
                        !matches.iter().enumerate().any(|(other_index, other)| {
                            other_index != *index && dominates(&other.0, &candidate.0)
                        })
                    })
                    .map(|(_, candidate)| candidate)
                    .collect::<Vec<_>>();
                let specificity = best.iter().map(|candidate| candidate.1).min();
                let best = best
                    .into_iter()
                    .filter(|candidate| Some(candidate.1) == specificity)
                    .collect::<Vec<_>>();
                let [(_, _, function, result, arguments)] = best.as_slice() else {
                    return Err(Diagnostic::new(
                        "E000206",
                        format!("call to `{name}` has no unique compatible declaration"),
                        Some(ast.span),
                    ));
                };
                Ok(Expression {
                    id: self.next_id(),
                    type_id: *result,
                    kind: ExpressionKind::Call {
                        callee: severian_hir::Callee::Direct {
                            function: self.function_definitions[function],
                            substitution: self.function_substitutions[function].clone(),
                        },
                        arguments: (*arguments).clone(),
                    },
                    span: ast.span,
                })
            }
            AstExpressionKind::Unary { operator, operand } => {
                if *operator == AstUnaryOperator::Copy {
                    let operand = self.expression(operand, expected)?;
                    let Some(element) = self.list_elements.get(&operand.type_id).copied() else {
                        return Ok(operand);
                    };
                    let list_type = operand.type_id;
                    let storage = self.list_storage_expression(operand, ast.span);
                    let storage_type = storage.type_id;
                    let suffix = self.list_runtime_suffix(element, ast.span)?;
                    let copied = self.runtime_call(
                        &format!("__sev_list_copy_{suffix}"),
                        &[storage_type],
                        storage_type,
                        vec![storage],
                        ast.span,
                    );
                    return Ok(Expression {
                        id: self.next_id(),
                        type_id: list_type,
                        kind: ExpressionKind::Aggregate {
                            class: list_type,
                            fields: vec![copied],
                        },
                        span: ast.span,
                    });
                }
                if *operator == AstUnaryOperator::Move {
                    return Err(Diagnostic::new(
                        "E000302",
                        "move checking is not implemented yet",
                        Some(ast.span),
                    ));
                }
                let operator = universal_unary(*operator);
                let prepared = self.prepare(operand)?;
                let resolved = self
                    .types
                    .resolve_unary(operator, prepared.constraint(), expected)
                    .map_err(|error| semantic_error(error.to_string(), ast.span))?;
                let operand = self.finish(prepared, resolved.operand)?;
                Ok(Expression {
                    id: self.next_id(),
                    type_id: resolved.result,
                    kind: ExpressionKind::Unary {
                        operator,
                        operand: Box::new(operand),
                    },
                    span: ast.span,
                })
            }
            AstExpressionKind::Binary {
                operator,
                left,
                right,
            } => {
                if *operator == AstBinaryOperator::Power
                    && matches!(
                        right.kind,
                        AstExpressionKind::Literal(AstLiteral::Integer(_))
                    )
                {
                    let left = self.expression(left, None)?;
                    let name = self
                        .types
                        .definition(left.type_id)
                        .map(|definition| definition.name.as_str());
                    if matches!(name, Some("float" | "f64")) {
                        let integer = self
                            .types
                            .resolve_name("int")
                            .expect("bootstrap defines int");
                        let right = self.expression(right, Some(integer))?;
                        return Ok(self.runtime_call(
                            "__sev_pow_f64_i64",
                            &[left.type_id, integer],
                            left.type_id,
                            vec![left, right],
                            ast.span,
                        ));
                    }
                }
                if matches!(
                    operator,
                    AstBinaryOperator::Equal
                        | AstBinaryOperator::NotEqual
                        | AstBinaryOperator::Identity
                ) {
                    let resolved_left = self.expression(left, None)?;
                    if let Some(element) = self.list_elements.get(&resolved_left.type_id).copied() {
                        let list_type = resolved_left.type_id;
                        let resolved_right = self.expression(right, Some(list_type))?;
                        let left_storage = self.list_storage_expression(resolved_left, ast.span);
                        let right_storage = self.list_storage_expression(resolved_right, ast.span);
                        let storage_type = left_storage.type_id;
                        let boolean = self
                            .types
                            .resolve_name("bool")
                            .expect("bootstrap defines bool");
                        let symbol = if *operator == AstBinaryOperator::Identity {
                            "__sev_list_identity".to_owned()
                        } else {
                            format!(
                                "__sev_list_equal_{}",
                                self.list_runtime_suffix(element, ast.span)?
                            )
                        };
                        let comparison = self.runtime_call(
                            &symbol,
                            &[storage_type, storage_type],
                            boolean,
                            vec![left_storage, right_storage],
                            ast.span,
                        );
                        if *operator == AstBinaryOperator::NotEqual {
                            return Ok(Expression {
                                id: self.next_id(),
                                type_id: boolean,
                                kind: ExpressionKind::Unary {
                                    operator: UnaryOperator::Not,
                                    operand: Box::new(comparison),
                                },
                                span: ast.span,
                            });
                        }
                        return Ok(comparison);
                    }
                    if *operator == AstBinaryOperator::Identity {
                        let resolved_right = self.expression(right, Some(resolved_left.type_id))?;
                        return Ok(Expression {
                            id: self.next_id(),
                            type_id: self
                                .types
                                .resolve_name("bool")
                                .expect("bootstrap defines bool"),
                            kind: ExpressionKind::Binary {
                                operator: BinaryOperator::Equal,
                                left: Box::new(resolved_left),
                                right: Box::new(resolved_right),
                            },
                            span: ast.span,
                        });
                    }
                }
                let operator = universal_binary(*operator);
                // Both operands remain constraints until a single signature is
                // selected; neither side gets an early default literal type.
                let left = self.prepare(left)?;
                let right = self.prepare(right)?;
                let resolved = self
                    .types
                    .resolve_binary(operator, left.constraint(), right.constraint(), expected)
                    .map_err(|error| semantic_error(error.to_string(), ast.span))?;
                let left = self.finish(left, resolved.left)?;
                let right = self.finish(right, resolved.right)?;
                Ok(Expression {
                    id: self.next_id(),
                    type_id: resolved.result,
                    kind: ExpressionKind::Binary {
                        operator,
                        left: Box::new(left),
                        right: Box::new(right),
                    },
                    span: ast.span,
                })
            }
        }
    }

    fn class_constructor(
        &mut self,
        class: &str,
        type_arguments: &[TypeAnnotation],
        arguments: &[severian_ast::CallArgument],
        expected: Option<TypeId>,
        span: severian_source::Span,
    ) -> Result<Expression, Diagnostic> {
        let instance = self.instantiate_class(class, type_arguments, span)?;
        if expected.is_some_and(|expected| expected != instance.ty) {
            return Err(semantic_error(
                "constructed class does not satisfy the expected type".into(),
                span,
            ));
        }
        let constructor = instance
            .constructors
            .iter()
            .find(|constructor| constructor.parameters.len() == arguments.len())
            .cloned();
        let fields = if let Some(constructor) = constructor {
            self.class_constructor_values(&instance, &constructor, arguments, span)?
        } else if !instance.constructors.is_empty() {
            return Err(Diagnostic::new(
                "E000221",
                format!(
                    "constructor `{class}` has no overload accepting {} argument(s)",
                    arguments.len()
                ),
                Some(span),
            ));
        } else if arguments.is_empty() && !instance.fields.is_empty() {
            instance
                .fields
                .iter()
                .map(|field| self.default_expression(field.ty, span))
                .collect::<Result<Vec<_>, _>>()?
        } else if arguments.len() != instance.fields.len() {
            return Err(Diagnostic::new(
                "E000221",
                format!(
                    "constructor `{class}` expects {} field value(s), received {}",
                    instance.fields.len(),
                    arguments.len()
                ),
                Some(span),
            ));
        } else {
            arguments
                .iter()
                .zip(&instance.fields)
                .map(|(argument, field)| self.expression(&argument.value, Some(field.ty)))
                .collect::<Result<Vec<_>, _>>()?
        };
        Ok(Expression {
            id: self.next_id(),
            type_id: instance.ty,
            kind: ExpressionKind::Aggregate {
                class: instance.ty,
                fields,
            },
            span,
        })
    }

    fn inferred_class_constructor(
        &mut self,
        class: &str,
        arguments: &[severian_ast::CallArgument],
        expected: Option<TypeId>,
        span: severian_source::Span,
    ) -> Result<Expression, Diagnostic> {
        let declaration = self.classes.get(class).cloned().ok_or_else(|| {
            Diagnostic::new(
                "E000204",
                format!("unknown generic class `{class}`"),
                Some(span),
            )
        })?;
        if arguments.len() != declaration.fields.len() {
            return Err(Diagnostic::new(
                "E000221",
                format!(
                    "constructor `{class}` expects {} field value(s), received {}",
                    declaration.fields.len(),
                    arguments.len()
                ),
                Some(span),
            ));
        }

        let parameters = declaration
            .type_parameters
            .iter()
            .map(String::as_str)
            .collect::<BTreeSet<_>>();
        let mut inferred = BTreeMap::<String, TypeId>::new();
        let mut fields = Vec::with_capacity(arguments.len());
        for (argument, field) in arguments.iter().zip(&declaration.fields) {
            let field_type_name = field.annotation.simple_name().ok_or_else(|| {
                Diagnostic::new(
                    "E000204",
                    "generic constructor inference currently requires named field types",
                    Some(field.annotation.span),
                )
            })?;
            if parameters.contains(field_type_name) {
                let expected_field = inferred.get(field_type_name).copied();
                let value = self.expression(&argument.value, expected_field)?;
                inferred
                    .entry(field_type_name.to_owned())
                    .or_insert(value.type_id);
                fields.push(value);
            } else {
                let field_type = resolve_type_annotation(self.types, &field.annotation)?;
                fields.push(self.expression(&argument.value, Some(field_type))?);
            }
        }
        let concrete = declaration
            .type_parameters
            .iter()
            .map(|parameter| {
                inferred.get(parameter).copied().ok_or_else(|| {
                    Diagnostic::new(
                        "E000206",
                        format!(
                            "cannot infer type parameter `{parameter}` for constructor `{class}`"
                        ),
                        Some(span),
                    )
                })
            })
            .collect::<Result<Vec<_>, _>>()?;
        let instance = self.instantiate_class_types(class, &concrete, span)?;
        if expected.is_some_and(|expected| expected != instance.ty) {
            return Err(semantic_error(
                "constructed class does not satisfy the expected type".into(),
                span,
            ));
        }
        Ok(Expression {
            id: self.next_id(),
            type_id: instance.ty,
            kind: ExpressionKind::Aggregate {
                class: instance.ty,
                fields,
            },
            span,
        })
    }

    fn instantiate_class(
        &mut self,
        name: &str,
        arguments: &[TypeAnnotation],
        span: severian_source::Span,
    ) -> Result<ClassInstance, Diagnostic> {
        let Some(declaration) = self.classes.get(name).cloned() else {
            return Err(Diagnostic::new(
                "E000204",
                format!("unknown generic class `{name}`"),
                Some(span),
            ));
        };
        if declaration.type_parameters.len() != arguments.len() {
            return Err(Diagnostic::new(
                "E000204",
                format!(
                    "class `{name}` expects {} type argument(s), received {}",
                    declaration.type_parameters.len(),
                    arguments.len()
                ),
                Some(span),
            ));
        }
        let concrete = arguments
            .iter()
            .map(|argument| resolve_type_annotation(self.types, argument))
            .collect::<Result<Vec<_>, _>>()?;
        self.instantiate_class_types(name, &concrete, span)
    }

    fn instantiate_class_types(
        &mut self,
        name: &str,
        concrete: &[TypeId],
        span: severian_source::Span,
    ) -> Result<ClassInstance, Diagnostic> {
        let declaration = self.classes.get(name).cloned().ok_or_else(|| {
            Diagnostic::new(
                "E000204",
                format!("unknown generic class `{name}`"),
                Some(span),
            )
        })?;
        if declaration.type_parameters.len() != concrete.len() {
            return Err(Diagnostic::new(
                "E000204",
                format!(
                    "class `{name}` expects {} type argument(s), received {}",
                    declaration.type_parameters.len(),
                    concrete.len()
                ),
                Some(span),
            ));
        }
        let key = (name.to_owned(), concrete.to_vec());
        if let Some(instance) = self.class_instances.get(&key) {
            return Ok(instance.clone());
        }
        let substitution = declaration
            .type_parameters
            .iter()
            .cloned()
            .zip(concrete.iter().copied())
            .collect::<BTreeMap<_, _>>();
        let mut fields = Vec::with_capacity(declaration.fields.len());
        for field in &declaration.fields {
            let ty = self.resolve_instantiated_type(&field.annotation, &substitution)?;
            fields.push(HirClassFieldDeclaration {
                name: field.name.clone(),
                ty,
            });
        }
        let ty = TypeId(self.next_class_type);
        self.next_class_type = self.next_class_type.saturating_sub(1);
        let instance = ClassInstance {
            ty,
            name: name.to_owned(),
            fields: fields.clone(),
            constructors: declaration.constructors.clone(),
            methods: declaration.methods.clone(),
        };
        self.class_instances.insert(key, instance.clone());
        self.class_instances_by_type.insert(ty, instance.clone());
        self.lowered_classes.push(HirClassDeclaration {
            id: ty,
            name: format!(
                "{}[{}]",
                name,
                concrete
                    .iter()
                    .map(|ty| {
                        self.types
                            .definition(*ty)
                            .map(|definition| definition.name.clone())
                            .unwrap_or_else(|| format!("type#{}", ty.0))
                    })
                    .collect::<Vec<_>>()
                    .join(", ")
            ),
            fields,
        });
        Ok(instance)
    }

    fn resolve_instantiated_type(
        &mut self,
        annotation: &TypeAnnotation,
        substitution: &BTreeMap<String, TypeId>,
    ) -> Result<TypeId, Diagnostic> {
        let Some((name, arguments)) = annotation.named_parts() else {
            return Err(Diagnostic::new(
                "E000204",
                "union class fields are not yet supported",
                Some(annotation.span),
            ));
        };
        if arguments.is_empty() {
            if let Some(ty) = substitution.get(name) {
                return Ok(*ty);
            }
            return resolve_type_annotation(self.types, annotation);
        }
        if name == "list" && arguments.len() == 1 {
            let element = self.resolve_instantiated_type(&arguments[0], substitution)?;
            return Ok(self.instantiate_list_type(element));
        }
        Err(Diagnostic::new(
            "E000204",
            format!("generic field type `{name}` is not yet supported"),
            Some(annotation.span),
        ))
    }

    fn instantiate_list_type(&mut self, element: TypeId) -> TypeId {
        if let Some(ty) = self.list_types.get(&element) {
            return *ty;
        }
        let ty = TypeId(self.next_class_type);
        self.next_class_type = self.next_class_type.saturating_sub(1);
        let storage = self
            .types
            .resolve_name("string")
            .expect("bootstrap defines pointer-backed string");
        let element_name = self
            .types
            .definition(element)
            .map(|definition| definition.name.clone())
            .unwrap_or_else(|| format!("type#{}", element.0));
        self.list_types.insert(element, ty);
        self.list_elements.insert(ty, element);
        self.lowered_classes.push(HirClassDeclaration {
            id: ty,
            name: format!("list[{element_name}]"),
            fields: vec![HirClassFieldDeclaration {
                name: "storage".into(),
                ty: storage,
            }],
        });
        ty
    }

    fn instantiate_tuple_type(&mut self, elements: &[TypeId]) -> TypeId {
        if let Some(ty) = self.tuple_types.get(elements) {
            return *ty;
        }
        let ty = TypeId(self.next_class_type);
        self.next_class_type = self.next_class_type.saturating_sub(1);
        let element_names = elements
            .iter()
            .map(|element| {
                self.types
                    .definition(*element)
                    .map(|definition| definition.name.clone())
                    .unwrap_or_else(|| format!("type#{}", element.0))
            })
            .collect::<Vec<_>>();
        let fields = elements
            .iter()
            .enumerate()
            .map(|(index, element)| HirClassFieldDeclaration {
                name: index.to_string(),
                ty: *element,
            })
            .collect::<Vec<_>>();
        self.tuple_types.insert(elements.to_vec(), ty);
        self.tuple_elements.insert(ty, elements.to_vec());
        self.lowered_classes.push(HirClassDeclaration {
            id: ty,
            name: format!("({})", element_names.join(", ")),
            fields,
        });
        ty
    }

    fn class_constructor_values(
        &mut self,
        instance: &ClassInstance,
        constructor: &severian_ast::FunctionDeclaration,
        arguments: &[severian_ast::CallArgument],
        span: severian_source::Span,
    ) -> Result<Vec<Expression>, Diagnostic> {
        let body = constructor.body.as_ref().ok_or_else(|| {
            Diagnostic::new(
                "E000211",
                format!("constructor `{}` has no implementation", constructor.name),
                Some(constructor.span),
            )
        })?;
        let resolved_arguments = arguments
            .iter()
            .zip(&constructor.parameters)
            .map(|(argument, parameter)| {
                let parameter_type = resolve_type_annotation(self.types, &parameter.annotation)?;
                self.expression(&argument.value, Some(parameter_type))
            })
            .collect::<Result<Vec<_>, Diagnostic>>()?;
        let previous = self.value_substitutions.clone();
        for (parameter, argument) in constructor.parameters.iter().zip(resolved_arguments) {
            self.value_substitutions
                .insert(parameter.name.clone(), argument);
        }
        let values = (|| {
            let mut values = Vec::with_capacity(instance.fields.len());
            for field in &instance.fields {
                let initializer = body.iter().find_map(|statement| {
                    let AstStatement::Binding(binding) = statement else {
                        return None;
                    };
                    (binding.name == field.name).then_some(&binding.value)
                });
                let Some(initializer) = initializer else {
                    return Err(Diagnostic::new(
                        "E000221",
                        format!(
                            "constructor `{}` does not initialize field `{}`",
                            constructor.name, field.name
                        ),
                        Some(span),
                    ));
                };
                values.push(self.expression(initializer, Some(field.ty))?);
            }
            Ok(values)
        })();
        self.value_substitutions = previous;
        values
    }

    fn empty_list_expression(
        &mut self,
        list_type: TypeId,
        span: severian_source::Span,
    ) -> Result<Expression, Diagnostic> {
        let storage_type = self
            .types
            .resolve_name("string")
            .expect("bootstrap defines pointer-backed string");
        let storage = self.runtime_call("__sev_list_create", &[], storage_type, Vec::new(), span);
        Ok(Expression {
            id: self.next_id(),
            type_id: list_type,
            kind: ExpressionKind::Aggregate {
                class: list_type,
                fields: vec![storage],
            },
            span,
        })
    }

    fn enum_constructor(
        &mut self,
        path: &str,
        arguments: &[severian_ast::CallArgument],
        expected: Option<TypeId>,
        span: severian_source::Span,
    ) -> Result<Expression, Diagnostic> {
        let (enum_name, ordinal) = self.enum_variants[path].clone();
        let instance = self.enums[&enum_name].clone();
        let variant = &instance.variants[ordinal];
        if arguments.len() != variant.fields.len() {
            return Err(Diagnostic::new(
                "E000221",
                format!(
                    "enum variant `{}` expects {} payload value(s), received {}",
                    variant.name,
                    variant.fields.len(),
                    arguments.len()
                ),
                Some(span),
            ));
        }
        if expected.is_some_and(|expected| expected != instance.ty) {
            return Err(semantic_error(
                "enum variant does not satisfy the expected type".into(),
                span,
            ));
        }
        let integer = self
            .types
            .resolve_name("int")
            .expect("bootstrap defines int");
        let mut values = Vec::with_capacity(instance.fields.len());
        values.push(self.integer_expression(&ordinal.to_string(), integer, span));
        for field in instance.fields.iter().skip(1) {
            if let Some((argument, _)) = arguments
                .iter()
                .zip(&variant.fields)
                .find(|(_, payload)| payload.name == field.name)
            {
                values.push(self.expression(&argument.value, Some(field.ty))?);
            } else {
                values.push(self.default_expression(field.ty, span)?);
            }
        }
        Ok(Expression {
            id: self.next_id(),
            type_id: instance.ty,
            kind: ExpressionKind::Aggregate {
                class: instance.ty,
                fields: values,
            },
            span,
        })
    }

    fn default_expression(
        &mut self,
        type_id: TypeId,
        span: severian_source::Span,
    ) -> Result<Expression, Diagnostic> {
        if self.list_elements.contains_key(&type_id) {
            return self.empty_list_expression(type_id, span);
        }
        let name = self
            .types
            .definition(type_id)
            .map(|definition| definition.name.as_str());
        let literal = match name {
            Some("string") => LiteralValue::String(String::new()),
            Some("float" | "f16" | "f32" | "f64" | "bf16") => LiteralValue::Float("0.0".into()),
            Some("bool") => LiteralValue::Boolean(false),
            Some("char") => LiteralValue::Character('\0'),
            Some("None") => LiteralValue::None,
            Some("unit") => LiteralValue::Unit,
            Some(_) => LiteralValue::Integer("0".into()),
            None => {
                return Err(Diagnostic::new(
                    "E000221",
                    "this enum payload type requires an explicit representation",
                    Some(span),
                ))
            }
        };
        Ok(Expression {
            id: self.next_id(),
            type_id,
            kind: ExpressionKind::Literal(literal),
            span,
        })
    }

    fn tuple_string(
        &mut self,
        tuple: Expression,
        span: severian_source::Span,
    ) -> Result<Expression, Diagnostic> {
        let elements = self.tuple_elements[&tuple.type_id].clone();
        let string = self
            .types
            .resolve_name("string")
            .expect("bootstrap defines string");
        let mut rendered = Expression {
            id: self.next_id(),
            type_id: string,
            kind: ExpressionKind::Literal(LiteralValue::String("(".into())),
            span,
        };
        for (index, element) in elements.iter().enumerate() {
            if index != 0 {
                let comma = Expression {
                    id: self.next_id(),
                    type_id: string,
                    kind: ExpressionKind::Literal(LiteralValue::String(",".into())),
                    span,
                };
                rendered = self.runtime_call(
                    "__sev_string_concat",
                    &[string, string],
                    string,
                    vec![rendered, comma],
                    span,
                );
            }
            let field = Expression {
                id: self.next_id(),
                type_id: *element,
                kind: ExpressionKind::Field {
                    object: Box::new(tuple.clone()),
                    index: index as u32,
                },
                span,
            };
            let name = self
                .types
                .definition(*element)
                .map(|definition| definition.name.as_str());
            let field = match name {
                Some("string") => field,
                Some("int" | "i64") => self.runtime_call(
                    "__sev_string_from_int",
                    &[*element],
                    string,
                    vec![field],
                    span,
                ),
                Some("usize") => self.runtime_call(
                    "__sev_string_from_usize",
                    &[*element],
                    string,
                    vec![field],
                    span,
                ),
                Some("float" | "f64") => self.runtime_call(
                    "__sev_string_from_float",
                    &[*element],
                    string,
                    vec![field],
                    span,
                ),
                _ => {
                    return Err(Diagnostic::new(
                        "E000211",
                        "tuple display does not support this element type",
                        Some(span),
                    ))
                }
            };
            rendered = self.runtime_call(
                "__sev_string_concat",
                &[string, string],
                string,
                vec![rendered, field],
                span,
            );
        }
        let close = Expression {
            id: self.next_id(),
            type_id: string,
            kind: ExpressionKind::Literal(LiteralValue::String(")".into())),
            span,
        };
        Ok(self.runtime_call(
            "__sev_string_concat",
            &[string, string],
            string,
            vec![rendered, close],
            span,
        ))
    }

    fn list_storage_expression(
        &mut self,
        list: Expression,
        span: severian_source::Span,
    ) -> Expression {
        let storage_type = self
            .types
            .resolve_name("string")
            .expect("bootstrap defines pointer-backed string");
        Expression {
            id: self.next_id(),
            type_id: storage_type,
            kind: ExpressionKind::Field {
                object: Box::new(list),
                index: 0,
            },
            span,
        }
    }

    fn list_runtime_suffix(
        &self,
        element: TypeId,
        span: severian_source::Span,
    ) -> Result<&'static str, Diagnostic> {
        let name = self
            .types
            .definition(element)
            .map(|definition| definition.name.as_str());
        match name {
            Some("int") | Some("i64") | Some("usize") => Ok("i64"),
            Some("string") => Ok("ptr"),
            _ => Err(Diagnostic::new(
                "E000211",
                "native list lowering currently supports `int`, `i64`, `usize`, and `string` elements",
                Some(span),
            )),
        }
    }

    fn select_runtime_suffix(
        &self,
        result: TypeId,
        span: severian_source::Span,
    ) -> Result<&'static str, Diagnostic> {
        let name = self
            .types
            .definition(result)
            .map(|definition| definition.name.as_str());
        match name {
            Some("string") => Ok("string"),
            Some("float" | "f64") => Ok("f64"),
            Some("f32") => Ok("f32"),
            Some("bool") => Ok("bool"),
            Some("int" | "i64" | "usize") => Ok("i64"),
            _ => Err(Diagnostic::new(
                "E000211",
                "enum return selection does not yet support this result type",
                Some(span),
            )),
        }
    }

    fn runtime_call(
        &mut self,
        symbol: &str,
        parameter_types: &[TypeId],
        result_type: TypeId,
        arguments: Vec<Expression>,
        span: severian_source::Span,
    ) -> Expression {
        let definition = self.ensure_runtime_function(symbol, parameter_types, result_type);
        Expression {
            id: self.next_id(),
            type_id: result_type,
            kind: ExpressionKind::Call {
                callee: severian_hir::Callee::Direct {
                    function: definition,
                    substitution: severian_universal::Substitution::default(),
                },
                arguments,
            },
            span,
        }
    }

    fn integer_expression(
        &mut self,
        spelling: &str,
        type_id: TypeId,
        span: severian_source::Span,
    ) -> Expression {
        Expression {
            id: self.next_id(),
            type_id,
            kind: ExpressionKind::Literal(LiteralValue::Integer(spelling.into())),
            span,
        }
    }

    fn ensure_runtime_function(
        &mut self,
        symbol: &str,
        parameter_types: &[TypeId],
        result_type: TypeId,
    ) -> DefId {
        if let Some(definition) = self.runtime_definitions.get(symbol) {
            return *definition;
        }
        let definition = synthetic_runtime_definition(symbol);
        let id = FunctionId(definition.declaration.0);
        let parameters = parameter_types
            .iter()
            .enumerate()
            .map(|(index, ty)| FunctionParameter {
                binding: self.new_binding_id(),
                name: format!("argument{index}"),
                contract: universal_boundary(*ty),
            })
            .collect();
        self.runtime_functions.push(FunctionDeclaration {
            id,
            definition,
            substitution: severian_universal::Substitution::default(),
            name: symbol.into(),
            type_parameters: Vec::new(),
            parameters,
            result: universal_boundary(result_type),
            compile_route: severian_universal::CompileRoute::Standard,
            call_type: severian_hir::CallType::External(severian_hir::ExternalCall {
                interface: severian_hir::InterfaceId("native-runtime".into()),
                symbol: severian_hir::SymbolId(symbol.into()),
                provider: None,
                ffi: severian_hir::FfiId("c".into()),
                abi: severian_hir::AbiId("native".into()),
            }),
            body: None,
        });
        self.runtime_definitions.insert(symbol.into(), definition);
        definition
    }

    fn class_method_call(
        &mut self,
        callee: &AstExpression,
        arguments: &[severian_ast::CallArgument],
        expected: Option<TypeId>,
        span: severian_source::Span,
    ) -> Result<Option<Expression>, Diagnostic> {
        let AstExpressionKind::Member { object, name } = &callee.kind else {
            return Ok(None);
        };
        if callable_path(callee).is_some_and(|path| self.functions.contains_key(&path)) {
            return Ok(None);
        }
        let object = self.expression(object, None)?;
        let string = self
            .types
            .resolve_name("string")
            .expect("bootstrap defines string");
        if object.type_id == string {
            let usize_type = self
                .types
                .resolve_name("usize")
                .expect("bootstrap defines usize");
            let bool_type = self
                .types
                .resolve_name("bool")
                .expect("bootstrap defines bool");
            let (symbol, parameters, result, resolved_arguments) = match name.as_str() {
                "length" if arguments.is_empty() => (
                    "__sev_string_length",
                    vec![string],
                    usize_type,
                    vec![object],
                ),
                "upper" if arguments.is_empty() => {
                    ("__sev_string_upper", vec![string], string, vec![object])
                }
                "contains" if arguments.len() == 1 && arguments[0].name.is_none() => {
                    let needle = self.expression(&arguments[0].value, Some(string))?;
                    (
                        "__sev_string_contains",
                        vec![string, string],
                        bool_type,
                        vec![object, needle],
                    )
                }
                "length" | "upper" | "contains" => {
                    return Err(Diagnostic::new(
                        "E000206",
                        format!("string method `{name}` received incompatible arguments"),
                        Some(span),
                    ));
                }
                _ => return Ok(None),
            };
            if expected.is_some_and(|expected| !self.types.assignable(result, expected)) {
                return Err(semantic_error(
                    "method result does not satisfy the expected type".into(),
                    span,
                ));
            }
            return Ok(Some(self.runtime_call(
                symbol,
                &parameters,
                result,
                resolved_arguments,
                span,
            )));
        }
        let Some(instance) = self.class_instances_by_type.get(&object.type_id).cloned() else {
            return Ok(None);
        };
        let Some(method) = instance.methods.iter().find(|method| method.name == *name) else {
            return Err(Diagnostic::new(
                "E000211",
                format!("class `{}` has no method `{name}`", instance.name),
                Some(callee.span),
            ));
        };
        if method.parameters.len() != arguments.len() {
            return Err(Diagnostic::new(
                "E000206",
                format!(
                    "method `{name}` expects {} argument(s), received {}",
                    method.parameters.len(),
                    arguments.len()
                ),
                Some(span),
            ));
        }
        let Some(body) = &method.body else {
            return Err(Diagnostic::new(
                "E000211",
                format!("method `{name}` has no implementation"),
                Some(method.span),
            ));
        };
        if let Some(field_name) = body.iter().find_map(|statement| {
            let AstStatement::Return {
                value: Some(value), ..
            } = statement
            else {
                return None;
            };
            let AstExpressionKind::Call { callee, arguments } = &value.kind else {
                return None;
            };
            let AstExpressionKind::Member {
                object: field,
                name: operation,
            } = &callee.kind
            else {
                return None;
            };
            let AstExpressionKind::Name(field) = &field.kind else {
                return None;
            };
            (operation == "pop" && arguments.is_empty()).then_some(field.as_str())
        }) {
            let Some((field, declaration)) = instance
                .fields
                .iter()
                .enumerate()
                .find(|(_, field)| field.name == field_name)
            else {
                return Err(Diagnostic::new(
                    "E000211",
                    format!("class `{}` has no field `{field_name}`", instance.name),
                    Some(method.span),
                ));
            };
            let Some(element) = self.list_elements.get(&declaration.ty).copied() else {
                return Err(Diagnostic::new(
                    "E000211",
                    format!("field `{field_name}` is not a list"),
                    Some(method.span),
                ));
            };
            if expected.is_some_and(|expected| !self.types.assignable(element, expected)) {
                return Err(semantic_error(
                    "method result does not satisfy the expected type".into(),
                    span,
                ));
            }
            let list = Expression {
                id: self.next_id(),
                type_id: declaration.ty,
                kind: ExpressionKind::Field {
                    object: Box::new(object),
                    index: field as u32,
                },
                span,
            };
            let storage = self.list_storage_expression(list, span);
            let storage_type = storage.type_id;
            let suffix = self.list_runtime_suffix(element, span)?;
            let symbol = format!("__sev_list_pop_{suffix}");
            return Ok(Some(self.runtime_call(
                &symbol,
                &[storage_type],
                element,
                vec![storage],
                span,
            )));
        }
        if let Some(return_value) = body.iter().find_map(|statement| match statement {
            AstStatement::Return {
                value: Some(value), ..
            } if !matches!(value.kind, AstExpressionKind::Name(_)) => Some(value),
            _ => None,
        }) {
            let result_type = self.resolve_source_type(&method.result)?;
            if expected.is_some_and(|expected| !self.types.assignable(result_type, expected)) {
                return Err(semantic_error(
                    "method result does not satisfy the expected type".into(),
                    span,
                ));
            }
            let previous = self.value_substitutions.clone();
            for (field, declaration) in instance.fields.iter().enumerate() {
                let id = self.next_id();
                self.value_substitutions.insert(
                    declaration.name.clone(),
                    Expression {
                        id,
                        type_id: declaration.ty,
                        kind: ExpressionKind::Field {
                            object: Box::new(object.clone()),
                            index: field as u32,
                        },
                        span,
                    },
                );
            }
            let resolved = (|| {
                for (parameter, argument) in method.parameters.iter().zip(arguments) {
                    let parameter_type = self.resolve_source_type(&parameter.annotation)?;
                    let value = self.expression(&argument.value, Some(parameter_type))?;
                    self.value_substitutions
                        .insert(parameter.name.clone(), value);
                }
                self.expression(return_value, Some(result_type))
            })();
            self.value_substitutions = previous;
            return resolved.map(Some);
        }
        let field_name = body.iter().find_map(|statement| {
            let AstStatement::Return {
                value: Some(value), ..
            } = statement
            else {
                return None;
            };
            let AstExpressionKind::Name(field) = &value.kind else {
                return None;
            };
            Some(field.as_str())
        });
        let Some((field, declaration)) = field_name.and_then(|field_name| {
            instance
                .fields
                .iter()
                .enumerate()
                .find(|(_, field)| field.name == field_name)
        }) else {
            return Err(Diagnostic::new(
                "E000211",
                format!("method `{name}` is not a field-returning method"),
                Some(method.span),
            ));
        };
        if expected.is_some_and(|expected| !self.types.assignable(declaration.ty, expected)) {
            return Err(semantic_error(
                "method result does not satisfy the expected type".into(),
                span,
            ));
        }
        Ok(Some(Expression {
            id: self.next_id(),
            type_id: declaration.ty,
            kind: ExpressionKind::Field {
                object: Box::new(object),
                index: field as u32,
            },
            span,
        }))
    }

    fn class_method_update(
        &mut self,
        expression: &AstExpression,
    ) -> Result<Option<Statement>, Diagnostic> {
        let AstExpressionKind::Call { callee, arguments } = &expression.kind else {
            return Ok(None);
        };
        let AstExpressionKind::Member { object, name } = &callee.kind else {
            return Ok(None);
        };
        let AstExpressionKind::Name(receiver) = &object.kind else {
            return Ok(None);
        };
        let Some((binding, ty)) = self.names.get(receiver).copied() else {
            return Ok(None);
        };
        let Some(instance) = self.class_instances_by_type.get(&ty).cloned() else {
            return Ok(None);
        };
        let Some(method) = instance.methods.iter().find(|method| method.name == *name) else {
            return Err(Diagnostic::new(
                "E000211",
                format!("class `{}` has no method `{name}`", instance.name),
                Some(callee.span),
            ));
        };
        if method.parameters.len() != arguments.len() {
            return Err(Diagnostic::new(
                "E000206",
                format!(
                    "method `{name}` expects {} argument(s), received {}",
                    method.parameters.len(),
                    arguments.len()
                ),
                Some(expression.span),
            ));
        }
        let Some(body) = &method.body else {
            return Err(Diagnostic::new(
                "E000211",
                format!("method `{name}` has no implementation"),
                Some(method.span),
            ));
        };
        if let Some(field_name) = body.iter().find_map(|statement| {
            let AstStatement::Expression(expression) = statement else {
                return None;
            };
            let AstExpressionKind::Call {
                callee,
                arguments: method_arguments,
            } = &expression.kind
            else {
                return None;
            };
            let AstExpressionKind::Member {
                object: field,
                name: operation,
            } = &callee.kind
            else {
                return None;
            };
            let AstExpressionKind::Name(field) = &field.kind else {
                return None;
            };
            let [argument] = method_arguments.as_slice() else {
                return None;
            };
            let AstExpressionKind::Name(argument_name) = &argument.value.kind else {
                return None;
            };
            (operation == "append"
                && method
                    .parameters
                    .first()
                    .is_some_and(|parameter| argument_name == &parameter.name))
            .then_some(field.as_str())
        }) {
            let Some((field, declaration)) = instance
                .fields
                .iter()
                .enumerate()
                .find(|(_, field)| field.name == field_name)
            else {
                return Err(Diagnostic::new(
                    "E000211",
                    format!("class `{}` has no field `{field_name}`", instance.name),
                    Some(method.span),
                ));
            };
            let Some(element) = self.list_elements.get(&declaration.ty).copied() else {
                return Err(Diagnostic::new(
                    "E000211",
                    format!("field `{field_name}` is not a list"),
                    Some(method.span),
                ));
            };
            let receiver = Expression {
                id: self.next_id(),
                type_id: ty,
                kind: ExpressionKind::Binding(binding),
                span: object.span,
            };
            let list = Expression {
                id: self.next_id(),
                type_id: declaration.ty,
                kind: ExpressionKind::Field {
                    object: Box::new(receiver),
                    index: field as u32,
                },
                span: object.span,
            };
            let storage = self.list_storage_expression(list, expression.span);
            let storage_type = storage.type_id;
            let value = self.expression(&arguments[0].value, Some(element))?;
            let suffix = self.list_runtime_suffix(element, expression.span)?;
            let symbol = format!("__sev_list_push_{suffix}");
            let unit = self
                .types
                .resolve_name("unit")
                .expect("bootstrap defines unit");
            return Ok(Some(Statement::Expression(self.runtime_call(
                &symbol,
                &[storage_type, element],
                unit,
                vec![storage, value],
                expression.span,
            ))));
        }
        let Some((field_name, operator, update_value)) = body.iter().find_map(|statement| {
            let severian_ast::Statement::Binding(update) = statement else {
                return None;
            };
            if !update.update {
                return None;
            }
            let AstExpressionKind::Binary {
                operator,
                left,
                right,
            } = &update.value.kind
            else {
                return None;
            };
            let AstExpressionKind::Name(left) = &left.kind else {
                return None;
            };
            (left == &update.name).then(|| (update.name.as_str(), *operator, right.as_ref()))
        }) else {
            return Err(Diagnostic::new(
                "E000211",
                format!("method `{name}` is not a field-update method"),
                Some(method.span),
            ));
        };
        let Some((field, declaration)) = instance
            .fields
            .iter()
            .enumerate()
            .find(|(_, field)| field.name == field_name)
        else {
            return Err(Diagnostic::new(
                "E000211",
                format!("class `{}` has no field `{field_name}`", instance.name),
                Some(method.span),
            ));
        };
        let previous = self.value_substitutions.clone();
        for (parameter, argument) in method.parameters.iter().zip(arguments) {
            let value = self.expression(&argument.value, Some(declaration.ty))?;
            self.value_substitutions
                .insert(parameter.name.clone(), value);
        }
        let value = self.expression(update_value, Some(declaration.ty));
        self.value_substitutions = previous;
        let value = value?;
        let operator = universal_binary(operator);
        if !self.types.supports_binary(operator, declaration.ty) {
            return Err(Diagnostic::new(
                "E000202",
                format!("field `{field_name}` does not support this update operator"),
                Some(method.span),
            ));
        }
        Ok(Some(Statement::FieldUpdate {
            binding,
            field: field as u32,
            operator,
            value,
        }))
    }
}

fn synthetic_definition(
    module: &str,
    function: &severian_ast::FunctionDeclaration,
    overload_ordinal: usize,
) -> DefId {
    DefId {
        package: 0,
        module: severian_universal::DeclarationId::from_path(module).0,
        declaration: severian_universal::DeclarationId::from_path(&format!(
            "{module}.function.{}.{overload_ordinal}",
            function.name
        )),
    }
}

fn synthetic_test_definition(module: &str, ordinal: usize) -> DefId {
    DefId {
        package: 0,
        module: severian_universal::DeclarationId::from_path(module).0,
        declaration: severian_universal::DeclarationId::from_path(&format!(
            "{module}.test.{ordinal}"
        )),
    }
}

fn synthetic_runtime_definition(symbol: &str) -> DefId {
    DefId {
        package: 0,
        module: severian_universal::DeclarationId::from_path("severian.runtime").0,
        declaration: severian_universal::DeclarationId::from_path(&format!(
            "severian.runtime.{symbol}"
        )),
    }
}

fn synthetic_trait_definition(module: &str, name: &str) -> DefId {
    DefId {
        package: 0,
        module: severian_universal::DeclarationId::from_path(module).0,
        declaration: severian_universal::DeclarationId::from_path(&format!(
            "{module}.trait.{name}"
        )),
    }
}

fn validate_trait_implementations(ast: &severian_ast::Module) -> Result<(), Diagnostic> {
    let traits = ast
        .items
        .iter()
        .filter_map(|item| match item {
            severian_ast::Item::Trait(declaration) => {
                Some((declaration.name.as_str(), declaration))
            }
            _ => None,
        })
        .collect::<BTreeMap<_, _>>();

    for class in ast.items.iter().filter_map(|item| match item {
        severian_ast::Item::Class(declaration) => Some(declaration),
        _ => None,
    }) {
        for implemented in &class.traits {
            let Some((trait_name, _)) = implemented.named_parts() else {
                continue;
            };
            let Some(contract) = traits.get(trait_name) else {
                continue;
            };
            for required in &contract.methods {
                let Some(provided) = class
                    .methods
                    .iter()
                    .find(|method| method.name == required.name)
                else {
                    return Err(Diagnostic::new(
                        "E000218",
                        format!(
                            "class `{}` does not implement required method `{}.{}`",
                            class.name, trait_name, required.name
                        ),
                        Some(class.span),
                    ));
                };
                let same_parameters = required.parameters.len() == provided.parameters.len()
                    && required.parameters.iter().zip(&provided.parameters).all(
                        |(required, provided)| {
                            same_type_annotation(&required.annotation, &provided.annotation)
                        },
                    );
                if !same_parameters || !same_type_annotation(&required.result, &provided.result) {
                    return Err(Diagnostic::new(
                        "E000218",
                        format!(
                            "method `{}.{}` does not match trait `{}`",
                            class.name, provided.name, trait_name
                        ),
                        Some(provided.span),
                    ));
                }
            }
        }
    }
    Ok(())
}

fn validate_class_declarations(ast: &severian_ast::Module) -> Result<(), Diagnostic> {
    for class in ast.items.iter().filter_map(|item| match item {
        severian_ast::Item::Class(declaration) => Some(declaration),
        _ => None,
    }) {
        let fields = class
            .fields
            .iter()
            .map(|field| field.name.as_str())
            .collect::<BTreeSet<_>>();
        for method in class.constructors.iter().chain(&class.methods) {
            if let Some(parameter) = method
                .parameters
                .iter()
                .find(|parameter| fields.contains(parameter.name.as_str()))
            {
                return Err(Diagnostic::new(
                    "E000220",
                    format!(
                        "parameter `{}` shadows a field of class `{}`",
                        parameter.name, class.name
                    ),
                    Some(parameter.span),
                ));
            }
        }
    }
    Ok(())
}

fn same_type_annotation(left: &TypeAnnotation, right: &TypeAnnotation) -> bool {
    match (&left.kind, &right.kind) {
        (
            severian_ast::TypeAnnotationKind::Named {
                name: left_name,
                arguments: left_arguments,
            },
            severian_ast::TypeAnnotationKind::Named {
                name: right_name,
                arguments: right_arguments,
            },
        ) => {
            left_name == right_name
                && left_arguments.len() == right_arguments.len()
                && left_arguments
                    .iter()
                    .zip(right_arguments)
                    .all(|(left, right)| same_type_annotation(left, right))
        }
        (
            severian_ast::TypeAnnotationKind::Union(left),
            severian_ast::TypeAnnotationKind::Union(right),
        ) => {
            left.len() == right.len()
                && left
                    .iter()
                    .zip(right)
                    .all(|(left, right)| same_type_annotation(left, right))
        }
        _ => false,
    }
}

fn callable_path(expression: &AstExpression) -> Option<String> {
    match &expression.kind {
        AstExpressionKind::Name(name) => Some(name.clone()),
        AstExpressionKind::Member { object, name } => {
            Some(format!("{}.{}", callable_path(object)?, name))
        }
        _ => None,
    }
}

fn class_application(expression: &AstExpression) -> Option<(&str, &[TypeAnnotation])> {
    let AstExpressionKind::TypeApplication { callee, arguments } = &expression.kind else {
        return None;
    };
    let AstExpressionKind::Name(name) = &callee.kind else {
        return None;
    };
    Some((name, arguments))
}

fn static_integer(expression: &AstExpression) -> Option<i64> {
    match &expression.kind {
        AstExpressionKind::Literal(AstLiteral::Integer(value)) => value.parse().ok(),
        AstExpressionKind::Unary {
            operator: AstUnaryOperator::Negative,
            operand,
        } => static_integer(operand)?.checked_neg(),
        _ => None,
    }
}

fn static_integer_in(
    expression: &AstExpression,
    environment: &BTreeMap<String, i64>,
) -> Option<i64> {
    match &expression.kind {
        AstExpressionKind::Name(name) => environment.get(name).copied(),
        AstExpressionKind::Literal(AstLiteral::Integer(value)) => value.parse().ok(),
        AstExpressionKind::Unary {
            operator: AstUnaryOperator::Negative,
            operand,
        } => static_integer_in(operand, environment)?.checked_neg(),
        AstExpressionKind::Binary {
            operator,
            left,
            right,
        } => {
            let left = static_integer_in(left, environment)?;
            let right = static_integer_in(right, environment)?;
            match operator {
                AstBinaryOperator::Add => left.checked_add(right),
                AstBinaryOperator::Subtract => left.checked_sub(right),
                AstBinaryOperator::Multiply => left.checked_mul(right),
                AstBinaryOperator::Divide => (right != 0).then(|| left / right),
                AstBinaryOperator::Remainder => (right != 0).then(|| left % right),
                _ => None,
            }
        }
        _ => None,
    }
}

fn static_boolean(expression: &AstExpression, environment: &BTreeMap<String, i64>) -> Option<bool> {
    match &expression.kind {
        AstExpressionKind::Literal(AstLiteral::Boolean(value)) => Some(*value),
        AstExpressionKind::Unary {
            operator: AstUnaryOperator::Not,
            operand,
        } => Some(!static_boolean(operand, environment)?),
        AstExpressionKind::Binary {
            operator: AstBinaryOperator::And,
            left,
            right,
        } => Some(static_boolean(left, environment)? && static_boolean(right, environment)?),
        AstExpressionKind::Binary {
            operator: AstBinaryOperator::Or,
            left,
            right,
        } => Some(static_boolean(left, environment)? || static_boolean(right, environment)?),
        AstExpressionKind::Binary {
            operator,
            left,
            right,
        } => {
            let left = static_integer_in(left, environment)?;
            let right = static_integer_in(right, environment)?;
            Some(match operator {
                AstBinaryOperator::Equal | AstBinaryOperator::Identity => left == right,
                AstBinaryOperator::NotEqual => left != right,
                AstBinaryOperator::Less => left < right,
                AstBinaryOperator::LessEqual => left <= right,
                AstBinaryOperator::Greater => left > right,
                AstBinaryOperator::GreaterEqual => left >= right,
                _ => return None,
            })
        }
        _ => None,
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum StaticLoopControl {
    Continue,
    Break,
    Fallthrough,
}

fn specialize_loop_body(
    statements: &[AstStatement],
    environment: &mut BTreeMap<String, i64>,
) -> (Vec<AstStatement>, StaticLoopControl) {
    let mut output = Vec::new();
    for statement in statements {
        match statement {
            AstStatement::Break { .. } => return (output, StaticLoopControl::Break),
            AstStatement::Continue { .. } => return (output, StaticLoopControl::Continue),
            AstStatement::Binding(binding) => {
                if let Some(value) = static_integer_in(&binding.value, environment) {
                    environment.insert(binding.name.clone(), value);
                }
                output.push(statement.clone());
            }
            AstStatement::If {
                condition,
                then_block,
                else_block,
                ..
            } if static_boolean(condition, environment).is_some() => {
                let selected = if static_boolean(condition, environment) == Some(true) {
                    then_block
                } else {
                    else_block
                };
                let (specialized, control) = specialize_loop_body(selected, environment);
                output.extend(specialized);
                if control != StaticLoopControl::Fallthrough {
                    return (output, control);
                }
            }
            _ => output.push(statement.clone()),
        }
    }
    (output, StaticLoopControl::Fallthrough)
}

fn static_range_values(iterable: &AstExpression) -> Option<Vec<i64>> {
    let AstExpressionKind::Call { callee, arguments } = &iterable.kind else {
        return None;
    };
    if callable_path(callee).as_deref() != Some("range") {
        return None;
    }
    let (start, end) = match arguments.as_slice() {
        [end] => (0, static_integer(&end.value)?),
        [start, end] => (static_integer(&start.value)?, static_integer(&end.value)?),
        _ => return None,
    };
    let count = end.saturating_sub(start).max(0);
    (count <= 10_000).then(|| (start..end).collect())
}

fn resolve_type_annotation(
    types: &TypeContext,
    annotation: &TypeAnnotation,
) -> Result<TypeId, Diagnostic> {
    if let severian_ast::TypeAnnotationKind::Union(members) = &annotation.kind {
        let mut concrete = members
            .iter()
            .filter_map(|member| {
                let name = member.simple_name()?;
                (!matches!(name, "None" | "absent")).then_some(member)
            })
            .map(|member| resolve_type_annotation(types, member))
            .collect::<Result<Vec<_>, _>>()?;
        concrete.sort();
        concrete.dedup();
        if let [only] = concrete.as_slice() {
            return Ok(*only);
        }
        if concrete.is_empty() {
            return types.resolve_name("None").ok_or_else(|| {
                Diagnostic::new("E000204", "unknown type `None`", Some(annotation.span))
            });
        }
        return Err(Diagnostic::new(
            "E000204",
            "unions with multiple concrete representations are not implemented",
            Some(annotation.span),
        ));
    }
    let Some(name) = annotation.simple_name() else {
        return Err(Diagnostic::new(
            "E000204",
            "this source type form is not yet supported by universal resolution",
            Some(annotation.span),
        ));
    };
    let name = if name == "absent" { "None" } else { name };
    types.resolve_name(name).ok_or_else(|| {
        Diagnostic::new(
            "E000204",
            format!("unknown type `{name}`"),
            Some(annotation.span),
        )
    })
}

fn resolve_trait_type(
    types: &TypeContext,
    annotation: &TypeAnnotation,
) -> Result<HirTraitType, Diagnostic> {
    if annotation.simple_name() == Some("Self") {
        Ok(HirTraitType::SelfType)
    } else {
        resolve_type_annotation(types, annotation).map(HirTraitType::Concrete)
    }
}

fn universal_boundary(type_id: TypeId) -> BoundaryType {
    BoundaryType {
        ty: type_id,
        modifiers: Vec::new(),
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum ConversionRank {
    Exact,
    Widening(u16),
    General,
}

fn conversion_rank(
    types: &TypeContext,
    actual: TypeId,
    expected: TypeId,
) -> Option<ConversionRank> {
    if actual == expected {
        return Some(ConversionRank::Exact);
    }
    if !types.assignable(actual, expected) {
        return None;
    }
    let actual = types.primitive(actual)?.representation;
    let expected = types.primitive(expected)?.representation;
    match (actual, expected) {
        (
            severian_universal::PrimitiveRepresentation::Integer {
                bits: severian_universal::IntegerWidth::Fixed(actual),
                ..
            },
            severian_universal::PrimitiveRepresentation::Integer {
                bits: severian_universal::IntegerWidth::Fixed(expected),
                ..
            },
        ) => Some(ConversionRank::Widening(expected - actual)),
        (
            severian_universal::PrimitiveRepresentation::Float {
                format: severian_universal::FloatFormat::Ieee(actual),
            },
            severian_universal::PrimitiveRepresentation::Float {
                format: severian_universal::FloatFormat::Ieee(expected),
            },
        ) => Some(ConversionRank::Widening(expected - actual)),
        _ => Some(ConversionRank::General),
    }
}

fn dominates(left: &[ConversionRank], right: &[ConversionRank]) -> bool {
    left.len() == right.len()
        && left.iter().zip(right).all(|(left, right)| left <= right)
        && left.iter().zip(right).any(|(left, right)| left < right)
}

fn test_mode(
    name: &str,
    span: severian_source::Span,
) -> Result<severian_hir::TestMode, Diagnostic> {
    match name {
        "property" => Ok(severian_hir::TestMode::Property),
        "bench" | "benchmark" => Ok(severian_hir::TestMode::Benchmark),
        "chaos" => Ok(severian_hir::TestMode::Chaos),
        "profile" => Ok(severian_hir::TestMode::Profile),
        "compiler" => Ok(severian_hir::TestMode::Compiler),
        "integ" | "integration" => Ok(severian_hir::TestMode::Integration),
        _ => Err(Diagnostic::new(
            "E000213",
            format!("unknown test runner `{name}`"),
            Some(span),
        )),
    }
}

fn internal_name_part(name: &str) -> String {
    name.chars()
        .map(|character| {
            if character.is_ascii_alphanumeric() {
                character
            } else {
                '_'
            }
        })
        .collect()
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ControlFlow {
    FallsThrough,
    Returns,
}

fn block_flow(statements: &[AstStatement]) -> ControlFlow {
    for statement in statements {
        let flow = match statement {
            AstStatement::Return { .. } => ControlFlow::Returns,
            AstStatement::If {
                then_block,
                else_block,
                ..
            } if !else_block.is_empty()
                && block_flow(then_block) == ControlFlow::Returns
                && block_flow(else_block) == ControlFlow::Returns =>
            {
                ControlFlow::Returns
            }
            AstStatement::Match { cases, .. }
                if !cases.is_empty()
                    && cases
                        .iter()
                        .all(|case| block_flow(&case.body) == ControlFlow::Returns) =>
            {
                ControlFlow::Returns
            }
            AstStatement::Binding(_)
            | AstStatement::FieldAssignment { .. }
            | AstStatement::Expression(_)
            | AstStatement::Assert { .. }
            | AstStatement::If { .. }
            | AstStatement::While { .. }
            | AstStatement::For { .. }
            | AstStatement::Break { .. }
            | AstStatement::Continue { .. }
            | AstStatement::Match { .. } => ControlFlow::FallsThrough,
        };
        if flow == ControlFlow::Returns {
            return flow;
        }
    }
    ControlFlow::FallsThrough
}

fn integration_expectation(statement: &AstStatement) -> Option<severian_hir::TestExpectation> {
    let AstStatement::Assert { condition, .. } = statement else {
        return None;
    };
    if let AstExpressionKind::Unary {
        operator: AstUnaryOperator::Not,
        operand,
    } = &condition.kind
    {
        let AstExpressionKind::Binary {
            operator: AstBinaryOperator::Contains,
            left,
            right,
        } = &operand.kind
        else {
            return None;
        };
        return Some(severian_hir::TestExpectation::Excludes {
            stream: test_stream(right)?,
            value: string_literal(left)?.to_owned(),
        });
    }
    let AstExpressionKind::Binary {
        operator,
        left,
        right,
    } = &condition.kind
    else {
        return None;
    };
    match operator {
        AstBinaryOperator::Contains => Some(severian_hir::TestExpectation::Contains {
            stream: test_stream(right)?,
            value: string_literal(left)?.to_owned(),
        }),
        AstBinaryOperator::Equal => {
            if let (Some(stream), Some(value)) = (test_stream(left), string_literal(right)) {
                Some(severian_hir::TestExpectation::Equals {
                    stream,
                    value: value.to_owned(),
                })
            } else {
                Some(severian_hir::TestExpectation::Equals {
                    stream: test_stream(right)?,
                    value: string_literal(left)?.to_owned(),
                })
            }
        }
        _ => None,
    }
}

fn test_stream(expression: &AstExpression) -> Option<severian_hir::TestStream> {
    match &expression.kind {
        AstExpressionKind::Name(name) if name == "stdout" => Some(severian_hir::TestStream::Stdout),
        AstExpressionKind::Name(name) if name == "stderr" => Some(severian_hir::TestStream::Stderr),
        _ => None,
    }
}

fn string_literal(expression: &AstExpression) -> Option<&str> {
    match &expression.kind {
        AstExpressionKind::Literal(AstLiteral::String(value)) => Some(value),
        _ => None,
    }
}

fn universal_literal(literal: &AstLiteral) -> LiteralValue {
    match literal {
        AstLiteral::Integer(value) => LiteralValue::Integer(value.clone()),
        AstLiteral::Float(value) => LiteralValue::Float(value.clone()),
        AstLiteral::Boolean(value) => LiteralValue::Boolean(*value),
        AstLiteral::Character(value) => LiteralValue::Character(*value),
        AstLiteral::String(value) => LiteralValue::String(value.clone()),
        AstLiteral::Bytes(value) => LiteralValue::Bytes(value.clone()),
        AstLiteral::None => LiteralValue::None,
        AstLiteral::Unit => LiteralValue::Unit,
    }
}

fn universal_unary(operator: AstUnaryOperator) -> UnaryOperator {
    match operator {
        AstUnaryOperator::Positive => UnaryOperator::Positive,
        AstUnaryOperator::Negative => UnaryOperator::Negative,
        AstUnaryOperator::Not => UnaryOperator::Not,
        AstUnaryOperator::Copy => unreachable!("copy is lowered before universal resolution"),
        AstUnaryOperator::Move => unreachable!("move is rejected before universal resolution"),
    }
}

fn universal_binary(operator: AstBinaryOperator) -> BinaryOperator {
    match operator {
        AstBinaryOperator::Add => BinaryOperator::Add,
        AstBinaryOperator::Subtract => BinaryOperator::Subtract,
        AstBinaryOperator::Multiply => BinaryOperator::Multiply,
        AstBinaryOperator::Divide => BinaryOperator::Divide,
        AstBinaryOperator::Remainder => BinaryOperator::Remainder,
        AstBinaryOperator::Power => BinaryOperator::Power,
        AstBinaryOperator::Equal => BinaryOperator::Equal,
        AstBinaryOperator::Identity => {
            unreachable!("identity is lowered before universal resolution")
        }
        AstBinaryOperator::NotEqual => BinaryOperator::NotEqual,
        AstBinaryOperator::Less => BinaryOperator::Less,
        AstBinaryOperator::LessEqual => BinaryOperator::LessEqual,
        AstBinaryOperator::Greater => BinaryOperator::Greater,
        AstBinaryOperator::GreaterEqual => BinaryOperator::GreaterEqual,
        AstBinaryOperator::Contains => BinaryOperator::Contains,
        AstBinaryOperator::And => BinaryOperator::And,
        AstBinaryOperator::Or => BinaryOperator::Or,
    }
}

fn semantic_error(message: String, span: severian_source::Span) -> Diagnostic {
    Diagnostic::new("E000202", message, Some(span))
}

#[cfg(test)]
mod tests {
    use super::*;
    use severian_source::SourceFile;

    fn analyze_source(source: &str) -> (Program, severian_universal::UniversalContext) {
        let context = severian_bootstrap::load().unwrap();
        let source = SourceFile::virtual_source("test.sev", source);
        let tokens = severian_lexer::scan(&source).unwrap();
        let ast = severian_parser::parse(&tokens).unwrap();
        let hir = analyze(&ast, &context.types).unwrap();
        (hir, context)
    }

    #[test]
    fn annotation_and_both_literal_orders_share_i32() {
        let (program, context) = analyze_source("x: i32 = 10\na = x + 1\nb = 1 + x\n");
        let i32 = context.types.resolve_name("i32").unwrap();
        assert!(program.modules[0]
            .bindings
            .iter()
            .all(|binding| binding.type_id == i32));
    }

    #[test]
    fn unconstrained_literals_default_only_after_operator_matching() {
        let (program, context) = analyze_source("a = 1 + 2\n");
        let int = context.types.resolve_name("int").unwrap();
        assert_eq!(program.modules[0].bindings[0].type_id, int);
    }

    #[test]
    fn question_equal_survives_semantic_analysis() {
        let (program, _) = analyze_source("result ?= 1\nordinary = 2\n");
        assert!(program.modules[0].bindings[0].preserve_error);
        assert!(!program.modules[0].bindings[1].preserve_error);
    }

    #[test]
    fn expected_binary_and_default_unary_constraints_are_ranked() {
        let (program, context) = analyze_source("a: i32 = 1 + 2\nb = -1\n");
        assert_eq!(
            program.modules[0].bindings[0].type_id,
            context.types.resolve_name("i32").unwrap()
        );
        assert_eq!(
            program.modules[0].bindings[1].type_id,
            context.types.resolve_name("int").unwrap()
        );
    }

    #[test]
    fn generic_class_instances_keep_distinct_layouts_and_field_updates() {
        let source = "class Box[T]:\n    value: T\n    def addition(addition: T):\n        value += addition\ndef main():\n    ints := Box[int](10)\n    floats := Box[f64](2.5)\n    ints.addition(20)\n    floats.addition(4.5)\n    observed_int := ints.value\n    observed_float := floats.value\n";
        let (program, _) = analyze_source(source);
        let module = &program.modules[0];
        assert_eq!(module.classes.len(), 2);
        assert_ne!(module.classes[0].id, module.classes[1].id);
        let main = module
            .functions
            .iter()
            .find(|function| function.name == "main")
            .unwrap();
        assert_eq!(
            main.body
                .as_ref()
                .unwrap()
                .statements
                .iter()
                .filter(|statement| matches!(statement, Statement::FieldUpdate { .. }))
                .count(),
            2
        );
    }

    #[test]
    fn class_parameters_cannot_shadow_fields() {
        let source = SourceFile::virtual_source(
            "shadow.sev",
            "class Box[T]:\n    value: T\n    def invalid(value: T):\n        pass\n",
        );
        let universal = severian_bootstrap::load().unwrap();
        let ast = severian_parser::parse(&severian_lexer::scan(&source).unwrap()).unwrap();
        let error = analyze(&ast, &universal.types).unwrap_err();
        assert_eq!(error.code, "E000220");
        assert!(error.message.contains("shadows a field"));
    }

    #[test]
    fn untyped_case_binding_is_a_catch_all_not_a_magic_result_variant() {
        let context = severian_bootstrap::load().unwrap();
        let source = SourceFile::virtual_source(
            "match.sev",
            "def invalid(result: int) -> int:\n    match result:\n        case value:\n            return value\n        case error: int:\n            return error\n",
        );
        let tokens = severian_lexer::scan(&source).unwrap();
        let ast = severian_parser::parse(&tokens).unwrap();
        let error = analyze(&ast, &context.types).unwrap_err();
        assert_eq!(error.code, "E000214");
        assert!(error.message.contains("unreachable"));
    }

    #[test]
    fn build_excludes_tests_and_test_mode_uses_descriptive_internal_names() {
        let context = severian_bootstrap::load().unwrap();
        let source = SourceFile::virtual_source(
            "checks.sev",
            "def helper():\n    pass\n\ntest with profile and compiler \"frontend diagnostics\":\n    reject:\n        missing()\n",
        );
        let tokens = severian_lexer::scan(&source).unwrap();
        let ast = severian_parser::parse(&tokens).unwrap();
        let build = analyze(&ast, &context.types).unwrap();
        assert!(build.modules[0].tests.is_empty());
        assert_eq!(build.modules[0].functions.len(), 1);

        let tests = analyze_with_context(
            &ast,
            &context.types,
            AnalysisContext {
                mode: AnalysisMode::Test,
                module_name: "package_checks",
            },
        )
        .unwrap();
        assert_eq!(tests.modules[0].tests.len(), 1);
        assert_eq!(
            tests.modules[0].functions[1].name,
            "__sev_package_checks_test_with_profile-compiler_frontend_diagnostics_0"
        );
    }

    #[test]
    fn compiler_tests_reject_unimplemented_body_and_diagnostic_assertions() {
        let context = severian_bootstrap::load().unwrap();
        for source_text in [
            "test with compiler:\n    assert(false)\n    reject:\n        missing()\n",
            "test with compiler:\n    reject error:\n        missing()\n",
        ] {
            let source = SourceFile::virtual_source("compiler-test.sev", source_text);
            let tokens = severian_lexer::scan(&source).unwrap();
            let ast = severian_parser::parse(&tokens).unwrap();
            let error = analyze_with_context(
                &ast,
                &context.types,
                AnalysisContext {
                    mode: AnalysisMode::Test,
                    module_name: "package_compiler_test",
                },
            )
            .unwrap_err();
            assert_eq!(error.code, "E000217");
        }
    }

    #[test]
    fn captured_stream_assertions_make_ordinary_tests_integration_tests() {
        let context = severian_bootstrap::load().unwrap();
        let source = SourceFile::virtual_source(
            "implicit-integration.sev",
            "test:\n    assert(\"captured\" in stdout)\n",
        );
        let tokens = severian_lexer::scan(&source).unwrap();
        let ast = severian_parser::parse(&tokens).unwrap();
        let program = analyze_with_context(
            &ast,
            &context.types,
            AnalysisContext {
                mode: AnalysisMode::Test,
                module_name: "implicit_integration",
            },
        )
        .unwrap();
        let test = &program.modules[0].tests[0];
        assert_eq!(test.modes, [severian_hir::TestMode::Integration]);
        assert_eq!(
            test.expectations,
            [severian_hir::TestExpectation::Contains {
                stream: severian_hir::TestStream::Stdout,
                value: "captured".to_owned(),
            }]
        );
    }

    #[test]
    fn function_scopes_can_read_and_shadow_globals() {
        let (program, _) = analyze_source(
            "value := 1\ndef use_global():\n    observed := value\ndef main():\n    value := 2\n    observed := value\n",
        );
        let module = &program.modules[0];
        let global = module.bindings[0].id;
        let first_body_binding = module.bindings[1].id;
        let shadow = module.bindings[2].id;
        let shadow_use = &module.bindings[3].value;
        assert_ne!(global, first_body_binding);
        assert_ne!(global, shadow);
        assert_eq!(shadow_use.kind, ExpressionKind::Binding(shadow));
    }

    #[test]
    fn return_analysis_is_control_flow_based_not_last_statement_based() {
        let (program, _) =
            analyze_source("def answer() -> int:\n    return 42\n    unreachable := 0\n");
        assert_eq!(program.modules[0].functions.len(), 1);
    }

    #[test]
    fn overload_resolution_prefers_exact_parameters_over_widening() {
        let (program, context) = analyze_source(
            "def choose(value: i32) -> i32:\n    return value\ndef choose(value: i64) -> i64:\n    return value\nsource: i32 = 1\nselected = choose(source)\n",
        );
        let selected = program.modules[0].bindings.last().unwrap();
        assert_eq!(selected.type_id, context.types.resolve_name("i32").unwrap());
    }

    #[test]
    fn crossed_widening_overloads_remain_ambiguous() {
        let context = severian_bootstrap::load().unwrap();
        let source = SourceFile::virtual_source(
            "ambiguous.sev",
            "def choose(left: i32, right: i128) -> i32:\n    return left\ndef choose(left: i64, right: i64) -> i64:\n    return left\nleft: i32 = 1\nright: i32 = 2\nselected = choose(left, right)\n",
        );
        let tokens = severian_lexer::scan(&source).unwrap();
        let ast = severian_parser::parse(&tokens).unwrap();
        let error = analyze(&ast, &context.types).unwrap_err();
        assert_eq!(error.code, "E000206");
    }

    #[test]
    fn conversion_categories_order_exact_before_widening_before_general() {
        let context = severian_bootstrap::load().unwrap();
        let resolve = |name| context.types.resolve_name(name).unwrap();
        let exact = conversion_rank(&context.types, resolve("i32"), resolve("i32")).unwrap();
        let widening = conversion_rank(&context.types, resolve("i32"), resolve("i64")).unwrap();
        let general = conversion_rank(&context.types, resolve("bf16"), resolve("f32")).unwrap();
        assert!(exact < widening);
        assert!(widening < general);
    }
}
