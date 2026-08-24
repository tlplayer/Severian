use crate::{Function, Module};
use severian_hir::Program as HirProgram;

/// Build executable MIR exactly once, as control-flow graphs. Source-level
/// metadata is copied beside those graphs; no parallel structured operation
/// body is maintained.
pub fn build(hir: &HirProgram) -> Result<Module, crate::VerifyError> {
    let (globals, initializer, mut function_bodies) = crate::cfg::lower_program(hir);
    let mut module = Module {
        globals,
        initializer,
        ..Module::default()
    };

    for hir_module in &hir.modules {
        module
            .classes
            .extend(hir_module.classes.iter().map(|declaration| {
                crate::ClassDeclaration {
                    id: declaration.id,
                    name: declaration.name.clone(),
                    fields: declaration
                        .fields
                        .iter()
                        .map(|field| crate::ClassFieldDeclaration {
                            name: field.name.clone(),
                            ty: field.ty,
                        })
                        .collect(),
                }
            }));
        module
            .traits
            .extend(hir_module.traits.iter().map(|declaration| {
                crate::TraitDeclaration {
                    definition: declaration.definition,
                    name: declaration.name.clone(),
                    methods: declaration
                        .methods
                        .iter()
                        .map(|method| crate::TraitMethodDeclaration {
                            name: method.name.clone(),
                            parameters: method
                                .parameters
                                .iter()
                                .map(|parameter| match parameter {
                                    severian_hir::TraitType::SelfType => crate::TraitType::SelfType,
                                    severian_hir::TraitType::Concrete(ty) => {
                                        crate::TraitType::Concrete(*ty)
                                    }
                                })
                                .collect(),
                            result: match method.result {
                                severian_hir::TraitType::SelfType => crate::TraitType::SelfType,
                                severian_hir::TraitType::Concrete(ty) => {
                                    crate::TraitType::Concrete(ty)
                                }
                            },
                        })
                        .collect(),
                }
            }));
        if hir_module.entry.is_some() {
            module.entry = hir_module.entry;
        }
        module
            .tests
            .extend(hir_module.tests.iter().map(|test| crate::TestDeclaration {
                name: test.name.clone(),
                modes: test.modes.clone(),
                function: test.function,
                expectations: test.expectations.clone(),
            }));
        module
            .functions
            .extend(hir_module.functions.iter().map(|function| {
                Function {
                    id: function.id,
                    definition: function.definition,
                    substitution: function.substitution.clone(),
                name: function.name.clone(),
                parameters: function
                    .parameters
                    .iter()
                    .map(|parameter| parameter.contract.ty)
                    .collect(),
                    result: function.result.ty,
                    body: function_bodies.remove(&function.id),
                    call_type: function.call_type.clone(),
                }
            }));
    }

    crate::verify::verify_structure(&module)?;
    Ok(module)
}
