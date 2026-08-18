use std::collections::HashMap;
use std::error::Error;
use std::fmt;

use crate::{
    CompileType, CompileTypeId, ExternalDeclaration, ExternalId, Implementation, Interface,
    ModuleId, ModuleInterface, ModulePath, PackageId, Symbol, SymbolId, SymbolKind, TypeId,
};

#[derive(Debug)]
pub struct InterfaceResolver {
    interfaces: Vec<Interface>,
    packages: HashMap<PackageId, usize>,
    modules: HashMap<ModuleId, (usize, usize)>,
    modules_by_path: HashMap<(PackageId, ModulePath), ModuleId>,
    symbols: HashMap<SymbolId, (usize, usize, usize)>,
    symbols_by_name: HashMap<(ModuleId, String), SymbolId>,
    externals: HashMap<ExternalId, (usize, usize)>,
    implementations_by_trait: HashMap<TypeId, Vec<(usize, usize)>>,
    compile_types: HashMap<CompileTypeId, (usize, usize)>,
    compile_types_by_name: HashMap<(PackageId, String), CompileTypeId>,
}

impl InterfaceResolver {
    pub fn new(interfaces: Vec<Interface>) -> Result<Self, ResolveError> {
        let mut resolver = Self {
            interfaces,
            packages: HashMap::new(),
            modules: HashMap::new(),
            modules_by_path: HashMap::new(),
            symbols: HashMap::new(),
            symbols_by_name: HashMap::new(),
            externals: HashMap::new(),
            implementations_by_trait: HashMap::new(),
            compile_types: HashMap::new(),
            compile_types_by_name: HashMap::new(),
        };
        resolver.index()?;
        Ok(resolver)
    }

    pub fn interface(&self, id: &PackageId) -> Option<&Interface> {
        self.packages.get(id).map(|index| &self.interfaces[*index])
    }

    pub fn resolve_module(&self, package: &PackageId, path: &ModulePath) -> Option<&ModuleInterface> {
        let id = self
            .modules_by_path
            .get(&(package.clone(), path.clone()))?;
        self.module(id)
    }

    pub fn module(&self, id: &ModuleId) -> Option<&ModuleInterface> {
        let (interface_index, module_index) = *self.modules.get(id)?;
        Some(&self.interfaces[interface_index].modules[module_index])
    }

    pub fn resolve_symbol(&self, module: &ModuleId, name: &str) -> Option<&Symbol> {
        let id = self
            .symbols_by_name
            .get(&(module.clone(), name.to_owned()))?;
        self.symbol(id)
    }

    pub fn resolve_export(&self, module: &ModuleId, name: &str) -> Option<&Symbol> {
        let module_interface = self.module(module)?;
        let symbol = self.resolve_symbol(module, name)?;
        module_interface
            .exports
            .contains(&symbol.id)
            .then_some(symbol)
    }

    pub fn symbol(&self, id: &SymbolId) -> Option<&Symbol> {
        let (interface_index, module_index, symbol_index) = *self.symbols.get(id)?;
        Some(&self.interfaces[interface_index].modules[module_index].symbols[symbol_index])
    }

    pub fn external(&self, id: &ExternalId) -> Option<&ExternalDeclaration> {
        let (interface_index, external_index) = *self.externals.get(id)?;
        Some(&self.interfaces[interface_index].externals[external_index])
    }

    pub fn compile_type(&self, id: &CompileTypeId) -> Option<&CompileType> {
        let (interface_index, compile_type_index) = *self.compile_types.get(id)?;
        Some(&self.interfaces[interface_index].compile_types[compile_type_index])
    }

    pub fn resolve_compile_type(&self, package: &PackageId, name: &str) -> Option<&CompileType> {
        let id = self
            .compile_types_by_name
            .get(&(package.clone(), name.to_owned()))?;
        self.compile_type(id)
    }

    pub fn compile_handler(&self, id: &CompileTypeId) -> Option<&Symbol> {
        let compile_type = self.compile_type(id)?;
        self.symbol(&compile_type.handler)
    }

    pub fn implementations_for_trait(&self, trait_id: &TypeId) -> Vec<&Implementation> {
        self.implementations_by_trait
            .get(trait_id)
            .into_iter()
            .flatten()
            .map(|(interface_index, implementation_index)| {
                &self.interfaces[*interface_index].implementations[*implementation_index]
            })
            .collect()
    }

    pub fn interfaces(&self) -> &[Interface] {
        &self.interfaces
    }

    fn index(&mut self) -> Result<(), ResolveError> {
        // First pass: packages, modules and symbols. Compile handlers are symbols,
        // so all symbol identities must exist before compile types are validated.
        for interface_index in 0..self.interfaces.len() {
            let interface = &self.interfaces[interface_index];
            let package = interface.id.clone();

            if self.packages.insert(package.clone(), interface_index).is_some() {
                return Err(ResolveError::DuplicatePackage(package));
            }

            let mut root_found = false;

            for module_index in 0..interface.modules.len() {
                let module = &interface.modules[module_index];

                if module.id.package != package {
                    return Err(ResolveError::ForeignModuleId {
                        package: package.clone(),
                        module: module.id.clone(),
                    });
                }

                if module.id == interface.root {
                    root_found = true;
                }

                if self
                    .modules
                    .insert(module.id.clone(), (interface_index, module_index))
                    .is_some()
                {
                    return Err(ResolveError::DuplicateModuleId(module.id.clone()));
                }

                let path_key = (package.clone(), module.path.clone());
                if self
                    .modules_by_path
                    .insert(path_key, module.id.clone())
                    .is_some()
                {
                    return Err(ResolveError::DuplicateModulePath {
                        package: package.clone(),
                        path: module.path.clone(),
                    });
                }

                for symbol_index in 0..module.symbols.len() {
                    let symbol = &module.symbols[symbol_index];

                    if symbol.id.module != module.id {
                        return Err(ResolveError::ForeignSymbolId {
                            module: module.id.clone(),
                            symbol: symbol.id.clone(),
                        });
                    }

                    if self
                        .symbols
                        .insert(
                            symbol.id.clone(),
                            (interface_index, module_index, symbol_index),
                        )
                        .is_some()
                    {
                        return Err(ResolveError::DuplicateSymbolId(symbol.id.clone()));
                    }

                    let name_key = (module.id.clone(), symbol.name.clone());
                    if self
                        .symbols_by_name
                        .insert(name_key, symbol.id.clone())
                        .is_some()
                    {
                        return Err(ResolveError::DuplicateSymbolName {
                            module: module.id.clone(),
                            name: symbol.name.clone(),
                        });
                    }
                }

                for export in &module.exports {
                    if export.module != module.id || !module.symbols.iter().any(|s| s.id == *export) {
                        return Err(ResolveError::UnknownExport {
                            module: module.id.clone(),
                            symbol: export.clone(),
                        });
                    }
                }
            }

            if !root_found {
                return Err(ResolveError::MissingRootModule {
                    package: package.clone(),
                    root: interface.root.clone(),
                });
            }
        }

        // Second pass: compile domains. This permits the handler symbol to live
        // anywhere in its owning package without depending on module ordering.
        for interface_index in 0..self.interfaces.len() {
            let interface = &self.interfaces[interface_index];
            let package = interface.id.clone();

            for compile_type_index in 0..interface.compile_types.len() {
                let compile_type = &interface.compile_types[compile_type_index];

                if compile_type.id.package != package {
                    return Err(ResolveError::ForeignCompileTypeId {
                        package: package.clone(),
                        compile_type: compile_type.id.clone(),
                    });
                }

                if compile_type.handler.module.package != package
                    || !self.symbols.contains_key(&compile_type.handler)
                {
                    return Err(ResolveError::UnknownCompileHandler {
                        compile_type: compile_type.id.clone(),
                        handler: compile_type.handler.clone(),
                    });
                }

                if self
                    .compile_types
                    .insert(
                        compile_type.id.clone(),
                        (interface_index, compile_type_index),
                    )
                    .is_some()
                {
                    return Err(ResolveError::DuplicateCompileTypeId(
                        compile_type.id.clone(),
                    ));
                }

                let name_key = (package.clone(), compile_type.name.clone());
                if self
                    .compile_types_by_name
                    .insert(name_key, compile_type.id.clone())
                    .is_some()
                {
                    return Err(ResolveError::DuplicateCompileTypeName {
                        package: package.clone(),
                        name: compile_type.name.clone(),
                    });
                }
            }
        }

        // Third pass: validate references to compile domains and index the
        // remaining package-level interface data.
        for interface_index in 0..self.interfaces.len() {
            let interface = &self.interfaces[interface_index];
            let package = interface.id.clone();

            for module in &interface.modules {
                for symbol in &module.symbols {
                    let compile_type = match &symbol.kind {
                        SymbolKind::Class(class) => class.compile_type.as_ref(),
                        SymbolKind::Function(function) => function.compile_type.as_ref(),
                        _ => None,
                    };

                    if let Some(compile_type) = compile_type {
                        if !self.compile_types.contains_key(compile_type) {
                            return Err(ResolveError::UnknownCompileType {
                                symbol: symbol.id.clone(),
                                compile_type: compile_type.clone(),
                            });
                        }
                    }
                }
            }

            for external_index in 0..interface.externals.len() {
                let external = &interface.externals[external_index];
                if external.id.package != package {
                    return Err(ResolveError::ForeignExternalId {
                        package: package.clone(),
                        external: external.id.clone(),
                    });
                }

                if self
                    .externals
                    .insert(external.id.clone(), (interface_index, external_index))
                    .is_some()
                {
                    return Err(ResolveError::DuplicateExternalId(external.id.clone()));
                }
            }

            for implementation_index in 0..interface.implementations.len() {
                let implementation = &interface.implementations[implementation_index];
                if implementation.id.package != package {
                    return Err(ResolveError::ForeignImplementationId {
                        package: package.clone(),
                        implementation: implementation.id.clone(),
                    });
                }

                if let Some(trait_id) = &implementation.trait_id {
                    self.implementations_by_trait
                        .entry(trait_id.clone())
                        .or_default()
                        .push((interface_index, implementation_index));
                }
            }
        }

        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ResolveError {
    DuplicatePackage(PackageId),
    DuplicateModuleId(ModuleId),
    DuplicateModulePath {
        package: PackageId,
        path: ModulePath,
    },
    DuplicateSymbolId(SymbolId),
    DuplicateSymbolName {
        module: ModuleId,
        name: String,
    },
    DuplicateExternalId(ExternalId),
    DuplicateCompileTypeId(CompileTypeId),
    DuplicateCompileTypeName {
        package: PackageId,
        name: String,
    },
    ForeignModuleId {
        package: PackageId,
        module: ModuleId,
    },
    ForeignSymbolId {
        module: ModuleId,
        symbol: SymbolId,
    },
    ForeignExternalId {
        package: PackageId,
        external: ExternalId,
    },
    ForeignCompileTypeId {
        package: PackageId,
        compile_type: CompileTypeId,
    },
    ForeignImplementationId {
        package: PackageId,
        implementation: crate::ImplementationId,
    },
    MissingRootModule {
        package: PackageId,
        root: ModuleId,
    },
    UnknownExport {
        module: ModuleId,
        symbol: SymbolId,
    },
    UnknownCompileHandler {
        compile_type: CompileTypeId,
        handler: SymbolId,
    },
    UnknownCompileType {
        symbol: SymbolId,
        compile_type: CompileTypeId,
    },
}

impl fmt::Display for ResolveError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::DuplicatePackage(package) => write!(f, "duplicate interface for `{package}`"),
            Self::DuplicateModuleId(module) => {
                write!(f, "duplicate module id `{}:{}`", module.package, module.local)
            }
            Self::DuplicateModulePath { package, path } => {
                write!(f, "duplicate module path `{package}:{path}`")
            }
            Self::DuplicateSymbolId(symbol) => write!(
                f,
                "duplicate symbol id `{}:{}:{}`",
                symbol.module.package, symbol.module.local, symbol.local
            ),
            Self::DuplicateSymbolName { module, name } => write!(
                f,
                "duplicate symbol `{name}` in module `{}:{}`",
                module.package, module.local
            ),
            Self::DuplicateExternalId(external) => write!(
                f,
                "duplicate external id `{}:{}`",
                external.package, external.local
            ),
            Self::DuplicateCompileTypeId(compile_type) => write!(
                f,
                "duplicate compile type id `{}:{}`",
                compile_type.package, compile_type.local
            ),
            Self::DuplicateCompileTypeName { package, name } => {
                write!(f, "duplicate compile type `{name}` in package `{package}`")
            }
            Self::ForeignModuleId { package, module } => write!(
                f,
                "module `{}:{}` does not belong to package `{package}`",
                module.package, module.local
            ),
            Self::ForeignSymbolId { module, symbol } => write!(
                f,
                "symbol `{}:{}:{}` does not belong to module `{}:{}`",
                symbol.module.package,
                symbol.module.local,
                symbol.local,
                module.package,
                module.local
            ),
            Self::ForeignExternalId { package, external } => write!(
                f,
                "external `{}:{}` does not belong to package `{package}`",
                external.package, external.local
            ),
            Self::ForeignCompileTypeId {
                package,
                compile_type,
            } => write!(
                f,
                "compile type `{}:{}` does not belong to package `{package}`",
                compile_type.package, compile_type.local
            ),
            Self::ForeignImplementationId {
                package,
                implementation,
            } => write!(
                f,
                "implementation `{}:{}` does not belong to package `{package}`",
                implementation.package, implementation.local
            ),
            Self::MissingRootModule { package, root } => write!(
                f,
                "root module `{}:{}` is missing from package `{package}`",
                root.package, root.local
            ),
            Self::UnknownExport { module, symbol } => write!(
                f,
                "module `{}:{}` exports unknown symbol `{}:{}:{}`",
                module.package,
                module.local,
                symbol.module.package,
                symbol.module.local,
                symbol.local
            ),
            Self::UnknownCompileHandler {
                compile_type,
                handler,
            } => write!(
                f,
                "compile type `{}:{}` references unknown handler `{}:{}:{}`",
                compile_type.package,
                compile_type.local,
                handler.module.package,
                handler.module.local,
                handler.local
            ),
            Self::UnknownCompileType {
                symbol,
                compile_type,
            } => write!(
                f,
                "symbol `{}:{}:{}` references unknown compile type `{}:{}`",
                symbol.module.package,
                symbol.module.local,
                symbol.local,
                compile_type.package,
                compile_type.local
            ),
        }
    }
}

impl Error for ResolveError {}
