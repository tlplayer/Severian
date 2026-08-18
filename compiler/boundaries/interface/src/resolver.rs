use std::collections::HashMap;
use std::error::Error;
use std::fmt;

use crate::{
    ExternalDeclaration, ExternalId, Implementation, Interface, ModuleId, ModuleInterface,
    ModulePath, PackageId, Symbol, SymbolId, TypeId,
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
                "interface `{package}` is missing root module `{}:{}`",
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
        }
    }
}

impl Error for ResolveError {}
