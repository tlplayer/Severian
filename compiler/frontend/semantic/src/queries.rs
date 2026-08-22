use crate::{DefId, DefKind, FunctionDecl, ProgramIndex, Resolution, TypedProgram};
use severian_hir::FunctionDeclaration;
use severian_modules::ModuleId;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ScopeId(pub ModuleId);

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum QueryError {
    UnknownModule(ModuleId),
    UnknownName { scope: ScopeId, name: String },
    Ambiguous(Vec<DefId>),
    NotFunction(DefId),
    MissingBody(DefId),
}

/// Pure item-keyed requests over the package semantic database. The current
/// implementation is eager; these keys are suitable for a cached query engine
/// without changing callers later.
pub struct SemanticQueries<'a> {
    program: &'a TypedProgram,
}

impl<'a> SemanticQueries<'a> {
    pub const fn new(program: &'a TypedProgram) -> Self {
        Self { program }
    }

    pub fn collect_declarations(&self, module: ModuleId) -> Result<&[DefId], QueryError> {
        self.program
            .index
            .modules
            .get(&module)
            .map(|module| module.items.as_slice())
            .ok_or(QueryError::UnknownModule(module))
    }

    pub fn resolve_name(&self, scope: ScopeId, name: &str) -> Result<DefId, QueryError> {
        let resolution = self
            .program
            .index
            .modules
            .get(&scope.0)
            .ok_or(QueryError::UnknownModule(scope.0))?
            .scope
            .bindings
            .get(name)
            .ok_or_else(|| QueryError::UnknownName {
                scope,
                name: name.to_owned(),
            })?;
        match resolution {
            Resolution::Def(definition) => Ok(*definition),
            Resolution::OverloadSet(definitions) | Resolution::Ambiguous(definitions) => {
                Err(QueryError::Ambiguous(definitions.clone()))
            }
            Resolution::Module(_) => Err(QueryError::UnknownName {
                scope,
                name: name.to_owned(),
            }),
        }
    }

    pub fn signature_of(&self, definition: DefId) -> Result<&FunctionDecl, QueryError> {
        match self
            .program
            .index
            .definitions
            .get(&definition)
            .map(|item| &item.kind)
        {
            Some(DefKind::Function(signature)) => Ok(signature),
            _ => Err(QueryError::NotFunction(definition)),
        }
    }

    pub fn type_check(&self, definition: DefId) -> Result<&FunctionDeclaration, QueryError> {
        self.program
            .hir
            .modules
            .iter()
            .flat_map(|module| &module.functions)
            .find(|function| function.definition == definition)
            .ok_or(QueryError::MissingBody(definition))
    }

    pub const fn index(&self) -> &ProgramIndex {
        &self.program.index
    }
}
