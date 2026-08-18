use severian_interface::CompileTypeId;

/// Routing metadata attached to an already-resolved HIR operation.
///
/// `None` means the normal Severian compiler pipeline owns the operation.
/// `Some(id)` means MIR must preserve the id so lowering can route the reduced
/// operation to the package handler declared by `severian-interface`.
#[derive(Clone, Debug, Default, PartialEq, Eq, Hash)]
pub struct CompileRoute {
    compile_type: Option<CompileTypeId>,
}

impl CompileRoute {
    pub const fn core() -> Self {
        Self { compile_type: None }
    }

    pub fn extension(compile_type: CompileTypeId) -> Self {
        Self {
            compile_type: Some(compile_type),
        }
    }

    pub fn compile_type(&self) -> Option<&CompileTypeId> {
        self.compile_type.as_ref()
    }

    pub fn is_core(&self) -> bool {
        self.compile_type.is_none()
    }
}

impl From<CompileTypeId> for CompileRoute {
    fn from(value: CompileTypeId) -> Self {
        Self::extension(value)
    }
}
