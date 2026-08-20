use severian_hir::{
    BindingRef, Decorator, FunctionId, HirId, MatchPattern, ScopedBehavior, ValueType,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct BlockId(pub u32);

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct LocalId(pub u32);

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Local {
    pub id: LocalId,
    pub binding: BindingRef,
    pub ty: ValueType,
}

#[derive(Clone, PartialEq, Eq)]
pub struct Program {
    pub(crate) hir: severian_hir::Program,
    pub functions: Vec<Function>,
}

impl Default for Program {
    fn default() -> Self {
        Self {
            hir: severian_hir::Program::default(),
            functions: Vec::new(),
        }
    }
}

impl std::fmt::Debug for Program {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("MirProgram")
            .field("functions", &self.functions)
            .finish()
    }
}

impl Program {
    /// The structured expression payload consumed by the current MLIR
    /// lowering. Control-flow ownership lives in MIR and consumers must enter
    /// lowering through this type.
    pub fn lowering_hir(&self) -> &severian_hir::Program {
        &self.hir
    }

    /// HIR-v2 metadata is carried through MIR as an inert sidecar. MIR and
    /// lowering do not interpret it yet, but downstream migrations can query
    /// canonical source spans and detailed types without recovering AST data.
    pub fn metadata(&self) -> &severian_hir::ProgramMetadata {
        &self.hir.metadata
    }

    pub fn source_span(&self, value: ValueRef) -> Option<severian_hir::SourceSpan> {
        value
            .id
            .and_then(|id| self.hir.metadata.sources.expression_span(id))
    }

    /// The resolved semantic type carried across the HIR -> MIR boundary.
    /// MIR never reconstructs this information from source spelling.
    pub fn resolved_type(&self, value: ValueRef) -> Option<severian_hir::TypeId> {
        value
            .id
            .and_then(|id| self.hir.metadata.expression_types.get(&id).copied())
    }

    pub fn primitive(&self, value: ValueRef) -> Option<severian_hir::PrimitiveId> {
        let ty = self.resolved_type(value)?;
        match self.hir.metadata.types.get(ty)? {
            severian_hir::TypeKind::Primitive(primitive) => Some(*primitive),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Function {
    pub id: FunctionId,
    pub name: String,
    pub native_symbol: Option<String>,
    pub decorators: Vec<Decorator>,
    pub parameters: Vec<LocalId>,
    pub locals: Vec<Local>,
    pub return_type: ValueType,
    pub(crate) source_tensor_intrinsics: usize,
    pub tensor_operations: Vec<TensorOp>,
    /// Typed calls crossing package-owned foreign ABI boundaries. MIR records
    /// argument/result identities without interpreting the requested domain.
    pub foreign_calls: Vec<ForeignCall>,
    pub blocks: Vec<BasicBlock>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ForeignCall {
    pub function: severian_abi::ExternalFunction,
    pub arguments: Vec<ValueRef>,
    pub result: ValueRef,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BasicBlock {
    pub id: BlockId,
    pub operations: Vec<Operation>,
    pub terminator: Terminator,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Operation {
    pub kind: OperationKind,
    pub operands: Vec<ValueRef>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum OperationKind {
    Bind(LocalId),
    TryBind(LocalId),
    Assign,
    Print,
    Assert,
    Evaluate,
    With,
    ScopeEnter(ScopedBehavior),
    ScopeExit(ScopedBehavior),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ValueRef {
    pub id: Option<HirId>,
    pub ty: Option<ValueType>,
    pub local: Option<LocalId>,
    pub tensor_op: Option<TensorOpId>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Terminator {
    Return(Option<ValueRef>),
    Goto(BlockId),
    Branch {
        condition: ValueRef,
        then_block: BlockId,
        else_block: BlockId,
    },
    Loop {
        condition: ValueRef,
        body: BlockId,
        exit: BlockId,
    },
    For {
        pattern: MatchPattern,
        iterable: ValueRef,
        body: BlockId,
        exit: BlockId,
    },
    Switch {
        values: Vec<ValueRef>,
        arms: Vec<BlockId>,
        exit: BlockId,
    },
    Break,
    Continue,
    Unreachable,
}
