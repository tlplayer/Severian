#![forbid(unsafe_code)]

mod build;
mod cfg;
#[path = "model/operation/mod.rs"]
mod operation;
mod ownership;
mod passes;
#[path = "model/value/mod.rs"]
mod value;
mod verify;

pub use build::build;
pub use cfg::{
    BasicBlock, BlockId, Body as CfgBody, Callee, Case, LocalDecl, LocalId, Operand, Place,
    Projection, Rvalue, Statement as CfgStatement, Terminator,
};
pub use operation::Operation;
pub use ownership::{
    analyze_ownership, elaborate_drops, Loan, LoanKind, OwnershipError, OwnershipReport,
};
pub use passes::{
    run_required_pipeline, AnalysisId, AnalysisManager, IrStage, Pass, PassContext, PassError,
    PassKind, PassManager, PassMetadata,
};
use severian_hir::BindingId;
pub use severian_hir::{CallType, FunctionId};
pub use value::{Value, ValueId};
pub use verify::{verify, VerifyError};

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct Module {
    pub values: Vec<Value>,
    pub bindings: Vec<(BindingId, ValueId)>,
    pub globals: Vec<ValueId>,
    pub initializer: Block,
    pub initializer_cfg: CfgBody,
    pub functions: Vec<Function>,
    pub entry: Option<FunctionId>,
    pub tests: Vec<TestDeclaration>,
    pub traits: Vec<TraitDeclaration>,
    pub classes: Vec<ClassDeclaration>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ClassFieldDeclaration {
    pub name: String,
    pub ty: severian_universal::TypeId,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ClassDeclaration {
    pub id: severian_universal::TypeId,
    pub name: String,
    pub fields: Vec<ClassFieldDeclaration>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TraitMethodDeclaration {
    pub name: String,
    pub parameters: Vec<TraitType>,
    pub result: TraitType,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TraitType {
    SelfType,
    Concrete(severian_universal::TypeId),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TraitDeclaration {
    pub definition: severian_universal::DefId,
    pub name: String,
    pub methods: Vec<TraitMethodDeclaration>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TestDeclaration {
    pub name: String,
    pub modes: Vec<TestMode>,
    pub function: FunctionId,
    pub expectations: Vec<TestExpectation>,
}

pub use severian_hir::{TestExpectation, TestMode, TestStream};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AssertionOrigin {
    pub statement_start: u32,
    pub condition_start: u32,
    pub condition_end: u32,
    pub location: Option<AssertionLocation>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AssertionLocation {
    pub file: String,
    pub line: u32,
    pub column: u32,
    pub expression: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum CoverageKind {
    Line,
    Branch,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct CoveragePoint {
    pub span_start: u32,
    pub kind: CoverageKind,
    pub ordinal: u32,
    pub key: Option<String>,
    pub file: Option<String>,
    pub line: Option<u32>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MatchArm {
    pub type_id: Option<severian_universal::TypeId>,
    pub body: Block,
}

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct Block {
    pub operations: Vec<Operation>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Function {
    pub id: FunctionId,
    pub definition: severian_universal::DefId,
    pub substitution: severian_universal::Substitution,
    pub name: String,
    pub parameters: Vec<ValueId>,
    pub result: severian_universal::TypeId,
    pub body: Option<Block>,
    pub cfg: Option<CfgBody>,
    pub call_type: CallType,
}
