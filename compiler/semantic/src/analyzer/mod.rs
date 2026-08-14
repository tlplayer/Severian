#![forbid(unsafe_code)]

use severian_ast::{
    AssignOp as AstAssignOp, BinaryOp as AstBinaryOp, Block, ElseBranch, Expr, ImportKind, Item,
    LetKind, Literal, Module, OwnershipOp as AstOwnershipOp, Pattern, Span, Stmt, Type, TypeArg,
    UnaryOp as AstUnaryOp,
};
use severian_hir::{
    AnyOrigin, AssignmentOp, BinaryOp, BindingId, BindingRef, CallTarget,
    ChaosAction as HirChaosAction, Class, ClassDefinition,
    ComprehensionClause as HirComprehensionClause, ContractClause as HirContractClause,
    Decorator as HirDecorator, DefinitionId, DetailedFunctionType, EnumDefinition, Expression,
    FieldDefinition, Function, FunctionContract as HirFunctionContract, FunctionId, FunctionType,
    Global, HirId, Instruction, MatchPattern, OwnershipOp, Parameter, Program, ProgramMetadata,
    ReceiverType, SourceRange, SourceSpan, SwitchArm as HirSwitchArm, TaskPlacement,
    TensorDimension, TensorElementType, TensorType, Test, TestMode as HirTestMode,
    TypeDefinitionId, TypeId, TypeKind, TypeTable, UnaryOp, ValueType, VariantDefinition,
    VariantId,
};
use severian_package::{local_import_exposed_name, local_import_module_name, PackageInterface};
use std::collections::{BTreeMap, HashMap, HashSet};
use std::fmt;
use std::path::PathBuf;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SemanticError {
    pub span: Span,
    pub message: String,
}

impl fmt::Display for SemanticError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "{} at bytes {}..{}",
            self.message, self.span.start, self.span.end
        )
    }
}

impl std::error::Error for SemanticError {}

#[derive(Clone)]
struct Signature {
    target: CallTarget,
    params: Vec<SignatureParameter>,
    returns: SignatureType,
}

#[derive(Clone)]
struct SignatureParameter {
    name: String,
    ty: SignatureType,
    function_return: Option<ValueType>,
    default: Option<Expr>,
    any_origin: Option<AnyOrigin>,
}

#[derive(Clone)]
enum SignatureType {
    Concrete(ValueType),
    TensorGeneric(GenericTensorType),
}

#[derive(Clone)]
struct GenericTensorType {
    variable: String,
    constraints: Vec<severian_hir::TensorElementConstraint>,
    rank: Option<u8>,
    dimensions: [TensorDimension; 8],
}

impl SignatureType {
    fn erased(&self) -> ValueType {
        match self {
            Self::Concrete(ty) => *ty,
            Self::TensorGeneric(_) => ValueType::TensorAny,
        }
    }
}

#[derive(Clone)]
struct Binding {
    reference: BindingRef,
    ty: ValueType,
    class: Option<String>,
    function_return: Option<ValueType>,
    collection_len: Option<usize>,
    mutable: bool,
    field: bool,
    integer_max: Option<i64>,
    known_integer: Option<i64>,
    any_origin: Option<AnyOrigin>,
}

fn source_binding(identifier: &severian_ast::Ident) -> BindingRef {
    BindingRef::source(
        identifier.name.clone(),
        identifier.span.start,
        identifier.span.end,
    )
}

fn named_binding(name: impl Into<String>, identity: impl AsRef<str>) -> BindingRef {
    BindingRef::new(BindingId::from_name(identity.as_ref()), name)
}

fn declared_any_origin(ty: Option<&Type>, resolved: ValueType) -> Option<AnyOrigin> {
    matches!(resolved, ValueType::Any | ValueType::TensorAny).then_some(if ty.is_some() {
        AnyOrigin::Explicit
    } else {
        AnyOrigin::InferenceFallback
    })
}

mod call;
mod contract;
mod expression;
mod expression_helpers;
mod metadata;
mod pattern;
mod pipeline;
mod resolve;
mod statement;
mod switch;
mod tensor;
mod traits;
mod type_resolution;
mod types;

use call::*;
use contract::*;
use expression::*;
use expression_helpers::*;
pub use metadata::*;
use pattern::*;
pub use pipeline::*;
use resolve::*;
use statement::*;
use switch::*;
use traits::*;
pub use type_resolution::enforce_type_resolution_policy;
use types::*;
