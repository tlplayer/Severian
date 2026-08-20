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
    Decorator as HirDecorator, DecoratorOption as HirDecoratorOption, DefinitionId,
    DetailedFunctionType, EnumDefinition, Expression, FieldDefinition, Function,
    FunctionContract as HirFunctionContract, FunctionId, FunctionType, Global, HirId, Instruction,
    MatchPattern, OwnershipOp, Parameter, PrimitiveCategory, PrimitiveDefinition, PrimitiveId,
    Program, ProgramMetadata, ReceiverType,
    ScopedBehavior as HirScopedBehavior, SemanticContext, SemanticMember, SourceRange, SourceSpan,
    SwitchArm as HirSwitchArm, TaskPlacement, TensorDimension, TensorElementType, TensorType, Test,
    TestMode as HirTestMode, TraitImplementationDefinition, TraitPropertyDefinition,
    TraitPropertyValue, TraitRegistryDefinition, TypeDefinitionId, TypeId, TypeKind, TypeTable,
    UnaryOp, ValueType, VariantDefinition, VariantId,
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
    result_ok: Option<SignatureType>,
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
    Declared(Type),
}

#[derive(Clone)]
struct GenericTensorType {
    variable: String,
    constraints: Vec<severian_hir::TensorElementConstraint>,
    rank: Option<u8>,
    dimensions: [TensorDimension; 8],
}

impl SignatureType {
    fn resolved(&self, aliases: &HashMap<String, String>) -> ValueType {
        match self {
            Self::Concrete(ty) => *ty,
            Self::TensorGeneric(_) => ValueType::TensorAny,
            Self::Declared(ty) => declared_value_type(ty, aliases),
        }
    }
}

#[derive(Clone)]
struct Binding {
    reference: BindingRef,
    ty: ValueType,
    class: Option<String>,
    /// Statically known variant of a transition-aware enum, when flow analysis
    /// can prove one.
    enum_variant: Option<String>,
    function_return: Option<ValueType>,
    collection_len: Option<usize>,
    mutable: bool,
    field: bool,
    integer_max: Option<i64>,
    known_integer: Option<i64>,
    any_origin: Option<AnyOrigin>,
}

#[derive(Clone, Default)]
struct TraitSemantics {
    decorators: HashMap<String, TraitDecoratorDefinition>,
    namespaces: HashMap<String, TraitSemanticNamespace>,
}

#[derive(Clone)]
struct TraitDecoratorDefinition {
    owner: String,
    policies: Vec<(String, String)>,
}

#[derive(Clone, Default)]
struct TraitSemanticNamespace {
    traits: Vec<String>,
    operators: BTreeMap<String, Vec<TraitMemberProvider>>,
    operations: BTreeMap<String, Vec<TraitMemberProvider>>,
    scoped_behaviors: Vec<TraitScopedBehaviorProvider>,
}

#[derive(Clone, PartialEq, Eq)]
struct TraitMemberProvider {
    trait_name: String,
    qualified_member: String,
}

#[derive(Clone)]
struct TraitScopedBehaviorProvider {
    trait_name: String,
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

mod contracts;
mod control;
mod expression;
mod generics;
mod pipeline;
#[path = "../types/mod.rs"]
mod types;
#[path = "../registry/mod.rs"]
mod registry;
#[path = "../resolve/mod.rs"]
mod resolve;

use contracts::*;
use control::*;
use expression::*;
use generics::*;
pub use pipeline::*;
use types::*;
pub use registry::*;
pub use resolve::enforce_type_resolution_policy;
use resolve::*;
