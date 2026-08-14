use crate::*;

/// Why an expression still has an erased dynamic type after semantic
/// resolution. `Explicit` is source intent; every other variant describes
/// compiler information that a strict package may forbid from escaping into
/// MIR.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum AnyOrigin {
    Explicit,
    InferenceFallback,
    UnresolvedType,
    UnresolvedGeneric,
    LostTypeInformation,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Expression {
    Typed {
        id: HirId,
        ty: ValueType,
        any_origin: Option<AnyOrigin>,
        expression: Box<Expression>,
    },
    Integer(i64),
    Float(u64),
    Boolean(bool),
    String(String),
    Variable(BindingRef),
    Function(CallTarget),
    Lambda {
        params: Vec<BindingRef>,
        body: Box<Expression>,
    },
    Closure {
        params: Vec<Parameter>,
        body: Vec<Instruction>,
        return_type: ValueType,
    },
    Ownership {
        op: OwnershipOp,
        value: Box<Expression>,
    },
    List(Vec<Expression>),
    Tuple(Vec<Expression>),
    Map(Vec<(Expression, Expression)>),
    Set(Vec<Expression>),
    Index {
        object: Box<Expression>,
        index: Box<Expression>,
    },
    Slice {
        object: Box<Expression>,
        start: Option<Box<Expression>>,
        end: Option<Box<Expression>>,
        step: Option<Box<Expression>>,
    },
    Format {
        template: String,
        args: Vec<Expression>,
        arg_types: Vec<ValueType>,
    },
    PrintArgs(Vec<Expression>),
    Construct {
        type_id: TypeDefinitionId,
        class: String,
        args: Vec<Expression>,
    },
    /// Field-oriented construction used by generated builders and structural
    /// conversion. Unlike constructor calls, values are named and may be sparse;
    /// class defaults are applied only after the complete construction plan is
    /// known.
    ConstructFields {
        type_id: TypeDefinitionId,
        class: String,
        fields: Vec<(String, Expression)>,
        /// False only for the hidden receiver used to invoke an explicit static
        /// conversion hook such as `Target.from(source)`.
        validate: bool,
    },
    /// A transactional copy/conversion. Fields not present in `fields` are read
    /// from `object` by name, then the resulting target object is validated once.
    ObjectUpdate {
        object: Box<Expression>,
        type_id: TypeDefinitionId,
        class: String,
        fields: Vec<(String, Expression)>,
        /// Read inherited fields from a canonical JSON document instead of a
        /// Severian class object.
        json_document: bool,
    },
    /// Materialize the public fields of an object as a canonical JSON map.
    ObjectDocument {
        object: Box<Expression>,
        fields: Vec<String>,
    },
    Member {
        object: Box<Expression>,
        member: String,
    },
    MethodCall {
        object: Box<Expression>,
        method: String,
        args: Vec<Expression>,
    },
    Variant {
        type_id: Option<TypeDefinitionId>,
        variant_id: VariantId,
        name: String,
        fields: Vec<Expression>,
    },
    Task {
        value: Box<Expression>,
        placement: TaskPlacement,
    },
    Await(Box<Expression>),
    Channel(Box<Expression>),
    Send {
        value: Box<Expression>,
        channel: Box<Expression>,
    },
    ChaosRule {
        function: CallTarget,
        action: ChaosAction,
        value: Box<Expression>,
    },
    ListComprehension {
        element: Box<Expression>,
        clauses: Vec<ComprehensionClause>,
    },
    SetComprehension {
        element: Box<Expression>,
        clauses: Vec<ComprehensionClause>,
    },
    MapComprehension {
        key: Box<Expression>,
        value: Box<Expression>,
        clauses: Vec<ComprehensionClause>,
    },
    Conditional {
        condition: Box<Expression>,
        then_expression: Box<Expression>,
        else_expression: Box<Expression>,
    },
    /// A package-declared elementwise pipeline implemented by one native ABI
    /// entry point. The HIR deliberately does not know model operation names.
    FusedPipeline {
        input: Box<Expression>,
        runtime_symbol: String,
        operations: Vec<u8>,
        packing_bits: u8,
    },
    Unary {
        op: UnaryOp,
        expression: Box<Expression>,
    },
    Binary {
        left: Box<Expression>,
        op: BinaryOp,
        right: Box<Expression>,
    },
    Call {
        target: CallTarget,
        args: Vec<Expression>,
    },
    CallValue {
        callee: Box<Expression>,
        args: Vec<Expression>,
        return_type: ValueType,
    },
}

impl Expression {
    pub fn kind(&self) -> &Self {
        match self {
            Self::Typed { expression, .. } => expression.kind(),
            expression => expression,
        }
    }

    pub fn ty(&self) -> Option<ValueType> {
        match self {
            Self::Typed { ty, .. } => Some(*ty),
            _ => None,
        }
    }

    pub fn hir_id(&self) -> Option<HirId> {
        match self {
            Self::Typed { id, .. } => Some(*id),
            _ => None,
        }
    }

    pub fn any_origin(&self) -> Option<AnyOrigin> {
        match self {
            Self::Typed { any_origin, .. } => *any_origin,
            _ => None,
        }
    }

    pub fn into_kind(self) -> Self {
        match self {
            Self::Typed { expression, .. } => expression.into_kind(),
            expression => expression,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct CallTarget {
    pub id: FunctionId,
    pub name: String,
    pub native_symbol: Option<String>,
    pub signature: Option<FunctionType>,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct FunctionType {
    pub parameters: Vec<ValueType>,
    pub parameter_any_origins: Vec<Option<AnyOrigin>>,
    pub returns: ValueType,
    pub return_any_origin: Option<AnyOrigin>,
}

impl CallTarget {
    pub fn source(name: impl Into<String>) -> Self {
        let name = name.into();
        Self {
            id: FunctionId::from_name(&name),
            name,
            native_symbol: None,
            signature: None,
        }
    }

    pub fn native(name: impl Into<String>, native_symbol: impl Into<String>) -> Self {
        let name = name.into();
        Self {
            id: FunctionId::from_name(&name),
            name,
            native_symbol: Some(native_symbol.into()),
            signature: None,
        }
    }

    pub fn with_signature(
        mut self,
        parameters: impl IntoIterator<Item = ValueType>,
        returns: ValueType,
    ) -> Self {
        let parameters = parameters.into_iter().collect::<Vec<_>>();
        self.signature = Some(FunctionType {
            parameter_any_origins: parameters
                .iter()
                .copied()
                .map(explicit_any_origin)
                .collect(),
            parameters,
            returns,
            return_any_origin: explicit_any_origin(returns),
        });
        self
    }

    pub fn with_signature_origins(
        mut self,
        parameter_any_origins: Vec<Option<AnyOrigin>>,
        return_any_origin: Option<AnyOrigin>,
    ) -> Self {
        if let Some(signature) = &mut self.signature {
            debug_assert_eq!(signature.parameters.len(), parameter_any_origins.len());
            signature.parameter_any_origins = parameter_any_origins;
            signature.return_any_origin = return_any_origin;
        }
        self
    }

    pub fn lowering_symbol(&self) -> &str {
        self.native_symbol.as_deref().unwrap_or(&self.name)
    }

    pub fn tensor_intrinsic(&self) -> Option<TensorIntrinsic> {
        self.native_symbol
            .as_deref()
            .and_then(TensorIntrinsic::from_native_symbol)
    }
}

fn explicit_any_origin(ty: ValueType) -> Option<AnyOrigin> {
    matches!(ty, ValueType::Any | ValueType::TensorAny).then_some(AnyOrigin::Explicit)
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ComprehensionClause {
    pub pattern: MatchPattern,
    pub iterable: Expression,
    pub condition: Option<Expression>,
}
