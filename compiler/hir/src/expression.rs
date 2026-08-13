use crate::*;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Expression {
    Typed {
        id: HirId,
        ty: ValueType,
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
    pub returns: ValueType,
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
        self.signature = Some(FunctionType {
            parameters: parameters.into_iter().collect(),
            returns,
        });
        self
    }

    pub fn lowering_symbol(&self) -> &str {
        self.native_symbol.as_deref().unwrap_or(&self.name)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ComprehensionClause {
    pub pattern: MatchPattern,
    pub iterable: Expression,
    pub condition: Option<Expression>,
}
