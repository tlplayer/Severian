#![forbid(unsafe_code)]

use std::{collections::BTreeMap, path::PathBuf};

macro_rules! stable_id {
    ($name:ident) => {
        #[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
        pub struct $name(pub u64);

        impl $name {
            pub fn from_name(name: &str) -> Self {
                Self(stable_name_hash(name))
            }
        }
    };
}

stable_id!(FunctionId);
stable_id!(TypeDefinitionId);
stable_id!(VariantId);

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct HirId(pub u64);

impl HirId {
    pub const fn from_source_range(start: usize, end: usize) -> Self {
        Self(((start as u64) << 32) ^ end as u64)
    }

    pub const fn synthetic(value: u64) -> Self {
        Self(u64::MAX - value)
    }

    pub fn from_source_span(file: SourceFileId, range: SourceRange) -> Self {
        let mut hash = 0xcbf29ce484222325_u64;
        for byte in file
            .0
            .to_le_bytes()
            .into_iter()
            .chain(range.start.to_le_bytes())
            .chain(range.end.to_le_bytes())
        {
            hash ^= u64::from(byte);
            hash = hash.wrapping_mul(0x100000001b3);
        }
        Self(hash)
    }

    pub const fn legacy_source_range(self) -> Option<SourceRange> {
        if self.0 > u64::MAX - (1 << 20) {
            return None;
        }
        Some(SourceRange {
            start: (self.0 >> 32) as usize,
            end: (self.0 & u32::MAX as u64) as usize,
        })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct SourceFileId(pub u32);

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct SourceRange {
    pub start: usize,
    pub end: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct SourceSpan {
    pub file: SourceFileId,
    pub range: SourceRange,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SourceFile {
    pub id: SourceFileId,
    pub path: PathBuf,
    pub source: String,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct SourceMap {
    files: Vec<SourceFile>,
    expression_spans: BTreeMap<HirId, SourceSpan>,
    definition_spans: BTreeMap<DefinitionId, SourceSpan>,
}

impl SourceMap {
    pub fn add_file(
        &mut self,
        path: impl Into<PathBuf>,
        source: impl Into<String>,
    ) -> SourceFileId {
        let path = path.into();
        if let Some(file) = self.files.iter().find(|file| file.path == path) {
            return file.id;
        }
        let id = SourceFileId(self.files.len() as u32);
        self.files.push(SourceFile {
            id,
            path,
            source: source.into(),
        });
        id
    }

    pub fn files(&self) -> &[SourceFile] {
        &self.files
    }

    pub fn file(&self, id: SourceFileId) -> Option<&SourceFile> {
        self.files.get(id.0 as usize)
    }

    pub fn expression_span(&self, id: HirId) -> Option<SourceSpan> {
        self.expression_spans.get(&id).copied()
    }

    pub fn definition_span(&self, id: DefinitionId) -> Option<SourceSpan> {
        self.definition_spans.get(&id).copied()
    }

    pub fn record_definition(&mut self, id: DefinitionId, span: SourceSpan) {
        self.definition_spans.insert(id, span);
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum DefinitionId {
    Function(FunctionId),
    Type(TypeDefinitionId),
    Variant {
        owner: TypeDefinitionId,
        variant: VariantId,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct TypeId(pub u32);

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum TypeKind {
    Int,
    Float,
    Bool,
    String,
    Unit,
    Any,
    List(TypeId),
    Tuple(Vec<TypeId>),
    Map {
        key: TypeId,
        value: TypeId,
    },
    Set(TypeId),
    Tensor(TensorType),
    Channel(TypeId),
    Function {
        parameters: Vec<TypeId>,
        returns: TypeId,
    },
    Result {
        ok: TypeId,
        error: TypeId,
    },
    Option(TypeId),
    Union(Vec<TypeId>),
    Future(TypeId),
    Reference {
        mutable: bool,
        inner: TypeId,
    },
    Named {
        definition: TypeDefinitionId,
        name: String,
        arguments: Vec<TypeId>,
    },
    Unresolved {
        name: String,
        arguments: Vec<TypeId>,
    },
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct TypeTable {
    types: Vec<TypeKind>,
}

impl TypeTable {
    pub fn intern(&mut self, kind: TypeKind) -> TypeId {
        if let Some(index) = self.types.iter().position(|existing| existing == &kind) {
            return TypeId(index as u32);
        }
        let id = TypeId(self.types.len() as u32);
        self.types.push(kind);
        id
    }

    pub fn get(&self, id: TypeId) -> Option<&TypeKind> {
        self.types.get(id.0 as usize)
    }

    pub fn iter(&self) -> impl Iterator<Item = (TypeId, &TypeKind)> {
        self.types
            .iter()
            .enumerate()
            .map(|(index, kind)| (TypeId(index as u32), kind))
    }

    pub fn legacy(&mut self, ty: ValueType) -> TypeId {
        let kind = match ty {
            ValueType::Int => TypeKind::Int,
            ValueType::Float => TypeKind::Float,
            ValueType::Bool => TypeKind::Bool,
            ValueType::String => TypeKind::String,
            ValueType::List => TypeKind::List(self.intern(TypeKind::Any)),
            ValueType::Tuple => TypeKind::Tuple(Vec::new()),
            ValueType::Map => {
                let any = self.intern(TypeKind::Any);
                TypeKind::Map {
                    key: any,
                    value: any,
                }
            }
            ValueType::Set => TypeKind::Set(self.intern(TypeKind::Any)),
            ValueType::Tensor(tensor) => TypeKind::Tensor(tensor),
            ValueType::TensorAny => TypeKind::Any,
            ValueType::Channel => TypeKind::Channel(self.intern(TypeKind::Any)),
            ValueType::Function => {
                let any = self.intern(TypeKind::Any);
                TypeKind::Function {
                    parameters: Vec::new(),
                    returns: any,
                }
            }
            ValueType::Result => {
                let any = self.intern(TypeKind::Any);
                TypeKind::Result {
                    ok: any,
                    error: any,
                }
            }
            ValueType::Option => TypeKind::Option(self.intern(TypeKind::Any)),
            ValueType::Any => TypeKind::Any,
            ValueType::Unit => TypeKind::Unit,
        };
        self.intern(kind)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DetailedFunctionType {
    pub parameters: Vec<TypeId>,
    pub returns: TypeId,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FieldDefinition {
    pub name: String,
    pub ty: TypeId,
    pub default: Option<HirId>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ClassDefinition {
    pub id: TypeDefinitionId,
    pub name: String,
    pub fields: Vec<FieldDefinition>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VariantDefinition {
    pub id: VariantId,
    pub name: String,
    pub fields: Vec<TypeId>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EnumDefinition {
    pub id: TypeDefinitionId,
    pub name: String,
    pub variants: Vec<VariantDefinition>,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ProgramMetadata {
    pub sources: SourceMap,
    pub types: TypeTable,
    pub expression_types: BTreeMap<HirId, TypeId>,
    pub globals: BTreeMap<String, TypeId>,
    pub functions: BTreeMap<FunctionId, DetailedFunctionType>,
    pub classes: BTreeMap<TypeDefinitionId, ClassDefinition>,
    pub enums: BTreeMap<TypeDefinitionId, EnumDefinition>,
}

fn stable_name_hash(name: &str) -> u64 {
    let mut hash = 0xcbf29ce484222325_u64;
    for byte in name.bytes() {
        hash ^= u64::from(byte);
        hash = hash.wrapping_mul(0x100000001b3);
    }
    hash
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Program {
    pub metadata: ProgramMetadata,
    pub globals: Vec<Global>,
    pub classes: Vec<Class>,
    pub functions: Vec<Function>,
}

impl Default for Program {
    fn default() -> Self {
        Self {
            metadata: ProgramMetadata::default(),
            globals: Vec::new(),
            classes: Vec::new(),
            functions: Vec::new(),
        }
    }
}

impl Program {
    pub fn attach_source_file(
        &mut self,
        path: impl Into<PathBuf>,
        source: impl Into<String>,
    ) -> SourceFileId {
        let mut metadata = std::mem::take(&mut self.metadata);
        let file = self.attach_source_file_to(&mut metadata, path, source);
        self.metadata = metadata;
        file
    }

    pub fn attach_source_file_to(
        &mut self,
        metadata: &mut ProgramMetadata,
        path: impl Into<PathBuf>,
        source: impl Into<String>,
    ) -> SourceFileId {
        let file = metadata.sources.add_file(path, source);
        let mut spans = Vec::new();
        let mut expression_types = Vec::new();
        self.visit_expressions_mut(&mut |expression| {
            let Expression::Typed { id, ty, .. } = expression else {
                return;
            };
            if metadata.sources.expression_span(*id).is_some() {
                return;
            }
            let Some(range) = id.legacy_source_range() else {
                return;
            };
            let remapped = HirId::from_source_span(file, range);
            *id = remapped;
            spans.push((remapped, SourceSpan { file, range }));
            expression_types.push((remapped, *ty));
        });
        metadata.sources.expression_spans.extend(spans);
        for (id, ty) in expression_types {
            let ty = metadata.types.legacy(ty);
            metadata.expression_types.insert(id, ty);
        }
        file
    }

    pub fn main(&self) -> Option<&Function> {
        self.functions
            .iter()
            .find(|function| function.name == "main")
    }

    pub fn test_count(&self) -> usize {
        self.functions
            .iter()
            .map(|function| function.tests.len())
            .sum::<usize>()
            + self
                .classes
                .iter()
                .flat_map(|class| class.methods.iter().chain(&class.constructors))
                .map(|function| function.tests.len())
                .sum::<usize>()
    }

    /// Visits every expression bottom-up, including expressions nested in tests.
    ///
    /// Compiler passes use this shared traversal so a new language construct has
    /// one authoritative place where recursive walking must be updated.
    pub fn visit_expressions_mut(&mut self, visitor: &mut impl FnMut(&mut Expression)) {
        for global in &mut self.globals {
            visit_expression_mut(&mut global.value, visitor);
        }
        for function in &mut self.functions {
            visit_function_expressions_mut(function, visitor);
        }
        for class in &mut self.classes {
            for default in class.field_defaults.iter_mut().flatten() {
                visit_expression_mut(default, visitor);
            }
            for function in class.methods.iter_mut().chain(&mut class.constructors) {
                visit_function_expressions_mut(function, visitor);
            }
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Class {
    pub id: TypeDefinitionId,
    pub name: String,
    pub decorators: Vec<Decorator>,
    pub fields: Vec<String>,
    pub field_types: Vec<ValueType>,
    pub field_classes: Vec<Option<String>>,
    pub field_defaults: Vec<Option<Expression>>,
    pub constructors: Vec<Function>,
    pub methods: Vec<Function>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Global {
    pub name: String,
    pub value: Expression,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Function {
    pub id: FunctionId,
    pub name: String,
    pub native_symbol: Option<String>,
    pub decorators: Vec<Decorator>,
    pub contract: Option<FunctionContract>,
    pub params: Vec<Parameter>,
    pub return_type: ValueType,
    pub instructions: Vec<Instruction>,
    pub tests: Vec<Test>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FunctionContract {
    pub requirements: Vec<Expression>,
    pub capabilities: Vec<Expression>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Decorator {
    pub package: String,
    pub symbols: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Parameter {
    pub name: String,
    pub ty: ValueType,
    pub default: Option<Expression>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Test {
    pub name: Option<String>,
    pub modes: Vec<TestMode>,
    pub instructions: Vec<Instruction>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TestMode {
    Property,
    Bench,
    Chaos,
    Integration,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ValueType {
    Int,
    Float,
    Bool,
    String,
    List,
    Tuple,
    Map,
    Set,
    Tensor(TensorType),
    /// A tensor of any element type and rank. Unlike `Any`, this remains a
    /// tensor-only type guard for dtype-polymorphic APIs such as `release`.
    TensorAny,
    Channel,
    Function,
    Result,
    Option,
    Any,
    Unit,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TensorElementType {
    BF16,
    F32,
    F64,
    I32,
    I64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TensorDimension {
    Static(u64),
    Dynamic,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TensorType {
    pub element: TensorElementType,
    /// `None` is dynamic rank. Ranked tensors use the first `rank` entries.
    pub rank: Option<u8>,
    pub dimensions: [TensorDimension; 8],
}

impl TensorType {
    pub const fn dynamic(element: TensorElementType) -> Self {
        Self {
            element,
            rank: None,
            dimensions: [TensorDimension::Dynamic; 8],
        }
    }

    pub fn ranked(
        element: TensorElementType,
        dimensions: &[TensorDimension],
    ) -> Result<Self, &'static str> {
        if dimensions.len() > 8 {
            return Err("tensor rank exceeds the supported maximum of 8");
        }
        let mut result = Self::dynamic(element);
        result.rank = Some(dimensions.len() as u8);
        result.dimensions[..dimensions.len()].copy_from_slice(dimensions);
        Ok(result)
    }

    pub fn is_compatible_with(self, expected: Self) -> bool {
        if self.element != expected.element {
            return false;
        }
        let (Some(actual_rank), Some(expected_rank)) = (self.rank, expected.rank) else {
            return true;
        };
        actual_rank == expected_rank
            && (0..actual_rank as usize).all(|axis| {
                self.dimensions[axis] == expected.dimensions[axis]
                    || self.dimensions[axis] == TensorDimension::Dynamic
                    || expected.dimensions[axis] == TensorDimension::Dynamic
            })
    }

    pub fn broadcast_with(self, right: Self) -> Result<Self, &'static str> {
        if self.element != right.element {
            return Err("tensor element types do not match");
        }
        let (Some(left_rank), Some(right_rank)) = (self.rank, right.rank) else {
            return Ok(Self::dynamic(self.element));
        };
        let rank = left_rank.max(right_rank) as usize;
        let mut dimensions = [TensorDimension::Dynamic; 8];
        for output_axis in 0..rank {
            let left_axis = output_axis.checked_sub(rank - left_rank as usize);
            let right_axis = output_axis.checked_sub(rank - right_rank as usize);
            let left = left_axis.map_or(TensorDimension::Static(1), |axis| self.dimensions[axis]);
            let right =
                right_axis.map_or(TensorDimension::Static(1), |axis| right.dimensions[axis]);
            dimensions[output_axis] = match (left, right) {
                (TensorDimension::Static(a), TensorDimension::Static(b)) if a == b => left,
                (TensorDimension::Static(1), other) | (other, TensorDimension::Static(1)) => other,
                (TensorDimension::Dynamic, _) | (_, TensorDimension::Dynamic) => {
                    TensorDimension::Dynamic
                }
                _ => return Err("tensor shapes cannot be broadcast"),
            };
        }
        Self::ranked(self.element, &dimensions[..rank])
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TaskPlacement {
    Default,
    Local,
    Gpu,
    Simd,
    Simt,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Instruction {
    Let {
        name: String,
        value: Expression,
    },
    TryLet {
        name: String,
        value: Expression,
    },
    Assign {
        target: Expression,
        op: AssignmentOp,
        value: Expression,
    },
    Print(Expression),
    Assert(Expression),
    Return(Option<Expression>),
    If {
        condition: Expression,
        then_instructions: Vec<Instruction>,
        else_instructions: Vec<Instruction>,
    },
    While {
        setup: Option<Box<Instruction>>,
        capabilities: Vec<Expression>,
        condition: Expression,
        instructions: Vec<Instruction>,
    },
    For {
        setup: Option<Box<Instruction>>,
        pattern: MatchPattern,
        iterable: Expression,
        instructions: Vec<Instruction>,
    },
    Switch {
        value: Expression,
        arms: Vec<SwitchArm>,
    },
    ChannelSwitch {
        channels: Vec<Expression>,
        setup: Option<Box<Instruction>>,
        repeat_condition: Option<Expression>,
        arms: Vec<SwitchArm>,
    },
    With {
        placement: TaskPlacement,
        resources: Vec<Expression>,
        instructions: Vec<Instruction>,
    },
    Break,
    Continue,
    Evaluate(Expression),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SwitchArm {
    pub source: Option<Expression>,
    pub pattern: MatchPattern,
    pub guard: Option<Expression>,
    pub instructions: Vec<Instruction>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MatchPattern {
    Wildcard,
    Bind(String),
    Integer(i64),
    Float(u64),
    Boolean(bool),
    String(String),
    Constructor {
        name: String,
        fields: Vec<MatchPattern>,
    },
}

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
    Variable(String),
    Function(String),
    Lambda {
        params: Vec<String>,
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
        function: String,
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

fn visit_function_expressions_mut(
    function: &mut Function,
    visitor: &mut impl FnMut(&mut Expression),
) {
    visit_instructions_mut(&mut function.instructions, visitor);
    for test in &mut function.tests {
        visit_instructions_mut(&mut test.instructions, visitor);
    }
}

fn visit_instructions_mut(
    instructions: &mut [Instruction],
    visitor: &mut impl FnMut(&mut Expression),
) {
    for instruction in instructions {
        match instruction {
            Instruction::Let { value, .. }
            | Instruction::TryLet { value, .. }
            | Instruction::Print(value)
            | Instruction::Assert(value)
            | Instruction::Evaluate(value) => visit_expression_mut(value, visitor),
            Instruction::Assign { target, value, .. } => {
                visit_expression_mut(target, visitor);
                visit_expression_mut(value, visitor);
            }
            Instruction::Return(value) => {
                if let Some(value) = value {
                    visit_expression_mut(value, visitor);
                }
            }
            Instruction::If {
                condition,
                then_instructions,
                else_instructions,
            } => {
                visit_expression_mut(condition, visitor);
                visit_instructions_mut(then_instructions, visitor);
                visit_instructions_mut(else_instructions, visitor);
            }
            Instruction::While {
                setup,
                capabilities,
                condition,
                instructions,
            } => {
                if let Some(setup) = setup {
                    visit_instructions_mut(std::slice::from_mut(setup), visitor);
                }
                for capability in capabilities {
                    visit_expression_mut(capability, visitor);
                }
                visit_expression_mut(condition, visitor);
                visit_instructions_mut(instructions, visitor);
            }
            Instruction::For {
                setup,
                iterable,
                instructions,
                ..
            } => {
                if let Some(setup) = setup {
                    visit_instructions_mut(std::slice::from_mut(setup), visitor);
                }
                visit_expression_mut(iterable, visitor);
                visit_instructions_mut(instructions, visitor);
            }
            Instruction::Switch { value, arms } => {
                visit_expression_mut(value, visitor);
                visit_arms_mut(arms, visitor);
            }
            Instruction::ChannelSwitch {
                channels,
                setup,
                repeat_condition,
                arms,
            } => {
                for channel in channels {
                    visit_expression_mut(channel, visitor);
                }
                if let Some(setup) = setup {
                    visit_instructions_mut(std::slice::from_mut(setup), visitor);
                }
                if let Some(condition) = repeat_condition {
                    visit_expression_mut(condition, visitor);
                }
                visit_arms_mut(arms, visitor);
            }
            Instruction::With {
                placement: _,
                resources,
                instructions,
            } => {
                for resource in resources {
                    visit_expression_mut(resource, visitor);
                }
                visit_instructions_mut(instructions, visitor);
            }
            Instruction::Break | Instruction::Continue => {}
        }
    }
}

fn visit_arms_mut(arms: &mut [SwitchArm], visitor: &mut impl FnMut(&mut Expression)) {
    for arm in arms {
        if let Some(source) = &mut arm.source {
            visit_expression_mut(source, visitor);
        }
        if let Some(guard) = &mut arm.guard {
            visit_expression_mut(guard, visitor);
        }
        visit_instructions_mut(&mut arm.instructions, visitor);
    }
}

fn visit_expression_mut(expression: &mut Expression, visitor: &mut impl FnMut(&mut Expression)) {
    match expression {
        Expression::Typed { expression, .. } => visit_expression_mut(expression, visitor),
        Expression::List(values)
        | Expression::Tuple(values)
        | Expression::Set(values)
        | Expression::PrintArgs(values)
        | Expression::Construct { args: values, .. }
        | Expression::Variant { fields: values, .. } => {
            for value in values {
                visit_expression_mut(value, visitor);
            }
        }
        Expression::Map(entries) => {
            for (key, value) in entries {
                visit_expression_mut(key, visitor);
                visit_expression_mut(value, visitor);
            }
        }
        Expression::Index { object, index } => {
            visit_expression_mut(object, visitor);
            visit_expression_mut(index, visitor);
        }
        Expression::Slice {
            object,
            start,
            end,
            step,
        } => {
            visit_expression_mut(object, visitor);
            for bound in [start, end, step].into_iter().flatten() {
                visit_expression_mut(bound, visitor);
            }
        }
        Expression::Member { object, .. }
        | Expression::Await(object)
        | Expression::Channel(object)
        | Expression::ChaosRule { value: object, .. }
        | Expression::FusedPipeline { input: object, .. } => {
            visit_expression_mut(object, visitor);
        }
        Expression::Ownership { value, .. } => visit_expression_mut(value, visitor),
        Expression::Lambda { body, .. } => visit_expression_mut(body, visitor),
        Expression::MethodCall { object, args, .. } => {
            visit_expression_mut(object, visitor);
            for arg in args {
                visit_expression_mut(arg, visitor);
            }
        }
        Expression::Task { value, .. } => visit_expression_mut(value, visitor),
        Expression::Send { value, channel } => {
            visit_expression_mut(value, visitor);
            visit_expression_mut(channel, visitor);
        }
        Expression::ListComprehension { element, clauses } => {
            visit_expression_mut(element, visitor);
            for clause in clauses {
                visit_expression_mut(&mut clause.iterable, visitor);
                if let Some(condition) = &mut clause.condition {
                    visit_expression_mut(condition, visitor);
                }
            }
        }
        Expression::SetComprehension { element, clauses } => {
            visit_expression_mut(element, visitor);
            for clause in clauses {
                visit_expression_mut(&mut clause.iterable, visitor);
                if let Some(condition) = &mut clause.condition {
                    visit_expression_mut(condition, visitor);
                }
            }
        }
        Expression::MapComprehension {
            key,
            value,
            clauses,
        } => {
            visit_expression_mut(key, visitor);
            visit_expression_mut(value, visitor);
            for clause in clauses {
                visit_expression_mut(&mut clause.iterable, visitor);
                if let Some(condition) = &mut clause.condition {
                    visit_expression_mut(condition, visitor);
                }
            }
        }
        Expression::Conditional {
            condition,
            then_expression,
            else_expression,
        } => {
            visit_expression_mut(condition, visitor);
            visit_expression_mut(then_expression, visitor);
            visit_expression_mut(else_expression, visitor);
        }
        Expression::Unary { expression, .. } => visit_expression_mut(expression, visitor),
        Expression::Binary { left, right, .. } => {
            visit_expression_mut(left, visitor);
            visit_expression_mut(right, visitor);
        }
        Expression::Call { args, .. } => {
            for arg in args {
                visit_expression_mut(arg, visitor);
            }
        }
        Expression::CallValue { callee, args, .. } => {
            visit_expression_mut(callee, visitor);
            for arg in args {
                visit_expression_mut(arg, visitor);
            }
        }
        Expression::Format { args, .. } => {
            for arg in args {
                visit_expression_mut(arg, visitor);
            }
        }
        Expression::Integer(_)
        | Expression::Float(_)
        | Expression::Boolean(_)
        | Expression::String(_)
        | Expression::Variable(_)
        | Expression::Function(_) => {}
    }
    visitor(expression);
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChaosAction {
    Return,
    Throw,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BinaryOp {
    Add,
    Sub,
    Mul,
    Div,
    Mod,
    Power,
    Equal,
    NotEqual,
    Less,
    LessEqual,
    Greater,
    GreaterEqual,
    And,
    Or,
    In,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UnaryOp {
    Negate,
    Not,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OwnershipOp {
    View,
    Borrow,
    Clone,
    Move,
    AddressOf,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AssignmentOp {
    Assign,
    Add,
    Sub,
    Mul,
    Div,
    Mod,
}
