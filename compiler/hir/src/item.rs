use crate::visitor::{visit_expression_mut, visit_function_expressions_mut};
use crate::*;
use std::collections::BTreeMap;
use std::path::PathBuf;

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

    /// Qualifies every resolved binding identity while preserving its source
    /// name for diagnostics. Package linking uses this before combining HIR so
    /// equal source spans in different modules cannot alias the same binding.
    pub fn namespace_bindings(&mut self, namespace: &str) {
        for global in &mut self.globals {
            namespace_binding(&mut global.name, namespace);
        }
        for function in &mut self.functions {
            namespace_function_bindings(function, namespace);
        }
        for class in &mut self.classes {
            for function in class.methods.iter_mut().chain(&mut class.constructors) {
                namespace_function_bindings(function, namespace);
            }
        }
        self.visit_expressions_mut(&mut |expression| match expression {
            Expression::Variable(binding) => namespace_binding(binding, namespace),
            Expression::Lambda { params, .. } => {
                for parameter in params {
                    namespace_binding(parameter, namespace);
                }
            }
            Expression::ListComprehension { clauses, .. }
            | Expression::SetComprehension { clauses, .. }
            | Expression::MapComprehension { clauses, .. } => {
                for clause in clauses {
                    namespace_pattern(&mut clause.pattern, namespace);
                }
            }
            _ => {}
        });
    }
}

fn namespace_binding(binding: &mut BindingRef, namespace: &str) {
    binding.id = binding.id.in_namespace(namespace);
}

fn namespace_function_bindings(function: &mut Function, namespace: &str) {
    for parameter in &mut function.params {
        namespace_binding(&mut parameter.name, namespace);
    }
    if let Some(contract) = &mut function.contract {
        namespace_contract_bindings(contract, namespace);
    }
    namespace_instruction_bindings(&mut function.instructions, namespace);
    for test in &mut function.tests {
        if let Some(contract) = &mut test.contract {
            namespace_contract_bindings(contract, namespace);
        }
        namespace_instruction_bindings(&mut test.instructions, namespace);
    }
}

fn namespace_contract_bindings(contract: &mut FunctionContract, namespace: &str) {
    for clause in &mut contract.clauses {
        for dependency in &mut clause.dependencies {
            namespace_binding(dependency, namespace);
        }
    }
}

fn namespace_instruction_bindings(instructions: &mut [Instruction], namespace: &str) {
    for instruction in instructions {
        match instruction {
            Instruction::Let { name, .. } | Instruction::TryLet { name, .. } => {
                namespace_binding(name, namespace);
            }
            Instruction::If {
                then_instructions,
                else_instructions,
                ..
            } => {
                namespace_instruction_bindings(then_instructions, namespace);
                namespace_instruction_bindings(else_instructions, namespace);
            }
            Instruction::While {
                setup,
                instructions,
                ..
            } => {
                if let Some(setup) = setup {
                    namespace_instruction_bindings(std::slice::from_mut(setup), namespace);
                }
                namespace_instruction_bindings(instructions, namespace);
            }
            Instruction::For {
                setup,
                pattern,
                instructions,
                ..
            } => {
                if let Some(setup) = setup {
                    namespace_instruction_bindings(std::slice::from_mut(setup), namespace);
                }
                namespace_pattern(pattern, namespace);
                namespace_instruction_bindings(instructions, namespace);
            }
            Instruction::Switch { arms, .. } | Instruction::ChannelSwitch { arms, .. } => {
                for arm in arms {
                    namespace_pattern(&mut arm.pattern, namespace);
                    arm.receivers = std::mem::take(&mut arm.receivers)
                        .into_iter()
                        .map(|(binding, receiver)| (binding.in_namespace(namespace), receiver))
                        .collect();
                    namespace_instruction_bindings(&mut arm.instructions, namespace);
                }
            }
            Instruction::With { instructions, .. } => {
                namespace_instruction_bindings(instructions, namespace);
            }
            Instruction::Assign { .. }
            | Instruction::Print(_)
            | Instruction::Assert(_)
            | Instruction::Return(_)
            | Instruction::Break
            | Instruction::Continue
            | Instruction::Evaluate(_) => {}
        }
    }
}

fn namespace_pattern(pattern: &mut MatchPattern, namespace: &str) {
    match pattern {
        MatchPattern::Bind(binding) => namespace_binding(binding, namespace),
        MatchPattern::Constructor { fields, .. } => {
            for field in fields {
                namespace_pattern(field, namespace);
            }
        }
        MatchPattern::Wildcard
        | MatchPattern::Integer(_)
        | MatchPattern::Float(_)
        | MatchPattern::Boolean(_)
        | MatchPattern::String(_) => {}
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
    pub method_return_classes: Vec<Option<String>>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Global {
    pub name: BindingRef,
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
    pub clauses: Vec<ContractClause>,
    pub capabilities: Vec<Expression>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ContractClause {
    pub condition: Expression,
    pub deferred: bool,
    pub message: Option<String>,
    pub location: bool,
    pub vars: bool,
    pub dependencies: Vec<BindingRef>,
    pub dependency_types: Vec<ValueType>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Decorator {
    pub package: String,
    pub symbols: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Parameter {
    pub name: BindingRef,
    pub ty: ValueType,
    pub default: Option<Expression>,
    pub receiver: Option<ReceiverType>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Test {
    pub name: Option<String>,
    pub modes: Vec<TestMode>,
    pub return_type: ValueType,
    pub contract: Option<FunctionContract>,
    pub instructions: Vec<Instruction>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TestMode {
    Property,
    Bench,
    Chaos,
    Integration,
    Profile,
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
    Bool,
    I8,
    I16,
    I32,
    I64,
    U8,
    U16,
    U32,
    U64,
    F8E4M3FN,
    F8E5M2,
    F16,
    BF16,
    F32,
    F64,
    C64,
    C128,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TensorElementClass {
    Boolean,
    SignedInteger,
    UnsignedInteger,
    Float,
    Complex,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TensorElementConstraint {
    Any,
    Numeric,
    Integer,
    SignedInteger,
    UnsignedInteger,
    Float,
    Complex,
}

impl TensorElementType {
    pub fn parse(name: &str) -> Option<Self> {
        Some(match name {
            "bool" => Self::Bool,
            "i8" => Self::I8,
            "i16" => Self::I16,
            "i32" => Self::I32,
            "i64" | "int" => Self::I64,
            "u8" => Self::U8,
            "u16" => Self::U16,
            "u32" => Self::U32,
            "u64" => Self::U64,
            "f8e4m3" | "f8e4m3fn" => Self::F8E4M3FN,
            "f8e5m2" => Self::F8E5M2,
            "f16" | "float16" => Self::F16,
            "bf16" | "bfloat16" => Self::BF16,
            "f32" | "float32" => Self::F32,
            "f64" | "float" | "float64" => Self::F64,
            "c64" | "complex64" => Self::C64,
            "c128" | "complex128" => Self::C128,
            _ => return None,
        })
    }

    pub const fn name(self) -> &'static str {
        match self {
            Self::Bool => "bool",
            Self::I8 => "i8",
            Self::I16 => "i16",
            Self::I32 => "i32",
            Self::I64 => "i64",
            Self::U8 => "u8",
            Self::U16 => "u16",
            Self::U32 => "u32",
            Self::U64 => "u64",
            Self::F8E4M3FN => "f8e4m3fn",
            Self::F8E5M2 => "f8e5m2",
            Self::F16 => "f16",
            Self::BF16 => "bf16",
            Self::F32 => "f32",
            Self::F64 => "f64",
            Self::C64 => "c64",
            Self::C128 => "c128",
        }
    }

    pub const fn mlir_name(self) -> &'static str {
        match self {
            Self::Bool => "i1",
            Self::F8E4M3FN => "f8E4M3FN",
            Self::F8E5M2 => "f8E5M2",
            Self::C64 => "complex<f32>",
            Self::C128 => "complex<f64>",
            _ => self.name(),
        }
    }

    pub const fn safetensors_name(self) -> &'static str {
        match self {
            Self::Bool => "BOOL",
            Self::I8 => "I8",
            Self::I16 => "I16",
            Self::I32 => "I32",
            Self::I64 => "I64",
            Self::U8 => "U8",
            Self::U16 => "U16",
            Self::U32 => "U32",
            Self::U64 => "U64",
            Self::F8E4M3FN => "F8_E4M3",
            Self::F8E5M2 => "F8_E5M2",
            Self::F16 => "F16",
            Self::BF16 => "BF16",
            Self::F32 => "F32",
            Self::F64 => "F64",
            Self::C64 => "C64",
            Self::C128 => "C128",
        }
    }

    pub const fn storage_bytes(self) -> u8 {
        match self {
            Self::Bool | Self::I8 | Self::U8 | Self::F8E4M3FN | Self::F8E5M2 => 1,
            Self::I16 | Self::U16 | Self::F16 | Self::BF16 => 2,
            Self::I32 | Self::U32 | Self::F32 => 4,
            Self::I64 | Self::U64 | Self::F64 | Self::C64 => 8,
            Self::C128 => 16,
        }
    }

    pub const fn class(self) -> TensorElementClass {
        match self {
            Self::Bool => TensorElementClass::Boolean,
            Self::I8 | Self::I16 | Self::I32 | Self::I64 => TensorElementClass::SignedInteger,
            Self::U8 | Self::U16 | Self::U32 | Self::U64 => {
                TensorElementClass::UnsignedInteger
            }
            Self::F8E4M3FN | Self::F8E5M2 | Self::F16 | Self::BF16 | Self::F32 | Self::F64 => {
                TensorElementClass::Float
            }
            Self::C64 | Self::C128 => TensorElementClass::Complex,
        }
    }

    pub const fn satisfies(self, constraint: TensorElementConstraint) -> bool {
        use TensorElementClass as Class;
        use TensorElementConstraint as Constraint;
        matches!(constraint, Constraint::Any)
            || matches!(
                (self.class(), constraint),
                (Class::SignedInteger | Class::UnsignedInteger | Class::Float | Class::Complex, Constraint::Numeric)
                    | (Class::SignedInteger | Class::UnsignedInteger, Constraint::Integer)
                    | (Class::SignedInteger, Constraint::SignedInteger)
                    | (Class::UnsignedInteger, Constraint::UnsignedInteger)
                    | (Class::Float, Constraint::Float)
                    | (Class::Complex, Constraint::Complex)
            )
    }

    /// Severian's language-level promotion rule. Backends must consume the
    /// resolved result instead of applying their own implicit conversions.
    pub const fn promote(left: Self, right: Self) -> Option<Self> {
        use TensorElementType as T;
        if left as u8 == right as u8 {
            return Some(left);
        }
        if matches!(left, T::Bool) || matches!(right, T::Bool) {
            return None;
        }
        match (left, right) {
            (T::C128, _) | (_, T::C128) => Some(T::C128),
            (T::C64, T::F64) | (T::F64, T::C64) => Some(T::C128),
            (T::C64, _) | (_, T::C64) => Some(T::C64),
            (T::F64, _) | (_, T::F64) => Some(T::F64),
            (T::F32, _) | (_, T::F32) => Some(T::F32),
            (T::BF16, T::F16) | (T::F16, T::BF16) => Some(T::F32),
            (T::BF16, _) | (_, T::BF16) => Some(T::BF16),
            (T::F16, _) | (_, T::F16) => Some(T::F16),
            (T::F8E4M3FN, T::F8E5M2) | (T::F8E5M2, T::F8E4M3FN) => Some(T::F16),
            (T::F8E4M3FN, _) | (_, T::F8E4M3FN) => Some(T::F8E4M3FN),
            (T::F8E5M2, _) | (_, T::F8E5M2) => Some(T::F8E5M2),
            _ => Self::promote_integers(left, right),
        }
    }

    const fn integer_width(self) -> Option<(u8, bool)> {
        match self {
            Self::I8 => Some((8, true)),
            Self::I16 => Some((16, true)),
            Self::I32 => Some((32, true)),
            Self::I64 => Some((64, true)),
            Self::U8 => Some((8, false)),
            Self::U16 => Some((16, false)),
            Self::U32 => Some((32, false)),
            Self::U64 => Some((64, false)),
            _ => None,
        }
    }

    const fn promote_integers(left: Self, right: Self) -> Option<Self> {
        let (Some((left_width, left_signed)), Some((right_width, right_signed))) =
            (left.integer_width(), right.integer_width())
        else {
            return None;
        };
        let width = if left_width > right_width {
            left_width
        } else {
            right_width
        };
        if left_signed == right_signed {
            return match (width, left_signed) {
                (8, true) => Some(Self::I8),
                (16, true) => Some(Self::I16),
                (32, true) => Some(Self::I32),
                (64, true) => Some(Self::I64),
                (8, false) => Some(Self::U8),
                (16, false) => Some(Self::U16),
                (32, false) => Some(Self::U32),
                (64, false) => Some(Self::U64),
                _ => None,
            };
        }
        // A signed result needs one more value bit than an unsigned operand of
        // the same width. Reject i64/u64 rather than silently losing values.
        let signed_width = if left_signed { left_width } else { right_width };
        let unsigned_width = if left_signed { right_width } else { left_width };
        let required = if signed_width > unsigned_width {
            signed_width
        } else if unsigned_width < 64 {
            unsigned_width * 2
        } else {
            return None;
        };
        match required {
            8 => Some(Self::I8),
            16 => Some(Self::I16),
            32 => Some(Self::I32),
            64 => Some(Self::I64),
            _ => None,
        }
    }
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
        name: BindingRef,
        value: Expression,
    },
    TryLet {
        name: BindingRef,
        value: Expression,
        receiver: Option<ReceiverType>,
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
    pub receivers: BTreeMap<BindingId, ReceiverType>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReceiverType {
    pub name: String,
    pub concrete: bool,
    pub methods: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MatchPattern {
    Wildcard,
    Bind(BindingRef),
    Integer(i64),
    Float(u64),
    Boolean(bool),
    String(String),
    Constructor {
        name: String,
        fields: Vec<MatchPattern>,
    },
}
