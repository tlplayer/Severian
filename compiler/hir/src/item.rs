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
            for constraint in &mut class.field_constraints {
                visit_expression_mut(constraint, visitor);
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
    /// Cross-field invariants evaluated against the completely assembled object.
    pub field_constraints: Vec<Expression>,
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

pub use severian_dtype::{
    DType, DTypeClass, DTypeConstraint, TensorElementClass, TensorElementConstraint,
    TensorElementType,
};

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
