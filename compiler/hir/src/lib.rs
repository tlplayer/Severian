#![forbid(unsafe_code)]

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Program {
    pub globals: Vec<Global>,
    pub classes: Vec<Class>,
    pub functions: Vec<Function>,
}

impl Program {
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
    Tensor,
    Channel,
    Function,
    Result,
    Option,
    Any,
    Unit,
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
        function: String,
        args: Vec<Expression>,
    },
    CallValue {
        callee: Box<Expression>,
        args: Vec<Expression>,
        return_type: ValueType,
    },
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
