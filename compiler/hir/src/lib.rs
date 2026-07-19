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
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Class {
    pub name: String,
    pub decorators: Vec<Decorator>,
    pub fields: Vec<String>,
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
    pub decorators: Vec<Decorator>,
    pub params: Vec<Parameter>,
    pub return_type: ValueType,
    pub instructions: Vec<Instruction>,
    pub tests: Vec<Test>,
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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ValueType {
    Int,
    Float,
    Bool,
    String,
    List,
    Tuple,
    Map,
    Set,
    Function,
    Result,
    Option,
    Any,
    Unit,
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
        condition: Expression,
        instructions: Vec<Instruction>,
    },
    For {
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
    List(Vec<Expression>),
    Tuple(Vec<Expression>),
    Map(Vec<(Expression, Expression)>),
    Set(Vec<Expression>),
    Index {
        object: Box<Expression>,
        index: Box<Expression>,
    },
    Format(String),
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
    Task(Box<Expression>),
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
        variable: String,
        iterable: Box<Expression>,
        condition: Option<Box<Expression>>,
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
    },
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
pub enum AssignmentOp {
    Assign,
    Add,
    Sub,
    Mul,
    Div,
    Mod,
}
