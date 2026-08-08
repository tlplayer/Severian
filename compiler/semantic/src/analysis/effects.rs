//! HIR effect analysis for optimization and lowering.
//!
//! The source semantic pass already validates language rules. This module
//! answers a different question: which HIR operations may be reordered,
//! duplicated, eliminated, fused, or moved across task/dispatch boundaries?

use severian_hir::{
    Expression, Function, Instruction, OwnershipOp, Program, SwitchArm,
};
use std::collections::HashMap;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct EffectSet(u32);

impl EffectSet {
    pub const NONE: Self = Self(0);

    pub const READ_MEMORY: Self = Self(1 << 0);
    pub const WRITE_MEMORY: Self = Self(1 << 1);
    pub const ALLOCATE: Self = Self(1 << 2);
    pub const CALL: Self = Self(1 << 3);
    pub const IO: Self = Self(1 << 4);
    pub const SPAWN_TASK: Self = Self(1 << 5);
    pub const AWAIT: Self = Self(1 << 6);
    pub const CHANNEL: Self = Self(1 << 7);
    pub const MAY_PANIC: Self = Self(1 << 8);
    pub const NONDETERMINISTIC: Self = Self(1 << 9);
    pub const CHAOS: Self = Self(1 << 10);
    pub const CONTROL_FLOW: Self = Self(1 << 11);
    pub const UNKNOWN: Self = Self(1 << 12);

    pub const fn from_bits(bits: u32) -> Self {
        Self(bits)
    }

    pub const fn bits(self) -> u32 {
        self.0
    }

    pub const fn contains(self, other: Self) -> bool {
        (self.0 & other.0) == other.0
    }

    pub const fn intersects(self, other: Self) -> bool {
        (self.0 & other.0) != 0
    }

    pub const fn union(self, other: Self) -> Self {
        Self(self.0 | other.0)
    }

    pub fn insert(&mut self, other: Self) {
        self.0 |= other.0;
    }

    pub const fn is_empty(self) -> bool {
        self.0 == 0
    }

    /// Strong purity suitable for duplication/elimination.
    pub const fn is_pure(self) -> bool {
        !self.intersects(
            Self::WRITE_MEMORY
                .union(Self::ALLOCATE)
                .union(Self::CALL)
                .union(Self::IO)
                .union(Self::SPAWN_TASK)
                .union(Self::AWAIT)
                .union(Self::CHANNEL)
                .union(Self::MAY_PANIC)
                .union(Self::NONDETERMINISTIC)
                .union(Self::CHAOS)
                .union(Self::UNKNOWN),
        )
    }

    /// Whether two operations can usually be reordered without changing
    /// externally visible behavior.
    pub const fn is_reorderable(self) -> bool {
        !self.intersects(
            Self::WRITE_MEMORY
                .union(Self::CALL)
                .union(Self::IO)
                .union(Self::SPAWN_TASK)
                .union(Self::AWAIT)
                .union(Self::CHANNEL)
                .union(Self::NONDETERMINISTIC)
                .union(Self::CHAOS)
                .union(Self::UNKNOWN),
        )
    }
}

impl std::ops::BitOr for EffectSet {
    type Output = Self;

    fn bitor(self, rhs: Self) -> Self::Output {
        self.union(rhs)
    }
}

impl std::ops::BitOrAssign for EffectSet {
    fn bitor_assign(&mut self, rhs: Self) {
        self.insert(rhs);
    }
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct FunctionEffects {
    pub function: String,
    pub effects: EffectSet,
    pub expression_count: usize,
    pub instruction_count: usize,
}

impl FunctionEffects {
    pub fn is_pure(&self) -> bool {
        self.effects.is_pure()
    }

    pub fn is_reorderable(&self) -> bool {
        self.effects.is_reorderable()
    }
}

#[derive(Debug, Clone, Default)]
pub struct EffectAnalysis {
    pub functions: HashMap<String, FunctionEffects>,
}

impl EffectAnalysis {
    pub fn function(&self, name: &str) -> Option<&FunctionEffects> {
        self.functions.get(name)
    }

    pub fn is_pure_function(&self, name: &str) -> bool {
        self.function(name).is_some_and(FunctionEffects::is_pure)
    }

    pub fn expression_effects(&self, expression: &Expression) -> EffectSet {
        expression_effects(expression, &self.functions)
    }
}

/// Computes interprocedural summaries to a fixed point.
///
/// Calls to known Severian functions inherit the callee summary. Native,
/// indirect, or unresolved calls remain conservative.
pub fn analyze(program: &Program) -> EffectAnalysis {
    let mut functions = HashMap::<String, FunctionEffects>::new();

    for function in &program.functions {
        functions.insert(
            function.name.clone(),
            FunctionEffects {
                function: function.name.clone(),
                ..FunctionEffects::default()
            },
        );
    }

    for class in &program.classes {
        for function in class.methods.iter().chain(&class.constructors) {
            let name = format!("{}::{}", class.name, function.name);
            functions.insert(
                name.clone(),
                FunctionEffects {
                    function: name,
                    ..FunctionEffects::default()
                },
            );
        }
    }

    // Effect sets only grow, so this terminates quickly.
    for _ in 0..64 {
        let previous = functions.clone();
        let mut changed = false;

        for function in &program.functions {
            let summary = analyze_function_with_summaries(function, &previous);
            changed |= functions.get(&function.name) != Some(&summary);
            functions.insert(function.name.clone(), summary);
        }

        for class in &program.classes {
            for function in class.methods.iter().chain(&class.constructors) {
                let qualified = format!("{}::{}", class.name, function.name);
                let mut summary = analyze_function_with_summaries(function, &previous);
                summary.function = qualified.clone();
                changed |= functions.get(&qualified) != Some(&summary);
                functions.insert(qualified, summary);
            }
        }

        if !changed {
            break;
        }
    }

    EffectAnalysis { functions }
}

pub fn analyze_function(function: &Function) -> FunctionEffects {
    analyze_function_with_summaries(function, &HashMap::new())
}

fn analyze_function_with_summaries(
    function: &Function,
    summaries: &HashMap<String, FunctionEffects>,
) -> FunctionEffects {
    let mut analyzer = Analyzer {
        summaries,
        effects: EffectSet::NONE,
        expression_count: 0,
        instruction_count: 0,
    };

    for parameter in &function.params {
        if let Some(default) = &parameter.default {
            analyzer.expression(default);
        }
    }

    if let Some(contract) = &function.contract {
        for requirement in &contract.requirements {
            analyzer.expression(requirement);
        }
        for capability in &contract.capabilities {
            analyzer.expression(capability);
        }
    }

    analyzer.instructions(&function.instructions);

    // Native functions cross an unknown ABI boundary even if their HIR body is
    // empty.
    if function.native_symbol.is_some() {
        analyzer.effects |= EffectSet::CALL | EffectSet::UNKNOWN;
    }

    FunctionEffects {
        function: function.name.clone(),
        effects: analyzer.effects,
        expression_count: analyzer.expression_count,
        instruction_count: analyzer.instruction_count,
    }
}

pub fn expression_effects(
    expression: &Expression,
    summaries: &HashMap<String, FunctionEffects>,
) -> EffectSet {
    let mut analyzer = Analyzer {
        summaries,
        effects: EffectSet::NONE,
        expression_count: 0,
        instruction_count: 0,
    };
    analyzer.expression(expression);
    analyzer.effects
}

struct Analyzer<'a> {
    summaries: &'a HashMap<String, FunctionEffects>,
    effects: EffectSet,
    expression_count: usize,
    instruction_count: usize,
}

impl Analyzer<'_> {
    fn instructions(&mut self, instructions: &[Instruction]) {
        for instruction in instructions {
            self.instruction(instruction);
        }
    }

    fn instruction(&mut self, instruction: &Instruction) {
        self.instruction_count += 1;

        match instruction {
            Instruction::Let { value, .. } | Instruction::TryLet { value, .. } => {
                self.expression(value);
            }

            Instruction::Assign { target, value, .. } => {
                self.effects |= EffectSet::WRITE_MEMORY;
                self.expression(target);
                self.expression(value);
            }

            Instruction::Print(value) => {
                self.effects |= EffectSet::IO;
                self.expression(value);
            }

            Instruction::Assert(value) => {
                self.effects |= EffectSet::MAY_PANIC;
                self.expression(value);
            }

            Instruction::Return(value) => {
                self.effects |= EffectSet::CONTROL_FLOW;
                if let Some(value) = value {
                    self.expression(value);
                }
            }

            Instruction::If {
                condition,
                then_instructions,
                else_instructions,
            } => {
                self.effects |= EffectSet::CONTROL_FLOW;
                self.expression(condition);
                self.instructions(then_instructions);
                self.instructions(else_instructions);
            }

            Instruction::While {
                setup,
                capabilities,
                condition,
                instructions,
            } => {
                self.effects |= EffectSet::CONTROL_FLOW;
                if let Some(setup) = setup {
                    self.instruction(setup);
                }
                for capability in capabilities {
                    self.expression(capability);
                }
                self.expression(condition);
                self.instructions(instructions);
            }

            Instruction::For {
                setup,
                iterable,
                instructions,
                ..
            } => {
                self.effects |= EffectSet::CONTROL_FLOW;
                if let Some(setup) = setup {
                    self.instruction(setup);
                }
                self.expression(iterable);
                self.instructions(instructions);
            }

            Instruction::Switch { value, arms } => {
                self.effects |= EffectSet::CONTROL_FLOW;
                self.expression(value);
                self.arms(arms);
            }

            Instruction::ChannelSwitch {
                channels,
                setup,
                repeat_condition,
                arms,
            } => {
                self.effects |=
                    EffectSet::CONTROL_FLOW | EffectSet::CHANNEL | EffectSet::NONDETERMINISTIC;
                for channel in channels {
                    self.expression(channel);
                }
                if let Some(setup) = setup {
                    self.instruction(setup);
                }
                if let Some(condition) = repeat_condition {
                    self.expression(condition);
                }
                self.arms(arms);
            }

            Instruction::With {
                resources,
                instructions,
                ..
            } => {
                self.effects |= EffectSet::CONTROL_FLOW;
                for resource in resources {
                    self.expression(resource);
                }
                self.instructions(instructions);
            }

            Instruction::Break | Instruction::Continue => {
                self.effects |= EffectSet::CONTROL_FLOW;
            }

            Instruction::Evaluate(value) => self.expression(value),
        }
    }

    fn arms(&mut self, arms: &[SwitchArm]) {
        for arm in arms {
            if let Some(source) = &arm.source {
                self.expression(source);
            }
            if let Some(guard) = &arm.guard {
                self.expression(guard);
            }
            self.instructions(&arm.instructions);
        }
    }

    fn expression(&mut self, expression: &Expression) {
        self.expression_count += 1;

        match expression {
            Expression::Integer(_)
            | Expression::Float(_)
            | Expression::Boolean(_)
            | Expression::String(_)
            | Expression::Variable(_)
            | Expression::Function(_) => {}

            Expression::Lambda { body, .. } => self.expression(body),

            Expression::Ownership { op, value } => {
                match op {
                    OwnershipOp::View | OwnershipOp::Borrow | OwnershipOp::AddressOf => {
                        self.effects |= EffectSet::READ_MEMORY;
                    }
                    OwnershipOp::Clone => {
                        self.effects |= EffectSet::READ_MEMORY | EffectSet::ALLOCATE;
                    }
                    OwnershipOp::Move => {
                        self.effects |= EffectSet::READ_MEMORY | EffectSet::WRITE_MEMORY;
                    }
                }
                self.expression(value);
            }

            Expression::List(values)
            | Expression::Tuple(values)
            | Expression::Set(values)
            | Expression::PrintArgs(values)
            | Expression::Construct { args: values, .. }
            | Expression::Variant { fields: values, .. } => {
                self.effects |= EffectSet::ALLOCATE;
                for value in values {
                    self.expression(value);
                }
            }

            Expression::Map(entries) => {
                self.effects |= EffectSet::ALLOCATE;
                for (key, value) in entries {
                    self.expression(key);
                    self.expression(value);
                }
            }

            Expression::Index { object, index } => {
                self.effects |= EffectSet::READ_MEMORY | EffectSet::MAY_PANIC;
                self.expression(object);
                self.expression(index);
            }

            Expression::Slice {
                object,
                start,
                end,
                step,
            } => {
                self.effects |= EffectSet::READ_MEMORY | EffectSet::ALLOCATE | EffectSet::MAY_PANIC;
                self.expression(object);
                for bound in [start, end, step].into_iter().flatten() {
                    self.expression(bound);
                }
            }

            Expression::Format { args, .. } => {
                self.effects |= EffectSet::ALLOCATE;
                for arg in args {
                    self.expression(arg);
                }
            }

            Expression::Member { object, .. } => {
                self.effects |= EffectSet::READ_MEMORY;
                self.expression(object);
            }

            Expression::MethodCall { object, args, .. } => {
                // Until method resolution attaches a qualified function identity
                // to HIR, method calls are conservative.
                self.effects |= EffectSet::CALL | EffectSet::UNKNOWN;
                self.expression(object);
                for arg in args {
                    self.expression(arg);
                }
            }

            Expression::Task { value, .. } => {
                self.effects |=
                    EffectSet::SPAWN_TASK | EffectSet::ALLOCATE | EffectSet::NONDETERMINISTIC;
                self.expression(value);
            }

            Expression::Await(value) => {
                self.effects |= EffectSet::AWAIT | EffectSet::NONDETERMINISTIC;
                self.expression(value);
            }

            Expression::Channel(capacity) => {
                self.effects |= EffectSet::CHANNEL | EffectSet::ALLOCATE;
                self.expression(capacity);
            }

            Expression::Send { value, channel } => {
                self.effects |=
                    EffectSet::CHANNEL | EffectSet::SPAWN_TASK | EffectSet::NONDETERMINISTIC;
                self.expression(value);
                self.expression(channel);
            }

            Expression::ChaosRule { value, .. } => {
                self.effects |=
                    EffectSet::CHAOS | EffectSet::NONDETERMINISTIC | EffectSet::MAY_PANIC;
                self.expression(value);
            }

            Expression::ListComprehension { element, clauses }
            | Expression::SetComprehension { element, clauses } => {
                self.effects |= EffectSet::ALLOCATE | EffectSet::CONTROL_FLOW;
                for clause in clauses {
                    self.expression(&clause.iterable);
                    if let Some(condition) = &clause.condition {
                        self.expression(condition);
                    }
                }
                self.expression(element);
            }

            Expression::MapComprehension {
                key,
                value,
                clauses,
            } => {
                self.effects |= EffectSet::ALLOCATE | EffectSet::CONTROL_FLOW;
                for clause in clauses {
                    self.expression(&clause.iterable);
                    if let Some(condition) = &clause.condition {
                        self.expression(condition);
                    }
                }
                self.expression(key);
                self.expression(value);
            }

            Expression::Conditional {
                condition,
                then_expression,
                else_expression,
            } => {
                self.effects |= EffectSet::CONTROL_FLOW;
                self.expression(condition);
                self.expression(then_expression);
                self.expression(else_expression);
            }

            Expression::FusedPipeline { input, .. } => {
                // Package native ABI call. It may be pure in practice, but the
                // package metadata does not yet carry an effect contract.
                self.effects |= EffectSet::CALL | EffectSet::UNKNOWN;
                self.expression(input);
            }

            Expression::Unary { expression, .. } => self.expression(expression),

            Expression::Binary { left, right, .. } => {
                self.expression(left);
                self.expression(right);
            }

            Expression::Call { function, args } => {
                self.effects |= EffectSet::CALL;
                for arg in args {
                    self.expression(arg);
                }

                match self.summaries.get(function) {
                    Some(summary) => self.effects |= summary.effects,
                    None => self.effects |= EffectSet::UNKNOWN,
                }
            }

            Expression::CallValue { callee, args, .. } => {
                self.effects |= EffectSet::CALL | EffectSet::UNKNOWN;
                self.expression(callee);
                for arg in args {
                    self.expression(arg);
                }
            }
        }
    }
}
