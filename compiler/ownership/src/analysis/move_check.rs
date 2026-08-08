use severian_hir::{
    Expression, Function, Instruction, MatchPattern, OwnershipOp, Program, SwitchArm,
};
use std::collections::HashMap;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MoveState {
    Available,
    Moved,
    MaybeMoved,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MoveDiagnosticKind {
    UseAfterMove,
    MoveAfterMove,
    MutationAfterMove,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MoveDiagnostic {
    pub binding: String,
    pub kind: MoveDiagnosticKind,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct MoveReport {
    pub diagnostics: Vec<MoveDiagnostic>,
    pub final_states: HashMap<String, MoveState>,
}

impl MoveReport {
    pub fn is_valid(&self) -> bool {
        self.diagnostics.is_empty()
    }
}

pub fn check(program: &Program) -> HashMap<String, MoveReport> {
    let mut reports = HashMap::new();

    for function in &program.functions {
        reports.insert(function.name.clone(), check_function(function));
    }

    for class in &program.classes {
        for function in class.methods.iter().chain(&class.constructors) {
            reports.insert(
                format!("{}::{}", class.name, function.name),
                check_function(function),
            );
        }
    }

    reports
}

pub fn check_function(function: &Function) -> MoveReport {
    let mut checker = Checker::default();

    for parameter in &function.params {
        checker.define(&parameter.name);
        if let Some(default) = &parameter.default {
            checker.expression(default, Access::Read);
        }
    }

    if let Some(contract) = &function.contract {
        for expression in contract
            .requirements
            .iter()
            .chain(&contract.capabilities)
        {
            checker.expression(expression, Access::Read);
        }
    }

    checker.instructions(&function.instructions);

    MoveReport {
        diagnostics: checker.diagnostics,
        final_states: checker.bindings,
    }
}

#[derive(Debug, Clone, Copy)]
enum Access {
    Read,
    Mutate,
}

#[derive(Debug, Clone, Default)]
struct Checker {
    bindings: HashMap<String, MoveState>,
    diagnostics: Vec<MoveDiagnostic>,
}

impl Checker {
    fn define(&mut self, name: &str) {
        self.bindings
            .insert(name.to_string(), MoveState::Available);
    }

    fn use_binding(&mut self, name: &str, access: Access) {
        match self.bindings.get(name).copied() {
            Some(MoveState::Moved | MoveState::MaybeMoved) => {
                self.diagnostics.push(MoveDiagnostic {
                    binding: name.to_string(),
                    kind: match access {
                        Access::Read => MoveDiagnosticKind::UseAfterMove,
                        Access::Mutate => MoveDiagnosticKind::MutationAfterMove,
                    },
                });
            }
            Some(MoveState::Available) | None => {}
        }
    }

    fn move_binding(&mut self, name: &str) {
        match self.bindings.get(name).copied() {
            Some(MoveState::Moved | MoveState::MaybeMoved) => {
                self.diagnostics.push(MoveDiagnostic {
                    binding: name.to_string(),
                    kind: MoveDiagnosticKind::MoveAfterMove,
                });
            }
            Some(MoveState::Available) | None => {}
        }

        self.bindings.insert(name.to_string(), MoveState::Moved);
    }

    fn instructions(&mut self, instructions: &[Instruction]) {
        for instruction in instructions {
            self.instruction(instruction);
        }
    }

    fn instruction(&mut self, instruction: &Instruction) {
        match instruction {
            Instruction::Let { name, value } | Instruction::TryLet { name, value } => {
                self.expression(value, Access::Read);
                self.define(name);
            }
            Instruction::Assign { target, value, .. } => {
                self.expression(value, Access::Read);
                self.expression(target, Access::Mutate);
            }
            Instruction::Print(value)
            | Instruction::Assert(value)
            | Instruction::Evaluate(value) => self.expression(value, Access::Read),
            Instruction::Return(value) => {
                if let Some(value) = value {
                    self.expression(value, Access::Read);
                }
            }
            Instruction::If {
                condition,
                then_instructions,
                else_instructions,
            } => {
                self.expression(condition, Access::Read);
                let before = self.clone();

                let mut then_checker = before.clone();
                then_checker.instructions(then_instructions);

                let mut else_checker = before;
                else_checker.instructions(else_instructions);

                self.diagnostics
                    .extend(then_checker.diagnostics.iter().skip(self.diagnostics.len()).cloned());
                self.diagnostics
                    .extend(else_checker.diagnostics.iter().skip(self.diagnostics.len()).cloned());

                self.bindings =
                    merge_states(&then_checker.bindings, &else_checker.bindings);
            }
            Instruction::While {
                setup,
                capabilities,
                condition,
                instructions,
            } => {
                if let Some(setup) = setup {
                    self.instruction(setup);
                }
                for capability in capabilities {
                    self.expression(capability, Access::Read);
                }
                self.expression(condition, Access::Read);

                let before = self.bindings.clone();
                let mut body = self.clone();
                body.instructions(instructions);
                self.diagnostics = body.diagnostics;
                self.bindings = merge_loop_states(&before, &body.bindings);
            }
            Instruction::For {
                setup,
                pattern,
                iterable,
                instructions,
            } => {
                if let Some(setup) = setup {
                    self.instruction(setup);
                }
                self.expression(iterable, Access::Read);

                let before = self.bindings.clone();
                let mut body = self.clone();
                body.define_pattern(pattern);
                body.instructions(instructions);
                self.diagnostics = body.diagnostics;
                self.bindings = merge_loop_states(&before, &body.bindings);
            }
            Instruction::Switch { value, arms } => {
                self.expression(value, Access::Read);
                self.switch_arms(arms);
            }
            Instruction::ChannelSwitch {
                channels,
                setup,
                repeat_condition,
                arms,
            } => {
                for channel in channels {
                    self.expression(channel, Access::Read);
                }
                if let Some(setup) = setup {
                    self.instruction(setup);
                }
                if let Some(condition) = repeat_condition {
                    self.expression(condition, Access::Read);
                }
                self.switch_arms(arms);
            }
            Instruction::With {
                resources,
                instructions,
                ..
            } => {
                for resource in resources {
                    self.expression(resource, Access::Read);
                }
                self.instructions(instructions);
            }
            Instruction::Break | Instruction::Continue => {}
        }
    }

    fn switch_arms(&mut self, arms: &[SwitchArm]) {
        if arms.is_empty() {
            return;
        }

        let before = self.clone();
        let mut branch_states = Vec::with_capacity(arms.len());

        for arm in arms {
            let mut branch = before.clone();
            if let Some(source) = &arm.source {
                branch.expression(source, Access::Read);
            }
            branch.define_pattern(&arm.pattern);
            if let Some(guard) = &arm.guard {
                branch.expression(guard, Access::Read);
            }
            branch.instructions(&arm.instructions);
            branch_states.push(branch);
        }

        for branch in &branch_states {
            self.diagnostics.extend(
                branch
                    .diagnostics
                    .iter()
                    .skip(before.diagnostics.len())
                    .cloned(),
            );
        }

        let mut merged = branch_states[0].bindings.clone();
        for branch in branch_states.iter().skip(1) {
            merged = merge_states(&merged, &branch.bindings);
        }
        self.bindings = merged;
    }

    fn define_pattern(&mut self, pattern: &MatchPattern) {
        match pattern {
            MatchPattern::Bind(name) => self.define(name),
            MatchPattern::Constructor { fields, .. } => {
                for field in fields {
                    self.define_pattern(field);
                }
            }
            MatchPattern::Wildcard
            | MatchPattern::Integer(_)
            | MatchPattern::Float(_)
            | MatchPattern::Boolean(_)
            | MatchPattern::String(_) => {}
        }
    }

    fn expression(&mut self, expression: &Expression, access: Access) {
        match expression {
            Expression::Variable(name) => self.use_binding(name, access),
            Expression::Ownership {
                op: OwnershipOp::Move,
                value,
            } => {
                if let Expression::Variable(name) = value.as_ref() {
                    self.move_binding(name);
                } else {
                    self.expression(value, Access::Read);
                }
            }
            Expression::Ownership {
                op: OwnershipOp::Clone,
                value,
            }
            | Expression::Ownership {
                op: OwnershipOp::View,
                value,
            }
            | Expression::Ownership {
                op: OwnershipOp::Borrow,
                value,
            }
            | Expression::Ownership {
                op: OwnershipOp::AddressOf,
                value,
            } => self.expression(value, Access::Read),
            Expression::Lambda { params, body } => {
                let saved = self.bindings.clone();
                for param in params {
                    self.define(param);
                }
                self.expression(body, Access::Read);
                self.bindings = saved;
            }
            Expression::List(values)
            | Expression::Tuple(values)
            | Expression::Set(values)
            | Expression::PrintArgs(values)
            | Expression::Construct { args: values, .. }
            | Expression::Variant { fields: values, .. } => {
                for value in values {
                    self.expression(value, Access::Read);
                }
            }
            Expression::Map(entries) => {
                for (key, value) in entries {
                    self.expression(key, Access::Read);
                    self.expression(value, Access::Read);
                }
            }
            Expression::Index { object, index } => {
                self.expression(object, access);
                self.expression(index, Access::Read);
            }
            Expression::Slice {
                object,
                start,
                end,
                step,
            } => {
                self.expression(object, access);
                for bound in [start, end, step].into_iter().flatten() {
                    self.expression(bound, Access::Read);
                }
            }
            Expression::Format { args, .. } | Expression::Call { args, .. } => {
                for arg in args {
                    self.expression(arg, Access::Read);
                }
            }
            Expression::Member { object, .. }
            | Expression::Await(object)
            | Expression::Channel(object)
            | Expression::Task { value: object, .. }
            | Expression::ChaosRule { value: object, .. }
            | Expression::FusedPipeline { input: object, .. }
            | Expression::Unary {
                expression: object, ..
            } => self.expression(object, access),
            Expression::MethodCall { object, args, .. } => {
                self.expression(object, Access::Read);
                for arg in args {
                    self.expression(arg, Access::Read);
                }
            }
            Expression::Send { value, channel } => {
                self.expression(value, Access::Read);
                self.expression(channel, Access::Mutate);
            }
            Expression::ListComprehension { element, clauses }
            | Expression::SetComprehension { element, clauses } => {
                for clause in clauses {
                    self.expression(&clause.iterable, Access::Read);
                    self.define_pattern(&clause.pattern);
                    if let Some(condition) = &clause.condition {
                        self.expression(condition, Access::Read);
                    }
                }
                self.expression(element, Access::Read);
            }
            Expression::MapComprehension {
                key,
                value,
                clauses,
            } => {
                for clause in clauses {
                    self.expression(&clause.iterable, Access::Read);
                    self.define_pattern(&clause.pattern);
                    if let Some(condition) = &clause.condition {
                        self.expression(condition, Access::Read);
                    }
                }
                self.expression(key, Access::Read);
                self.expression(value, Access::Read);
            }
            Expression::Conditional {
                condition,
                then_expression,
                else_expression,
            } => {
                self.expression(condition, Access::Read);

                let before = self.clone();
                let mut then_checker = before.clone();
                then_checker.expression(then_expression, access);

                let mut else_checker = before;
                else_checker.expression(else_expression, access);

                self.bindings =
                    merge_states(&then_checker.bindings, &else_checker.bindings);
            }
            Expression::Binary { left, right, .. } => {
                self.expression(left, Access::Read);
                self.expression(right, Access::Read);
            }
            Expression::CallValue { callee, args, .. } => {
                self.expression(callee, Access::Read);
                for arg in args {
                    self.expression(arg, Access::Read);
                }
            }
            Expression::Integer(_)
            | Expression::Float(_)
            | Expression::Boolean(_)
            | Expression::String(_)
            | Expression::Function(_) => {}
        }
    }
}

fn merge_states(
    left: &HashMap<String, MoveState>,
    right: &HashMap<String, MoveState>,
) -> HashMap<String, MoveState> {
    let mut result = HashMap::new();

    for name in left.keys().chain(right.keys()) {
        let a = left.get(name).copied().unwrap_or(MoveState::Available);
        let b = right.get(name).copied().unwrap_or(MoveState::Available);
        let merged = match (a, b) {
            (MoveState::Available, MoveState::Available) => MoveState::Available,
            (MoveState::Moved, MoveState::Moved) => MoveState::Moved,
            (a, b) if a == b => a,
            _ => MoveState::MaybeMoved,
        };
        result.insert(name.clone(), merged);
    }

    result
}

fn merge_loop_states(
    before: &HashMap<String, MoveState>,
    after_one_iteration: &HashMap<String, MoveState>,
) -> HashMap<String, MoveState> {
    merge_states(before, after_one_iteration)
}

