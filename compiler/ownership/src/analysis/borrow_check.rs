use severian_hir::{
    Expression, Function, Instruction, MatchPattern, OwnershipOp, Program, SwitchArm,
};
use std::collections::HashMap;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BorrowKind {
    Shared,
    Mutable,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Loan {
    pub owner: String,
    pub binding: String,
    pub kind: BorrowKind,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BorrowDiagnosticKind {
    SharedWhileMutablyBorrowed,
    MutableWhileBorrowed,
    MoveWhileBorrowed,
    MutateWhileSharedBorrowed,
    ReturnBorrowedValue,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BorrowDiagnostic {
    pub owner: String,
    pub kind: BorrowDiagnosticKind,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct BorrowReport {
    pub diagnostics: Vec<BorrowDiagnostic>,
    pub loans: Vec<Loan>,
}

impl BorrowReport {
    pub fn is_valid(&self) -> bool {
        self.diagnostics.is_empty()
    }
}

pub fn check(program: &Program) -> HashMap<String, BorrowReport> {
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

pub fn check_function(function: &Function) -> BorrowReport {
    let mut checker = Checker::default();

    for parameter in &function.params {
        checker.define(parameter.name.clone());
    }

    checker.instructions(&function.instructions);

    BorrowReport {
        diagnostics: checker.diagnostics,
        loans: checker
            .loans
            .values()
            .flat_map(|loans| loans.iter().cloned())
            .collect(),
    }
}

#[derive(Debug, Clone, Default)]
struct Binding {
    loan_of: Option<(String, BorrowKind)>,
}

#[derive(Debug, Clone, Default)]
struct Checker {
    bindings: HashMap<String, Binding>,
    loans: HashMap<String, Vec<Loan>>,
    diagnostics: Vec<BorrowDiagnostic>,
}

impl Checker {
    fn define(&mut self, name: String) {
        self.release_binding(&name);
        self.bindings.insert(name, Binding::default());
    }

    fn release_binding(&mut self, name: &str) {
        let Some(binding) = self.bindings.remove(name) else {
            return;
        };

        if let Some((owner, _)) = binding.loan_of {
            if let Some(loans) = self.loans.get_mut(&owner) {
                loans.retain(|loan| loan.binding != name);
                if loans.is_empty() {
                    self.loans.remove(&owner);
                }
            }
        }
    }

    fn root_owner(&self, name: &str) -> String {
        let mut current = name.to_string();
        let mut remaining = self.bindings.len() + 1;

        while remaining > 0 {
            remaining -= 1;
            let Some(binding) = self.bindings.get(&current) else {
                break;
            };
            let Some((owner, _)) = &binding.loan_of else {
                break;
            };
            current = owner.clone();
        }

        current
    }

    fn borrow(&mut self, source: &str, destination: &str, kind: BorrowKind) {
        let owner = self.root_owner(source);
        let existing = self.loans.get(&owner).cloned().unwrap_or_default();

        match kind {
            BorrowKind::Shared => {
                if existing.iter().any(|loan| loan.kind == BorrowKind::Mutable) {
                    self.diagnostics.push(BorrowDiagnostic {
                        owner: owner.clone(),
                        kind: BorrowDiagnosticKind::SharedWhileMutablyBorrowed,
                    });
                }
            }
            BorrowKind::Mutable => {
                if !existing.is_empty() {
                    self.diagnostics.push(BorrowDiagnostic {
                        owner: owner.clone(),
                        kind: BorrowDiagnosticKind::MutableWhileBorrowed,
                    });
                }
            }
        }

        self.release_binding(destination);
        self.bindings.insert(
            destination.to_string(),
            Binding {
                loan_of: Some((owner.clone(), kind)),
            },
        );
        self.loans.entry(owner.clone()).or_default().push(Loan {
            owner,
            binding: destination.to_string(),
            kind,
        });
    }

    fn ensure_can_move(&mut self, name: &str) {
        let owner = self.root_owner(name);
        if self.loans.get(&owner).is_some_and(|loans| !loans.is_empty()) {
            self.diagnostics.push(BorrowDiagnostic {
                owner,
                kind: BorrowDiagnosticKind::MoveWhileBorrowed,
            });
        }
    }

    fn ensure_can_mutate(&mut self, name: &str) {
        let owner = self.root_owner(name);
        if self
            .loans
            .get(&owner)
            .is_some_and(|loans| loans.iter().any(|loan| loan.kind == BorrowKind::Shared))
        {
            self.diagnostics.push(BorrowDiagnostic {
                owner,
                kind: BorrowDiagnosticKind::MutateWhileSharedBorrowed,
            });
        }
    }

    fn initializer(&mut self, destination: &str, expression: &Expression) {
        match expression {
            Expression::Ownership { op, value } => {
                if let Expression::Variable(source) = value.as_ref() {
                    match op {
                        OwnershipOp::View | OwnershipOp::AddressOf => {
                            self.borrow(source, destination, BorrowKind::Shared);
                        }
                        OwnershipOp::Borrow => {
                            self.borrow(source, destination, BorrowKind::Mutable);
                        }
                        OwnershipOp::Move => {
                            self.ensure_can_move(source);
                            self.define(destination.to_string());
                        }
                        OwnershipOp::Clone => {
                            self.define(destination.to_string());
                        }
                    }
                    return;
                }

                self.expression(value, Access::Read);
                self.define(destination.to_string());
            }
            _ => {
                self.expression(expression, Access::Read);
                self.define(destination.to_string());
            }
        }
    }

    fn instructions(&mut self, instructions: &[Instruction]) {
        for instruction in instructions {
            self.instruction(instruction);
        }
    }

    fn instruction(&mut self, instruction: &Instruction) {
        match instruction {
            Instruction::Let { name, value } | Instruction::TryLet { name, value } => {
                self.initializer(name, value);
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
                    if let Expression::Variable(name) = value {
                        if let Some(binding) = self.bindings.get(name) {
                            if let Some((owner, _)) = &binding.loan_of {
                                self.diagnostics.push(BorrowDiagnostic {
                                    owner: owner.clone(),
                                    kind: BorrowDiagnosticKind::ReturnBorrowedValue,
                                });
                            }
                        }
                    }
                    self.expression(value, Access::Read);
                }
            }
            Instruction::If {
                condition,
                then_instructions,
                else_instructions,
            } => {
                self.expression(condition, Access::Read);

                let mut then_checker = self.clone();
                then_checker.instructions(then_instructions);

                let mut else_checker = self.clone();
                else_checker.instructions(else_instructions);

                self.merge_branches(&[then_checker, else_checker]);
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

                let mut body = self.clone();
                body.instructions(instructions);
                self.merge_branches(&[body]);
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

                let mut body = self.clone();
                body.define_pattern(pattern);
                body.instructions(instructions);
                self.merge_branches(&[body]);
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

    fn merge_branches(&mut self, branches: &[Checker]) {
        for branch in branches {
            let base = self.diagnostics.len();
            self.diagnostics
                .extend(branch.diagnostics.iter().skip(base).cloned());
        }

        // Conservative union: any loan active on any outgoing branch remains
        // potentially active at the join.
        for branch in branches {
            for (owner, loans) in &branch.loans {
                let entry = self.loans.entry(owner.clone()).or_default();
                for loan in loans {
                    if !entry.contains(loan) {
                        entry.push(loan.clone());
                    }
                }
            }
        }
    }

    fn switch_arms(&mut self, arms: &[SwitchArm]) {
        let mut branches = Vec::with_capacity(arms.len());

        for arm in arms {
            let mut branch = self.clone();
            if let Some(source) = &arm.source {
                branch.expression(source, Access::Read);
            }
            branch.define_pattern(&arm.pattern);
            if let Some(guard) = &arm.guard {
                branch.expression(guard, Access::Read);
            }
            branch.instructions(&arm.instructions);
            branches.push(branch);
        }

        self.merge_branches(&branches);
    }

    fn define_pattern(&mut self, pattern: &MatchPattern) {
        match pattern {
            MatchPattern::Bind(name) => self.define(name.clone()),
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
            Expression::Variable(name) => {
                if matches!(access, Access::Mutate) {
                    self.ensure_can_mutate(name);
                }
            }
            Expression::Ownership { op, value } => {
                if let Expression::Variable(name) = value.as_ref() {
                    match op {
                        OwnershipOp::Move => self.ensure_can_move(name),
                        OwnershipOp::Borrow => {
                            // Temporary mutable borrow.
                            let owner = self.root_owner(name);
                            if self.loans.get(&owner).is_some_and(|loans| !loans.is_empty()) {
                                self.diagnostics.push(BorrowDiagnostic {
                                    owner,
                                    kind: BorrowDiagnosticKind::MutableWhileBorrowed,
                                });
                            }
                        }
                        OwnershipOp::View | OwnershipOp::AddressOf => {
                            let owner = self.root_owner(name);
                            if self.loans.get(&owner).is_some_and(|loans| {
                                loans.iter().any(|loan| loan.kind == BorrowKind::Mutable)
                            }) {
                                self.diagnostics.push(BorrowDiagnostic {
                                    owner,
                                    kind: BorrowDiagnosticKind::SharedWhileMutablyBorrowed,
                                });
                            }
                        }
                        OwnershipOp::Clone => {}
                    }
                } else {
                    self.expression(value, Access::Read);
                }
            }
            Expression::Lambda { body, .. } => self.expression(body, Access::Read),
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
                self.expression(then_expression, access);
                self.expression(else_expression, access);
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

#[derive(Debug, Clone, Copy)]
enum Access {
    Read,
    Mutate,
}

