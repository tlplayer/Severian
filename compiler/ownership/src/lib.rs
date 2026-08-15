#![forbid(unsafe_code)]

use severian_hir::{
    BindingId, BindingRef, CallTarget, Expression, Function, FunctionId, Instruction, MatchPattern,
    OwnershipOp, Program, SwitchArm,
};
use std::collections::HashMap;

/// Check the ownership rules encoded in HIR.
///
/// The pass is deliberately separate from name and type checking.  It tracks
/// owners through control flow, treats a move on either branch as a move at the
/// join, and keeps loans live until the last use of their binding.
pub fn check(program: &Program) -> Result<(), OwnershipError> {
    let mut globals = Checker {
        effects: infer_function_effects(program),
        ..Checker::default()
    };
    for global in &program.globals {
        globals.check_expression(&global.value, Access::Read)?;
        globals.define(global.name.clone(), None);
    }

    for function in &program.functions {
        check_function(function, &globals, None, &[])?;
    }
    for class in &program.classes {
        for default in class.field_defaults.iter().flatten() {
            globals.check_expression(default, Access::Read)?;
        }
        for function in class.methods.iter().chain(&class.constructors) {
            check_function(function, &globals, Some(&class.name), &class.fields)?;
        }
    }

    Ok(())
}

fn check_function(
    function: &Function,
    globals: &Checker,
    class: Option<&str>,
    fields: &[String],
) -> Result<(), OwnershipError> {
    let mut checker = globals.clone();
    checker.remaining = count_instruction_uses(&function.instructions);
    if let Some(class) = class {
        checker.define(
            BindingRef::new(BindingId::from_name(&format!("{class}.self")), "self"),
            None,
        );
    }
    for field in fields {
        checker.define(
            BindingRef::new(
                BindingId::from_name(&format!("{}.{field}", class.unwrap_or("<field>"))),
                field,
            ),
            None,
        );
    }
    for parameter in &function.params {
        if let Some(default) = &parameter.default {
            checker.check_expression(default, Access::Read)?;
        }
        checker.define(parameter.name.clone(), None);
    }
    if let Some(contract) = &function.contract {
        for expression in contract
            .clauses
            .iter()
            .map(|clause| &clause.condition)
            .chain(&contract.capabilities)
        {
            checker.check_expression(expression, Access::Read)?;
        }
    }
    checker.check_instructions(&function.instructions)?;

    for test in &function.tests {
        let mut test_checker = globals.clone();
        test_checker.remaining = count_instruction_uses(&test.instructions);
        if let Some(class) = class {
            test_checker.define(
                BindingRef::new(BindingId::from_name(&format!("{class}.self")), "self"),
                None,
            );
        }
        for field in fields {
            test_checker.define(
                BindingRef::new(
                    BindingId::from_name(&format!("{}.{field}", class.unwrap_or("<field>"))),
                    field,
                ),
                None,
            );
        }
        if let Some(contract) = &test.contract {
            for expression in contract
                .clauses
                .iter()
                .map(|clause| &clause.condition)
                .chain(&contract.capabilities)
            {
                test_checker.check_expression(expression, Access::Read)?;
            }
        }
        test_checker.check_instructions(&test.instructions)?;
    }
    Ok(())
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Access {
    Read,
    Mutate,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum LoanKind {
    Shared,
    Mutable,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum ParameterEffect {
    View,
    Borrow,
    Move,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct Loan {
    owner: BindingId,
    kind: LoanKind,
}

#[derive(Debug, Clone, Default)]
struct BindingState {
    name: String,
    moved: bool,
    loan: Option<Loan>,
}

#[derive(Debug, Clone, Default)]
struct Checker {
    bindings: HashMap<BindingId, BindingState>,
    remaining: HashMap<BindingId, usize>,
    temporary_loans: Vec<Loan>,
    effects: HashMap<FunctionId, Vec<ParameterEffect>>,
}

impl Checker {
    fn define(&mut self, binding: BindingRef, loan: Option<Loan>) {
        self.bindings.insert(
            binding.id,
            BindingState {
                name: binding.name,
                moved: false,
                loan,
            },
        );
    }

    fn check_instructions(&mut self, instructions: &[Instruction]) -> Result<(), OwnershipError> {
        for instruction in instructions {
            self.check_instruction(instruction)?;
        }
        Ok(())
    }

    fn check_instruction(&mut self, instruction: &Instruction) -> Result<(), OwnershipError> {
        match instruction {
            Instruction::Let { name, value } | Instruction::TryLet { name, value, .. } => {
                let loan = self.check_initializer(name, value)?;
                self.define(name.clone(), loan);
            }
            Instruction::Assign { target, value, .. } => {
                self.check_expression(value, Access::Read)?;
                self.check_expression(target, Access::Mutate)?;
            }
            Instruction::Print(value)
            | Instruction::Assert(value)
            | Instruction::Evaluate(value) => self.check_expression(value, Access::Read)?,
            Instruction::Return(value) => {
                if let Some(value) = value {
                    self.ensure_return_does_not_escape_loan(value)?;
                    self.check_expression(value, Access::Read)?;
                }
            }
            Instruction::If {
                condition,
                then_instructions,
                else_instructions,
            } => {
                self.check_expression(condition, Access::Read)?;
                self.check_branches([then_instructions.as_slice(), else_instructions.as_slice()])?;
            }
            Instruction::While {
                setup,
                capabilities,
                condition,
                instructions,
            } => {
                if let Some(setup) = setup {
                    self.check_instruction(setup)?;
                }
                for capability in capabilities {
                    self.check_expression(capability, Access::Read)?;
                }
                self.check_expression(condition, Access::Read)?;
                self.check_branches([instructions.as_slice()])?;
            }
            Instruction::For {
                setup,
                pattern,
                iterable,
                instructions,
            } => {
                if let Some(setup) = setup {
                    self.check_instruction(setup)?;
                }
                self.check_expression(iterable, Access::Read)?;
                let mut branch = self.clone();
                define_pattern(&mut branch, pattern);
                branch.check_instructions(instructions)?;
                self.merge_from(&branch);
            }
            Instruction::Switch { value, arms } => {
                self.check_expression(value, Access::Read)?;
                self.check_arms(arms)?;
            }
            Instruction::ChannelSwitch {
                channels,
                setup,
                repeat_condition,
                arms,
            } => {
                for channel in channels {
                    self.check_expression(channel, Access::Read)?;
                }
                if let Some(setup) = setup {
                    self.check_instruction(setup)?;
                }
                if let Some(condition) = repeat_condition {
                    self.check_expression(condition, Access::Read)?;
                }
                self.check_arms(arms)?;
            }
            Instruction::With {
                resources,
                instructions,
                ..
            } => {
                for resource in resources {
                    self.check_expression(resource, Access::Read)?;
                }
                self.check_instructions(instructions)?;
            }
            Instruction::Break | Instruction::Continue => {}
        }
        Ok(())
    }

    fn check_arms(&mut self, arms: &[SwitchArm]) -> Result<(), OwnershipError> {
        let mut branches = Vec::with_capacity(arms.len());
        for arm in arms {
            let mut branch = self.clone();
            if let Some(source) = &arm.source {
                branch.check_expression(source, Access::Read)?;
            }
            define_pattern(&mut branch, &arm.pattern);
            if let Some(guard) = &arm.guard {
                branch.check_expression(guard, Access::Read)?;
            }
            branch.check_instructions(&arm.instructions)?;
            branches.push(branch);
        }
        for branch in branches {
            self.merge_from(&branch);
        }
        Ok(())
    }

    fn check_branches<const N: usize>(
        &mut self,
        branches: [&[Instruction]; N],
    ) -> Result<(), OwnershipError> {
        let mut results = Vec::with_capacity(N);
        for instructions in branches {
            let mut branch = self.clone();
            branch.check_instructions(instructions)?;
            results.push(branch);
        }
        for branch in results {
            self.merge_from(&branch);
        }
        Ok(())
    }

    fn merge_from(&mut self, branch: &Checker) {
        for (name, state) in &mut self.bindings {
            if branch.bindings.get(name).is_some_and(|other| other.moved) {
                state.moved = true;
            }
        }
        for (name, count) in &mut self.remaining {
            *count = (*count).min(branch.remaining.get(name).copied().unwrap_or(0));
        }
    }

    fn check_initializer(
        &mut self,
        destination: &BindingRef,
        expression: &Expression,
    ) -> Result<Option<Loan>, OwnershipError> {
        let Expression::Ownership { op, value } = expression.kind() else {
            self.check_expression(expression, Access::Read)?;
            return Ok(None);
        };
        let Expression::Variable(source) = value.kind() else {
            self.check_expression(value, Access::Read)?;
            return Ok(None);
        };

        self.ensure_alive(source)?;
        let owner = self.root_owner(source);
        let loan = match op {
            OwnershipOp::View | OwnershipOp::AddressOf => {
                self.ensure_can_borrow(&owner, LoanKind::Shared)?;
                Some(Loan {
                    owner,
                    kind: LoanKind::Shared,
                })
            }
            OwnershipOp::Borrow => {
                self.ensure_can_borrow(&owner, LoanKind::Mutable)?;
                Some(Loan {
                    owner,
                    kind: LoanKind::Mutable,
                })
            }
            OwnershipOp::Clone => {
                self.check_variable(source, Access::Read)?;
                None
            }
            OwnershipOp::Move => {
                self.ensure_unborrowed(&owner, "move")?;
                let state = self
                    .bindings
                    .get_mut(&source.id)
                    .ok_or_else(|| unknown(source))?;
                state.moved = true;
                None
            }
        };
        self.consume(source);

        // An unused loan has no live range and therefore cannot block its owner.
        if matches!(loan, Some(_)) && self.remaining.get(&destination.id).copied().unwrap_or(0) == 0
        {
            return Ok(None);
        }
        Ok(loan)
    }

    fn check_expression(
        &mut self,
        expression: &Expression,
        access: Access,
    ) -> Result<(), OwnershipError> {
        match expression {
            Expression::Variable(name) => self.check_variable(name, access)?,
            Expression::Ownership {
                op: OwnershipOp::Move,
                value,
            } => {
                if let Expression::Variable(name) = value.kind() {
                    self.ensure_alive(name)?;
                    let owner = self.root_owner(name);
                    self.ensure_unborrowed(&owner, "move")?;
                    self.bindings
                        .get_mut(&name.id)
                        .ok_or_else(|| unknown(name))?
                        .moved = true;
                    self.consume(name);
                } else {
                    self.check_expression(value, Access::Read)?;
                }
            }
            Expression::Ownership { value, .. }
            | Expression::Member { object: value, .. }
            | Expression::Await(value)
            | Expression::Channel(value)
            | Expression::Task { value, .. }
            | Expression::ChaosRule { value, .. }
            | Expression::FusedPipeline { input: value, .. }
            | Expression::Unary {
                expression: value, ..
            } => {
                self.check_expression(value, access)?;
            }
            Expression::Lambda { params, body } => self.check_lambda(params, body)?,
            Expression::Closure { params, body, .. } => self.check_closure(params, body)?,
            Expression::List(values)
            | Expression::Tuple(values)
            | Expression::Set(values)
            | Expression::PrintArgs(values)
            | Expression::Construct { args: values, .. }
            | Expression::Variant { fields: values, .. } => {
                for value in values {
                    self.check_expression(value, Access::Read)?;
                }
            }
            Expression::ConstructFields { fields, .. } => {
                for (_, value) in fields {
                    self.check_expression(value, Access::Read)?;
                }
            }
            Expression::ObjectUpdate { object, fields, .. } => {
                self.check_expression(object, Access::Read)?;
                for (_, value) in fields {
                    self.check_expression(value, Access::Read)?;
                }
            }
            Expression::ObjectDocument { object, .. } => {
                self.check_expression(object, Access::Read)?;
            }
            Expression::Map(entries) => {
                for (key, value) in entries {
                    self.check_expression(key, Access::Read)?;
                    self.check_expression(value, Access::Read)?;
                }
            }
            Expression::Index { object, index } => {
                self.check_expression(object, access)?;
                self.check_expression(index, Access::Read)?;
            }
            Expression::Slice {
                object,
                start,
                end,
                step,
            } => {
                self.check_expression(object, access)?;
                for bound in [start, end, step].into_iter().flatten() {
                    self.check_expression(bound, Access::Read)?;
                }
            }
            Expression::Format { args, .. } => {
                for arg in args {
                    self.check_expression(arg, Access::Read)?;
                }
            }
            Expression::Call { target, args } => self.check_call(target, args)?,
            Expression::ForeignCall { args, .. } => {
                for argument in args {
                    self.check_expression(argument, Access::Read)?;
                }
            }
            Expression::Typed { expression, .. } => self.check_expression(expression, access)?,
            Expression::MethodCall {
                object,
                method,
                args,
            } => {
                let object_access = if mutating_method(method) {
                    Access::Mutate
                } else {
                    Access::Read
                };
                self.check_expression(object, object_access)?;
                for arg in args {
                    self.check_expression(arg, Access::Read)?;
                }
            }
            Expression::Send { value, channel } => {
                self.check_expression(value, Access::Read)?;
                self.check_expression(channel, Access::Mutate)?;
            }
            Expression::ListComprehension { element, clauses } => {
                let previous = self.bindings.clone();
                for clause in clauses {
                    self.check_expression(&clause.iterable, Access::Read)?;
                    define_pattern(self, &clause.pattern);
                    if let Some(condition) = &clause.condition {
                        self.check_expression(condition, Access::Read)?;
                    }
                }
                self.check_expression(element, Access::Read)?;
                self.bindings = previous;
            }
            Expression::SetComprehension { element, clauses } => {
                let previous = self.bindings.clone();
                for clause in clauses {
                    self.check_expression(&clause.iterable, Access::Read)?;
                    define_pattern(self, &clause.pattern);
                    if let Some(condition) = &clause.condition {
                        self.check_expression(condition, Access::Read)?;
                    }
                }
                self.check_expression(element, Access::Read)?;
                self.bindings = previous;
            }
            Expression::MapComprehension {
                key,
                value,
                clauses,
            } => {
                let previous = self.bindings.clone();
                for clause in clauses {
                    self.check_expression(&clause.iterable, Access::Read)?;
                    define_pattern(self, &clause.pattern);
                    if let Some(condition) = &clause.condition {
                        self.check_expression(condition, Access::Read)?;
                    }
                }
                self.check_expression(key, Access::Read)?;
                self.check_expression(value, Access::Read)?;
                self.bindings = previous;
            }
            Expression::Conditional {
                condition,
                then_expression,
                else_expression,
            } => {
                self.check_expression(condition, Access::Read)?;
                let mut then_checker = self.clone();
                then_checker.check_expression(then_expression, access)?;
                let mut else_checker = self.clone();
                else_checker.check_expression(else_expression, access)?;
                self.merge_from(&then_checker);
                self.merge_from(&else_checker);
            }
            Expression::RegistryLookup { key, fallback, .. } => {
                self.check_expression(key, Access::Read)?;
                self.check_expression(fallback, access)?;
            }
            Expression::Binary { left, right, .. } => {
                self.check_expression(left, Access::Read)?;
                self.check_expression(right, Access::Read)?;
            }
            Expression::CallValue { callee, args, .. } => {
                self.check_expression(callee, Access::Read)?;
                for arg in args {
                    self.check_expression(arg, Access::Read)?;
                }
            }
            Expression::Integer(_)
            | Expression::Float(_)
            | Expression::Boolean(_)
            | Expression::String(_)
            | Expression::Function(_) => {}
        }
        Ok(())
    }

    fn check_variable(
        &mut self,
        binding: &BindingRef,
        access: Access,
    ) -> Result<(), OwnershipError> {
        // HIR may contain compiler-provided capabilities (for example
        // `stdout` and `chaos`) which are intentionally absent from the local
        // binding table. Name resolution has already validated ordinary reads;
        // only explicit ownership operations require a tracked local owner.
        if !self.bindings.contains_key(&binding.id) {
            self.consume(binding);
            return Ok(());
        }
        self.ensure_alive(binding)?;
        let state = self
            .bindings
            .get(&binding.id)
            .ok_or_else(|| unknown(binding))?;
        if let Some(loan) = &state.loan {
            if access == Access::Mutate && loan.kind != LoanKind::Mutable {
                return Err(ownership_error(
                    "E000302",
                    format!("shared view `{binding}` cannot be mutated"),
                ));
            }
        } else {
            let owner = self.root_owner(binding);
            let owner_name = self.binding_name(owner);
            match access {
                Access::Read if self.has_live_loan(&owner, Some(LoanKind::Mutable)) => {
                    return Err(ownership_error(
                        "E000303",
                        format!(
                            "owner `{owner_name}` cannot be read while an exclusive borrow is live"
                        ),
                    ));
                }
                Access::Mutate if self.has_live_loan(&owner, None) => {
                    return Err(ownership_error(
                        "E000302",
                        format!(
                            "owner `{owner_name}` cannot be mutated while a borrow is live. An owner cannot be structurally mutated while an immutable borrow is live."
                        ),
                    ));
                }
                _ => {}
            }
        }
        self.consume(binding);
        Ok(())
    }

    fn ensure_alive(&self, binding: &BindingRef) -> Result<(), OwnershipError> {
        let state = self
            .bindings
            .get(&binding.id)
            .ok_or_else(|| unknown(binding))?;
        if state.moved {
            return Err(ownership_error(
                "E000301",
                format!(
                    "binding `{binding}` cannot be used after its ownership was moved. A binding cannot be read after ownership has moved to another binding."
                ),
            ));
        }
        Ok(())
    }

    fn ensure_can_borrow(&self, owner: &BindingId, kind: LoanKind) -> Result<(), OwnershipError> {
        let conflict = match kind {
            LoanKind::Shared => self.has_live_loan(owner, Some(LoanKind::Mutable)),
            LoanKind::Mutable => self.has_live_loan(owner, None),
        };
        if conflict {
            return Err(ownership_error(
                "E000303",
                format!(
                    "conflicting borrow of `{}` while another borrow is live",
                    self.binding_name(*owner)
                ),
            ));
        }
        Ok(())
    }

    fn ensure_unborrowed(&self, owner: &BindingId, operation: &str) -> Result<(), OwnershipError> {
        if self.has_live_loan(owner, None) {
            return Err(ownership_error(
                "E000304",
                format!(
                    "cannot {operation} `{}` while it is borrowed",
                    self.binding_name(*owner)
                ),
            ));
        }
        Ok(())
    }

    fn root_owner(&self, binding: &BindingRef) -> BindingId {
        self.bindings
            .get(&binding.id)
            .and_then(|state| state.loan.as_ref())
            .map_or(binding.id, |loan| loan.owner)
    }

    fn has_live_loan(&self, owner: &BindingId, kind: Option<LoanKind>) -> bool {
        self.temporary_loans
            .iter()
            .any(|loan| loan.owner == *owner && kind.is_none_or(|kind| loan.kind == kind))
            || self.bindings.iter().any(|(name, state)| {
                state.loan.as_ref().is_some_and(|loan| {
                    loan.owner == *owner
                        && kind.is_none_or(|kind| loan.kind == kind)
                        && self.remaining.get(name).copied().unwrap_or(0) > 0
                })
            })
    }

    fn consume(&mut self, binding: &BindingRef) {
        if let Some(remaining) = self.remaining.get_mut(&binding.id) {
            *remaining = remaining.saturating_sub(1);
        }
    }

    fn binding_name(&self, id: BindingId) -> &str {
        self.bindings
            .get(&id)
            .map_or("<unknown binding>", |state| state.name.as_str())
    }

    fn check_call(
        &mut self,
        function: &CallTarget,
        args: &[Expression],
    ) -> Result<(), OwnershipError> {
        let effects = self.effects.get(&function.id).cloned().unwrap_or_default();
        let loan_boundary = self.temporary_loans.len();

        for (index, argument) in args.iter().enumerate() {
            let effect = effects.get(index).copied().unwrap_or(ParameterEffect::View);
            self.check_call_argument(&function.name, index, argument, effect)?;
        }

        self.temporary_loans.truncate(loan_boundary);
        Ok(())
    }

    fn check_call_argument(
        &mut self,
        function: &str,
        index: usize,
        argument: &Expression,
        effect: ParameterEffect,
    ) -> Result<(), OwnershipError> {
        let (explicit, value) = match argument.kind() {
            Expression::Ownership { op, value } => (Some(*op), value.as_ref()),
            value => (None, value),
        };
        let Expression::Variable(source) = value.kind() else {
            self.check_expression(argument, Access::Read)?;
            return Ok(());
        };

        self.ensure_alive(source)?;
        let owner = self.root_owner(source);
        let requested = match explicit {
            Some(OwnershipOp::View | OwnershipOp::AddressOf) => ParameterEffect::View,
            Some(OwnershipOp::Borrow) => ParameterEffect::Borrow,
            Some(OwnershipOp::Move) => ParameterEffect::Move,
            Some(OwnershipOp::Clone) => {
                self.check_variable(source, Access::Read)?;
                return Ok(());
            }
            None => effect,
        };

        if requested < effect {
            return Err(ownership_error(
                "E000306",
                format!(
                    "argument {} to `{function}` requires {}, but `{source}` is only passed as {}",
                    index + 1,
                    effect_name(effect),
                    effect_name(requested),
                ),
            ));
        }

        match requested {
            ParameterEffect::View => {
                self.ensure_can_borrow(&owner, LoanKind::Shared)?;
                self.consume(source);
                self.temporary_loans.push(Loan {
                    owner,
                    kind: LoanKind::Shared,
                });
            }
            ParameterEffect::Borrow => {
                self.ensure_can_borrow(&owner, LoanKind::Mutable)?;
                self.consume(source);
                self.temporary_loans.push(Loan {
                    owner,
                    kind: LoanKind::Mutable,
                });
            }
            ParameterEffect::Move => {
                self.ensure_unborrowed(&owner, "move")?;
                self.bindings
                    .get_mut(&source.id)
                    .ok_or_else(|| unknown(source))?
                    .moved = true;
                self.consume(source);
            }
        }
        Ok(())
    }

    fn ensure_return_does_not_escape_loan(
        &self,
        expression: &Expression,
    ) -> Result<(), OwnershipError> {
        let borrowed = match expression.kind() {
            Expression::Variable(name) => self
                .bindings
                .get(&name.id)
                .and_then(|state| state.loan.as_ref())
                .map(|_| name.name.as_str()),
            Expression::Ownership {
                op: OwnershipOp::View | OwnershipOp::Borrow | OwnershipOp::AddressOf,
                value,
            } => match value.kind() {
                Expression::Variable(name) => Some(name.name.as_str()),
                _ => None,
            },
            _ => None,
        };
        if let Some(name) = borrowed {
            return Err(ownership_error(
                "E000305",
                format!("borrowed value `{name}` cannot escape through a return"),
            ));
        }
        Ok(())
    }
}

mod closure;
mod effects;
use effects::*;
