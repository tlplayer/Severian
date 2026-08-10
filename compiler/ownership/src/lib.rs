#![forbid(unsafe_code)]

use severian_hir::{
    Expression, Function, Instruction, MatchPattern, OwnershipOp, Program, SwitchArm,
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
        check_function(function, &globals, &[])?;
    }
    for class in &program.classes {
        for default in class.field_defaults.iter().flatten() {
            globals.check_expression(default, Access::Read)?;
        }
        for function in class.methods.iter().chain(&class.constructors) {
            check_function(function, &globals, &class.fields)?;
        }
    }

    Ok(())
}

fn check_function(
    function: &Function,
    globals: &Checker,
    fields: &[String],
) -> Result<(), OwnershipError> {
    let mut checker = globals.clone();
    checker.remaining = count_instruction_uses(&function.instructions);
    for field in fields {
        checker.define(field.clone(), None);
    }
    for parameter in &function.params {
        if let Some(default) = &parameter.default {
            checker.check_expression(default, Access::Read)?;
        }
        checker.define(parameter.name.clone(), None);
    }
    if let Some(contract) = &function.contract {
        for expression in contract.requirements.iter().chain(&contract.capabilities) {
            checker.check_expression(expression, Access::Read)?;
        }
    }
    checker.check_instructions(&function.instructions)?;

    for test in &function.tests {
        let mut test_checker = globals.clone();
        test_checker.remaining = count_instruction_uses(&test.instructions);
        for field in fields {
            test_checker.define(field.clone(), None);
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
    owner: String,
    kind: LoanKind,
}

#[derive(Debug, Clone, Default)]
struct BindingState {
    moved: bool,
    loan: Option<Loan>,
}

#[derive(Debug, Clone, Default)]
struct Checker {
    bindings: HashMap<String, BindingState>,
    remaining: HashMap<String, usize>,
    temporary_loans: Vec<Loan>,
    effects: HashMap<String, Vec<ParameterEffect>>,
}

impl Checker {
    fn define(&mut self, name: String, loan: Option<Loan>) {
        self.bindings
            .insert(name, BindingState { moved: false, loan });
    }

    fn check_instructions(&mut self, instructions: &[Instruction]) -> Result<(), OwnershipError> {
        for instruction in instructions {
            self.check_instruction(instruction)?;
        }
        Ok(())
    }

    fn check_instruction(&mut self, instruction: &Instruction) -> Result<(), OwnershipError> {
        match instruction {
            Instruction::Let { name, value } | Instruction::TryLet { name, value } => {
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
        destination: &str,
        expression: &Expression,
    ) -> Result<Option<Loan>, OwnershipError> {
        let Expression::Ownership { op, value } = expression else {
            self.check_expression(expression, Access::Read)?;
            return Ok(None);
        };
        let Expression::Variable(source) = value.as_ref() else {
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
                    .get_mut(source)
                    .ok_or_else(|| unknown(source))?;
                state.moved = true;
                None
            }
        };
        self.consume(source);

        // An unused loan has no live range and therefore cannot block its owner.
        if matches!(loan, Some(_)) && self.remaining.get(destination).copied().unwrap_or(0) == 0 {
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
                if let Expression::Variable(name) = value.as_ref() {
                    self.ensure_alive(name)?;
                    let owner = self.root_owner(name);
                    self.ensure_unborrowed(&owner, "move")?;
                    self.bindings
                        .get_mut(name)
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
            Expression::Lambda { params, body } => {
                let previous = params
                    .iter()
                    .map(|param| (param.clone(), self.bindings.remove(param)))
                    .collect::<Vec<_>>();
                for param in params {
                    self.define(param.clone(), None);
                }
                self.check_expression(body, Access::Read)?;
                for (param, state) in previous {
                    if let Some(state) = state {
                        self.bindings.insert(param, state);
                    } else {
                        self.bindings.remove(&param);
                    }
                }
            }
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
            Expression::Call { function, args } => self.check_call(function, args)?,
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

    fn check_variable(&mut self, name: &str, access: Access) -> Result<(), OwnershipError> {
        // HIR may contain compiler-provided capabilities (for example
        // `stdout` and `chaos`) which are intentionally absent from the local
        // binding table. Name resolution has already validated ordinary reads;
        // only explicit ownership operations require a tracked local owner.
        if !self.bindings.contains_key(name) {
            self.consume(name);
            return Ok(());
        }
        self.ensure_alive(name)?;
        let state = self.bindings.get(name).ok_or_else(|| unknown(name))?;
        if let Some(loan) = &state.loan {
            if access == Access::Mutate && loan.kind != LoanKind::Mutable {
                return Err(ownership_error(
                    "E0302",
                    format!("shared view `{name}` cannot be mutated"),
                ));
            }
        } else {
            let owner = self.root_owner(name);
            match access {
                Access::Read if self.has_live_loan(&owner, Some(LoanKind::Mutable)) => {
                    return Err(ownership_error(
                        "E0303",
                        format!("owner `{owner}` cannot be read while an exclusive borrow is live"),
                    ));
                }
                Access::Mutate if self.has_live_loan(&owner, None) => {
                    return Err(ownership_error(
                        "E0302",
                        format!(
                            "owner `{owner}` cannot be mutated while a borrow is live. An owner cannot be structurally mutated while an immutable borrow is live."
                        ),
                    ));
                }
                _ => {}
            }
        }
        self.consume(name);
        Ok(())
    }

    fn ensure_alive(&self, name: &str) -> Result<(), OwnershipError> {
        let state = self.bindings.get(name).ok_or_else(|| unknown(name))?;
        if state.moved {
            return Err(ownership_error(
                "E0301",
                format!(
                    "binding `{name}` cannot be used after its ownership was moved. A binding cannot be read after ownership has moved to another binding."
                ),
            ));
        }
        Ok(())
    }

    fn ensure_can_borrow(&self, owner: &str, kind: LoanKind) -> Result<(), OwnershipError> {
        let conflict = match kind {
            LoanKind::Shared => self.has_live_loan(owner, Some(LoanKind::Mutable)),
            LoanKind::Mutable => self.has_live_loan(owner, None),
        };
        if conflict {
            return Err(ownership_error(
                "E0303",
                format!("conflicting borrow of `{owner}` while another borrow is live"),
            ));
        }
        Ok(())
    }

    fn ensure_unborrowed(&self, owner: &str, operation: &str) -> Result<(), OwnershipError> {
        if self.has_live_loan(owner, None) {
            return Err(ownership_error(
                "E0304",
                format!("cannot {operation} `{owner}` while it is borrowed"),
            ));
        }
        Ok(())
    }

    fn root_owner(&self, name: &str) -> String {
        self.bindings
            .get(name)
            .and_then(|state| state.loan.as_ref())
            .map_or_else(|| name.to_owned(), |loan| loan.owner.clone())
    }

    fn has_live_loan(&self, owner: &str, kind: Option<LoanKind>) -> bool {
        self.temporary_loans
            .iter()
            .any(|loan| loan.owner == owner && kind.is_none_or(|kind| loan.kind == kind))
            || self.bindings.iter().any(|(name, state)| {
                state.loan.as_ref().is_some_and(|loan| {
                    loan.owner == owner
                        && kind.is_none_or(|kind| loan.kind == kind)
                        && self.remaining.get(name).copied().unwrap_or(0) > 0
                })
            })
    }

    fn consume(&mut self, name: &str) {
        if let Some(remaining) = self.remaining.get_mut(name) {
            *remaining = remaining.saturating_sub(1);
        }
    }

    fn check_call(&mut self, function: &str, args: &[Expression]) -> Result<(), OwnershipError> {
        let effects = self.effects.get(function).cloned().unwrap_or_default();
        let loan_boundary = self.temporary_loans.len();

        for (index, argument) in args.iter().enumerate() {
            let effect = effects.get(index).copied().unwrap_or(ParameterEffect::View);
            self.check_call_argument(function, index, argument, effect)?;
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
        let (explicit, value) = match argument {
            Expression::Ownership { op, value } => (Some(*op), value.as_ref()),
            value => (None, value),
        };
        let Expression::Variable(source) = value else {
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
                "E0306",
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
                    .get_mut(source)
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
        let borrowed = match expression {
            Expression::Variable(name) => self
                .bindings
                .get(name)
                .and_then(|state| state.loan.as_ref())
                .map(|_| name.as_str()),
            Expression::Ownership {
                op: OwnershipOp::View | OwnershipOp::Borrow | OwnershipOp::AddressOf,
                value,
            } => match value.as_ref() {
                Expression::Variable(name) => Some(name.as_str()),
                _ => None,
            },
            _ => None,
        };
        if let Some(name) = borrowed {
            return Err(ownership_error(
                "E0305",
                format!("borrowed value `{name}` cannot escape through a return"),
            ));
        }
        Ok(())
    }
}

fn effect_name(effect: ParameterEffect) -> &'static str {
    match effect {
        ParameterEffect::View => "shared view",
        ParameterEffect::Borrow => "exclusive borrow",
        ParameterEffect::Move => "ownership transfer",
    }
}

fn infer_function_effects(program: &Program) -> HashMap<String, Vec<ParameterEffect>> {
    let mut effects = HashMap::new();
    for function in &program.functions {
        effects.insert(function.name.clone(), infer_parameter_effects(function));
    }
    for class in &program.classes {
        for function in class.methods.iter().chain(&class.constructors) {
            effects
                .entry(function.name.clone())
                .or_insert_with(|| infer_parameter_effects(function));
        }
    }
    effects
}

fn infer_parameter_effects(function: &Function) -> Vec<ParameterEffect> {
    let parameters = function
        .params
        .iter()
        .enumerate()
        .map(|(index, parameter)| (parameter.name.as_str(), index))
        .collect::<HashMap<_, _>>();
    let mut effects = vec![ParameterEffect::View; function.params.len()];
    infer_instruction_effects(&function.instructions, &parameters, &mut effects);
    effects
}

fn infer_instruction_effects(
    instructions: &[Instruction],
    parameters: &HashMap<&str, usize>,
    effects: &mut [ParameterEffect],
) {
    for instruction in instructions {
        match instruction {
            Instruction::Let { value, .. }
            | Instruction::TryLet { value, .. }
            | Instruction::Print(value)
            | Instruction::Assert(value)
            | Instruction::Evaluate(value) => {
                infer_expression_effect(value, Access::Read, parameters, effects)
            }
            Instruction::Assign { target, value, .. } => {
                infer_expression_effect(value, Access::Read, parameters, effects);
                infer_expression_effect(target, Access::Mutate, parameters, effects);
            }
            Instruction::Return(value) => {
                if let Some(value) = value {
                    infer_expression_effect(value, Access::Read, parameters, effects);
                }
            }
            Instruction::If {
                condition,
                then_instructions,
                else_instructions,
            } => {
                infer_expression_effect(condition, Access::Read, parameters, effects);
                infer_instruction_effects(then_instructions, parameters, effects);
                infer_instruction_effects(else_instructions, parameters, effects);
            }
            Instruction::While {
                setup,
                capabilities,
                condition,
                instructions,
            } => {
                if let Some(setup) = setup {
                    infer_instruction_effects(std::slice::from_ref(setup), parameters, effects);
                }
                for capability in capabilities {
                    infer_expression_effect(capability, Access::Read, parameters, effects);
                }
                infer_expression_effect(condition, Access::Read, parameters, effects);
                infer_instruction_effects(instructions, parameters, effects);
            }
            Instruction::For {
                setup,
                iterable,
                instructions,
                ..
            } => {
                if let Some(setup) = setup {
                    infer_instruction_effects(std::slice::from_ref(setup), parameters, effects);
                }
                infer_expression_effect(iterable, Access::Read, parameters, effects);
                infer_instruction_effects(instructions, parameters, effects);
            }
            Instruction::Switch { value, arms } => {
                infer_expression_effect(value, Access::Read, parameters, effects);
                infer_arm_effects(arms, parameters, effects);
            }
            Instruction::ChannelSwitch {
                channels,
                setup,
                repeat_condition,
                arms,
            } => {
                for channel in channels {
                    infer_expression_effect(channel, Access::Read, parameters, effects);
                }
                if let Some(setup) = setup {
                    infer_instruction_effects(std::slice::from_ref(setup), parameters, effects);
                }
                if let Some(condition) = repeat_condition {
                    infer_expression_effect(condition, Access::Read, parameters, effects);
                }
                infer_arm_effects(arms, parameters, effects);
            }
            Instruction::With {
                resources,
                instructions,
                ..
            } => {
                for resource in resources {
                    infer_expression_effect(resource, Access::Read, parameters, effects);
                }
                infer_instruction_effects(instructions, parameters, effects);
            }
            Instruction::Break | Instruction::Continue => {}
        }
    }
}

fn infer_arm_effects(
    arms: &[SwitchArm],
    parameters: &HashMap<&str, usize>,
    effects: &mut [ParameterEffect],
) {
    for arm in arms {
        if let Some(source) = &arm.source {
            infer_expression_effect(source, Access::Read, parameters, effects);
        }
        if let Some(guard) = &arm.guard {
            infer_expression_effect(guard, Access::Read, parameters, effects);
        }
        infer_instruction_effects(&arm.instructions, parameters, effects);
    }
}

fn infer_expression_effect(
    expression: &Expression,
    access: Access,
    parameters: &HashMap<&str, usize>,
    effects: &mut [ParameterEffect],
) {
    match expression {
        Expression::Variable(name) => mark_parameter_effect(
            name,
            if access == Access::Mutate {
                ParameterEffect::Borrow
            } else {
                ParameterEffect::View
            },
            parameters,
            effects,
        ),
        Expression::Ownership { op, value } => {
            let effect = match op {
                OwnershipOp::Move => ParameterEffect::Move,
                OwnershipOp::Borrow => ParameterEffect::Borrow,
                OwnershipOp::View | OwnershipOp::Clone | OwnershipOp::AddressOf => {
                    ParameterEffect::View
                }
            };
            if let Expression::Variable(name) = value.as_ref() {
                mark_parameter_effect(name, effect, parameters, effects);
            } else {
                infer_expression_effect(value, Access::Read, parameters, effects);
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
        } => infer_expression_effect(object, access, parameters, effects),
        Expression::Lambda { body, .. } => {
            infer_expression_effect(body, Access::Read, parameters, effects)
        }
        Expression::List(values)
        | Expression::Tuple(values)
        | Expression::Set(values)
        | Expression::PrintArgs(values)
        | Expression::Construct { args: values, .. }
        | Expression::Variant { fields: values, .. } => {
            for value in values {
                infer_expression_effect(value, Access::Read, parameters, effects);
            }
        }
        Expression::Map(entries) => {
            for (key, value) in entries {
                infer_expression_effect(key, Access::Read, parameters, effects);
                infer_expression_effect(value, Access::Read, parameters, effects);
            }
        }
        Expression::Index { object, index } => {
            infer_expression_effect(object, access, parameters, effects);
            infer_expression_effect(index, Access::Read, parameters, effects);
        }
        Expression::Slice {
            object,
            start,
            end,
            step,
        } => {
            infer_expression_effect(object, access, parameters, effects);
            for bound in [start, end, step].into_iter().flatten() {
                infer_expression_effect(bound, Access::Read, parameters, effects);
            }
        }
        Expression::MethodCall {
            object,
            method,
            args,
        } => {
            let receiver_access = if mutating_method(method) {
                Access::Mutate
            } else {
                Access::Read
            };
            infer_expression_effect(object, receiver_access, parameters, effects);
            for arg in args {
                infer_expression_effect(arg, Access::Read, parameters, effects);
            }
        }
        Expression::Send { value, channel } => {
            infer_expression_effect(value, Access::Read, parameters, effects);
            infer_expression_effect(channel, Access::Mutate, parameters, effects);
        }
        Expression::ListComprehension { element, clauses } => {
            infer_expression_effect(element, Access::Read, parameters, effects);
            for clause in clauses {
                infer_expression_effect(&clause.iterable, Access::Read, parameters, effects);
                if let Some(condition) = &clause.condition {
                    infer_expression_effect(condition, Access::Read, parameters, effects);
                }
            }
        }
        Expression::SetComprehension { element, clauses } => {
            infer_expression_effect(element, Access::Read, parameters, effects);
            for clause in clauses {
                infer_expression_effect(&clause.iterable, Access::Read, parameters, effects);
                if let Some(condition) = &clause.condition {
                    infer_expression_effect(condition, Access::Read, parameters, effects);
                }
            }
        }
        Expression::MapComprehension {
            key,
            value,
            clauses,
        } => {
            infer_expression_effect(key, Access::Read, parameters, effects);
            infer_expression_effect(value, Access::Read, parameters, effects);
            for clause in clauses {
                infer_expression_effect(&clause.iterable, Access::Read, parameters, effects);
                if let Some(condition) = &clause.condition {
                    infer_expression_effect(condition, Access::Read, parameters, effects);
                }
            }
        }
        Expression::Conditional {
            condition,
            then_expression,
            else_expression,
        } => {
            infer_expression_effect(condition, Access::Read, parameters, effects);
            infer_expression_effect(then_expression, access, parameters, effects);
            infer_expression_effect(else_expression, access, parameters, effects);
        }
        Expression::Binary { left, right, .. } => {
            infer_expression_effect(left, Access::Read, parameters, effects);
            infer_expression_effect(right, Access::Read, parameters, effects);
        }
        Expression::Format { args, .. } | Expression::Call { args, .. } => {
            for arg in args {
                infer_expression_effect(arg, Access::Read, parameters, effects);
            }
        }
        Expression::CallValue { callee, args, .. } => {
            infer_expression_effect(callee, Access::Read, parameters, effects);
            for arg in args {
                infer_expression_effect(arg, Access::Read, parameters, effects);
            }
        }
        Expression::Integer(_)
        | Expression::Float(_)
        | Expression::Boolean(_)
        | Expression::String(_)
        | Expression::Function(_) => {}
    }
}

fn mark_parameter_effect(
    name: &str,
    effect: ParameterEffect,
    parameters: &HashMap<&str, usize>,
    effects: &mut [ParameterEffect],
) {
    if let Some(index) = parameters.get(name) {
        effects[*index] = effects[*index].max(effect);
    }
}

fn define_pattern(checker: &mut Checker, pattern: &MatchPattern) {
    match pattern {
        MatchPattern::Bind(name) => checker.define(name.clone(), None),
        MatchPattern::Constructor { fields, .. } => {
            for field in fields {
                define_pattern(checker, field);
            }
        }
        _ => {}
    }
}

fn mutating_method(method: &str) -> bool {
    matches!(
        method,
        "append"
            | "appendleft"
            | "extend"
            | "push"
            | "pop"
            | "popleft"
            | "remove"
            | "clear"
            | "insert"
            | "sort"
            | "reverse"
            | "heapPush"
            | "heapPop"
            | "setDefault"
    )
}

fn count_instruction_uses(instructions: &[Instruction]) -> HashMap<String, usize> {
    let mut counts = HashMap::new();
    count_instructions(instructions, &mut counts);
    counts
}

fn count_instructions(instructions: &[Instruction], counts: &mut HashMap<String, usize>) {
    for instruction in instructions {
        match instruction {
            Instruction::Let { value, .. }
            | Instruction::TryLet { value, .. }
            | Instruction::Print(value)
            | Instruction::Assert(value)
            | Instruction::Evaluate(value) => count_expression(value, counts),
            Instruction::Assign { target, value, .. } => {
                count_expression(target, counts);
                count_expression(value, counts);
            }
            Instruction::Return(value) => {
                if let Some(value) = value {
                    count_expression(value, counts);
                }
            }
            Instruction::If {
                condition,
                then_instructions,
                else_instructions,
            } => {
                count_expression(condition, counts);
                count_instructions(then_instructions, counts);
                count_instructions(else_instructions, counts);
            }
            Instruction::While {
                setup,
                capabilities,
                condition,
                instructions,
            } => {
                if let Some(setup) = setup {
                    count_instructions(std::slice::from_ref(setup), counts);
                }
                for capability in capabilities {
                    count_expression(capability, counts);
                }
                count_expression(condition, counts);
                count_instructions(instructions, counts);
            }
            Instruction::For {
                setup,
                iterable,
                instructions,
                ..
            } => {
                if let Some(setup) = setup {
                    count_instructions(std::slice::from_ref(setup), counts);
                }
                count_expression(iterable, counts);
                count_instructions(instructions, counts);
            }
            Instruction::Switch { value, arms } => {
                count_expression(value, counts);
                count_arms(arms, counts);
            }
            Instruction::ChannelSwitch {
                channels,
                setup,
                repeat_condition,
                arms,
            } => {
                for channel in channels {
                    count_expression(channel, counts);
                }
                if let Some(setup) = setup {
                    count_instructions(std::slice::from_ref(setup), counts);
                }
                if let Some(condition) = repeat_condition {
                    count_expression(condition, counts);
                }
                count_arms(arms, counts);
            }
            Instruction::With {
                resources,
                instructions,
                ..
            } => {
                for resource in resources {
                    count_expression(resource, counts);
                }
                count_instructions(instructions, counts);
            }
            Instruction::Break | Instruction::Continue => {}
        }
    }
}

fn count_arms(arms: &[SwitchArm], counts: &mut HashMap<String, usize>) {
    for arm in arms {
        if let Some(source) = &arm.source {
            count_expression(source, counts);
        }
        if let Some(guard) = &arm.guard {
            count_expression(guard, counts);
        }
        count_instructions(&arm.instructions, counts);
    }
}

fn count_expression(expression: &Expression, counts: &mut HashMap<String, usize>) {
    match expression {
        Expression::Variable(name) => *counts.entry(name.clone()).or_default() += 1,
        Expression::Ownership { value, .. }
        | Expression::Member { object: value, .. }
        | Expression::Await(value)
        | Expression::Channel(value)
        | Expression::Task { value, .. }
        | Expression::ChaosRule { value, .. }
        | Expression::FusedPipeline { input: value, .. }
        | Expression::Unary {
            expression: value, ..
        } => count_expression(value, counts),
        Expression::Lambda { body, .. } => count_expression(body, counts),
        Expression::List(values)
        | Expression::Tuple(values)
        | Expression::Set(values)
        | Expression::PrintArgs(values)
        | Expression::Construct { args: values, .. }
        | Expression::Variant { fields: values, .. } => {
            for value in values {
                count_expression(value, counts);
            }
        }
        Expression::Map(entries) => {
            for (key, value) in entries {
                count_expression(key, counts);
                count_expression(value, counts);
            }
        }
        Expression::Index { object, index }
        | Expression::Binary {
            left: object,
            right: index,
            ..
        } => {
            count_expression(object, counts);
            count_expression(index, counts);
        }
        Expression::Slice {
            object,
            start,
            end,
            step,
        } => {
            count_expression(object, counts);
            for bound in [start, end, step].into_iter().flatten() {
                count_expression(bound, counts);
            }
        }
        Expression::Format { args, .. } | Expression::Call { args, .. } => {
            for arg in args {
                count_expression(arg, counts);
            }
        }
        Expression::MethodCall { object, args, .. } => {
            count_expression(object, counts);
            for arg in args {
                count_expression(arg, counts);
            }
        }
        Expression::Send { value, channel } => {
            count_expression(value, counts);
            count_expression(channel, counts);
        }
        Expression::ListComprehension { element, clauses } => {
            count_expression(element, counts);
            for clause in clauses {
                count_expression(&clause.iterable, counts);
                if let Some(condition) = &clause.condition {
                    count_expression(condition, counts);
                }
            }
        }
        Expression::SetComprehension { element, clauses } => {
            count_expression(element, counts);
            for clause in clauses {
                count_expression(&clause.iterable, counts);
                if let Some(condition) = &clause.condition {
                    count_expression(condition, counts);
                }
            }
        }
        Expression::MapComprehension {
            key,
            value,
            clauses,
        } => {
            count_expression(key, counts);
            count_expression(value, counts);
            for clause in clauses {
                count_expression(&clause.iterable, counts);
                if let Some(condition) = &clause.condition {
                    count_expression(condition, counts);
                }
            }
        }
        Expression::Conditional {
            condition,
            then_expression,
            else_expression,
        } => {
            count_expression(condition, counts);
            count_expression(then_expression, counts);
            count_expression(else_expression, counts);
        }
        Expression::CallValue { callee, args, .. } => {
            count_expression(callee, counts);
            for arg in args {
                count_expression(arg, counts);
            }
        }
        Expression::Integer(_)
        | Expression::Float(_)
        | Expression::Boolean(_)
        | Expression::String(_)
        | Expression::Function(_) => {}
    }
}

fn ownership_error(code: &str, message: String) -> OwnershipError {
    OwnershipError {
        message: format!("{code}: {message}"),
    }
}

fn unknown(name: &str) -> OwnershipError {
    ownership_error(
        "E0300",
        format!("ownership operation references unknown binding `{name}`"),
    )
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OwnershipError {
    pub message: String,
}

impl std::fmt::Display for OwnershipError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(&self.message)
    }
}

impl std::error::Error for OwnershipError {}
