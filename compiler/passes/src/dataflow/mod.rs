use crate::{Pass, PassError};
use severian_hir::{BindingId, Expression, Function, Instruction, MatchPattern, Program};
use std::collections::{HashMap, HashSet};

/// Straight-line local data-flow simplification.
///
/// Tracks immutable literal/copy bindings within a basic structured block,
/// substitutes known values, and invalidates facts conservatively across
/// assignments and control-flow joins.
#[derive(Debug, Default, Clone, Copy)]
pub struct LocalDataflow;

impl Pass for LocalDataflow {
    fn name(&self) -> &'static str {
        "local-dataflow"
    }

    fn run(&self, program: &mut Program) -> Result<(), PassError> {
        for function in &mut program.functions {
            optimize_function(function);
        }

        for class in &mut program.classes {
            for function in class.methods.iter_mut().chain(&mut class.constructors) {
                optimize_function(function);
            }
        }

        Ok(())
    }
}

#[derive(Debug, Clone, Default)]
struct Facts {
    values: HashMap<BindingId, Expression>,
}

impl Facts {
    fn invalidate(&mut self, binding: BindingId) {
        self.values.remove(&binding);

        // A copy fact whose value references `name` is now stale.
        self.values.retain(
            |_, value| !matches!(value, Expression::Variable(source) if source.id == binding),
        );
    }

    fn bind(&mut self, binding: BindingId, value: &Expression) {
        self.invalidate(binding);

        if is_propagatable(value) {
            self.values.insert(binding, value.clone());
        }
    }

    fn clear(&mut self) {
        self.values.clear();
    }
}

pub fn optimize_function(function: &mut Function) {
    let mut facts = Facts::default();

    for parameter in &function.params {
        facts.invalidate(parameter.name.id);
    }

    optimize_block(&mut function.instructions, &mut facts);

    for test in &mut function.tests {
        optimize_block(&mut test.instructions, &mut Facts::default());
    }
}

fn optimize_block(instructions: &mut [Instruction], facts: &mut Facts) {
    for instruction in instructions {
        match instruction {
            Instruction::Let { name, value } | Instruction::TryLet { name, value, .. } => {
                substitute_expression(value, facts, &mut HashSet::new());
                facts.bind(name.id, value);
            }
            Instruction::Assign { target, value, .. } => {
                substitute_expression(value, facts, &mut HashSet::new());

                if let Expression::Variable(name) = target.kind() {
                    facts.invalidate(name.id);
                } else {
                    substitute_expression(target, facts, &mut HashSet::new());
                    facts.clear();
                }
            }
            Instruction::Print(value)
            | Instruction::Assert(value)
            | Instruction::Evaluate(value) => {
                substitute_expression(value, facts, &mut HashSet::new());
            }
            Instruction::Return(value) => {
                if let Some(value) = value {
                    substitute_expression(value, facts, &mut HashSet::new());
                }
            }
            Instruction::If {
                condition,
                then_instructions,
                else_instructions,
            } => {
                substitute_expression(condition, facts, &mut HashSet::new());

                let mut then_facts = facts.clone();
                optimize_block(then_instructions, &mut then_facts);

                let mut else_facts = facts.clone();
                optimize_block(else_instructions, &mut else_facts);

                *facts = intersect_facts(&then_facts, &else_facts);
            }
            Instruction::While {
                setup,
                capabilities,
                condition,
                instructions,
            } => {
                if let Some(setup) = setup {
                    optimize_instruction(setup, facts);
                }
                for capability in capabilities {
                    substitute_expression(capability, facts, &mut HashSet::new());
                }
                substitute_expression(condition, facts, &mut HashSet::new());

                let mut loop_facts = facts.clone();
                optimize_block(instructions, &mut loop_facts);

                let mutated = assigned_names(instructions);
                for name in mutated {
                    facts.invalidate(name);
                }
            }
            Instruction::For {
                setup,
                pattern,
                iterable,
                instructions,
            } => {
                if let Some(setup) = setup {
                    optimize_instruction(setup, facts);
                }
                substitute_expression(iterable, facts, &mut HashSet::new());

                let mut loop_facts = facts.clone();
                for name in pattern_names(pattern) {
                    loop_facts.invalidate(name);
                }
                optimize_block(instructions, &mut loop_facts);

                for name in assigned_names(instructions) {
                    facts.invalidate(name);
                }
                for name in pattern_names(pattern) {
                    facts.invalidate(name);
                }
            }
            Instruction::Switch { value, arms } => {
                substitute_expression(value, facts, &mut HashSet::new());
                optimize_arms(arms, facts);
            }
            Instruction::ChannelSwitch {
                channels,
                setup,
                repeat_condition,
                arms,
            } => {
                for channel in channels {
                    substitute_expression(channel, facts, &mut HashSet::new());
                }
                if let Some(setup) = setup {
                    optimize_instruction(setup, facts);
                }
                if let Some(condition) = repeat_condition {
                    substitute_expression(condition, facts, &mut HashSet::new());
                }
                optimize_arms(arms, facts);
            }
            Instruction::With {
                resources,
                instructions,
                ..
            } => {
                for resource in resources {
                    substitute_expression(resource, facts, &mut HashSet::new());
                }

                let mut nested = facts.clone();
                optimize_block(instructions, &mut nested);

                for name in assigned_names(instructions) {
                    facts.invalidate(name);
                }
            }
            Instruction::Break | Instruction::Continue => {}
        }
    }
}

fn optimize_instruction(instruction: &mut Instruction, facts: &mut Facts) {
    optimize_block(std::slice::from_mut(instruction), facts);
}

fn optimize_arms(arms: &mut [severian_hir::SwitchArm], facts: &mut Facts) {
    if arms.is_empty() {
        return;
    }

    let incoming = facts.clone();
    let mut outgoing = Vec::with_capacity(arms.len());

    for arm in arms {
        let mut branch = incoming.clone();

        if let Some(source) = &mut arm.source {
            substitute_expression(source, &branch, &mut HashSet::new());
        }

        for name in pattern_names(&arm.pattern) {
            branch.invalidate(name);
        }

        if let Some(guard) = &mut arm.guard {
            substitute_expression(guard, &branch, &mut HashSet::new());
        }

        optimize_block(&mut arm.instructions, &mut branch);
        outgoing.push(branch);
    }

    let mut merged = outgoing.remove(0);
    for branch in outgoing {
        merged = intersect_facts(&merged, &branch);
    }
    *facts = merged;
}

fn intersect_facts(left: &Facts, right: &Facts) -> Facts {
    let mut values = HashMap::new();

    for (name, value) in &left.values {
        if right.values.get(name) == Some(value) {
            values.insert(name.clone(), value.clone());
        }
    }

    Facts { values }
}

fn substitute_expression(
    expression: &mut Expression,
    facts: &Facts,
    visiting: &mut HashSet<BindingId>,
) {
    match expression {
        Expression::Typed { expression, .. } => substitute_expression(expression, facts, visiting),
        Expression::Variable(name) => {
            if !visiting.insert(name.id) {
                return;
            }

            if let Some(replacement) = facts.values.get(&name.id).cloned() {
                let original = name.id;
                *expression = replacement;
                substitute_expression(expression, facts, visiting);
                visiting.remove(&original);
            } else {
                visiting.remove(&name.id);
            }
        }
        Expression::Lambda { params, body } => {
            let mut local = facts.clone();
            for param in params {
                local.invalidate(param.id);
            }
            substitute_expression(body, &local, visiting);
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
        } => substitute_expression(value, facts, visiting),
        Expression::List(values)
        | Expression::Tuple(values)
        | Expression::Set(values)
        | Expression::PrintArgs(values)
        | Expression::Construct { args: values, .. }
        | Expression::Variant { fields: values, .. } => {
            for value in values {
                substitute_expression(value, facts, visiting);
            }
        }
        Expression::ConstructFields { fields, .. } => {
            for (_, value) in fields {
                substitute_expression(value, facts, visiting);
            }
        }
        Expression::ObjectUpdate { object, fields, .. } => {
            substitute_expression(object, facts, visiting);
            for (_, value) in fields {
                substitute_expression(value, facts, visiting);
            }
        }
        Expression::ObjectDocument { object, .. } => {
            substitute_expression(object, facts, visiting);
        }
        Expression::Map(entries) => {
            for (key, value) in entries {
                substitute_expression(key, facts, visiting);
                substitute_expression(value, facts, visiting);
            }
        }
        Expression::Index { object, index } => {
            substitute_expression(object, facts, visiting);
            substitute_expression(index, facts, visiting);
        }
        Expression::Slice {
            object,
            start,
            end,
            step,
        } => {
            substitute_expression(object, facts, visiting);
            for bound in [start, end, step].into_iter().flatten() {
                substitute_expression(bound, facts, visiting);
            }
        }
        Expression::Format { args, .. } | Expression::Call { args, .. } => {
            for arg in args {
                substitute_expression(arg, facts, visiting);
            }
        }
        Expression::MethodCall { object, args, .. } => {
            substitute_expression(object, facts, visiting);
            for arg in args {
                substitute_expression(arg, facts, visiting);
            }
        }
        Expression::Send { value, channel } => {
            substitute_expression(value, facts, visiting);
            substitute_expression(channel, facts, visiting);
        }
        Expression::ListComprehension { element, clauses }
        | Expression::SetComprehension { element, clauses } => {
            let mut local = facts.clone();
            for clause in clauses {
                substitute_expression(&mut clause.iterable, &local, visiting);
                for name in pattern_names(&clause.pattern) {
                    local.invalidate(name);
                }
                if let Some(condition) = &mut clause.condition {
                    substitute_expression(condition, &local, visiting);
                }
            }
            substitute_expression(element, &local, visiting);
        }
        Expression::MapComprehension {
            key,
            value,
            clauses,
        } => {
            let mut local = facts.clone();
            for clause in clauses {
                substitute_expression(&mut clause.iterable, &local, visiting);
                for name in pattern_names(&clause.pattern) {
                    local.invalidate(name);
                }
                if let Some(condition) = &mut clause.condition {
                    substitute_expression(condition, &local, visiting);
                }
            }
            substitute_expression(key, &local, visiting);
            substitute_expression(value, &local, visiting);
        }
        Expression::Conditional {
            condition,
            then_expression,
            else_expression,
        } => {
            substitute_expression(condition, facts, visiting);
            substitute_expression(then_expression, facts, visiting);
            substitute_expression(else_expression, facts, visiting);
        }
        Expression::Binary { left, right, .. } => {
            substitute_expression(left, facts, visiting);
            substitute_expression(right, facts, visiting);
        }
        Expression::CallValue { callee, args, .. } => {
            substitute_expression(callee, facts, visiting);
            for arg in args {
                substitute_expression(arg, facts, visiting);
            }
        }
        Expression::Integer(_)
        | Expression::Float(_)
        | Expression::Boolean(_)
        | Expression::String(_)
        | Expression::Function(_) => {}
    }
}

fn is_propagatable(expression: &Expression) -> bool {
    matches!(
        expression,
        Expression::Integer(_)
            | Expression::Float(_)
            | Expression::Boolean(_)
            | Expression::String(_)
            | Expression::Function(_)
            | Expression::Variable(_)
    )
}

fn pattern_names(pattern: &MatchPattern) -> Vec<BindingId> {
    let mut names = Vec::new();
    collect_pattern_names(pattern, &mut names);
    names
}

fn collect_pattern_names(pattern: &MatchPattern, names: &mut Vec<BindingId>) {
    match pattern {
        MatchPattern::Bind(name) => names.push(name.id),
        MatchPattern::Constructor { fields, .. } => {
            for field in fields {
                collect_pattern_names(field, names);
            }
        }
        MatchPattern::Wildcard
        | MatchPattern::Integer(_)
        | MatchPattern::Float(_)
        | MatchPattern::Boolean(_)
        | MatchPattern::String(_) => {}
    }
}

fn assigned_names(instructions: &[Instruction]) -> HashSet<BindingId> {
    let mut names = HashSet::new();

    for instruction in instructions {
        match instruction {
            Instruction::Let { name, .. } | Instruction::TryLet { name, .. } => {
                names.insert(name.id);
            }
            Instruction::Assign {
                target: Expression::Variable(name),
                ..
            } => {
                names.insert(name.id);
            }
            Instruction::If {
                then_instructions,
                else_instructions,
                ..
            } => {
                names.extend(assigned_names(then_instructions));
                names.extend(assigned_names(else_instructions));
            }
            Instruction::While { instructions, .. }
            | Instruction::For { instructions, .. }
            | Instruction::With { instructions, .. } => {
                names.extend(assigned_names(instructions));
            }
            Instruction::Switch { arms, .. } | Instruction::ChannelSwitch { arms, .. } => {
                for arm in arms {
                    names.extend(assigned_names(&arm.instructions));
                }
            }
            _ => {}
        }
    }

    names
}
