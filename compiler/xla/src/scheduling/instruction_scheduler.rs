use severian_hir::{Expression, Function, Instruction};
use std::collections::{BTreeSet, HashMap, HashSet, VecDeque};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ScheduleEntry {
    pub original_index: usize,
    pub dependencies: Vec<usize>,
    pub estimated_cost: u32,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Schedule {
    pub order: Vec<usize>,
    pub entries: Vec<ScheduleEntry>,
}

#[derive(Debug, Clone, Copy)]
pub struct InstructionScheduler {
    /// Prefer higher-cost ready instructions first to expose independent work.
    pub cost_priority: bool,
}

impl Default for InstructionScheduler {
    fn default() -> Self {
        Self { cost_priority: true }
    }
}

impl InstructionScheduler {
    pub fn schedule(&self, function: &Function) -> Schedule {
        schedule_with_policy(function, self.cost_priority)
    }
}

pub fn schedule_function(function: &Function) -> Schedule {
    InstructionScheduler::default().schedule(function)
}

fn schedule_with_policy(function: &Function, cost_priority: bool) -> Schedule {
    let instructions = &function.instructions;
    let mut definitions = HashMap::<String, usize>::new();

    for (index, instruction) in instructions.iter().enumerate() {
        if let Some(name) = defined_binding(instruction) {
            definitions.insert(name.to_string(), index);
        }
    }

    let mut entries = Vec::with_capacity(instructions.len());

    for (index, instruction) in instructions.iter().enumerate() {
        let mut dependencies = BTreeSet::new();

        collect_instruction_uses(instruction, &mut |name| {
            if let Some(&producer) = definitions.get(name) {
                if producer < index {
                    dependencies.insert(producer);
                }
            }
        });

        // Preserve ordering around side-effecting/control-flow instructions.
        if index > 0 && is_barrier(instruction) {
            dependencies.insert(index - 1);
        }

        entries.push(ScheduleEntry {
            original_index: index,
            dependencies: dependencies.into_iter().collect(),
            estimated_cost: estimate_cost(instruction),
        });
    }

    let order = topological_order(&entries, cost_priority);

    Schedule { order, entries }
}

fn topological_order(entries: &[ScheduleEntry], cost_priority: bool) -> Vec<usize> {
    let mut indegree = vec![0usize; entries.len()];
    let mut users = vec![Vec::<usize>::new(); entries.len()];

    for entry in entries {
        indegree[entry.original_index] = entry.dependencies.len();
        for &dependency in &entry.dependencies {
            if dependency < users.len() {
                users[dependency].push(entry.original_index);
            }
        }
    }

    let mut ready = VecDeque::new();
    for (index, &degree) in indegree.iter().enumerate() {
        if degree == 0 {
            ready.push_back(index);
        }
    }

    let mut order = Vec::with_capacity(entries.len());

    while !ready.is_empty() {
        let next = if cost_priority {
            let best_pos = ready
                .iter()
                .enumerate()
                .max_by_key(|(_, index)| entries[**index].estimated_cost)
                .map(|(position, _)| position)
                .unwrap_or(0);
            ready.remove(best_pos).unwrap()
        } else {
            ready.pop_front().unwrap()
        };

        order.push(next);

        for &user in &users[next] {
            indegree[user] = indegree[user].saturating_sub(1);
            if indegree[user] == 0 {
                ready.push_back(user);
            }
        }
    }

    if order.len() != entries.len() {
        // Conservative fallback for an unexpected dependency cycle.
        return (0..entries.len()).collect();
    }

    order
}

fn defined_binding(instruction: &Instruction) -> Option<&str> {
    match instruction {
        Instruction::Let { name, .. } | Instruction::TryLet { name, .. } => Some(name),
        _ => None,
    }
}

fn estimate_cost(instruction: &Instruction) -> u32 {
    match instruction {
        Instruction::Let { value, .. }
        | Instruction::TryLet { value, .. }
        | Instruction::Evaluate(value) => expression_cost(value),

        Instruction::Assign { value, .. } => 1 + expression_cost(value),

        Instruction::Print(_)
        | Instruction::Assert(_)
        | Instruction::Return(_)
        | Instruction::Break
        | Instruction::Continue => 1,

        Instruction::If { .. }
        | Instruction::While { .. }
        | Instruction::For { .. }
        | Instruction::Switch { .. }
        | Instruction::ChannelSwitch { .. }
        | Instruction::With { .. } => 16,
    }
}

fn expression_cost(expression: &Expression) -> u32 {
    match expression {
        Expression::Call { .. } | Expression::CallValue { .. } | Expression::MethodCall { .. } => 8,
        Expression::FusedPipeline { operations, .. } => 2 + operations.len() as u32,
        Expression::Binary { left, right, .. } => {
            1 + expression_cost(left) + expression_cost(right)
        }
        Expression::Unary { expression, .. } => 1 + expression_cost(expression),
        Expression::Task { .. }
        | Expression::Await(_)
        | Expression::Send { .. }
        | Expression::Channel(_) => 8,
        _ => 1,
    }
}

fn is_barrier(instruction: &Instruction) -> bool {
    matches!(
        instruction,
        Instruction::Assign { .. }
            | Instruction::Print(_)
            | Instruction::Assert(_)
            | Instruction::Return(_)
            | Instruction::If { .. }
            | Instruction::While { .. }
            | Instruction::For { .. }
            | Instruction::Switch { .. }
            | Instruction::ChannelSwitch { .. }
            | Instruction::With { .. }
            | Instruction::Break
            | Instruction::Continue
    )
}

fn collect_instruction_uses(instruction: &Instruction, visitor: &mut impl FnMut(&str)) {
    match instruction {
        Instruction::Let { value, .. }
        | Instruction::TryLet { value, .. }
        | Instruction::Print(value)
        | Instruction::Assert(value)
        | Instruction::Evaluate(value) => collect_expression_uses(value, visitor),

        Instruction::Assign { target, value, .. } => {
            collect_expression_uses(target, visitor);
            collect_expression_uses(value, visitor);
        }

        Instruction::Return(value) => {
            if let Some(value) = value {
                collect_expression_uses(value, visitor);
            }
        }

        Instruction::If { condition, .. } => collect_expression_uses(condition, visitor),

        Instruction::While {
            capabilities,
            condition,
            ..
        } => {
            for capability in capabilities {
                collect_expression_uses(capability, visitor);
            }
            collect_expression_uses(condition, visitor);
        }

        Instruction::For { iterable, .. } => collect_expression_uses(iterable, visitor),

        Instruction::Switch { value, .. } => collect_expression_uses(value, visitor),

        Instruction::ChannelSwitch {
            channels,
            repeat_condition,
            ..
        } => {
            for channel in channels {
                collect_expression_uses(channel, visitor);
            }
            if let Some(condition) = repeat_condition {
                collect_expression_uses(condition, visitor);
            }
        }

        Instruction::With { resources, .. } => {
            for resource in resources {
                collect_expression_uses(resource, visitor);
            }
        }

        Instruction::Break | Instruction::Continue => {}
    }
}

fn collect_expression_uses(expression: &Expression, visitor: &mut impl FnMut(&str)) {
    match expression {
        Expression::Variable(name) => visitor(name),

        Expression::Unary { expression, .. }
        | Expression::Ownership {
            value: expression, ..
        }
        | Expression::Member {
            object: expression, ..
        }
        | Expression::Await(expression)
        | Expression::Channel(expression)
        | Expression::Task {
            value: expression, ..
        }
        | Expression::ChaosRule {
            value: expression, ..
        }
        | Expression::FusedPipeline {
            input: expression, ..
        } => collect_expression_uses(expression, visitor),

        Expression::Binary { left, right, .. } => {
            collect_expression_uses(left, visitor);
            collect_expression_uses(right, visitor);
        }

        Expression::List(values)
        | Expression::Tuple(values)
        | Expression::Set(values)
        | Expression::PrintArgs(values)
        | Expression::Construct { args: values, .. }
        | Expression::Variant { fields: values, .. }
        | Expression::Format { args: values, .. }
        | Expression::Call { args: values, .. } => {
            for value in values {
                collect_expression_uses(value, visitor);
            }
        }

        Expression::Map(entries) => {
            for (key, value) in entries {
                collect_expression_uses(key, visitor);
                collect_expression_uses(value, visitor);
            }
        }

        Expression::Index { object, index } => {
            collect_expression_uses(object, visitor);
            collect_expression_uses(index, visitor);
        }

        Expression::Slice {
            object,
            start,
            end,
            step,
        } => {
            collect_expression_uses(object, visitor);
            for bound in [start, end, step].into_iter().flatten() {
                collect_expression_uses(bound, visitor);
            }
        }

        Expression::MethodCall { object, args, .. } => {
            collect_expression_uses(object, visitor);
            for arg in args {
                collect_expression_uses(arg, visitor);
            }
        }

        Expression::Send { value, channel } => {
            collect_expression_uses(value, visitor);
            collect_expression_uses(channel, visitor);
        }

        Expression::Conditional {
            condition,
            then_expression,
            else_expression,
        } => {
            collect_expression_uses(condition, visitor);
            collect_expression_uses(then_expression, visitor);
            collect_expression_uses(else_expression, visitor);
        }

        Expression::CallValue { callee, args, .. } => {
            collect_expression_uses(callee, visitor);
            for arg in args {
                collect_expression_uses(arg, visitor);
            }
        }

        Expression::Lambda { body, .. } => collect_expression_uses(body, visitor),

        Expression::ListComprehension { element, clauses }
        | Expression::SetComprehension { element, clauses } => {
            for clause in clauses {
                collect_expression_uses(&clause.iterable, visitor);
                if let Some(condition) = &clause.condition {
                    collect_expression_uses(condition, visitor);
                }
            }
            collect_expression_uses(element, visitor);
        }

        Expression::MapComprehension {
            key,
            value,
            clauses,
        } => {
            for clause in clauses {
                collect_expression_uses(&clause.iterable, visitor);
                if let Some(condition) = &clause.condition {
                    collect_expression_uses(condition, visitor);
                }
            }
            collect_expression_uses(key, visitor);
            collect_expression_uses(value, visitor);
        }

        Expression::Integer(_)
        | Expression::Float(_)
        | Expression::Boolean(_)
        | Expression::String(_)
        | Expression::Function(_) => {}
    }
}
