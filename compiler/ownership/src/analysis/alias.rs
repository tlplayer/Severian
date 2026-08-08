use severian_hir::{
    Expression, Function, Instruction, MatchPattern, OwnershipOp, Program, SwitchArm,
};
use std::collections::{HashMap, HashSet};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AliasKind {
    Direct,
    SharedView,
    MutableBorrow,
    Address,
    MoveTransfer,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct AliasEdge {
    pub source: String,
    pub destination: String,
    pub kind: AliasKind,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct AliasAnalysis {
    pub edges: Vec<AliasEdge>,
    pub bindings: HashSet<String>,
}

impl AliasAnalysis {
    pub fn aliases_of<'a>(&'a self, name: &'a str) -> impl Iterator<Item = &'a AliasEdge> {
        self.edges
            .iter()
            .filter(move |edge| edge.source == name || edge.destination == name)
    }

    pub fn may_alias(&self, left: &str, right: &str) -> bool {
        if left == right {
            return true;
        }

        let mut stack = vec![left.to_string()];
        let mut visited = HashSet::new();

        while let Some(current) = stack.pop() {
            if !visited.insert(current.clone()) {
                continue;
            }

            for edge in self.aliases_of(&current) {
                let next = if edge.source == current {
                    &edge.destination
                } else {
                    &edge.source
                };

                if next == right {
                    return true;
                }
                stack.push(next.clone());
            }
        }

        false
    }
}

pub fn analyze(program: &Program) -> HashMap<String, AliasAnalysis> {
    let mut result = HashMap::new();

    for function in &program.functions {
        result.insert(function.name.clone(), analyze_function(function));
    }

    for class in &program.classes {
        for function in class.methods.iter().chain(&class.constructors) {
            result.insert(
                format!("{}::{}", class.name, function.name),
                analyze_function(function),
            );
        }
    }

    result
}

pub fn analyze_function(function: &Function) -> AliasAnalysis {
    let mut analyzer = Analyzer::default();

    for parameter in &function.params {
        analyzer.analysis.bindings.insert(parameter.name.clone());
    }

    analyzer.instructions(&function.instructions);
    analyzer.analysis
}

#[derive(Default)]
struct Analyzer {
    analysis: AliasAnalysis,
}

impl Analyzer {
    fn edge(&mut self, source: &str, destination: &str, kind: AliasKind) {
        self.analysis.bindings.insert(source.to_string());
        self.analysis.bindings.insert(destination.to_string());

        let edge = AliasEdge {
            source: source.to_string(),
            destination: destination.to_string(),
            kind,
        };

        if !self.analysis.edges.contains(&edge) {
            self.analysis.edges.push(edge);
        }
    }

    fn bind_initializer(&mut self, destination: &str, expression: &Expression) {
        self.analysis.bindings.insert(destination.to_string());

        match expression {
            Expression::Variable(source) => {
                self.edge(source, destination, AliasKind::Direct);
            }
            Expression::Ownership { op, value } => {
                let Expression::Variable(source) = value.as_ref() else {
                    self.expression(value);
                    return;
                };

                match op {
                    OwnershipOp::View => {
                        self.edge(source, destination, AliasKind::SharedView);
                    }
                    OwnershipOp::Borrow => {
                        self.edge(source, destination, AliasKind::MutableBorrow);
                    }
                    OwnershipOp::AddressOf => {
                        self.edge(source, destination, AliasKind::Address);
                    }
                    OwnershipOp::Move => {
                        self.edge(source, destination, AliasKind::MoveTransfer);
                    }
                    OwnershipOp::Clone => {
                        // Clone establishes a new value, not an alias.
                    }
                }
            }
            _ => self.expression(expression),
        }
    }

    fn assign(&mut self, target: &Expression, value: &Expression) {
        match (target, value) {
            (Expression::Variable(destination), Expression::Variable(source)) => {
                self.edge(source, destination, AliasKind::Direct);
            }
            (
                Expression::Variable(destination),
                Expression::Ownership { op, value },
            ) => {
                if let Expression::Variable(source) = value.as_ref() {
                    let kind = match op {
                        OwnershipOp::View => Some(AliasKind::SharedView),
                        OwnershipOp::Borrow => Some(AliasKind::MutableBorrow),
                        OwnershipOp::AddressOf => Some(AliasKind::Address),
                        OwnershipOp::Move => Some(AliasKind::MoveTransfer),
                        OwnershipOp::Clone => None,
                    };
                    if let Some(kind) = kind {
                        self.edge(source, destination, kind);
                    }
                } else {
                    self.expression(value);
                }
            }
            _ => {
                self.expression(target);
                self.expression(value);
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
                self.bind_initializer(name, value);
            }
            Instruction::Assign { target, value, .. } => self.assign(target, value),
            Instruction::Print(value)
            | Instruction::Assert(value)
            | Instruction::Evaluate(value) => self.expression(value),
            Instruction::Return(value) => {
                if let Some(value) = value {
                    self.expression(value);
                }
            }
            Instruction::If {
                condition,
                then_instructions,
                else_instructions,
            } => {
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
                pattern,
                iterable,
                instructions,
            } => {
                if let Some(setup) = setup {
                    self.instruction(setup);
                }
                self.expression(iterable);
                self.pattern(pattern);
                self.instructions(instructions);
            }
            Instruction::Switch { value, arms } => {
                self.expression(value);
                self.arms(arms);
            }
            Instruction::ChannelSwitch {
                channels,
                setup,
                repeat_condition,
                arms,
            } => {
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
                for resource in resources {
                    self.expression(resource);
                }
                self.instructions(instructions);
            }
            Instruction::Break | Instruction::Continue => {}
        }
    }

    fn arms(&mut self, arms: &[SwitchArm]) {
        for arm in arms {
            if let Some(source) = &arm.source {
                self.expression(source);
            }
            self.pattern(&arm.pattern);
            if let Some(guard) = &arm.guard {
                self.expression(guard);
            }
            self.instructions(&arm.instructions);
        }
    }

    fn pattern(&mut self, pattern: &MatchPattern) {
        match pattern {
            MatchPattern::Bind(name) => {
                self.analysis.bindings.insert(name.clone());
            }
            MatchPattern::Constructor { fields, .. } => {
                for field in fields {
                    self.pattern(field);
                }
            }
            MatchPattern::Wildcard
            | MatchPattern::Integer(_)
            | MatchPattern::Float(_)
            | MatchPattern::Boolean(_)
            | MatchPattern::String(_) => {}
        }
    }

    fn expression(&mut self, expression: &Expression) {
        match expression {
            Expression::Variable(name) => {
                self.analysis.bindings.insert(name.clone());
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
            } => self.expression(value),
            Expression::Lambda { body, .. } => self.expression(body),
            Expression::List(values)
            | Expression::Tuple(values)
            | Expression::Set(values)
            | Expression::PrintArgs(values)
            | Expression::Construct { args: values, .. }
            | Expression::Variant { fields: values, .. } => {
                for value in values {
                    self.expression(value);
                }
            }
            Expression::Map(entries) => {
                for (key, value) in entries {
                    self.expression(key);
                    self.expression(value);
                }
            }
            Expression::Index { object, index } => {
                self.expression(object);
                self.expression(index);
            }
            Expression::Slice {
                object,
                start,
                end,
                step,
            } => {
                self.expression(object);
                for bound in [start, end, step].into_iter().flatten() {
                    self.expression(bound);
                }
            }
            Expression::Format { args, .. } | Expression::Call { args, .. } => {
                for arg in args {
                    self.expression(arg);
                }
            }
            Expression::MethodCall { object, args, .. } => {
                self.expression(object);
                for arg in args {
                    self.expression(arg);
                }
            }
            Expression::Send { value, channel } => {
                self.expression(value);
                self.expression(channel);
            }
            Expression::ListComprehension { element, clauses }
            | Expression::SetComprehension { element, clauses } => {
                for clause in clauses {
                    self.expression(&clause.iterable);
                    self.pattern(&clause.pattern);
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
                for clause in clauses {
                    self.expression(&clause.iterable);
                    self.pattern(&clause.pattern);
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
                self.expression(condition);
                self.expression(then_expression);
                self.expression(else_expression);
            }
            Expression::Binary { left, right, .. } => {
                self.expression(left);
                self.expression(right);
            }
            Expression::CallValue { callee, args, .. } => {
                self.expression(callee);
                for arg in args {
                    self.expression(arg);
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
