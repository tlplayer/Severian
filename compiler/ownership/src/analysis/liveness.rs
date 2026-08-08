use severian_hir::{
    Expression, Function, Instruction, MatchPattern, Program, SwitchArm,
};
use std::collections::{BTreeSet, HashMap};

/// A half-open lexical live range `[definition, last_use + 1)`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LiveRange {
    pub definition: usize,
    pub last_use: usize,
}

impl LiveRange {
    pub fn contains(self, point: usize) -> bool {
        self.definition <= point && point <= self.last_use
    }

    pub fn overlaps(self, other: Self) -> bool {
        self.definition <= other.last_use && other.definition <= self.last_use
    }
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct BindingLiveness {
    pub definition: Option<usize>,
    pub uses: Vec<usize>,
    pub live_range: Option<LiveRange>,
    pub captured: bool,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct FunctionLiveness {
    pub function: String,
    pub bindings: HashMap<String, BindingLiveness>,
    pub program_points: usize,
}

impl FunctionLiveness {
    pub fn binding(&self, name: &str) -> Option<&BindingLiveness> {
        self.bindings.get(name)
    }

    pub fn is_live_at(&self, name: &str, point: usize) -> bool {
        self.binding(name)
            .and_then(|binding| binding.live_range)
            .is_some_and(|range| range.contains(point))
    }

    pub fn last_use(&self, name: &str) -> Option<usize> {
        self.binding(name)
            .and_then(|binding| binding.uses.last().copied())
    }
}

pub fn analyze(program: &Program) -> HashMap<String, FunctionLiveness> {
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

pub fn analyze_function(function: &Function) -> FunctionLiveness {
    let mut analyzer = Analyzer {
        result: FunctionLiveness {
            function: function.name.clone(),
            ..FunctionLiveness::default()
        },
        point: 0,
        lambda_scopes: Vec::new(),
    };

    for parameter in &function.params {
        analyzer.define(&parameter.name);
        if let Some(default) = &parameter.default {
            analyzer.expression(default);
        }
    }

    if let Some(contract) = &function.contract {
        for expression in contract
            .requirements
            .iter()
            .chain(&contract.capabilities)
        {
            analyzer.expression(expression);
        }
    }

    analyzer.instructions(&function.instructions);

    for test in &function.tests {
        analyzer.instructions(&test.instructions);
    }

    analyzer.finish()
}

struct Analyzer {
    result: FunctionLiveness,
    point: usize,
    /// Names introduced by nested lambdas. A variable use not present in the
    /// current lambda scope is a capture of an outer binding.
    lambda_scopes: Vec<BTreeSet<String>>,
}

impl Analyzer {
    fn finish(mut self) -> FunctionLiveness {
        for binding in self.result.bindings.values_mut() {
            if let Some(definition) = binding.definition {
                let last_use = binding.uses.last().copied().unwrap_or(definition);
                binding.live_range = Some(LiveRange {
                    definition,
                    last_use,
                });
            }
        }
        self.result.program_points = self.point;
        self.result
    }

    fn tick(&mut self) -> usize {
        let point = self.point;
        self.point += 1;
        point
    }

    fn define(&mut self, name: &str) {
        let point = self.tick();
        self.result
            .bindings
            .entry(name.to_string())
            .or_default()
            .definition
            .get_or_insert(point);
    }

    fn use_name(&mut self, name: &str) {
        let point = self.tick();
        let binding = self.result.bindings.entry(name.to_string()).or_default();
        binding.uses.push(point);

        if !self.lambda_scopes.is_empty()
            && !self
                .lambda_scopes
                .last()
                .is_some_and(|scope| scope.contains(name))
        {
            binding.captured = true;
        }
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

    fn instructions(&mut self, instructions: &[Instruction]) {
        for instruction in instructions {
            self.instruction(instruction);
        }
    }

    fn instruction(&mut self, instruction: &Instruction) {
        self.tick();

        match instruction {
            Instruction::Let { name, value } | Instruction::TryLet { name, value } => {
                self.expression(value);
                self.define(name);
            }
            Instruction::Assign { target, value, .. } => {
                self.expression(value);
                self.expression(target);
            }
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
                self.define_pattern(pattern);
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
            self.define_pattern(&arm.pattern);
            if let Some(guard) = &arm.guard {
                self.expression(guard);
            }
            self.instructions(&arm.instructions);
        }
    }

    fn comprehension_clauses(
        &mut self,
        clauses: &[severian_hir::ComprehensionClause],
    ) {
        for clause in clauses {
            self.expression(&clause.iterable);
            self.define_pattern(&clause.pattern);
            if let Some(condition) = &clause.condition {
                self.expression(condition);
            }
        }
    }

    fn expression(&mut self, expression: &Expression) {
        self.tick();

        match expression {
            Expression::Variable(name) => self.use_name(name),
            Expression::Lambda { params, body } => {
                let scope = params.iter().cloned().collect::<BTreeSet<_>>();
                self.lambda_scopes.push(scope);
                for param in params {
                    self.define(param);
                }
                self.expression(body);
                self.lambda_scopes.pop();
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
            Expression::Format { args, .. }
            | Expression::Call { args, .. } => {
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
                self.comprehension_clauses(clauses);
                self.expression(element);
            }
            Expression::MapComprehension {
                key,
                value,
                clauses,
            } => {
                self.comprehension_clauses(clauses);
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

