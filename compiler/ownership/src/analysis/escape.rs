use severian_hir::{Expression, Function, Instruction, MatchPattern, OwnershipOp, Program, SwitchArm};
use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet, VecDeque};

/// Why a binding's storage/value must outlive an ordinary local use.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum EscapeReason {
    /// The value (or an alias of it) reaches a function return.
    Return,
    /// The value is captured by a lambda/closure.
    LambdaCapture,
    /// The value is captured by an asynchronous task.
    TaskCapture,
    /// The value is sent through a channel.
    ChannelSend,
    /// The value is passed to a call whose retention behavior is unknown.
    UnknownCall,
    /// The value is passed across a native ABI boundary.
    NativeCall,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum EscapeClass {
    Local,
    Captured,
    Escapes,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum FlowKind {
    Value,
    Move,
    View,
    Borrow,
    Address,
    Store,
    Pattern,
    ReturnAlias,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FlowEdge {
    pub source: String,
    pub destination: String,
    pub kind: FlowKind,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BindingEscape {
    pub name: String,
    pub class: EscapeClass,
    pub reasons: BTreeSet<EscapeReason>,
    pub address_taken: bool,
    pub borrowed: bool,
    pub viewed: bool,
    pub moved: bool,
    pub mutated: bool,
    pub reassigned: bool,
}

impl BindingEscape {
    fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            class: EscapeClass::Local,
            reasons: BTreeSet::new(),
            address_taken: false,
            borrowed: false,
            viewed: false,
            moved: false,
            mutated: false,
            reassigned: false,
        }
    }

    fn recompute_class(&mut self) {
        self.class = if self.reasons.iter().any(|reason| {
            matches!(
                reason,
                EscapeReason::Return
                    | EscapeReason::TaskCapture
                    | EscapeReason::ChannelSend
                    | EscapeReason::UnknownCall
                    | EscapeReason::NativeCall
            )
        }) {
            EscapeClass::Escapes
        } else if self.reasons.contains(&EscapeReason::LambdaCapture) {
            EscapeClass::Captured
        } else {
            EscapeClass::Local
        };
    }

    pub fn escapes(&self) -> bool {
        self.class == EscapeClass::Escapes
    }

    pub fn is_captured(&self) -> bool {
        matches!(self.class, EscapeClass::Captured | EscapeClass::Escapes)
            && (self.reasons.contains(&EscapeReason::LambdaCapture)
                || self.reasons.contains(&EscapeReason::TaskCapture))
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ParameterEscape {
    pub name: String,
    pub class: EscapeClass,
    pub reasons: BTreeSet<EscapeReason>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FunctionEscapeAnalysis {
    pub function: String,
    pub bindings: BTreeMap<String, BindingEscape>,
    pub parameters: Vec<ParameterEscape>,
    pub flows: Vec<FlowEdge>,
}

impl FunctionEscapeAnalysis {
    pub fn binding(&self, name: &str) -> Option<&BindingEscape> {
        self.bindings.get(name)
    }

    pub fn escapes(&self, name: &str) -> bool {
        self.binding(name).is_some_and(BindingEscape::escapes)
    }

    pub fn escaping_bindings(&self) -> impl Iterator<Item = &BindingEscape> {
        self.bindings.values().filter(|binding| binding.escapes())
    }

    /// Borrow/address escapes are useful to later ownership diagnostics and
    /// allocation/lifetime lowering.
    pub fn escaping_borrows(&self) -> impl Iterator<Item = &BindingEscape> {
        self.bindings.values().filter(|binding| {
            binding.escapes() && (binding.address_taken || binding.borrowed || binding.viewed)
        })
    }
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ProgramEscapeAnalysis {
    /// Top-level functions use their ordinary name. Methods/constructors use
    /// `Class::function` to avoid collisions in the public report.
    pub functions: BTreeMap<String, FunctionEscapeAnalysis>,
}

impl ProgramEscapeAnalysis {
    pub fn function(&self, name: &str) -> Option<&FunctionEscapeAnalysis> {
        self.functions.get(name)
    }
}

/// Analyze escape behavior for the complete HIR program.
///
/// The analysis is conservative. It follows explicit ownership/alias flows,
/// propagates escape reasons backwards through those flows, and iterates
/// function summaries so direct Severian calls can transfer parameter escape
/// behavior to callers.
pub fn analyze(program: &Program) -> ProgramEscapeAnalysis {
    let models = collect_function_models(program);
    let known = build_known_functions(&models);
    let mut summaries: HashMap<String, Vec<BTreeSet<EscapeReason>>> = HashMap::new();

    // Escape summaries form a finite monotone set, so fixed-point iteration is
    // bounded. The explicit cap is defensive against future changes that add
    // non-monotone metadata by mistake.
    for _ in 0..64 {
        let mut changed = false;
        let mut next = summaries.clone();

        for model in &models {
            let result = Analyzer::new(&known, &summaries)
                .analyze_function(model.function, model.fields.as_slice());
            let parameter_summary = result
                .parameters
                .iter()
                .map(|parameter| parameter.reasons.clone())
                .collect::<Vec<_>>();

            match next.get(&model.function.name) {
                Some(previous) if *previous == parameter_summary => {}
                _ => {
                    next.insert(model.function.name.clone(), parameter_summary);
                    changed = true;
                }
            }
        }

        summaries = next;
        if !changed {
            break;
        }
    }

    let mut functions = BTreeMap::new();
    for model in models {
        let result = Analyzer::new(&known, &summaries)
            .analyze_function(model.function, model.fields.as_slice());
        functions.insert(model.report_name, result);
    }

    ProgramEscapeAnalysis { functions }
}

/// Analyze one function without whole-program call summaries.
///
/// Use [`analyze`] when a `Program` is available; it can resolve direct calls
/// between Severian functions and is therefore more precise.
pub fn analyze_function(function: &Function) -> FunctionEscapeAnalysis {
    Analyzer::new(&HashMap::new(), &HashMap::new()).analyze_function(function, &[])
}

struct FunctionModel<'a> {
    report_name: String,
    function: &'a Function,
    fields: Vec<String>,
}

#[derive(Debug, Clone, Copy)]
struct KnownFunction {
    native: bool,
    parameter_count: usize,
}

fn collect_function_models(program: &Program) -> Vec<FunctionModel<'_>> {
    let mut models = Vec::new();

    for function in &program.functions {
        models.push(FunctionModel {
            report_name: function.name.clone(),
            function,
            fields: Vec::new(),
        });
    }

    for class in &program.classes {
        for function in class.methods.iter().chain(&class.constructors) {
            models.push(FunctionModel {
                report_name: format!("{}::{}", class.name, function.name),
                function,
                fields: class.fields.clone(),
            });
        }
    }

    models
}

fn build_known_functions(models: &[FunctionModel<'_>]) -> HashMap<String, KnownFunction> {
    let mut known = HashMap::new();
    for model in models {
        known.entry(model.function.name.clone()).or_insert(KnownFunction {
            native: model.function.native_symbol.is_some(),
            parameter_count: model.function.params.len(),
        });
    }
    known
}

#[derive(Debug, Clone, Default)]
struct BindingState {
    info: Option<BindingEscape>,
}

struct Analyzer<'a> {
    bindings: HashMap<String, BindingState>,
    flows: Vec<FlowEdge>,
    known: &'a HashMap<String, KnownFunction>,
    summaries: &'a HashMap<String, Vec<BTreeSet<EscapeReason>>>,
}

impl<'a> Analyzer<'a> {
    fn new(
        known: &'a HashMap<String, KnownFunction>,
        summaries: &'a HashMap<String, Vec<BTreeSet<EscapeReason>>>,
    ) -> Self {
        Self {
            bindings: HashMap::new(),
            flows: Vec::new(),
            known,
            summaries,
        }
    }

    fn analyze_function(
        mut self,
        function: &Function,
        implicit_fields: &[String],
    ) -> FunctionEscapeAnalysis {
        for field in implicit_fields {
            self.ensure_binding(field);
        }
        for parameter in &function.params {
            self.ensure_binding(&parameter.name);
            if let Some(default) = &parameter.default {
                self.flow_expression(default, Some(&parameter.name), None, FlowKind::Value);
            }
        }

        if let Some(contract) = &function.contract {
            for expression in contract.requirements.iter().chain(&contract.capabilities) {
                self.flow_expression(expression, None, None, FlowKind::Value);
            }
        }

        self.analyze_instructions(&function.instructions);
        self.solve();

        let mut bindings = self
            .bindings
            .into_iter()
            .filter_map(|(name, state)| state.info.map(|info| (name, info)))
            .collect::<BTreeMap<_, _>>();
        for binding in bindings.values_mut() {
            binding.recompute_class();
        }

        let parameters = function
            .params
            .iter()
            .map(|parameter| {
                let info = bindings
                    .get(&parameter.name)
                    .cloned()
                    .unwrap_or_else(|| BindingEscape::new(parameter.name.clone()));
                ParameterEscape {
                    name: parameter.name.clone(),
                    class: info.class,
                    reasons: info.reasons,
                }
            })
            .collect();

        FunctionEscapeAnalysis {
            function: function.name.clone(),
            bindings,
            parameters,
            flows: self.flows,
        }
    }

    fn analyze_instructions(&mut self, instructions: &[Instruction]) {
        for instruction in instructions {
            self.analyze_instruction(instruction);
        }
    }

    fn analyze_instruction(&mut self, instruction: &Instruction) {
        match instruction {
            Instruction::Let { name, value } | Instruction::TryLet { name, value } => {
                self.ensure_binding(name);
                self.flow_expression(value, Some(name), None, FlowKind::Value);
            }
            Instruction::Assign { target, value, .. } => {
                if let Some(root) = root_binding(target) {
                    self.ensure_binding(root);
                    self.binding_mut(root).mutated = true;
                    if matches!(target, Expression::Variable(_)) {
                        self.binding_mut(root).reassigned = true;
                    }
                    self.flow_expression(value, Some(root), None, FlowKind::Store);
                    self.flow_expression(target, None, None, FlowKind::Value);
                } else {
                    self.flow_expression(value, None, None, FlowKind::Value);
                    self.flow_expression(target, None, None, FlowKind::Value);
                }
            }
            Instruction::Print(value)
            | Instruction::Assert(value)
            | Instruction::Evaluate(value) => {
                self.flow_expression(value, None, None, FlowKind::Value);
            }
            Instruction::Return(value) => {
                if let Some(value) = value {
                    self.flow_expression(
                        value,
                        None,
                        Some(EscapeReason::Return),
                        FlowKind::ReturnAlias,
                    );
                }
            }
            Instruction::If {
                condition,
                then_instructions,
                else_instructions,
            } => {
                self.flow_expression(condition, None, None, FlowKind::Value);
                self.analyze_instructions(then_instructions);
                self.analyze_instructions(else_instructions);
            }
            Instruction::While {
                setup,
                capabilities,
                condition,
                instructions,
            } => {
                if let Some(setup) = setup {
                    self.analyze_instruction(setup);
                }
                for capability in capabilities {
                    self.flow_expression(capability, None, None, FlowKind::Value);
                }
                self.flow_expression(condition, None, None, FlowKind::Value);
                self.analyze_instructions(instructions);
            }
            Instruction::For {
                setup,
                pattern,
                iterable,
                instructions,
            } => {
                if let Some(setup) = setup {
                    self.analyze_instruction(setup);
                }
                self.bind_pattern(pattern);
                self.flow_into_pattern(iterable, pattern);
                self.flow_expression(iterable, None, None, FlowKind::Value);
                self.analyze_instructions(instructions);
            }
            Instruction::Switch { value, arms } => {
                self.flow_expression(value, None, None, FlowKind::Value);
                for arm in arms {
                    self.analyze_arm(arm, Some(value));
                }
            }
            Instruction::ChannelSwitch {
                channels,
                setup,
                repeat_condition,
                arms,
            } => {
                for channel in channels {
                    self.flow_expression(channel, None, None, FlowKind::Value);
                }
                if let Some(setup) = setup {
                    self.analyze_instruction(setup);
                }
                if let Some(condition) = repeat_condition {
                    self.flow_expression(condition, None, None, FlowKind::Value);
                }
                for arm in arms {
                    self.analyze_arm(arm, arm.source.as_ref());
                }
            }
            Instruction::With {
                resources,
                instructions,
                ..
            } => {
                for resource in resources {
                    self.flow_expression(resource, None, None, FlowKind::Value);
                }
                self.analyze_instructions(instructions);
            }
            Instruction::Break | Instruction::Continue => {}
        }
    }

    fn analyze_arm(&mut self, arm: &SwitchArm, fallback_source: Option<&Expression>) {
        self.bind_pattern(&arm.pattern);
        if let Some(source) = arm.source.as_ref().or(fallback_source) {
            self.flow_into_pattern(source, &arm.pattern);
            self.flow_expression(source, None, None, FlowKind::Value);
        }
        if let Some(guard) = &arm.guard {
            self.flow_expression(guard, None, None, FlowKind::Value);
        }
        self.analyze_instructions(&arm.instructions);
    }

    fn flow_expression(
        &mut self,
        expression: &Expression,
        destination: Option<&str>,
        seed: Option<EscapeReason>,
        kind: FlowKind,
    ) {
        match expression {
            Expression::Variable(name) => {
                self.ensure_binding(name);
                if let Some(destination) = destination {
                    self.add_flow(name, destination, kind);
                }
                if let Some(reason) = seed {
                    self.add_reason(name, reason);
                }
            }
            Expression::Ownership { op, value } => {
                if let Expression::Variable(name) = value.as_ref() {
                    self.ensure_binding(name);
                    match op {
                        OwnershipOp::View => self.binding_mut(name).viewed = true,
                        OwnershipOp::Borrow => self.binding_mut(name).borrowed = true,
                        OwnershipOp::AddressOf => self.binding_mut(name).address_taken = true,
                        OwnershipOp::Move => self.binding_mut(name).moved = true,
                        OwnershipOp::Clone => {}
                    }
                }

                match op {
                    OwnershipOp::Clone => {
                        // Clone produces independent storage. Reads/effects in the
                        // source still matter, but escape of the clone does not
                        // imply escape of the original binding.
                        self.flow_expression(value, None, None, FlowKind::Value);
                    }
                    OwnershipOp::Move => {
                        self.flow_expression(value, destination, seed, FlowKind::Move);
                    }
                    OwnershipOp::View => {
                        self.flow_expression(value, destination, seed, FlowKind::View);
                    }
                    OwnershipOp::Borrow => {
                        self.flow_expression(value, destination, seed, FlowKind::Borrow);
                    }
                    OwnershipOp::AddressOf => {
                        self.flow_expression(value, destination, seed, FlowKind::Address);
                    }
                }
            }
            Expression::Lambda { params, body } => {
                let free = free_variables(body, params);
                for name in free {
                    self.ensure_binding(&name);
                    self.add_reason(&name, EscapeReason::LambdaCapture);
                    if let Some(destination) = destination {
                        self.add_flow(&name, destination, FlowKind::Value);
                    }
                    if let Some(reason) = seed {
                        self.add_reason(&name, reason);
                    }
                }
            }
            Expression::Task { value, .. } => {
                for name in free_variables(value, &[]) {
                    self.ensure_binding(&name);
                    self.add_reason(&name, EscapeReason::TaskCapture);
                    if let Some(reason) = seed {
                        self.add_reason(&name, reason);
                    }
                }
                self.flow_expression(value, destination, None, FlowKind::Value);
            }
            Expression::Send { value, channel } => {
                self.flow_expression(
                    value,
                    None,
                    Some(EscapeReason::ChannelSend),
                    FlowKind::Value,
                );
                self.flow_expression(channel, None, None, FlowKind::Value);
            }
            Expression::Call { function, args } => {
                self.flow_direct_call(function, args, destination, seed);
            }
            Expression::CallValue { callee, args, .. } => {
                self.flow_expression(callee, None, None, FlowKind::Value);
                for argument in args {
                    self.flow_expression(
                        argument,
                        None,
                        Some(EscapeReason::UnknownCall),
                        FlowKind::Value,
                    );
                }
            }
            Expression::MethodCall { object, args, .. } => {
                // HIR does not currently carry the receiver's resolved class/method
                // identity here, so retention cannot be summarized precisely.
                self.flow_expression(object, None, None, FlowKind::Value);
                for argument in args {
                    self.flow_expression(
                        argument,
                        None,
                        Some(EscapeReason::UnknownCall),
                        FlowKind::Value,
                    );
                }
            }
            Expression::List(values)
            | Expression::Tuple(values)
            | Expression::Set(values)
            | Expression::PrintArgs(values)
            | Expression::Construct { args: values, .. }
            | Expression::Variant { fields: values, .. } => {
                for value in values {
                    self.flow_expression(value, destination, seed, kind);
                }
            }
            Expression::Map(entries) => {
                for (key, value) in entries {
                    self.flow_expression(key, destination, seed, kind);
                    self.flow_expression(value, destination, seed, kind);
                }
            }
            Expression::Index { object, index } => {
                self.flow_expression(object, destination, seed, kind);
                self.flow_expression(index, None, None, FlowKind::Value);
            }
            Expression::Slice {
                object,
                start,
                end,
                step,
            } => {
                self.flow_expression(object, destination, seed, kind);
                for bound in [start, end, step].into_iter().flatten() {
                    self.flow_expression(bound, None, None, FlowKind::Value);
                }
            }
            Expression::Format { args, .. } => {
                // Formatting consumes values to produce independent string storage.
                for argument in args {
                    self.flow_expression(argument, None, None, FlowKind::Value);
                }
            }
            Expression::Member { object, .. }
            | Expression::Await(object)
            | Expression::Channel(object)
            | Expression::ChaosRule { value: object, .. }
            | Expression::FusedPipeline { input: object, .. } => {
                self.flow_expression(object, destination, seed, kind);
            }
            Expression::ListComprehension { element, clauses }
            | Expression::SetComprehension { element, clauses } => {
                for clause in clauses {
                    self.bind_pattern(&clause.pattern);
                    self.flow_into_pattern(&clause.iterable, &clause.pattern);
                    self.flow_expression(&clause.iterable, None, None, FlowKind::Value);
                    if let Some(condition) = &clause.condition {
                        self.flow_expression(condition, None, None, FlowKind::Value);
                    }
                }
                self.flow_expression(element, destination, seed, kind);
            }
            Expression::MapComprehension {
                key,
                value,
                clauses,
            } => {
                for clause in clauses {
                    self.bind_pattern(&clause.pattern);
                    self.flow_into_pattern(&clause.iterable, &clause.pattern);
                    self.flow_expression(&clause.iterable, None, None, FlowKind::Value);
                    if let Some(condition) = &clause.condition {
                        self.flow_expression(condition, None, None, FlowKind::Value);
                    }
                }
                self.flow_expression(key, destination, seed, kind);
                self.flow_expression(value, destination, seed, kind);
            }
            Expression::Conditional {
                condition,
                then_expression,
                else_expression,
            } => {
                self.flow_expression(condition, None, None, FlowKind::Value);
                self.flow_expression(then_expression, destination, seed, kind);
                self.flow_expression(else_expression, destination, seed, kind);
            }
            Expression::Unary { expression, .. } => {
                self.flow_expression(expression, None, None, FlowKind::Value);
            }
            Expression::Binary { left, right, .. } => {
                self.flow_expression(left, None, None, FlowKind::Value);
                self.flow_expression(right, None, None, FlowKind::Value);
            }
            Expression::Integer(_)
            | Expression::Float(_)
            | Expression::Boolean(_)
            | Expression::String(_)
            | Expression::Function(_) => {}
        }
    }

    fn flow_direct_call(
        &mut self,
        function: &str,
        args: &[Expression],
        destination: Option<&str>,
        outer_seed: Option<EscapeReason>,
    ) {
        let known = self.known.get(function).copied();
        let summary = self.summaries.get(function).cloned();

        match known {
            None => {
                for argument in args {
                    self.flow_expression(
                        argument,
                        None,
                        Some(EscapeReason::UnknownCall),
                        FlowKind::Value,
                    );
                }
            }
            Some(known) if known.native => {
                for argument in args {
                    self.flow_expression(
                        argument,
                        None,
                        Some(EscapeReason::NativeCall),
                        FlowKind::Value,
                    );
                }
            }
            Some(known) => {
                for (index, argument) in args.iter().enumerate() {
                    self.flow_expression(argument, None, None, FlowKind::Value);
                    let Some(reasons) = summary.as_ref().and_then(|values| values.get(index)) else {
                        continue;
                    };

                    for reason in reasons {
                        if *reason == EscapeReason::Return {
                            if let Some(destination) = destination {
                                self.flow_expression(
                                    argument,
                                    Some(destination),
                                    outer_seed,
                                    FlowKind::ReturnAlias,
                                );
                            } else if let Some(seed) = outer_seed {
                                self.flow_expression(
                                    argument,
                                    None,
                                    Some(seed),
                                    FlowKind::ReturnAlias,
                                );
                            }
                        } else {
                            self.flow_expression(argument, None, Some(*reason), FlowKind::Value);
                        }
                    }
                }

                // A malformed/incomplete call should stay conservative instead of
                // silently dropping extra arguments from escape consideration.
                if args.len() > known.parameter_count {
                    for argument in &args[known.parameter_count..] {
                        self.flow_expression(
                            argument,
                            None,
                            Some(EscapeReason::UnknownCall),
                            FlowKind::Value,
                        );
                    }
                }
            }
        }
    }

    fn bind_pattern(&mut self, pattern: &MatchPattern) {
        match pattern {
            MatchPattern::Bind(name) => self.ensure_binding(name),
            MatchPattern::Constructor { fields, .. } => {
                for field in fields {
                    self.bind_pattern(field);
                }
            }
            MatchPattern::Wildcard
            | MatchPattern::Integer(_)
            | MatchPattern::Float(_)
            | MatchPattern::Boolean(_)
            | MatchPattern::String(_) => {}
        }
    }

    fn flow_into_pattern(&mut self, source: &Expression, pattern: &MatchPattern) {
        match pattern {
            MatchPattern::Bind(name) => {
                self.ensure_binding(name);
                self.flow_expression(source, Some(name), None, FlowKind::Pattern);
            }
            MatchPattern::Constructor { fields, .. } => {
                for field in fields {
                    self.flow_into_pattern(source, field);
                }
            }
            MatchPattern::Wildcard
            | MatchPattern::Integer(_)
            | MatchPattern::Float(_)
            | MatchPattern::Boolean(_)
            | MatchPattern::String(_) => {}
        }
    }

    fn ensure_binding(&mut self, name: &str) {
        self.bindings.entry(name.to_owned()).or_insert_with(|| BindingState {
            info: Some(BindingEscape::new(name)),
        });
    }

    fn binding_mut(&mut self, name: &str) -> &mut BindingEscape {
        self.ensure_binding(name);
        self.bindings
            .get_mut(name)
            .and_then(|state| state.info.as_mut())
            .expect("binding created by ensure_binding")
    }

    fn add_reason(&mut self, name: &str, reason: EscapeReason) {
        self.binding_mut(name).reasons.insert(reason);
    }

    fn add_flow(&mut self, source: &str, destination: &str, kind: FlowKind) {
        self.ensure_binding(source);
        self.ensure_binding(destination);
        if source == destination {
            return;
        }
        self.flows.push(FlowEdge {
            source: source.to_owned(),
            destination: destination.to_owned(),
            kind,
        });
    }

    fn solve(&mut self) {
        // Build reverse adjacency: if source flows into destination and the
        // destination escapes, the source must satisfy the same lifetime.
        let mut reverse: HashMap<String, Vec<String>> = HashMap::new();
        for edge in &self.flows {
            reverse
                .entry(edge.destination.clone())
                .or_default()
                .push(edge.source.clone());
        }

        let mut queue = self
            .bindings
            .iter()
            .filter_map(|(name, state)| {
                state
                    .info
                    .as_ref()
                    .filter(|info| !info.reasons.is_empty())
                    .map(|_| name.clone())
            })
            .collect::<VecDeque<_>>();
        let mut queued = queue.iter().cloned().collect::<HashSet<_>>();

        while let Some(destination) = queue.pop_front() {
            queued.remove(&destination);
            let reasons = self
                .bindings
                .get(&destination)
                .and_then(|state| state.info.as_ref())
                .map(|info| info.reasons.clone())
                .unwrap_or_default();

            let Some(sources) = reverse.get(&destination).cloned() else {
                continue;
            };

            for source in sources {
                let info = self.binding_mut(&source);
                let before = info.reasons.len();
                info.reasons.extend(reasons.iter().copied());
                if info.reasons.len() != before && queued.insert(source.clone()) {
                    queue.push_back(source);
                }
            }
        }
    }
}

fn root_binding(expression: &Expression) -> Option<&str> {
    match expression {
        Expression::Variable(name) => Some(name),
        Expression::Member { object, .. }
        | Expression::Index { object, .. }
        | Expression::Slice { object, .. }
        | Expression::Ownership { value: object, .. } => root_binding(object),
        _ => None,
    }
}

fn free_variables(expression: &Expression, initial_bound: &[String]) -> BTreeSet<String> {
    let mut bound = initial_bound.iter().cloned().collect::<HashSet<_>>();
    let mut free = BTreeSet::new();
    collect_free_variables(expression, &mut bound, &mut free);
    free
}

fn collect_free_variables(
    expression: &Expression,
    bound: &mut HashSet<String>,
    free: &mut BTreeSet<String>,
) {
    match expression {
        Expression::Variable(name) => {
            if !bound.contains(name) {
                free.insert(name.clone());
            }
        }
        Expression::Lambda { params, body } => {
            let mut nested = bound.clone();
            nested.extend(params.iter().cloned());
            collect_free_variables(body, &mut nested, free);
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
        } => collect_free_variables(value, bound, free),
        Expression::List(values)
        | Expression::Tuple(values)
        | Expression::Set(values)
        | Expression::PrintArgs(values)
        | Expression::Construct { args: values, .. }
        | Expression::Variant { fields: values, .. } => {
            for value in values {
                collect_free_variables(value, bound, free);
            }
        }
        Expression::Map(entries) => {
            for (key, value) in entries {
                collect_free_variables(key, bound, free);
                collect_free_variables(value, bound, free);
            }
        }
        Expression::Index { object, index } => {
            collect_free_variables(object, bound, free);
            collect_free_variables(index, bound, free);
        }
        Expression::Slice {
            object,
            start,
            end,
            step,
        } => {
            collect_free_variables(object, bound, free);
            for value in [start, end, step].into_iter().flatten() {
                collect_free_variables(value, bound, free);
            }
        }
        Expression::Format { args, .. } => {
            for value in args {
                collect_free_variables(value, bound, free);
            }
        }
        Expression::MethodCall { object, args, .. } => {
            collect_free_variables(object, bound, free);
            for argument in args {
                collect_free_variables(argument, bound, free);
            }
        }
        Expression::Send { value, channel } => {
            collect_free_variables(value, bound, free);
            collect_free_variables(channel, bound, free);
        }
        Expression::ListComprehension { element, clauses }
        | Expression::SetComprehension { element, clauses } => {
            let mut nested = bound.clone();
            for clause in clauses {
                collect_free_variables(&clause.iterable, &mut nested, free);
                add_pattern_bindings(&clause.pattern, &mut nested);
                if let Some(condition) = &clause.condition {
                    collect_free_variables(condition, &mut nested, free);
                }
            }
            collect_free_variables(element, &mut nested, free);
        }
        Expression::MapComprehension {
            key,
            value,
            clauses,
        } => {
            let mut nested = bound.clone();
            for clause in clauses {
                collect_free_variables(&clause.iterable, &mut nested, free);
                add_pattern_bindings(&clause.pattern, &mut nested);
                if let Some(condition) = &clause.condition {
                    collect_free_variables(condition, &mut nested, free);
                }
            }
            collect_free_variables(key, &mut nested, free);
            collect_free_variables(value, &mut nested, free);
        }
        Expression::Conditional {
            condition,
            then_expression,
            else_expression,
        } => {
            collect_free_variables(condition, bound, free);
            collect_free_variables(then_expression, bound, free);
            collect_free_variables(else_expression, bound, free);
        }
        Expression::Binary { left, right, .. } => {
            collect_free_variables(left, bound, free);
            collect_free_variables(right, bound, free);
        }
        Expression::Call { args, .. } => {
            for argument in args {
                collect_free_variables(argument, bound, free);
            }
        }
        Expression::CallValue { callee, args, .. } => {
            collect_free_variables(callee, bound, free);
            for argument in args {
                collect_free_variables(argument, bound, free);
            }
        }
        Expression::Integer(_)
        | Expression::Float(_)
        | Expression::Boolean(_)
        | Expression::String(_)
        | Expression::Function(_) => {}
    }
}

fn add_pattern_bindings(pattern: &MatchPattern, bound: &mut HashSet<String>) {
    match pattern {
        MatchPattern::Bind(name) => {
            bound.insert(name.clone());
        }
        MatchPattern::Constructor { fields, .. } => {
            for field in fields {
                add_pattern_bindings(field, bound);
            }
        }
        MatchPattern::Wildcard
        | MatchPattern::Integer(_)
        | MatchPattern::Float(_)
        | MatchPattern::Boolean(_)
        | MatchPattern::String(_) => {}
    }
}

