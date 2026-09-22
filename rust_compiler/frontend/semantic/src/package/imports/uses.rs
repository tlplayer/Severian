use super::ImportRequirements;
use severian_ast::*;
use severian_modules::ModuleGraph;
use std::collections::BTreeSet;

pub fn collect_import_requirements(graph: &ModuleGraph) -> ImportRequirements {
    graph
        .modules
        .iter()
        .map(|module| {
            let mut uses = Uses::default();
            for item in &module.ast.items {
                uses.item(item);
            }
            (module.id, uses.names)
        })
        .collect()
}

#[derive(Default)]
struct Uses {
    names: BTreeSet<String>,
    bound: BTreeSet<String>,
}
impl Uses {
    fn name(&mut self, name: &str) {
        if !self.bound.contains(name.split('.').next().unwrap_or(name)) {
            self.names.insert(name.to_owned());
        }
    }
    fn scope(&mut self, names: impl IntoIterator<Item = String>, visit: impl FnOnce(&mut Self)) {
        let saved = self.bound.clone();
        self.bound.extend(names);
        visit(self);
        self.bound = saved;
    }
    fn annotation(&mut self, annotation: &TypeAnnotation) {
        match &annotation.kind {
            TypeAnnotationKind::Named { name, arguments } => {
                self.name(name);
                for a in arguments {
                    self.annotation(a);
                }
            }
            TypeAnnotationKind::Function { parameters, result } => {
                for p in parameters {
                    self.annotation(p);
                }
                self.annotation(result);
            }
            TypeAnnotationKind::Union(members) => {
                for m in members {
                    self.annotation(m);
                }
            }
            TypeAnnotationKind::ShapeSpread(name) => self.name(name),
            TypeAnnotationKind::DimensionConstant(_) | TypeAnnotationKind::DimensionRuntime(_) => {}
        }
    }
    fn constraints(&mut self, constraints: &[GenericConstraint]) {
        for constraint in constraints {
            match constraint {
                GenericConstraint::Parameter { bound, .. } => self.annotation(bound),
                GenericConstraint::Predicate(e) => self.expression(e),
                GenericConstraint::VariadicPack { .. } => {}
            }
        }
    }
    fn decorators(&mut self, decorators: &[Decorator]) {
        for decorator in decorators {
            self.name(&decorator.name);
            for argument in &decorator.arguments {
                if let DecoratorValue::Name(name) = &argument.value {
                    self.name(name);
                }
            }
        }
    }
    fn contracts(&mut self, contracts: &[FunctionContract]) {
        for c in contracts {
            self.expression(&c.condition);
            self.optional(c.failure.as_ref());
        }
    }
    fn property(&mut self, p: &PropertyDeclaration) {
        self.annotation(&p.annotation);
        self.optional(p.default.as_ref());
        for c in &p.constraints {
            self.expression(&c.condition);
            self.optional(c.failure.as_ref());
        }
    }
    fn function(&mut self, f: &FunctionDeclaration) {
        self.decorators(&f.decorators);
        self.scope(f.type_parameters.clone(), |s| {
            s.constraints(&f.constraints);
            for p in &f.parameters {
                s.annotation(&p.annotation);
                s.optional(p.default.as_ref());
            }
            s.annotation(&f.result);
            s.scope(f.parameters.iter().map(|p| p.name.clone()), |s| {
                s.contracts(&f.contracts);
                if let Some(hook) = &f.hook {
                    s.name(&hook.context);
                    s.block(&hook.with_phase);
                    s.block(&hook.without_phase);
                }
                if let Some(body) = &f.body {
                    s.block(body);
                }
            });
        });
    }
    fn operator(&mut self, op: &OperatorImplementation) {
        self.decorators(&op.decorators);
        self.scope(op.type_parameters.clone(), |s| {
            s.constraints(&op.constraints);
            for p in &op.parameters {
                s.annotation(&p.annotation);
            }
            s.annotation(&op.result);
            s.scope(op.parameters.iter().map(|p| p.name.clone()), |s| {
                s.contracts(&op.contracts);
                s.block(&op.body);
            });
        });
    }
    fn test(&mut self, test: &TestDeclaration) {
        for row in &test.cases {
            for e in row {
                self.expression(e);
            }
        }
        self.scope(test.parameters.clone(), |s| {
            s.contracts(&test.contracts);
            s.block(&test.body);
            for case in &test.compiler_cases {
                for item in &case.items {
                    s.item(item);
                }
                s.block(&case.body);
            }
        });
    }
    fn item(&mut self, item: &Item) {
        match item {
            Item::Import(_) => {}
            Item::Expression(e) => self.expression(e),
            Item::Binding(b) => {
                if let Some(a) = &b.annotation {
                    self.annotation(a);
                }
                self.expression(&b.value);
            }
            Item::Function(f) => self.function(f),
            Item::Test(t) => self.test(t),
            Item::Type(t) => {
                self.decorators(&t.decorators);
                self.scope(t.type_parameters.clone(), |s| {
                    s.constraints(&t.constraints);
                    if let Some(a) = &t.definition {
                        s.annotation(a);
                    }
                });
            }
            Item::Enum(e) => {
                for v in &e.variants {
                    for p in &v.fields {
                        self.property(p);
                    }
                }
            }
            Item::Class(c) => {
                self.decorators(&c.decorators);
                self.scope(
                    c.type_parameters.iter().cloned().chain(["Self".to_owned()]),
                    |s| {
                        s.constraints(&c.constraints);
                        for a in c
                            .type_parameter_defaults
                            .iter()
                            .flatten()
                            .chain(&c.traits)
                            .chain(&c.aliases)
                        {
                            s.annotation(a);
                        }
                        for p in &c.fields {
                            s.property(p);
                        }
                        for f in c.methods.iter().chain(&c.constructors) {
                            s.function(f);
                        }
                        for op in &c.operators {
                            s.operator(op);
                        }
                        for t in &c.tests {
                            s.test(t);
                        }
                    },
                );
            }
            Item::Trait(t) => {
                self.decorators(&t.decorators);
                self.decorators(&t.namespaces);
                self.scope(
                    t.type_parameters.iter().cloned().chain(["Self".to_owned()]),
                    |s| {
                        s.constraints(&t.constraints);
                        for a in &t.bases {
                            s.annotation(a);
                        }
                        for p in &t.properties {
                            s.property(p);
                        }
                        for f in &t.methods {
                            s.function(f);
                        }
                        for op in &t.operators {
                            s.decorators(&op.decorators);
                            s.scope(op.type_parameters.clone(), |s| {
                                s.constraints(&op.constraints);
                                for p in &op.parameters {
                                    s.annotation(&p.annotation);
                                }
                                s.annotation(&op.result);
                            });
                        }
                    },
                );
            }
            Item::Extension(e) => {
                self.decorators(&e.decorators);
                self.scope(
                    e.type_parameters.iter().cloned().chain(["Self".to_owned()]),
                    |s| {
                        s.annotation(&e.target);
                        s.constraints(&e.constraints);
                        for f in &e.methods {
                            s.function(f);
                        }
                        for op in &e.operators {
                            s.operator(op);
                        }
                    },
                );
            }
        }
    }
    fn optional(&mut self, expression: Option<&Expression>) {
        if let Some(e) = expression {
            self.expression(e);
        }
    }
    fn expression(&mut self, expression: &Expression) {
        use ExpressionKind::*;
        match &expression.kind {
            Literal(_) | Symbol(_) => {}
            Name(name) => self.name(name),
            Member { object, .. } => {
                fn path(e: &Expression) -> Option<String> {
                    match &e.kind {
                        Name(n) => Some(n.clone()),
                        Member { object, name } => Some(format!("{}.{}", path(object)?, name)),
                        _ => None,
                    }
                }
                if let Some(name) = path(expression) {
                    self.name(&name);
                }
                self.expression(object);
            }
            List(elements) | Set(elements) | Tuple(elements) => {
                for e in elements {
                    self.expression(e);
                }
            }
            Map(entries) => {
                for e in entries {
                    self.expression(&e.key);
                    self.expression(&e.value);
                }
            }
            ListComprehension { value, clauses } | SetComprehension { value, clauses } => {
                self.comprehension(clauses, |s| s.expression(value))
            }
            MapComprehension {
                key,
                value,
                clauses,
            } => self.comprehension(clauses, |s| {
                s.expression(key);
                s.expression(value);
            }),
            Mock { cases, fallback } => {
                for c in cases {
                    self.expression(&c.call);
                    self.expression(&c.result);
                }
                self.expression(fallback);
            }
            Lambda { parameters, body } => self.scope(parameters.clone(), |s| s.expression(body)),
            Index { object, index } => {
                self.expression(object);
                self.expression(index);
            }
            Slice {
                object,
                start,
                end,
                step,
                ..
            } => {
                self.expression(object);
                self.optional(start.as_deref());
                self.optional(end.as_deref());
                self.optional(step.as_deref());
            }
            TypeApplication { callee, arguments } => {
                self.expression(callee);
                for a in arguments {
                    self.annotation(a);
                }
            }
            Call { callee, arguments } => {
                self.expression(callee);
                for a in arguments {
                    self.expression(&a.value);
                    self.optional(a.expected_error.as_ref());
                }
            }
            Async { expression, .. } | Await { expression } => self.expression(expression),
            Conditional {
                value,
                condition,
                fallback,
            } => {
                self.expression(value);
                self.expression(condition);
                self.expression(fallback);
            }
            Fallback { value, fallback } => {
                self.expression(value);
                self.expression(fallback);
            }
            Throw { error } => self.expression(error),
            Unary { operand, .. } => self.expression(operand),
            Binary { left, right, .. } => {
                self.expression(left);
                self.expression(right);
            }
        }
    }
    fn comprehension(&mut self, clauses: &[ComprehensionClause], value: impl FnOnce(&mut Self)) {
        self.scope([], |s| {
            for clause in clauses {
                s.expression(&clause.iterable);
                s.bound.extend(clause.bindings.iter().cloned());
                s.optional(clause.condition.as_ref());
            }
            value(s);
        });
    }
    fn binding(&mut self, binding: &Binding) {
        if let Some(a) = &binding.annotation {
            self.annotation(a);
        }
        self.expression(&binding.value);
        if binding.update {
            self.name(&binding.name);
        }
        self.bound.insert(binding.name.clone());
    }
    fn block(&mut self, statements: &[Statement]) {
        self.scope([], |s| s.statements(statements));
    }
    fn statements(&mut self, statements: &[Statement]) {
        use Statement::*;
        for statement in statements {
            match statement {
                Binding(b) => self.binding(b),
                Destructure { names, value, .. } => {
                    self.expression(value);
                    self.bound.extend(names.iter().cloned());
                }
                FieldAssignment { object, value, .. } => {
                    self.expression(object);
                    self.expression(value);
                }
                IndexAssignment {
                    object,
                    index,
                    value,
                    ..
                } => {
                    self.expression(object);
                    self.expression(index);
                    self.expression(value);
                }
                Expression(e) | Defer { expression: e, .. } | Yield { value: e, .. } => {
                    self.expression(e)
                }
                Return { value, .. } => self.optional(value.as_ref()),
                Assert {
                    condition, message, ..
                } => {
                    self.expression(condition);
                    self.optional(message.as_ref());
                }
                Unsafe { body, .. } => self.statements(body),
                Placement { policy, body, .. } => {
                    self.name(policy);
                    self.block(body);
                }
                Try {
                    body,
                    catch_binding,
                    catch_annotation,
                    catch_body,
                    ..
                } => {
                    self.block(body);
                    if let Some(a) = catch_annotation {
                        self.annotation(a);
                    }
                    self.scope([catch_binding.clone()], |s| s.block(catch_body));
                }
                FallibleElse {
                    value,
                    error_binding,
                    body,
                    ..
                } => {
                    self.expression(value);
                    self.scope([error_binding.clone()], |s| s.block(body));
                }
                If {
                    condition,
                    then_block,
                    else_block,
                    ..
                } => {
                    self.expression(condition);
                    self.block(then_block);
                    self.block(else_block);
                }
                While {
                    condition,
                    initializer,
                    guards,
                    body,
                    ..
                } => self.scope([], |s| {
                    if let Some(b) = initializer {
                        s.binding(b);
                    }
                    s.expression(condition);
                    for g in guards {
                        s.expression(&g.condition);
                    }
                    s.block(body);
                }),
                For {
                    binding,
                    second_binding,
                    iterable,
                    initializer,
                    placement,
                    body,
                    ..
                } => self.scope([], |s| {
                    if let Some(b) = initializer {
                        s.binding(b);
                    }
                    s.expression(iterable);
                    if let Some(p) = placement {
                        s.name(p);
                    }
                    s.bound.insert(binding.clone());
                    s.bound.extend(second_binding.iter().cloned());
                    s.block(body);
                }),
                Match { subject, cases, .. } => {
                    self.expression(subject);
                    for c in cases {
                        if let Some(a) = &c.annotation {
                            self.annotation(a);
                        }
                        self.scope(c.binding.iter().cloned(), |s| s.block(&c.body));
                    }
                }
                Select {
                    limit,
                    cases,
                    error_body,
                    ..
                } => {
                    self.expression(limit);
                    for c in cases {
                        self.expression(&c.channel);
                        self.scope([c.binding.clone()], |s| s.block(&c.body));
                    }
                    self.block(error_body);
                }
                Break { .. } | Continue { .. } => {}
            }
        }
    }
}
