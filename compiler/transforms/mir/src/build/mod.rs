use crate::{Block, Function, Module, Operation, Value, ValueId};
use severian_hir::{BindingId, Expression, ExpressionKind, Program as HirProgram, Statement};
use std::collections::BTreeMap;

pub fn build(hir: &HirProgram) -> Result<Module, crate::VerifyError> {
    let function_ids = hir
        .modules
        .iter()
        .flat_map(|module| &module.functions)
        .map(|function| {
            (
                (function.definition, function.substitution.clone()),
                function.id,
            )
        })
        .collect();
    let mut builder = Builder {
        module: Module::default(),
        bindings: BTreeMap::new(),
        variables: BTreeMap::new(),
        function_ids,
    };
    let (initializer_cfg, mut function_cfgs) = crate::cfg::lower_program(hir);
    builder.module.initializer_cfg = initializer_cfg;
    let mut global_values = Vec::new();
    for hir_module in &hir.modules {
        builder
            .module
            .classes
            .extend(hir_module.classes.iter().map(|declaration| {
                crate::ClassDeclaration {
                    id: declaration.id,
                    name: declaration.name.clone(),
                    fields: declaration
                        .fields
                        .iter()
                        .map(|field| crate::ClassFieldDeclaration {
                            name: field.name.clone(),
                            ty: field.ty,
                        })
                        .collect(),
                }
            }));
        builder
            .module
            .traits
            .extend(hir_module.traits.iter().map(|declaration| {
                crate::TraitDeclaration {
                    definition: declaration.definition,
                    name: declaration.name.clone(),
                    methods: declaration
                        .methods
                        .iter()
                        .map(|method| crate::TraitMethodDeclaration {
                            name: method.name.clone(),
                            parameters: method
                                .parameters
                                .iter()
                                .map(|parameter| match parameter {
                                    severian_hir::TraitType::SelfType => crate::TraitType::SelfType,
                                    severian_hir::TraitType::Concrete(ty) => {
                                        crate::TraitType::Concrete(*ty)
                                    }
                                })
                                .collect(),
                            result: match method.result {
                                severian_hir::TraitType::SelfType => crate::TraitType::SelfType,
                                severian_hir::TraitType::Concrete(ty) => {
                                    crate::TraitType::Concrete(ty)
                                }
                            },
                        })
                        .collect(),
                }
            }));
        if hir_module.entry.is_some() {
            builder.module.entry = hir_module.entry;
        }
        builder
            .module
            .tests
            .extend(hir_module.tests.iter().map(|test| crate::TestDeclaration {
                name: test.name.clone(),
                modes: test.modes.clone(),
                function: test.function,
                expectations: test.expectations.clone(),
            }));
        let function_base = builder.module.functions.len();
        for function in &hir_module.functions {
            let parameters = function
                .parameters
                .iter()
                .map(|parameter| builder.value(parameter.contract.ty))
                .collect();
            builder.module.functions.push(Function {
                id: function.id,
                definition: function.definition,
                substitution: function.substitution.clone(),
                name: function.name.clone(),
                parameters,
                result: function.result.ty,
                body: None,
                cfg: function_cfgs.remove(&function.id),
                call_type: function.call_type.clone(),
            });
        }

        let mut initializer = Block::default();
        if hir_module.initializer.statements.is_empty() && hir_module.functions.is_empty() {
            for binding in &hir_module.bindings {
                builder.binding(binding, &mut initializer);
            }
        } else {
            for statement in &hir_module.initializer.statements {
                builder.statement(statement, hir_module, &mut initializer);
            }
        }
        builder
            .module
            .initializer
            .operations
            .extend(initializer.operations);
        let globals = builder.bindings.clone();
        let global_variables = builder.variables.clone();
        global_values.extend(globals.values().copied());

        for (index, function) in hir_module.functions.iter().enumerate() {
            let Some(hir_body) = &function.body else {
                continue;
            };
            builder.bindings.clone_from(&globals);
            builder.variables.clone_from(&global_variables);
            for (parameter, value) in function.parameters.iter().zip(
                builder.module.functions[function_base + index]
                    .parameters
                    .iter()
                    .copied(),
            ) {
                builder.bindings.insert(parameter.binding, value);
                builder.variables.insert(
                    severian_hir::VariableId(parameter.binding.0),
                    value,
                );
            }
            let mut body = Block::default();
            for statement in &hir_body.statements {
                builder.statement(statement, hir_module, &mut body);
            }
            builder.module.functions[function_base + index].body = Some(body);
        }
    }
    global_values.sort_unstable();
    global_values.dedup();
    builder.module.globals = global_values;
    crate::verify::verify_structure(&builder.module)?;
    Ok(builder.module)
}

struct Builder {
    module: Module,
    bindings: BTreeMap<BindingId, ValueId>,
    variables: BTreeMap<severian_hir::VariableId, ValueId>,
    function_ids: BTreeMap<
        (severian_universal::DefId, severian_universal::Substitution),
        severian_hir::FunctionId,
    >,
}

impl Builder {
    fn statement(
        &mut self,
        statement: &Statement,
        module: &severian_hir::Module,
        block: &mut Block,
    ) {
        let span_start = match statement {
            Statement::Sequence(_) => None,
            Statement::FieldUpdate { value, .. } | Statement::FieldSet { value, .. } => {
                Some(value.span.start)
            }
            Statement::Binding(id) => module
                .bindings
                .iter()
                .find(|binding| binding.id == *id)
                .map(|binding| binding.span.start),
            Statement::Expression(expression) => Some(expression.span.start),
            Statement::Return(Some(expression)) => Some(expression.span.start),
            Statement::Return(None) => None,
            Statement::Assert { span, .. } => Some(span.start),
            Statement::If { condition, .. }
            | Statement::Match {
                subject: condition, ..
            } => Some(condition.span.start),
            Statement::While { span, .. }
            | Statement::Break { span }
            | Statement::Continue { span } => Some(span.start),
        };
        if let Some(span_start) = span_start {
            block.operations.push(Operation::Coverage {
                point: crate::CoveragePoint {
                    span_start,
                    kind: crate::CoverageKind::Line,
                    ordinal: 0,
                    key: None,
                    file: None,
                    line: None,
                },
            });
        }
        match statement {
            Statement::Sequence(sequence) => {
                for statement in &sequence.statements {
                    self.statement(statement, module, block);
                }
            }
            Statement::FieldUpdate {
                binding,
                field,
                operator,
                value,
            } => {
                let object = self.bindings[binding];
                let object_type = self.module.values[object.0 as usize].type_id;
                let field_type = self
                    .module
                    .classes
                    .iter()
                    .find(|class| class.id == object_type)
                    .and_then(|class| class.fields.get(*field as usize))
                    .expect("typed field update references class field")
                    .ty;
                let old_field = self.value(field_type);
                block.operations.push(Operation::FieldGet {
                    object,
                    field: *field,
                    result: old_field,
                });
                let value = self.expression(value, block);
                let updated_field = self.value(field_type);
                block.operations.push(Operation::Binary {
                    operator: *operator,
                    left: old_field,
                    right: value,
                    result: updated_field,
                });
                let updated_object = self.value(object_type);
                block.operations.push(Operation::FieldSet {
                    object,
                    field: *field,
                    value: updated_field,
                    result: updated_object,
                });
                block.operations.push(Operation::Assign {
                    target: object,
                    value: updated_object,
                });
            }
            Statement::FieldSet {
                binding,
                field,
                value,
            } => {
                let object = self.bindings[binding];
                let object_type = self.module.values[object.0 as usize].type_id;
                let value = self.expression(value, block);
                let updated_object = self.value(object_type);
                block.operations.push(Operation::FieldSet {
                    object,
                    field: *field,
                    value,
                    result: updated_object,
                });
                block.operations.push(Operation::Assign {
                    target: object,
                    value: updated_object,
                });
            }
            Statement::Binding(id) => {
                let binding = module
                    .bindings
                    .iter()
                    .find(|binding| binding.id == *id)
                    .expect("HIR statement references a binding");
                self.binding(binding, block);
            }
            Statement::Expression(expression) => {
                self.expression(expression, block);
            }
            Statement::Return(value) => {
                let value = value.as_ref().map(|value| self.expression(value, block));
                block.operations.push(Operation::Return { value });
            }
            Statement::Assert {
                condition,
                message,
                span,
                condition_span,
            } => {
                let condition = self.expression(condition, block);
                let message = message
                    .as_ref()
                    .map(|message| self.expression(message, block));
                block.operations.push(Operation::Assert {
                    condition,
                    message,
                    origin: crate::AssertionOrigin {
                        statement_start: span.start,
                        condition_start: condition_span.start,
                        condition_end: condition_span.end,
                        location: None,
                    },
                });
            }
            Statement::If {
                condition,
                then_block,
                else_block,
            } => {
                let condition_span = condition.span.start;
                let condition = self.expression(condition, block);
                let outer_bindings = self.bindings.clone();
                let outer_variables = self.variables.clone();
                let mut then_mir = Block::default();
                then_mir.operations.push(Operation::Coverage {
                    point: crate::CoveragePoint {
                        span_start: condition_span,
                        kind: crate::CoverageKind::Branch,
                        ordinal: 0,
                        key: None,
                        file: None,
                        line: None,
                    },
                });
                for statement in &then_block.statements {
                    self.statement(statement, module, &mut then_mir);
                }
                self.bindings.clone_from(&outer_bindings);
                self.variables.clone_from(&outer_variables);
                let mut else_mir = Block::default();
                else_mir.operations.push(Operation::Coverage {
                    point: crate::CoveragePoint {
                        span_start: condition_span,
                        kind: crate::CoverageKind::Branch,
                        ordinal: 1,
                        key: None,
                        file: None,
                        line: None,
                    },
                });
                for statement in &else_block.statements {
                    self.statement(statement, module, &mut else_mir);
                }
                self.bindings = outer_bindings;
                self.variables = outer_variables;
                block.operations.push(Operation::If {
                    condition,
                    then_block: then_mir,
                    else_block: else_mir,
                });
            }
            Statement::While {
                condition, body, ..
            } => {
                let mut condition_block = Block::default();
                let condition = self.expression(condition, &mut condition_block);
                let outer_bindings = self.bindings.clone();
                let outer_variables = self.variables.clone();
                let mut body_mir = Block::default();
                for statement in &body.statements {
                    self.statement(statement, module, &mut body_mir);
                }
                self.bindings = outer_bindings;
                self.variables = outer_variables;
                block.operations.push(Operation::While {
                    condition_block,
                    condition,
                    body: body_mir,
                });
            }
            Statement::Break { .. } => block.operations.push(Operation::Break),
            Statement::Continue { .. } => block.operations.push(Operation::Continue),
            Statement::Match { subject, arms } => {
                let subject_span = subject.span.start;
                let subject = self.expression(subject, block);
                let outer_bindings = self.bindings.clone();
                let outer_variables = self.variables.clone();
                let mut mir_arms = Vec::new();
                for (ordinal, arm) in arms.iter().enumerate() {
                    self.bindings.clone_from(&outer_bindings);
                    self.variables.clone_from(&outer_variables);
                    if let Some(binding) = arm.binding {
                        self.bindings.insert(binding, subject);
                        self.module.bindings.push((binding, subject));
                    }
                    let mut body = Block::default();
                    body.operations.push(Operation::Coverage {
                        point: crate::CoveragePoint {
                            span_start: subject_span,
                            kind: crate::CoverageKind::Branch,
                            ordinal: ordinal as u32,
                            key: None,
                            file: None,
                            line: None,
                        },
                    });
                    for statement in &arm.body.statements {
                        self.statement(statement, module, &mut body);
                    }
                    mir_arms.push(crate::MatchArm {
                        type_id: arm.type_id,
                        body,
                    });
                }
                self.bindings = outer_bindings;
                self.variables = outer_variables;
                block.operations.push(Operation::Match {
                    subject,
                    arms: mir_arms,
                });
            }
            // Loops and their control transfers are represented by the CFG
            // body built above. This legacy structured body has no edge model;
            // duplicating loop lowering here would recreate the old split
            // semantics rather than preserving a single control-flow source.
        }
    }

    fn binding(&mut self, binding: &severian_hir::Binding, block: &mut Block) {
        let value = self.expression(&binding.value, block);
        let target = if let Some(target) = self.variables.get(&binding.variable).copied() {
            block.operations.push(Operation::Assign { target, value });
            target
        } else {
            self.variables.insert(binding.variable, value);
            value
        };
        self.bindings.insert(binding.id, target);
        self.module.bindings.push((binding.id, target));
    }

    fn value(&mut self, type_id: severian_universal::TypeId) -> ValueId {
        let id = ValueId(self.module.values.len() as u32);
        self.module.values.push(Value { id, type_id });
        id
    }

    fn expression(&mut self, expression: &Expression, block: &mut Block) -> ValueId {
        match &expression.kind {
            ExpressionKind::Aggregate { class, fields } => {
                let fields = fields
                    .iter()
                    .map(|field| self.expression(field, block))
                    .collect();
                let result = self.value(expression.type_id);
                block.operations.push(Operation::Aggregate {
                    class: *class,
                    fields,
                    result,
                });
                result
            }
            ExpressionKind::Field { object, index } => {
                let object = self.expression(object, block);
                let result = self.value(expression.type_id);
                block.operations.push(Operation::FieldGet {
                    object,
                    field: *index,
                    result,
                });
                result
            }
            ExpressionKind::Literal(value) => {
                let result = self.value(expression.type_id);
                block.operations.push(Operation::Constant {
                    value: value.clone(),
                    result,
                });
                result
            }
            ExpressionKind::Binding(binding) => self.bindings[binding],
            ExpressionKind::Call { callee, arguments } => {
                let arguments = arguments
                    .iter()
                    .map(|argument| self.expression(argument, block))
                    .collect();
                let result = self.value(expression.type_id);
                let severian_hir::Callee::Direct {
                    function,
                    substitution,
                } = callee
                else {
                    panic!("non-direct calls lower through CFG MIR")
                };
                block.operations.push(Operation::Call {
                    function: self.function_ids[&(*function, substitution.clone())],
                    arguments,
                    result,
                });
                result
            }
            ExpressionKind::Async {
                expression: task,
                owner,
                locked,
            } => {
                let ExpressionKind::Call { callee, arguments } = &task.kind else {
                    panic!("async expressions are required to contain a call")
                };
                let arguments = arguments
                    .iter()
                    .map(|argument| self.expression(argument, block))
                    .collect();
                let severian_hir::Callee::Direct {
                    function,
                    substitution,
                } = callee
                else {
                    panic!("non-direct async calls lower through CFG MIR")
                };
                let result = self.value(expression.type_id);
                block.operations.push(Operation::Spawn {
                    function: self.function_ids[&(*function, substitution.clone())],
                    arguments,
                    result,
                    owner: *owner,
                    locked: *locked,
                });
                result
            }
            ExpressionKind::Await(task) => {
                let task = self.expression(task, block);
                let result = self.value(expression.type_id);
                block.operations.push(Operation::Await { task, result });
                result
            }
            ExpressionKind::Convert { operand, .. } => self.expression(operand, block),
            ExpressionKind::Borrow { operand, .. } | ExpressionKind::Move(operand) => {
                self.expression(operand, block)
            }
            ExpressionKind::Function(_) => {
                panic!("function values lower through CFG MIR")
            }
            ExpressionKind::Unary { operator, operand } => {
                let operand = self.expression(operand, block);
                let result = self.value(expression.type_id);
                block.operations.push(Operation::Unary {
                    operator: *operator,
                    operand,
                    result,
                });
                result
            }
            ExpressionKind::Binary {
                operator,
                left,
                right,
            } => {
                let left = self.expression(left, block);
                let right = self.expression(right, block);
                let result = self.value(expression.type_id);
                block.operations.push(Operation::Binary {
                    operator: *operator,
                    left,
                    right,
                    result,
                });
                result
            }
        }
    }
}
