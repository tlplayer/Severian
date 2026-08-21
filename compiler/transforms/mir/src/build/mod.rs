use crate::{Block, Function, Module, Operation, Value, ValueId};
use severian_hir::{BindingId, Expression, ExpressionKind, Program as HirProgram, Statement};
use std::collections::BTreeMap;

pub fn build(hir: &HirProgram) -> Module {
    let mut builder = Builder {
        module: Module::default(),
        bindings: BTreeMap::new(),
    };
    for hir_module in &hir.modules {
        builder.module.entry = hir_module.entry;
        builder
            .module
            .tests
            .extend(hir_module.tests.iter().map(|test| crate::TestDeclaration {
                name: test.name.clone(),
                modes: test.modes.clone(),
                function: test.function,
                expectations: test.expectations.clone(),
            }));
        for function in &hir_module.functions {
            let parameters = function
                .parameters
                .iter()
                .map(|parameter| {
                    let severian_hir::SemanticType::Universal(type_id) = parameter.contract.ty
                    else {
                        panic!("declared source types must be lowered before MIR")
                    };
                    builder.value(type_id)
                })
                .collect();
            builder.module.functions.push(Function {
                id: function.id,
                name: function.name.clone(),
                parameters,
                result: match function.result.ty {
                    severian_hir::SemanticType::Universal(type_id) => type_id,
                    severian_hir::SemanticType::Declared(_) => {
                        panic!("declared result types must be lowered before MIR")
                    }
                },
                body: None,
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
        builder.module.initializer = initializer;
        builder.module.globals = builder
            .module
            .bindings
            .iter()
            .map(|(_, value)| *value)
            .collect();
        let globals = builder.bindings.clone();

        for (index, function) in hir_module.functions.iter().enumerate() {
            let Some(hir_body) = &function.body else {
                continue;
            };
            builder.bindings.clone_from(&globals);
            for (parameter, value) in function
                .parameters
                .iter()
                .zip(builder.module.functions[index].parameters.iter().copied())
            {
                builder.bindings.insert(parameter.binding, value);
            }
            let mut body = Block::default();
            for statement in &hir_body.statements {
                builder.statement(statement, hir_module, &mut body);
            }
            builder.module.functions[index].body = Some(body);
        }
    }
    builder.module
}

struct Builder {
    module: Module,
    bindings: BTreeMap<BindingId, ValueId>,
}

impl Builder {
    fn statement(
        &mut self,
        statement: &Statement,
        module: &severian_hir::Module,
        block: &mut Block,
    ) {
        match statement {
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
                let condition = self.expression(condition, block);
                let outer_bindings = self.bindings.clone();
                let mut then_mir = Block::default();
                for statement in &then_block.statements {
                    self.statement(statement, module, &mut then_mir);
                }
                self.bindings.clone_from(&outer_bindings);
                let mut else_mir = Block::default();
                for statement in &else_block.statements {
                    self.statement(statement, module, &mut else_mir);
                }
                self.bindings = outer_bindings;
                block.operations.push(Operation::If {
                    condition,
                    then_block: then_mir,
                    else_block: else_mir,
                });
            }
        }
    }

    fn binding(&mut self, binding: &severian_hir::Binding, block: &mut Block) {
        let value = self.expression(&binding.value, block);
        self.bindings.insert(binding.id, value);
        self.module.bindings.push((binding.id, value));
    }

    fn value(&mut self, type_id: severian_universal::TypeId) -> ValueId {
        let id = ValueId(self.module.values.len() as u32);
        self.module.values.push(Value { id, type_id });
        id
    }

    fn expression(&mut self, expression: &Expression, block: &mut Block) -> ValueId {
        match &expression.kind {
            ExpressionKind::Literal(value) => {
                let result = self.value(expression.type_id);
                block.operations.push(Operation::Constant {
                    value: value.clone(),
                    result,
                });
                result
            }
            ExpressionKind::Binding(binding) => self.bindings[binding],
            ExpressionKind::Call {
                function,
                arguments,
            } => {
                let arguments = arguments
                    .iter()
                    .map(|argument| self.expression(argument, block))
                    .collect();
                let result = self.value(expression.type_id);
                block.operations.push(Operation::Call {
                    function: *function,
                    arguments,
                    result,
                });
                result
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
