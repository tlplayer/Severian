use crate::{Module, Operation, Value, ValueId};
use severian_hir::{BindingId, Expression, ExpressionKind, Program as HirProgram};
use std::collections::BTreeMap;

pub fn build(hir: &HirProgram) -> Module {
    let mut builder = Builder {
        module: Module::default(),
        bindings: BTreeMap::new(),
    };
    for hir_module in &hir.modules {
        for binding in &hir_module.bindings {
            let value = builder.expression(&binding.value);
            builder.bindings.insert(binding.id, value);
            builder.module.bindings.push((binding.id, value));
        }
    }
    builder.module
}

struct Builder {
    module: Module,
    bindings: BTreeMap<BindingId, ValueId>,
}

impl Builder {
    fn value(&mut self, type_id: severian_universal::TypeId) -> ValueId {
        let id = ValueId(self.module.values.len() as u32);
        self.module.values.push(Value { id, type_id });
        id
    }

    fn expression(&mut self, expression: &Expression) -> ValueId {
        match &expression.kind {
            ExpressionKind::Literal(value) => {
                let result = self.value(expression.type_id);
                self.module.operations.push(Operation::Constant {
                    value: value.clone(),
                    result,
                });
                result
            }
            ExpressionKind::Binding(binding) => self.bindings[binding],
            ExpressionKind::Unary { operator, operand } => {
                let operand = self.expression(operand);
                let result = self.value(expression.type_id);
                self.module.operations.push(Operation::Unary {
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
                let left = self.expression(left);
                let right = self.expression(right);
                let result = self.value(expression.type_id);
                self.module.operations.push(Operation::Binary {
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
