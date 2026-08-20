use crate::{Module, Operation, Value, ValueId};
use severian_hir::{Expression, ExpressionKind, HirId, Program as HirProgram};
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
    bindings: BTreeMap<HirId, ValueId>,
}

impl Builder {
    fn value(&mut self, type_id: severian_hir::TypeId) -> ValueId {
        let id = ValueId(self.module.values.len() as u32);
        self.module.values.push(Value { id, type_id });
        id
    }

    fn expression(&mut self, expression: &Expression) -> ValueId {
        match &expression.kind {
            ExpressionKind::Integer(value) => {
                let result = self.value(expression.type_id);
                self.module.operations.push(Operation::ConstantInt {
                    value: *value,
                    result,
                });
                result
            }
            ExpressionKind::Binding(binding) => self.bindings[binding],
            ExpressionKind::Add { left, right } => {
                let left = self.expression(left);
                let right = self.expression(right);
                let result = self.value(expression.type_id);
                self.module.operations.push(Operation::AddInt {
                    left,
                    right,
                    result,
                });
                result
            }
        }
    }
}
