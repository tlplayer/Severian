use super::{default_layout, Layout};
use severian_hir::{Function, ValueType};
use std::collections::HashMap;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LayoutAssignment {
    pub value: String,
    pub layout: Layout,
}

#[derive(Debug, Clone, Default)]
pub struct LayoutPlan {
    pub assignments: Vec<LayoutAssignment>,
    pub by_value: HashMap<String, Layout>,
}

impl LayoutPlan {
    pub fn layout_of(&self, value: &str) -> Option<&Layout> {
        self.by_value.get(value)
    }

    fn insert(&mut self, value: String, layout: Layout) {
        self.by_value.insert(value.clone(), layout.clone());
        self.assignments.push(LayoutAssignment { value, layout });
    }
}

pub fn assign_function_layouts(function: &Function) -> LayoutPlan {
    let mut plan = LayoutPlan::default();

    for parameter in &function.params {
        if let ValueType::Tensor(tensor) = parameter.ty {
            if let Some(layout) = default_layout(tensor) {
                plan.insert(parameter.name.clone(), layout);
            }
        }
    }

    if let ValueType::Tensor(tensor) = function.return_type {
        if let Some(layout) = default_layout(tensor) {
            plan.insert("$return".to_string(), layout);
        }
    }

    plan
}
