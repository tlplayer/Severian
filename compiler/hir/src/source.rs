use severian_source::SourceSpan;
use std::collections::BTreeMap;

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum HirSourceKey {
    Program,
    Global(String),
    Class(String),
    ClassField { class: String, field: String },
    Function(String),
    Parameter { function: String, parameter: String },
    Instruction { function: String, path: Vec<usize> },
    Expression {
        function: String,
        instruction_path: Vec<usize>,
        expression_path: Vec<usize>,
    },
    Test { function: String, test_index: usize },
}

#[derive(Debug, Clone, Default)]
pub struct HirSourceMap {
    entries: BTreeMap<HirSourceKey, SourceSpan>,
}

impl HirSourceMap {
    pub fn new() -> Self { Self::default() }

    pub fn insert(
        &mut self,
        key: HirSourceKey,
        span: SourceSpan,
    ) -> Option<SourceSpan> {
        self.entries.insert(key, span)
    }

    pub fn get(&self, key: &HirSourceKey) -> Option<SourceSpan> {
        self.entries.get(key).copied()
    }

    pub fn function(&self, function: &str) -> Option<SourceSpan> {
        self.get(&HirSourceKey::Function(function.to_owned()))
    }

    pub fn instruction(
        &self,
        function: &str,
        path: &[usize],
    ) -> Option<SourceSpan> {
        self.get(&HirSourceKey::Instruction {
            function: function.to_owned(),
            path: path.to_vec(),
        })
    }

    pub fn expression(
        &self,
        function: &str,
        instruction_path: &[usize],
        expression_path: &[usize],
    ) -> Option<SourceSpan> {
        self.get(&HirSourceKey::Expression {
            function: function.to_owned(),
            instruction_path: instruction_path.to_vec(),
            expression_path: expression_path.to_vec(),
        })
    }

    pub fn entries(
        &self,
    ) -> impl Iterator<Item = (&HirSourceKey, &SourceSpan)> {
        self.entries.iter()
    }

    pub fn merge_from(&mut self, other: &Self) {
        for (key, span) in &other.entries {
            self.entries.insert(key.clone(), *span);
        }
    }
}
