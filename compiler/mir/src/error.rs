use severian_hir::{HirId, TensorIntrinsic};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MirLoweringError {
    pub function: Option<String>,
    pub expression: Option<HirId>,
    pub intrinsic: TensorIntrinsic,
    pub message: String,
}

impl MirLoweringError {
    pub(crate) fn tensor(intrinsic: TensorIntrinsic, message: impl Into<String>) -> Self {
        Self {
            function: None,
            expression: None,
            intrinsic,
            message: message.into(),
        }
    }

    pub(crate) fn at_expression(mut self, expression: Option<HirId>) -> Self {
        if self.expression.is_none() {
            self.expression = expression;
        }
        self
    }

    pub(crate) fn in_function(mut self, function: impl Into<String>) -> Self {
        if self.function.is_none() {
            self.function = Some(function.into());
        }
        self
    }
}

impl std::fmt::Display for MirLoweringError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            formatter,
            "could not construct MIR tensor operation `{}`",
            self.intrinsic.name()
        )?;
        if let Some(function) = &self.function {
            write!(formatter, " in function `{function}`")?;
        }
        if let Some(expression) = self.expression {
            write!(formatter, " at HIR expression {expression:?}")?;
        }
        write!(formatter, ": {}", self.message)
    }
}

impl std::error::Error for MirLoweringError {}
