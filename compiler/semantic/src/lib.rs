#![forbid(unsafe_code)]

use severian_ast::{Expr, Item, Literal, Module, Span, Stmt};
use severian_hir::{Function, Instruction, Program};
use std::fmt;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SemanticError {
    pub span: Span,
    pub message: String,
}

impl fmt::Display for SemanticError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "{} at bytes {}..{}",
            self.message, self.span.start, self.span.end
        )
    }
}

impl std::error::Error for SemanticError {}

pub fn analyze(module: &Module) -> Result<Program, SemanticError> {
    let mut functions = Vec::new();

    for item in &module.items {
        let Item::Function(function) = item else {
            return Err(SemanticError {
                span: item.span(),
                message: "only functions are supported in the first compiler slice".into(),
            });
        };

        if !is_lower_camel_case(&function.name.name) {
            return Err(SemanticError {
                span: function.name.span,
                message: format!("function `{}` must use lowerCamelCase", function.name.name),
            });
        }

        if !function.params.is_empty() || function.return_type.is_some() {
            return Err(SemanticError {
                span: function.span,
                message: "function parameters and return types are not supported yet".into(),
            });
        }

        let mut instructions = Vec::new();
        for statement in &function.body.statements {
            instructions.push(analyze_statement(statement)?);
        }

        functions.push(Function {
            name: function.name.name.clone(),
            instructions,
        });
    }

    let program = Program { functions };
    if program.main().is_none() {
        return Err(SemanticError {
            span: module.span,
            message: "program must define `main`".into(),
        });
    }

    Ok(program)
}

fn is_lower_camel_case(name: &str) -> bool {
    name.as_bytes().first().is_some_and(u8::is_ascii_lowercase)
        && name.bytes().all(|byte| byte.is_ascii_alphanumeric())
}

fn analyze_statement(statement: &Stmt) -> Result<Instruction, SemanticError> {
    let Stmt::Expr(Expr::Call(call)) = statement else {
        return Err(SemanticError {
            span: statement.span(),
            message: "only calls to `print` are supported in function bodies yet".into(),
        });
    };

    let Expr::Identifier(callee) = call.callee.as_ref() else {
        return Err(SemanticError {
            span: call.callee.span(),
            message: "call target must be a function name".into(),
        });
    };

    if callee.name != "print" {
        return Err(SemanticError {
            span: callee.span,
            message: format!("unknown function `{}`", callee.name),
        });
    }

    if call.args.len() != 1 {
        return Err(SemanticError {
            span: call.span,
            message: "`print` currently expects exactly one argument".into(),
        });
    }

    match &call.args[0].value {
        Expr::Literal(Literal::String { value, .. }) => Ok(Instruction::Print(value.clone())),
        value => Err(SemanticError {
            span: value.span(),
            message: "`print` currently expects a string literal".into(),
        }),
    }
}
