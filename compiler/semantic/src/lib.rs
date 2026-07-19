#![forbid(unsafe_code)]

use severian_ast::{
    BinaryOp as AstBinaryOp, Block, ElseBranch, Expr, Item, Literal, Module, Span, Stmt, Type,
};
use severian_hir::{
    BinaryOp, Expression, Function, Instruction, Parameter, Program, Test, ValueType,
};
use std::collections::HashMap;
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

#[derive(Clone)]
struct Signature {
    params: Vec<ValueType>,
    returns: ValueType,
}

pub fn analyze(module: &Module) -> Result<Program, SemanticError> {
    let mut signatures = HashMap::new();
    for item in &module.items {
        let Item::Function(function) = item else {
            return Err(error(
                item.span(),
                "only functions are supported in this compiler slice",
            ));
        };
        if !is_lower_camel_case(&function.name.name) {
            return Err(error(
                function.name.span,
                format!("function `{}` must use lowerCamelCase", function.name.name),
            ));
        }
        let params = function
            .params
            .iter()
            .map(|param| {
                param
                    .ty
                    .as_ref()
                    .ok_or_else(|| error(param.span, "parameters require a type"))
                    .and_then(lower_type)
            })
            .collect::<Result<Vec<_>, _>>()?;
        let returns = function
            .return_type
            .as_ref()
            .map(lower_type)
            .transpose()?
            .unwrap_or(ValueType::Unit);
        if signatures
            .insert(function.name.name.clone(), Signature { params, returns })
            .is_some()
        {
            return Err(error(
                function.name.span,
                format!("duplicate function `{}`", function.name.name),
            ));
        }
    }

    let mut functions = Vec::new();
    for item in &module.items {
        let Item::Function(function) = item else {
            unreachable!()
        };
        let signature = signatures.get(&function.name.name).unwrap();
        let params = function
            .params
            .iter()
            .zip(&signature.params)
            .map(|(param, ty)| Parameter {
                name: param.name.name.clone(),
                ty: *ty,
            })
            .collect::<Vec<_>>();
        let mut scope = params
            .iter()
            .map(|param| (param.name.clone(), param.ty))
            .collect();
        let instructions = lower_block(&function.body, &mut scope, signature.returns, &signatures)?;
        if signature.returns != ValueType::Unit && !always_returns(&instructions) {
            return Err(error(
                function.span,
                format!("function `{}` must return a value", function.name.name),
            ));
        }

        let mut tests = Vec::new();
        for test in &function.tests {
            let mut test_scope = HashMap::new();
            tests.push(Test {
                name: test.name.as_ref().map(|name| name.name.clone()),
                instructions: lower_block(
                    &test.body,
                    &mut test_scope,
                    ValueType::Unit,
                    &signatures,
                )?,
            });
        }
        functions.push(Function {
            name: function.name.name.clone(),
            params,
            return_type: signature.returns,
            instructions,
            tests,
        });
    }
    Ok(Program { functions })
}

fn lower_block(
    block: &Block,
    scope: &mut HashMap<String, ValueType>,
    return_type: ValueType,
    signatures: &HashMap<String, Signature>,
) -> Result<Vec<Instruction>, SemanticError> {
    let mut instructions = Vec::new();
    for statement in &block.statements {
        match statement {
            Stmt::Let(binding) => {
                let value = binding
                    .value
                    .as_ref()
                    .ok_or_else(|| error(binding.span, "binding requires a value"))?;
                let (value, ty) = lower_expression(value, scope, signatures)?;
                if scope.insert(binding.name.name.clone(), ty).is_some() {
                    return Err(error(
                        binding.name.span,
                        format!("duplicate binding `{}`", binding.name.name),
                    ));
                }
                instructions.push(Instruction::Let {
                    name: binding.name.name.clone(),
                    value,
                });
            }
            Stmt::Expr(expression) => {
                let (expression, _) = lower_expression(expression, scope, signatures)?;
                if let Expression::Call { function, args } = expression {
                    if function == "print" {
                        instructions.push(Instruction::Print(args.into_iter().next().unwrap()));
                    } else {
                        instructions
                            .push(Instruction::Evaluate(Expression::Call { function, args }));
                    }
                } else {
                    instructions.push(Instruction::Evaluate(expression));
                }
            }
            Stmt::Assert(assertion) => {
                let (condition, ty) = lower_expression(&assertion.condition, scope, signatures)?;
                require_type(assertion.condition.span(), ty, ValueType::Bool)?;
                instructions.push(Instruction::Assert(condition));
            }
            Stmt::Return(return_statement) => {
                let value = return_statement
                    .value
                    .as_ref()
                    .map(|value| lower_expression(value, scope, signatures))
                    .transpose()?;
                let actual = value.as_ref().map(|(_, ty)| *ty).unwrap_or(ValueType::Unit);
                require_type(return_statement.span, actual, return_type)?;
                instructions.push(Instruction::Return(value.map(|(expression, _)| expression)));
            }
            Stmt::If(if_statement) => {
                let (condition, ty) = lower_expression(&if_statement.condition, scope, signatures)?;
                require_type(if_statement.condition.span(), ty, ValueType::Bool)?;
                let mut then_scope = scope.clone();
                let then_instructions = lower_block(
                    &if_statement.then_block,
                    &mut then_scope,
                    return_type,
                    signatures,
                )?;
                let else_instructions = match &if_statement.else_branch {
                    None => Vec::new(),
                    Some(ElseBranch::Block(block)) => {
                        let mut else_scope = scope.clone();
                        lower_block(block, &mut else_scope, return_type, signatures)?
                    }
                    Some(ElseBranch::If(_)) => {
                        return Err(error(if_statement.span, "else-if is not supported yet"))
                    }
                };
                instructions.push(Instruction::If {
                    condition,
                    then_instructions,
                    else_instructions,
                });
            }
            _ => {
                return Err(error(
                    statement.span(),
                    "statement is not supported in this compiler slice yet",
                ))
            }
        }
    }
    Ok(instructions)
}

fn lower_expression(
    expression: &Expr,
    scope: &HashMap<String, ValueType>,
    signatures: &HashMap<String, Signature>,
) -> Result<(Expression, ValueType), SemanticError> {
    match expression {
        Expr::Literal(Literal::Integer { value, .. }) => {
            Ok((Expression::Integer(*value), ValueType::Int))
        }
        Expr::Literal(Literal::Boolean { value, .. }) => {
            Ok((Expression::Boolean(*value), ValueType::Bool))
        }
        Expr::Literal(Literal::String { value, .. }) => {
            Ok((Expression::String(value.clone()), ValueType::String))
        }
        Expr::Identifier(identifier) => scope
            .get(&identifier.name)
            .copied()
            .map(|ty| (Expression::Variable(identifier.name.clone()), ty))
            .ok_or_else(|| {
                error(
                    identifier.span,
                    format!("unknown binding `{}`", identifier.name),
                )
            }),
        Expr::Binary(binary) => {
            let (left, left_type) = lower_expression(&binary.left, scope, signatures)?;
            let (right, right_type) = lower_expression(&binary.right, scope, signatures)?;
            if left_type != right_type {
                return Err(error(
                    binary.span,
                    "binary operands must have the same type",
                ));
            }
            let (op, result_type) = match binary.op {
                AstBinaryOp::Add => (BinaryOp::Add, ValueType::Int),
                AstBinaryOp::Sub => (BinaryOp::Sub, ValueType::Int),
                AstBinaryOp::Mul => (BinaryOp::Mul, ValueType::Int),
                AstBinaryOp::Div => (BinaryOp::Div, ValueType::Int),
                AstBinaryOp::Mod => (BinaryOp::Mod, ValueType::Int),
                AstBinaryOp::Equal => (BinaryOp::Equal, ValueType::Bool),
                AstBinaryOp::NotEqual => (BinaryOp::NotEqual, ValueType::Bool),
                AstBinaryOp::Less => (BinaryOp::Less, ValueType::Bool),
                AstBinaryOp::LessEqual => (BinaryOp::LessEqual, ValueType::Bool),
                AstBinaryOp::Greater => (BinaryOp::Greater, ValueType::Bool),
                AstBinaryOp::GreaterEqual => (BinaryOp::GreaterEqual, ValueType::Bool),
                _ => return Err(error(binary.span, "operator is not supported yet")),
            };
            if !matches!(binary.op, AstBinaryOp::Equal | AstBinaryOp::NotEqual) {
                require_type(binary.span, left_type, ValueType::Int)?;
            }
            Ok((
                Expression::Binary {
                    left: Box::new(left),
                    op,
                    right: Box::new(right),
                },
                result_type,
            ))
        }
        Expr::Call(call) => {
            let Expr::Identifier(callee) = call.callee.as_ref() else {
                return Err(error(
                    call.callee.span(),
                    "call target must be a function name",
                ));
            };
            if callee.name == "print" {
                if call.args.len() != 1 {
                    return Err(error(call.span, "`print` expects exactly one argument"));
                }
                let (argument, _) = lower_expression(&call.args[0].value, scope, signatures)?;
                return Ok((
                    Expression::Call {
                        function: "print".into(),
                        args: vec![argument],
                    },
                    ValueType::Unit,
                ));
            }
            let signature = signatures
                .get(&callee.name)
                .ok_or_else(|| error(callee.span, format!("unknown function `{}`", callee.name)))?;
            if call.args.len() != signature.params.len() {
                return Err(error(
                    call.span,
                    format!(
                        "`{}` expects {} arguments",
                        callee.name,
                        signature.params.len()
                    ),
                ));
            }
            let mut args = Vec::new();
            for (argument, expected) in call.args.iter().zip(&signature.params) {
                let (argument, actual) = lower_expression(&argument.value, scope, signatures)?;
                require_type(call.span, actual, *expected)?;
                args.push(argument);
            }
            Ok((
                Expression::Call {
                    function: callee.name.clone(),
                    args,
                },
                signature.returns,
            ))
        }
        _ => Err(error(
            expression.span(),
            "expression is not supported in this compiler slice yet",
        )),
    }
}

fn lower_type(ty: &Type) -> Result<ValueType, SemanticError> {
    let Type::Named(path) = ty else {
        return Err(error(ty.span(), "type is not supported yet"));
    };
    let name = path
        .segments
        .first()
        .map(|segment| segment.name.as_str())
        .unwrap_or("");
    match name {
        "int" => Ok(ValueType::Int),
        "bool" => Ok(ValueType::Bool),
        "string" => Ok(ValueType::String),
        "unit" => Ok(ValueType::Unit),
        _ => Err(error(path.span, format!("unknown type `{name}`"))),
    }
}

fn require_type(span: Span, actual: ValueType, expected: ValueType) -> Result<(), SemanticError> {
    if actual == expected {
        Ok(())
    } else {
        Err(error(
            span,
            format!("expected {expected:?}, found {actual:?}"),
        ))
    }
}

fn always_returns(instructions: &[Instruction]) -> bool {
    instructions.iter().any(|instruction| match instruction {
        Instruction::Return(_) => true,
        Instruction::If {
            then_instructions,
            else_instructions,
            ..
        } => always_returns(then_instructions) && always_returns(else_instructions),
        _ => false,
    })
}

fn is_lower_camel_case(name: &str) -> bool {
    name.as_bytes().first().is_some_and(u8::is_ascii_lowercase)
        && name.bytes().all(|byte| byte.is_ascii_alphanumeric())
}

fn error(span: Span, message: impl Into<String>) -> SemanticError {
    SemanticError {
        span,
        message: message.into(),
    }
}
