#![forbid(unsafe_code)]

use severian_ast::{
    AssignOp as AstAssignOp, BinaryOp as AstBinaryOp, Block, ElseBranch, Expr, ImportKind, Item,
    LetKind, Literal, Module, OwnershipOp as AstOwnershipOp, Pattern, Span, Stmt, Type,
    UnaryOp as AstUnaryOp,
};
use severian_hir::{
    AssignmentOp, BinaryOp, ChaosAction as HirChaosAction, Class, Expression, Function, Global,
    Instruction, MatchPattern, Parameter, Program, SwitchArm as HirSwitchArm, Test,
    TestMode as HirTestMode, UnaryOp, ValueType,
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
    params: Vec<SignatureParameter>,
    returns: ValueType,
}

#[derive(Clone)]
struct SignatureParameter {
    name: String,
    ty: ValueType,
    default: Option<Expr>,
}

#[derive(Clone, Copy)]
struct Binding {
    ty: ValueType,
    mutable: bool,
    field: bool,
}

pub fn analyze(module: &Module) -> Result<Program, SemanticError> {
    let mut aliases = collect_imports(module);
    for item in &module.items {
        if let Item::Class(class) = item {
            aliases.insert(
                class.name.name.clone(),
                format!("__class.{}", class.name.name),
            );
        }
    }
    let mut signatures = HashMap::new();
    for item in &module.items {
        let Item::Function(function) = item else {
            continue;
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
                Ok(SignatureParameter {
                    name: param.name.name.clone(),
                    ty: param
                        .ty
                        .as_ref()
                        .ok_or_else(|| error(param.span, "parameters require a type"))
                        .and_then(lower_type)?,
                    default: param.default.clone(),
                })
            })
            .collect::<Result<Vec<_>, SemanticError>>()?;
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

    let mut global_scope = HashMap::new();
    let mut globals = Vec::new();
    for item in &module.items {
        if let Item::Statement(Stmt::Let(binding)) = item {
            let declared = binding.ty.as_ref().map(lower_type).transpose()?;
            let source = binding
                .value
                .as_ref()
                .ok_or_else(|| error(binding.span, "global requires a value"))?;
            let (value, inferred) = lower_expression(source, &global_scope, &signatures, &aliases)?;
            if let Some(declared) = declared {
                compatible(binding.span, inferred, declared)?;
            }
            let ty = declared.unwrap_or(inferred);
            global_scope.insert(
                binding.name.name.clone(),
                Binding {
                    ty,
                    mutable: false,
                    field: false,
                },
            );
            globals.push(Global {
                name: binding.name.name.clone(),
                value,
            });
        }
    }

    let mut functions = Vec::new();
    for item in &module.items {
        let Item::Function(function) = item else {
            continue;
        };
        let signature = signatures.get(&function.name.name).unwrap();
        let mut scope = global_scope.clone();
        let mut params = Vec::new();
        for parameter in &signature.params {
            let default = parameter
                .default
                .as_ref()
                .map(|value| {
                    let (value, ty) = lower_expression(value, &scope, &signatures, &aliases)?;
                    compatible(value_span(&parameter.default), ty, parameter.ty)?;
                    Ok(value)
                })
                .transpose()?;
            scope.insert(
                parameter.name.clone(),
                Binding {
                    ty: parameter.ty,
                    mutable: false,
                    field: false,
                },
            );
            params.push(Parameter {
                name: parameter.name.clone(),
                ty: parameter.ty,
                default,
            });
        }
        let instructions = lower_block(
            &function.body,
            &mut scope,
            signature.returns,
            &signatures,
            &aliases,
        )?;
        if signature.returns != ValueType::Unit && !always_returns(&instructions) {
            return Err(error(
                function.span,
                format!("function `{}` must return a value", function.name.name),
            ));
        }
        let mut tests = Vec::new();
        for test in &function.tests {
            let mut test_scope = global_scope.clone();
            add_test_bindings(&mut test_scope);
            tests.push(Test {
                name: test.name.as_ref().map(|name| name.name.clone()),
                modes: lower_test_modes(&test.modes),
                instructions: lower_block(
                    &test.body,
                    &mut test_scope,
                    ValueType::Unit,
                    &signatures,
                    &aliases,
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
    let mut classes = Vec::new();
    for item in &module.items {
        let Item::Class(class) = item else { continue };
        let fields = class
            .fields
            .iter()
            .map(|field| field.name.name.clone())
            .collect::<Vec<_>>();
        let mut constructors = Vec::new();
        for constructor in &class.constructors {
            constructors.push(lower_class_function(
                &class.name.name,
                &fields,
                &constructor.name.name,
                &constructor.params,
                &constructor.body,
                &constructor.tests,
                ValueType::Unit,
                &global_scope,
                &signatures,
                &aliases,
            )?);
        }
        let mut methods = Vec::new();
        for method in &class.methods {
            let returns = method
                .return_type
                .as_ref()
                .map(lower_type)
                .transpose()?
                .unwrap_or(ValueType::Unit);
            methods.push(lower_class_function(
                &class.name.name,
                &fields,
                &method.name.name,
                &method.params,
                &method.body,
                &method.tests,
                returns,
                &global_scope,
                &signatures,
                &aliases,
            )?);
        }
        classes.push(Class {
            name: class.name.name.clone(),
            fields,
            constructors,
            methods,
        });
    }
    Ok(Program {
        globals,
        classes,
        functions,
    })
}

#[allow(clippy::too_many_arguments)]
fn lower_class_function(
    class_name: &str,
    fields: &[String],
    name: &str,
    source_params: &[severian_ast::Parameter],
    body: &Block,
    source_tests: &[severian_ast::TestBlock],
    return_type: ValueType,
    global_scope: &HashMap<String, Binding>,
    signatures: &HashMap<String, Signature>,
    aliases: &HashMap<String, String>,
) -> Result<Function, SemanticError> {
    let mut scope = global_scope.clone();
    for field in fields {
        scope.insert(
            field.clone(),
            Binding {
                ty: ValueType::Any,
                mutable: true,
                field: true,
            },
        );
    }
    let mut params = Vec::new();
    for param in source_params {
        let ty = param
            .ty
            .as_ref()
            .map(lower_type)
            .transpose()?
            .unwrap_or(ValueType::Any);
        let default = param
            .default
            .as_ref()
            .map(|value| {
                lower_expression(value, &scope, signatures, aliases).map(|(value, _)| value)
            })
            .transpose()?;
        scope.insert(
            param.name.name.clone(),
            Binding {
                ty,
                mutable: false,
                field: false,
            },
        );
        params.push(Parameter {
            name: param.name.name.clone(),
            ty,
            default,
        });
    }
    let instructions = lower_block(body, &mut scope, return_type, signatures, aliases)?;
    if return_type != ValueType::Unit && !always_returns(&instructions) {
        return Err(error(
            body.span,
            format!("method `{class_name}.{name}` must return a value"),
        ));
    }
    let mut tests = Vec::new();
    for test in source_tests {
        let mut test_scope = global_scope.clone();
        add_test_bindings(&mut test_scope);
        tests.push(Test {
            name: test.name.as_ref().map(|name| name.name.clone()),
            modes: lower_test_modes(&test.modes),
            instructions: lower_block(
                &test.body,
                &mut test_scope,
                ValueType::Unit,
                signatures,
                aliases,
            )?,
        });
    }
    Ok(Function {
        name: name.into(),
        params,
        return_type,
        instructions,
        tests,
    })
}

fn collect_imports(module: &Module) -> HashMap<String, String> {
    let mut aliases = HashMap::new();
    for item in &module.items {
        let Item::Import(import) = item else { continue };
        if let ImportKind::From { module, names } = &import.kind {
            let module = module
                .iter()
                .map(|part| part.name.as_str())
                .collect::<Vec<_>>()
                .join(".");
            for name in names {
                let exposed = name.alias.as_ref().unwrap_or(&name.name).name.clone();
                aliases.insert(exposed, format!("{module}.{}", name.name.name));
            }
        }
    }
    aliases
}

fn lower_block(
    block: &Block,
    scope: &mut HashMap<String, Binding>,
    return_type: ValueType,
    signatures: &HashMap<String, Signature>,
    aliases: &HashMap<String, String>,
) -> Result<Vec<Instruction>, SemanticError> {
    let mut instructions = Vec::new();
    for statement in &block.statements {
        match statement {
            Stmt::Let(binding) => {
                let source = binding
                    .value
                    .as_ref()
                    .ok_or_else(|| error(binding.span, "binding requires a value"))?;
                let (value, inferred) = lower_expression(source, scope, signatures, aliases)?;
                let declared = binding.ty.as_ref().map(lower_type).transpose()?;
                if let Some(declared) = declared {
                    compatible(binding.span, inferred, declared)?;
                }
                let ty = declared.unwrap_or(inferred);
                if scope
                    .get(&binding.name.name)
                    .is_some_and(|existing| existing.field || existing.mutable)
                {
                    instructions.push(Instruction::Assign {
                        target: Expression::Variable(binding.name.name.clone()),
                        op: AssignmentOp::Assign,
                        value,
                    });
                    continue;
                }
                if scope
                    .insert(
                        binding.name.name.clone(),
                        Binding {
                            ty,
                            mutable: binding.kind == LetKind::Changeable,
                            field: false,
                        },
                    )
                    .is_some()
                {
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
            Stmt::Assign(assignment) => {
                let (target, target_type) =
                    lower_expression(&assignment.target, scope, signatures, aliases)?;
                if let Expr::Identifier(name) = &assignment.target {
                    if !scope.get(&name.name).is_some_and(|binding| binding.mutable) {
                        return Err(error(
                            name.span,
                            format!("binding `{}` is not changeable", name.name),
                        ));
                    }
                } else if !matches!(assignment.target, Expr::Index(_)) {
                    return Err(error(
                        assignment.target.span(),
                        "assignment target is not mutable",
                    ));
                }
                let (value, value_type) =
                    lower_expression(&assignment.value, scope, signatures, aliases)?;
                if target_type != ValueType::Any && value_type != ValueType::Any {
                    compatible(assignment.span, value_type, target_type)?;
                }
                instructions.push(Instruction::Assign {
                    target,
                    op: match assignment.op {
                        AstAssignOp::Assign => AssignmentOp::Assign,
                        AstAssignOp::AddAssign => AssignmentOp::Add,
                        AstAssignOp::SubAssign => AssignmentOp::Sub,
                        AstAssignOp::MulAssign => AssignmentOp::Mul,
                        AstAssignOp::DivAssign => AssignmentOp::Div,
                        AstAssignOp::ModAssign => AssignmentOp::Mod,
                    },
                    value,
                });
            }
            Stmt::TryBind(binding) => {
                let (value, _) = lower_expression(&binding.value, scope, signatures, aliases)?;
                scope.insert(
                    binding.name.name.clone(),
                    Binding {
                        ty: ValueType::Any,
                        mutable: false,
                        field: false,
                    },
                );
                instructions.push(Instruction::TryLet {
                    name: binding.name.name.clone(),
                    value,
                });
            }
            Stmt::Expr(expression) => {
                let (expression, _) = lower_expression(expression, scope, signatures, aliases)?;
                match expression {
                    Expression::Call { function, mut args } if function == "print" => {
                        let value = if args.len() == 1 {
                            args.remove(0)
                        } else {
                            Expression::PrintArgs(args)
                        };
                        instructions.push(Instruction::Print(value));
                    }
                    expression => instructions.push(Instruction::Evaluate(expression)),
                }
            }
            Stmt::Assert(assertion) => {
                let (condition, ty) =
                    lower_expression(&assertion.condition, scope, signatures, aliases)?;
                compatible(assertion.condition.span(), ty, ValueType::Bool)?;
                instructions.push(Instruction::Assert(condition));
            }
            Stmt::Return(statement) => {
                let value = statement
                    .value
                    .as_ref()
                    .map(|value| lower_expression(value, scope, signatures, aliases))
                    .transpose()?;
                let actual = value.as_ref().map_or(ValueType::Unit, |(_, ty)| *ty);
                let value = value.map(|(value, _)| value);
                let value = match (return_type, actual, value) {
                    (ValueType::Result, ValueType::Result, value) => value,
                    (ValueType::Result, _, Some(value)) => Some(Expression::Variant {
                        name: "ok".into(),
                        fields: vec![value],
                    }),
                    (ValueType::Option, ValueType::Option, value) => value,
                    (_, _, value) => {
                        compatible(statement.span, actual, return_type)?;
                        value
                    }
                };
                instructions.push(Instruction::Return(value));
            }
            Stmt::If(statement) => {
                let (condition, ty) =
                    lower_expression(&statement.condition, scope, signatures, aliases)?;
                compatible(statement.condition.span(), ty, ValueType::Bool)?;
                let mut then_scope = scope.clone();
                let then_instructions = lower_block(
                    &statement.then_block,
                    &mut then_scope,
                    return_type,
                    signatures,
                    aliases,
                )?;
                let else_instructions = match &statement.else_branch {
                    None => Vec::new(),
                    Some(ElseBranch::Block(block)) => {
                        let mut else_scope = scope.clone();
                        lower_block(block, &mut else_scope, return_type, signatures, aliases)?
                    }
                    Some(ElseBranch::If(_)) => {
                        return Err(error(statement.span, "else-if is not supported yet"))
                    }
                };
                instructions.push(Instruction::If {
                    condition,
                    then_instructions,
                    else_instructions,
                });
            }
            Stmt::While(statement) => {
                let setup = statement
                    .setup
                    .as_ref()
                    .map(|setup| {
                        let lowered = lower_block(
                            &Block {
                                span: setup.span(),
                                statements: vec![(**setup).clone()],
                            },
                            scope,
                            return_type,
                            signatures,
                            aliases,
                        )?;
                        Ok(Box::new(lowered.into_iter().next().unwrap()))
                    })
                    .transpose()?;
                let (condition, ty) =
                    lower_expression(&statement.condition, scope, signatures, aliases)?;
                compatible(statement.condition.span(), ty, ValueType::Bool)?;
                let mut body_scope = scope.clone();
                let body = lower_block(
                    &statement.body,
                    &mut body_scope,
                    return_type,
                    signatures,
                    aliases,
                )?;
                instructions.push(Instruction::While {
                    setup,
                    condition,
                    instructions: body,
                });
            }
            Stmt::For(statement) => {
                let (iterable, _) =
                    lower_expression(&statement.iterable, scope, signatures, aliases)?;
                let mut body_scope = scope.clone();
                let pattern = lower_pattern(&statement.pattern, &mut body_scope)?;
                let body = lower_block(
                    &statement.body,
                    &mut body_scope,
                    return_type,
                    signatures,
                    aliases,
                )?;
                instructions.push(Instruction::For {
                    pattern,
                    iterable,
                    instructions: body,
                });
            }
            Stmt::Switch(statement) => {
                if statement.values.len() != 1 {
                    return Err(error(
                        statement.span,
                        "only single-value switches are supported yet",
                    ));
                }
                let value = lower_expression(&statement.values[0], scope, signatures, aliases)?.0;
                let mut arms = Vec::new();
                for arm in &statement.arms {
                    let mut arm_scope = scope.clone();
                    let pattern = lower_pattern(&arm.pattern, &mut arm_scope)?;
                    let guard = arm
                        .guard
                        .as_ref()
                        .map(|guard| {
                            lower_expression(guard, &arm_scope, signatures, aliases)
                                .map(|(guard, _)| guard)
                        })
                        .transpose()?;
                    let arm_instructions =
                        lower_block(&arm.body, &mut arm_scope, return_type, signatures, aliases)?;
                    arms.push(HirSwitchArm {
                        pattern,
                        guard,
                        instructions: arm_instructions,
                    });
                }
                instructions.push(Instruction::Switch { value, arms });
            }
            Stmt::Unsafe(block) => {
                instructions.extend(lower_block(
                    &block.body,
                    scope,
                    return_type,
                    signatures,
                    aliases,
                )?);
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
    scope: &HashMap<String, Binding>,
    signatures: &HashMap<String, Signature>,
    aliases: &HashMap<String, String>,
) -> Result<(Expression, ValueType), SemanticError> {
    match expression {
        Expr::Literal(Literal::Integer { value, .. }) => {
            Ok((Expression::Integer(*value), ValueType::Int))
        }
        Expr::Literal(Literal::Float { value, .. }) => {
            Ok((Expression::Float(value.to_bits()), ValueType::Float))
        }
        Expr::Literal(Literal::Boolean { value, .. }) => {
            Ok((Expression::Boolean(*value), ValueType::Bool))
        }
        Expr::Literal(Literal::String { value, .. }) => {
            Ok((Expression::String(value.clone()), ValueType::String))
        }
        Expr::Identifier(identifier) => {
            if let Some(binding) = scope.get(&identifier.name) {
                Ok((Expression::Variable(identifier.name.clone()), binding.ty))
            } else if signatures.contains_key(&identifier.name) {
                Ok((
                    Expression::Function(identifier.name.clone()),
                    ValueType::Function,
                ))
            } else if identifier.name == "absent" {
                Ok((
                    Expression::Variant {
                        name: "absent".into(),
                        fields: Vec::new(),
                    },
                    ValueType::Option,
                ))
            } else if identifier.name == "None" {
                Ok((
                    Expression::Variant {
                        name: "None".into(),
                        fields: Vec::new(),
                    },
                    ValueType::Option,
                ))
            } else if identifier
                .name
                .as_bytes()
                .first()
                .is_some_and(u8::is_ascii_uppercase)
            {
                Ok((
                    Expression::Variant {
                        name: identifier.name.clone(),
                        fields: Vec::new(),
                    },
                    ValueType::Any,
                ))
            } else {
                Err(error(
                    identifier.span,
                    format!("unknown binding `{}`", identifier.name),
                ))
            }
        }
        Expr::Unary(unary) => {
            let (expression, ty) = lower_expression(&unary.expr, scope, signatures, aliases)?;
            let (op, expected, result) = match unary.op {
                AstUnaryOp::Negate => (UnaryOp::Negate, ty, ty),
                AstUnaryOp::Not => (UnaryOp::Not, ValueType::Bool, ValueType::Bool),
            };
            compatible(unary.span, ty, expected)?;
            Ok((
                Expression::Unary {
                    op,
                    expression: Box::new(expression),
                },
                result,
            ))
        }
        Expr::Binary(binary) => {
            let (left, left_type) = lower_expression(&binary.left, scope, signatures, aliases)?;
            let (right, right_type) = lower_expression(&binary.right, scope, signatures, aliases)?;
            let (op, result_type) = match binary.op {
                AstBinaryOp::Add => (
                    BinaryOp::Add,
                    merge_numeric(left_type, right_type, binary.span)?,
                ),
                AstBinaryOp::Sub => (
                    BinaryOp::Sub,
                    merge_numeric(left_type, right_type, binary.span)?,
                ),
                AstBinaryOp::Mul => (
                    BinaryOp::Mul,
                    merge_numeric(left_type, right_type, binary.span)?,
                ),
                AstBinaryOp::Div => (
                    BinaryOp::Div,
                    merge_numeric(left_type, right_type, binary.span)?,
                ),
                AstBinaryOp::Mod => (
                    BinaryOp::Mod,
                    merge_numeric(left_type, right_type, binary.span)?,
                ),
                AstBinaryOp::Equal => (BinaryOp::Equal, ValueType::Bool),
                AstBinaryOp::NotEqual => (BinaryOp::NotEqual, ValueType::Bool),
                AstBinaryOp::Less => (BinaryOp::Less, ValueType::Bool),
                AstBinaryOp::LessEqual => (BinaryOp::LessEqual, ValueType::Bool),
                AstBinaryOp::Greater => (BinaryOp::Greater, ValueType::Bool),
                AstBinaryOp::GreaterEqual => (BinaryOp::GreaterEqual, ValueType::Bool),
                AstBinaryOp::And => (BinaryOp::And, ValueType::Bool),
                AstBinaryOp::Or => (BinaryOp::Or, ValueType::Bool),
                AstBinaryOp::In => (BinaryOp::In, ValueType::Bool),
                _ => return Err(error(binary.span, "operator is not supported yet")),
            };
            Ok((
                Expression::Binary {
                    left: Box::new(left),
                    op,
                    right: Box::new(right),
                },
                result_type,
            ))
        }
        Expr::Call(call) => lower_call(call, scope, signatures, aliases),
        Expr::List(collection) => {
            lower_collection(&collection.elements, scope, signatures, aliases)
                .map(|elements| (Expression::List(elements), ValueType::List))
        }
        Expr::Tuple(collection) => {
            lower_collection(&collection.elements, scope, signatures, aliases)
                .map(|elements| (Expression::Tuple(elements), ValueType::Tuple))
        }
        Expr::Set(collection) => lower_collection(&collection.elements, scope, signatures, aliases)
            .map(|elements| (Expression::Set(elements), ValueType::Set)),
        Expr::Map(map) => {
            let entries = map
                .entries
                .iter()
                .map(|entry| {
                    Ok((
                        lower_expression(&entry.key, scope, signatures, aliases)?.0,
                        lower_expression(&entry.value, scope, signatures, aliases)?.0,
                    ))
                })
                .collect::<Result<Vec<_>, SemanticError>>()?;
            Ok((Expression::Map(entries), ValueType::Map))
        }
        Expr::Index(index) => {
            let object = lower_expression(&index.object, scope, signatures, aliases)?.0;
            let index_value = lower_expression(&index.index, scope, signatures, aliases)?.0;
            Ok((
                Expression::Index {
                    object: Box::new(object),
                    index: Box::new(index_value),
                },
                ValueType::Any,
            ))
        }
        Expr::Member(member) => {
            let object = lower_expression(&member.object, scope, signatures, aliases)?.0;
            Ok((
                Expression::Member {
                    object: Box::new(object),
                    member: member.member.name.clone(),
                },
                ValueType::Any,
            ))
        }
        Expr::ListComprehension(comprehension) => {
            let iterable = lower_expression(&comprehension.iterable, scope, signatures, aliases)?.0;
            let mut inner_scope = scope.clone();
            inner_scope.insert(
                comprehension.variable.name.clone(),
                Binding {
                    ty: ValueType::Any,
                    mutable: false,
                    field: false,
                },
            );
            let element =
                lower_expression(&comprehension.element, &inner_scope, signatures, aliases)?.0;
            let condition = comprehension
                .condition
                .as_ref()
                .map(|condition| {
                    lower_expression(condition, &inner_scope, signatures, aliases)
                        .map(|(condition, _)| Box::new(condition))
                })
                .transpose()?;
            Ok((
                Expression::ListComprehension {
                    element: Box::new(element),
                    variable: comprehension.variable.name.clone(),
                    iterable: Box::new(iterable),
                    condition,
                },
                ValueType::List,
            ))
        }
        Expr::Async(task) => {
            let (value, _) = lower_expression(&task.value, scope, signatures, aliases)?;
            Ok((Expression::Task(Box::new(value)), ValueType::Any))
        }
        Expr::Await(task) => {
            let (value, _) = lower_expression(&task.value, scope, signatures, aliases)?;
            Ok((Expression::Await(Box::new(value)), ValueType::Any))
        }
        Expr::Channel(channel) => {
            let capacity = lower_expression(&channel.capacity, scope, signatures, aliases)?.0;
            Ok((Expression::Channel(Box::new(capacity)), ValueType::Any))
        }
        Expr::Send(send) => {
            let value = lower_expression(&send.value, scope, signatures, aliases)?.0;
            let channel = lower_expression(&send.channel, scope, signatures, aliases)?.0;
            Ok((
                Expression::Send {
                    value: Box::new(value),
                    channel: Box::new(channel),
                },
                ValueType::Unit,
            ))
        }
        Expr::Ownership(ownership) => {
            let (value, ty) = lower_expression(&ownership.value, scope, signatures, aliases)?;
            match ownership.op {
                AstOwnershipOp::View
                | AstOwnershipOp::Borrow
                | AstOwnershipOp::Clone
                | AstOwnershipOp::Move => Ok((value, ty)),
            }
        }
        Expr::ChaosRule(rule) => {
            let (function, return_type) =
                lower_expression(&rule.function, scope, signatures, aliases)?;
            let Expression::Function(function) = function else {
                return Err(error(
                    rule.function.span(),
                    "chaos injection target must be a function",
                ));
            };
            let (value, value_type) = lower_expression(&rule.value, scope, signatures, aliases)?;
            if rule.action == severian_ast::ChaosAction::Return {
                let declared_return = signatures
                    .get(&function)
                    .map_or(return_type, |signature| signature.returns);
                compatible(rule.value.span(), value_type, declared_return)?;
            }
            Ok((
                Expression::ChaosRule {
                    function,
                    action: match rule.action {
                        severian_ast::ChaosAction::Return => HirChaosAction::Return,
                        severian_ast::ChaosAction::Throw => HirChaosAction::Throw,
                    },
                    value: Box::new(value),
                },
                ValueType::Any,
            ))
        }
        _ => Err(error(
            expression.span(),
            "expression is not supported in this compiler slice yet",
        )),
    }
}

fn add_test_bindings(scope: &mut HashMap<String, Binding>) {
    scope.insert(
        "chaos".into(),
        Binding {
            ty: ValueType::Any,
            mutable: false,
            field: false,
        },
    );
}

fn lower_call(
    call: &severian_ast::CallExpr,
    scope: &HashMap<String, Binding>,
    signatures: &HashMap<String, Signature>,
    aliases: &HashMap<String, String>,
) -> Result<(Expression, ValueType), SemanticError> {
    if let Expr::Member(member) = call.callee.as_ref() {
        if let Expr::Identifier(object) = member.object.as_ref() {
            if object.name == "int" && member.member.name == "parse" {
                let args = call
                    .args
                    .iter()
                    .map(|arg| {
                        lower_expression(&arg.value, scope, signatures, aliases).map(|(arg, _)| arg)
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                return Ok((
                    Expression::Call {
                        function: "int.parse".into(),
                        args,
                    },
                    ValueType::Result,
                ));
            }
            if object.name == "http" && member.member.name == "get" {
                let args = call
                    .args
                    .iter()
                    .map(|arg| {
                        lower_expression(&arg.value, scope, signatures, aliases).map(|(arg, _)| arg)
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                return Ok((
                    Expression::Call {
                        function: "http.get".into(),
                        args,
                    },
                    ValueType::Result,
                ));
            }
            if member.member.name == "zero" && !scope.contains_key(&object.name) {
                return Ok((
                    Expression::Call {
                        function: "Number.zero".into(),
                        args: Vec::new(),
                    },
                    ValueType::Any,
                ));
            }
        }
        let object = lower_expression(&member.object, scope, signatures, aliases)?.0;
        let args = call
            .args
            .iter()
            .map(|arg| lower_expression(&arg.value, scope, signatures, aliases).map(|(arg, _)| arg))
            .collect::<Result<Vec<_>, _>>()?;
        return Ok((
            Expression::MethodCall {
                object: Box::new(object),
                method: member.member.name.clone(),
                args,
            },
            ValueType::Any,
        ));
    }
    let Expr::Identifier(callee) = call.callee.as_ref() else {
        let (callee, ty) = lower_expression(&call.callee, scope, signatures, aliases)?;
        if ty != ValueType::Function {
            return Err(error(call.callee.span(), "value is not callable"));
        }
        let args = call
            .args
            .iter()
            .map(|arg| lower_expression(&arg.value, scope, signatures, aliases).map(|(arg, _)| arg))
            .collect::<Result<Vec<_>, _>>()?;
        return Ok((
            Expression::CallValue {
                callee: Box::new(callee),
                args,
            },
            ValueType::Any,
        ));
    };
    let canonical = aliases
        .get(&callee.name)
        .map(String::as_str)
        .unwrap_or(&callee.name);
    if let Some(class) = canonical.strip_prefix("__class.") {
        let args = call
            .args
            .iter()
            .map(|arg| lower_expression(&arg.value, scope, signatures, aliases).map(|(arg, _)| arg))
            .collect::<Result<Vec<_>, _>>()?;
        return Ok((
            Expression::Construct {
                class: class.into(),
                args,
            },
            ValueType::Any,
        ));
    }
    let builtin = match canonical {
        "print" | "io.print" => Some(("print", ValueType::Unit)),
        "sqrt" | "math.sqrt" => Some(("sqrt", ValueType::Float)),
        "float" => Some(("float", ValueType::Float)),
        "range" => Some(("range", ValueType::List)),
        "indices" => Some(("indices", ValueType::List)),
        "read" if !signatures.contains_key(&callee.name) => Some(("read", ValueType::Result)),
        "size" => Some(("size", ValueType::Int)),
        "present" => {
            let fields = call
                .args
                .iter()
                .map(|arg| {
                    lower_expression(&arg.value, scope, signatures, aliases).map(|(arg, _)| arg)
                })
                .collect::<Result<Vec<_>, _>>()?;
            return Ok((
                Expression::Variant {
                    name: "present".into(),
                    fields,
                },
                ValueType::Option,
            ));
        }
        "failure" => {
            let fields = call
                .args
                .iter()
                .map(|arg| {
                    lower_expression(&arg.value, scope, signatures, aliases).map(|(arg, _)| arg)
                })
                .collect::<Result<Vec<_>, _>>()?;
            return Ok((
                Expression::Variant {
                    name: "failure".into(),
                    fields,
                },
                ValueType::Result,
            ));
        }
        "__format" => {
            let Expr::Literal(Literal::String { value, .. }) = &call.args[0].value else {
                unreachable!()
            };
            return Ok((Expression::Format(value.clone()), ValueType::String));
        }
        _ => None,
    };
    if let Some((name, returns)) = builtin {
        let args = call
            .args
            .iter()
            .map(|arg| lower_expression(&arg.value, scope, signatures, aliases).map(|(arg, _)| arg))
            .collect::<Result<Vec<_>, _>>()?;
        return Ok((
            Expression::Call {
                function: name.into(),
                args,
            },
            returns,
        ));
    }
    if let Some(signature) = signatures.get(&callee.name) {
        let mut supplied: Vec<Option<Expression>> = vec![None; signature.params.len()];
        let mut positional = 0;
        for argument in &call.args {
            let index = if let Some(name) = &argument.name {
                signature
                    .params
                    .iter()
                    .position(|param| param.name == name.name)
                    .ok_or_else(|| error(name.span, format!("unknown argument `{}`", name.name)))?
            } else {
                let index = positional;
                positional += 1;
                index
            };
            if index >= supplied.len() || supplied[index].is_some() {
                return Err(error(
                    argument.span,
                    format!("invalid arguments for `{}`", callee.name),
                ));
            }
            let (value, ty) = lower_expression(&argument.value, scope, signatures, aliases)?;
            compatible(argument.span, ty, signature.params[index].ty)?;
            supplied[index] = Some(value);
        }
        let args = supplied
            .into_iter()
            .zip(&signature.params)
            .map(|(value, param)| {
                if let Some(value) = value {
                    Ok(value)
                } else if let Some(default) = &param.default {
                    lower_expression(default, scope, signatures, aliases).map(|(value, _)| value)
                } else {
                    Err(error(
                        call.span,
                        format!("missing argument `{}`", param.name),
                    ))
                }
            })
            .collect::<Result<Vec<_>, _>>()?;
        return Ok((
            Expression::Call {
                function: callee.name.clone(),
                args,
            },
            signature.returns,
        ));
    }
    if scope
        .get(&callee.name)
        .is_some_and(|binding| binding.ty == ValueType::Function)
    {
        let args = call
            .args
            .iter()
            .map(|arg| lower_expression(&arg.value, scope, signatures, aliases).map(|(arg, _)| arg))
            .collect::<Result<Vec<_>, _>>()?;
        return Ok((
            Expression::CallValue {
                callee: Box::new(Expression::Variable(callee.name.clone())),
                args,
            },
            ValueType::Any,
        ));
    }
    if callee
        .name
        .as_bytes()
        .first()
        .is_some_and(u8::is_ascii_uppercase)
    {
        let fields = call
            .args
            .iter()
            .map(|arg| lower_expression(&arg.value, scope, signatures, aliases).map(|(arg, _)| arg))
            .collect::<Result<Vec<_>, _>>()?;
        return Ok((
            Expression::Variant {
                name: callee.name.clone(),
                fields,
            },
            ValueType::Any,
        ));
    }
    Err(error(
        callee.span,
        format!("unknown function `{}`", callee.name),
    ))
}

fn lower_collection(
    elements: &[Expr],
    scope: &HashMap<String, Binding>,
    signatures: &HashMap<String, Signature>,
    aliases: &HashMap<String, String>,
) -> Result<Vec<Expression>, SemanticError> {
    elements
        .iter()
        .map(|element| {
            lower_expression(element, scope, signatures, aliases).map(|(element, _)| element)
        })
        .collect()
}

fn lower_type(ty: &Type) -> Result<ValueType, SemanticError> {
    match ty {
        Type::Union { .. } => Ok(ValueType::Any),
        Type::Named(path) => {
            let name = path
                .segments
                .first()
                .map(|segment| segment.name.as_str())
                .unwrap_or("");
            match name {
                "int" => Ok(ValueType::Int),
                "float" => Ok(ValueType::Float),
                "bool" => Ok(ValueType::Bool),
                "string" => Ok(ValueType::String),
                "unit" => Ok(ValueType::Unit),
                "list" => Ok(ValueType::List),
                "map" => Ok(ValueType::Map),
                "set" => Ok(ValueType::Set),
                "fn" => Ok(ValueType::Function),
                "Result" => Ok(ValueType::Result),
                "Option" => Ok(ValueType::Option),
                _ => Ok(ValueType::Any),
            }
        }
        _ => Err(error(ty.span(), "type is not supported yet")),
    }
}

fn merge_numeric(
    left: ValueType,
    right: ValueType,
    span: Span,
) -> Result<ValueType, SemanticError> {
    if left == ValueType::Any || right == ValueType::Any {
        return Ok(ValueType::Any);
    }
    if left == right && matches!(left, ValueType::Int | ValueType::Float | ValueType::String) {
        Ok(left)
    } else {
        Err(error(span, "operator requires matching numeric values"))
    }
}

fn compatible(span: Span, actual: ValueType, expected: ValueType) -> Result<(), SemanticError> {
    if actual == expected
        || actual == ValueType::Any
        || expected == ValueType::Any
        || (expected == ValueType::Result && actual != ValueType::Unit)
    {
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

fn value_span(value: &Option<Expr>) -> Span {
    value.as_ref().map_or(Span::dummy(), Expr::span)
}

fn lower_test_modes(modes: &[severian_ast::TestMode]) -> Vec<HirTestMode> {
    modes
        .iter()
        .map(|mode| match mode {
            severian_ast::TestMode::Property => HirTestMode::Property,
            severian_ast::TestMode::Bench => HirTestMode::Bench,
            severian_ast::TestMode::Chaos => HirTestMode::Chaos,
        })
        .collect()
}

fn error(span: Span, message: impl Into<String>) -> SemanticError {
    SemanticError {
        span,
        message: message.into(),
    }
}

fn lower_pattern(
    pattern: &Pattern,
    scope: &mut HashMap<String, Binding>,
) -> Result<MatchPattern, SemanticError> {
    match pattern {
        Pattern::Wildcard(_) => Ok(MatchPattern::Wildcard),
        Pattern::Identifier(name) => {
            scope.insert(
                name.name.clone(),
                Binding {
                    ty: ValueType::Any,
                    mutable: false,
                    field: false,
                },
            );
            Ok(MatchPattern::Bind(name.name.clone()))
        }
        Pattern::Literal(Literal::Integer { value, .. }) => Ok(MatchPattern::Integer(*value)),
        Pattern::Literal(Literal::Float { value, .. }) => Ok(MatchPattern::Float(value.to_bits())),
        Pattern::Literal(Literal::Boolean { value, .. }) => Ok(MatchPattern::Boolean(*value)),
        Pattern::Literal(Literal::String { value, .. }) => Ok(MatchPattern::String(value.clone())),
        Pattern::Constructor { name, fields, .. } => {
            let Type::Named(path) = name else {
                return Err(error(name.span(), "invalid constructor pattern"));
            };
            let name = path.segments.first().unwrap().name.clone();
            if fields.is_empty()
                && name.as_bytes().first().is_some_and(u8::is_ascii_lowercase)
                && !matches!(name.as_str(), "absent" | "ok" | "failure" | "present")
            {
                scope.insert(
                    name.clone(),
                    Binding {
                        ty: ValueType::Any,
                        mutable: false,
                        field: false,
                    },
                );
                return Ok(MatchPattern::Bind(name));
            }
            let fields = fields
                .iter()
                .map(|field| lower_pattern(field, scope))
                .collect::<Result<Vec<_>, _>>()?;
            Ok(MatchPattern::Constructor { name, fields })
        }
        Pattern::Tuple { elements, .. } => {
            let fields = elements
                .iter()
                .map(|field| lower_pattern(field, scope))
                .collect::<Result<Vec<_>, _>>()?;
            Ok(MatchPattern::Constructor {
                name: "tuple".into(),
                fields,
            })
        }
        _ => Err(error(pattern.span(), "pattern is not supported yet")),
    }
}
