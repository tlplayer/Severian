use super::*;

pub(super) fn lower_test_modes(modes: &[severian_ast::TestMode]) -> Vec<HirTestMode> {
    modes
        .iter()
        .map(|mode| match mode {
            severian_ast::TestMode::Property => HirTestMode::Property,
            severian_ast::TestMode::Bench => HirTestMode::Bench,
            severian_ast::TestMode::Chaos => HirTestMode::Chaos,
            severian_ast::TestMode::Integration => HirTestMode::Integration,
            severian_ast::TestMode::Profile => HirTestMode::Profile,
        })
        .collect()
}

pub(super) fn lower_test_return_type(
    test: &severian_ast::TestBlock,
) -> Result<ValueType, SemanticError> {
    let Some(return_type) = &test.return_type else {
        return Ok(ValueType::Unit);
    };
    let Type::Named(path) = return_type else {
        return Err(error(
            return_type.span(),
            "a test result annotation must be `TestResult`",
        ));
    };
    if path.segments.len() != 1 || path.segments[0].name != "TestResult" || !path.args.is_empty() {
        return Err(error(
            return_type.span(),
            "a test result annotation must be `TestResult`",
        ));
    }
    Ok(ValueType::Result)
}

pub(super) fn enforce_function_contract(
    instructions: &mut Vec<Instruction>,
    contract: Option<&HirFunctionContract>,
) {
    let Some(contract) = contract else { return };
    insert_deferred_contract_checks(instructions, &contract.clauses);
    let mut entry = contract
        .clauses
        .iter()
        .map(contract_check_instruction)
        .collect::<Vec<_>>();
    entry.append(instructions);
    *instructions = entry;
}

pub(super) fn insert_deferred_contract_checks(
    instructions: &mut Vec<Instruction>,
    clauses: &[HirContractClause],
) {
    for instruction in instructions.iter_mut() {
        match instruction {
            Instruction::If {
                then_instructions,
                else_instructions,
                ..
            } => {
                insert_deferred_contract_checks(then_instructions, clauses);
                insert_deferred_contract_checks(else_instructions, clauses);
            }
            Instruction::While { instructions, .. }
            | Instruction::For { instructions, .. }
            | Instruction::With { instructions, .. } => {
                insert_deferred_contract_checks(instructions, clauses);
            }
            Instruction::Switch { arms, .. } | Instruction::ChannelSwitch { arms, .. } => {
                for arm in arms {
                    insert_deferred_contract_checks(&mut arm.instructions, clauses);
                }
            }
            _ => {}
        }
    }

    let previous = std::mem::take(instructions);
    for instruction in previous {
        let changed = changed_contract_binding(&instruction);
        instructions.push(instruction);
        let Some(changed) = changed else { continue };
        instructions.extend(
            clauses
                .iter()
                .filter(|clause| clause.deferred && clause.dependencies.contains(&changed))
                .map(contract_check_instruction),
        );
    }
}

pub(super) fn changed_contract_binding(instruction: &Instruction) -> Option<BindingRef> {
    match instruction {
        Instruction::Assign { target, .. } => root_variable(target).cloned(),
        Instruction::Evaluate(expression) => match expression.kind() {
            Expression::MethodCall { object, method, .. } if contract_mutating_method(method) => {
                root_variable(object).cloned()
            }
            _ => None,
        },
        _ => None,
    }
}

pub(super) fn contract_mutating_method(method: &str) -> bool {
    matches!(
        method,
        "append"
            | "append_left"
            | "appendleft"
            | "extend"
            | "insert"
            | "remove"
            | "pop"
            | "pop_left"
            | "popleft"
            | "heap_push"
            | "heapPush"
            | "heap_pop"
            | "heapPop"
            | "clear"
            | "sort"
            | "reverse"
            | "set"
            | "set_default"
            | "setDefault"
    )
}

pub(super) fn root_variable(expression: &Expression) -> Option<&BindingRef> {
    match expression.kind() {
        Expression::Variable(name) => Some(name),
        Expression::Member { object, .. } | Expression::Index { object, .. } => {
            root_variable(object)
        }
        _ => None,
    }
}

pub(super) fn contract_check_instruction(clause: &HirContractClause) -> Instruction {
    let range = clause
        .condition
        .hir_id()
        .and_then(HirId::legacy_source_range);
    let location = if clause.location {
        range.map_or_else(
            || "contract source".into(),
            |range| format!("contract source bytes {}..{}", range.start, range.end),
        )
    } else {
        String::new()
    };
    let mut arguments = vec![
        synthetic_string(
            100,
            clause
                .message
                .clone()
                .unwrap_or_else(|| "contract condition was not satisfied".into()),
        ),
        synthetic_string(101, location),
    ];
    arguments.push(if clause.vars && !clause.dependencies.is_empty() {
        Expression::Typed {
            id: HirId::synthetic(102),
            ty: ValueType::String,
            expression: Box::new(Expression::Format {
                template: clause
                    .dependencies
                    .iter()
                    .map(|name| format!("{name}={{}}"))
                    .collect::<Vec<_>>()
                    .join(", "),
                args: clause
                    .dependencies
                    .iter()
                    .zip(&clause.dependency_types)
                    .enumerate()
                    .map(|(index, (name, ty))| Expression::Typed {
                        id: HirId::synthetic(120 + index as u64),
                        ty: *ty,
                        expression: Box::new(Expression::Variable(name.clone())),
                    })
                    .collect(),
                arg_types: clause.dependency_types.clone(),
            }),
        }
    } else {
        synthetic_string(102, String::new())
    });
    let failure = Expression::Typed {
        id: HirId::synthetic(110),
        ty: ValueType::Unit,
        expression: Box::new(Expression::Call {
            target: CallTarget {
                id: FunctionId::from_name("__sev_contract_fail"),
                name: "__sev_contract_fail".into(),
                native_symbol: Some("__sev_contract_fail".into()),
                signature: Some(FunctionType {
                    parameters: vec![ValueType::String; 3],
                    returns: ValueType::Unit,
                }),
            },
            args: arguments,
        }),
    };
    Instruction::If {
        condition: clause.condition.clone(),
        then_instructions: Vec::new(),
        else_instructions: vec![Instruction::Evaluate(failure)],
    }
}

pub(super) fn synthetic_string(id: u64, value: String) -> Expression {
    Expression::Typed {
        id: HirId::synthetic(id),
        ty: ValueType::String,
        expression: Box::new(Expression::String(value)),
    }
}

pub(super) fn collect_contract_dependencies(
    expression: &Expression,
    dependencies: &mut Vec<BindingRef>,
) {
    match expression.kind() {
        Expression::Variable(name) => dependencies.push(name.clone()),
        Expression::Binary { left, right, .. } => {
            collect_contract_dependencies(left, dependencies);
            collect_contract_dependencies(right, dependencies);
        }
        Expression::Unary { expression, .. }
        | Expression::Ownership {
            value: expression, ..
        }
        | Expression::Await(expression)
        | Expression::Channel(expression)
        | Expression::Task {
            value: expression, ..
        }
        | Expression::Member {
            object: expression, ..
        }
        | Expression::FusedPipeline {
            input: expression, ..
        } => {
            collect_contract_dependencies(expression, dependencies);
        }
        Expression::Index { object, index } => {
            collect_contract_dependencies(object, dependencies);
            collect_contract_dependencies(index, dependencies);
        }
        Expression::Slice {
            object,
            start,
            end,
            step,
        } => {
            collect_contract_dependencies(object, dependencies);
            for bound in [start, end, step].into_iter().flatten() {
                collect_contract_dependencies(bound, dependencies);
            }
        }
        Expression::List(values)
        | Expression::Tuple(values)
        | Expression::Set(values)
        | Expression::PrintArgs(values)
        | Expression::Construct { args: values, .. }
        | Expression::Variant { fields: values, .. } => {
            for value in values {
                collect_contract_dependencies(value, dependencies);
            }
        }
        Expression::ConstructFields { fields, .. } => {
            for (_, value) in fields {
                collect_contract_dependencies(value, dependencies);
            }
        }
        Expression::ObjectUpdate { object, fields, .. } => {
            collect_contract_dependencies(object, dependencies);
            for (_, value) in fields {
                collect_contract_dependencies(value, dependencies);
            }
        }
        Expression::ObjectDocument { object, .. } => {
            collect_contract_dependencies(object, dependencies);
        }
        Expression::Map(entries) => {
            for (key, value) in entries {
                collect_contract_dependencies(key, dependencies);
                collect_contract_dependencies(value, dependencies);
            }
        }
        Expression::MethodCall { object, args, .. } => {
            collect_contract_dependencies(object, dependencies);
            for argument in args {
                collect_contract_dependencies(argument, dependencies);
            }
        }
        Expression::Call { args, .. } => {
            for argument in args {
                collect_contract_dependencies(argument, dependencies);
            }
        }
        Expression::CallValue { callee, args, .. } => {
            collect_contract_dependencies(callee, dependencies);
            for argument in args {
                collect_contract_dependencies(argument, dependencies);
            }
        }
        _ => {}
    }
}

pub(super) fn error(span: Span, message: impl Into<String>) -> SemanticError {
    SemanticError {
        span,
        message: message.into(),
    }
}
