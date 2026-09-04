use severian_diagnostics::Diagnostic;
use severian_hir::{BindingId, Expression, ExpressionKind, Module, Program, Statement};
use std::collections::{BTreeMap, BTreeSet};

pub fn validate(program: &Program) -> Result<(), Diagnostic> {
    for module in &program.modules {
        validate_module(module)?;
        validate_slice_borrows(module)?;
    }
    Ok(())
}

#[derive(Debug, Clone, Copy)]
struct SliceRegion {
    owner: BindingId,
    start: Option<i64>,
    end: Option<i64>,
    active: bool,
}

fn validate_slice_borrows(module: &Module) -> Result<(), Diagnostic> {
    let bindings = module
        .bindings
        .iter()
        .map(|binding| (binding.id, binding))
        .collect::<BTreeMap<_, _>>();
    let slice_types = module
        .classes
        .iter()
        .filter(|class| class.name == "slice" || class.name.starts_with("slice["))
        .map(|class| class.id)
        .collect::<BTreeSet<_>>();
    if slice_types.is_empty() {
        return Ok(());
    }
    let array_types = module
        .classes
        .iter()
        .filter(|class| class.name == "array" || class.name.starts_with("array["))
        .map(|class| class.id)
        .collect::<BTreeSet<_>>();
    let call_names = module
        .functions
        .iter()
        .map(|function| (function.definition, function.name.as_str()))
        .collect::<BTreeMap<_, _>>();
    validate_slice_block(
        &module.initializer,
        &bindings,
        &slice_types,
        &array_types,
        &call_names,
        &BTreeSet::new(),
        &mut BTreeMap::new(),
    )?;
    for function in &module.functions {
        let Some(body) = &function.body else {
            continue;
        };
        let parameters = function
            .parameters
            .iter()
            .map(|parameter| parameter.binding)
            .collect::<BTreeSet<_>>();
        validate_slice_block(
            body,
            &bindings,
            &slice_types,
            &array_types,
            &call_names,
            &parameters,
            &mut BTreeMap::new(),
        )?;
    }
    Ok(())
}

fn validate_slice_block(
    block: &severian_hir::Block,
    bindings: &BTreeMap<BindingId, &severian_hir::Binding>,
    slice_types: &BTreeSet<severian_universal::TypeId>,
    array_types: &BTreeSet<severian_universal::TypeId>,
    call_names: &BTreeMap<severian_universal::DefId, &str>,
    parameters: &BTreeSet<BindingId>,
    regions: &mut BTreeMap<BindingId, SliceRegion>,
) -> Result<(), Diagnostic> {
    for statement in &block.statements {
        match statement {
            Statement::Binding(id) => {
                let binding = bindings[id];
                if slice_types.contains(&binding.type_id) {
                    if let Some(region) =
                        slice_region(&binding.value, bindings, slice_types, array_types, regions)
                    {
                        regions.insert(*id, region);
                    }
                }
                validate_slice_expression(
                    &binding.value,
                    bindings,
                    slice_types,
                    array_types,
                    call_names,
                    regions,
                )?;
            }
            Statement::Expression(expression) => {
                if let ExpressionKind::Move(operand) = &expression.kind {
                    if let ExpressionKind::Binding(binding) = operand.kind {
                        if let Some(region) = regions.get_mut(&binding) {
                            region.active = false;
                            continue;
                        }
                    }
                }
                validate_slice_expression(
                    expression,
                    bindings,
                    slice_types,
                    array_types,
                    call_names,
                    regions,
                )?;
            }
            Statement::Return(Some(value)) => {
                if slice_types.contains(&value.type_id) {
                    if let Some(region) =
                        slice_region(value, bindings, slice_types, array_types, regions)
                    {
                        if !parameters.contains(&region.owner) {
                            return Err(Diagnostic::new(
                                "E000302",
                                "a slice cannot outlive the local owner that produced it",
                                Some(value.span),
                            ));
                        }
                    }
                }
                validate_slice_expression(
                    value,
                    bindings,
                    slice_types,
                    array_types,
                    call_names,
                    regions,
                )?;
            }
            Statement::Sequence(nested) | Statement::Placement { body: nested, .. } => {
                validate_slice_block(
                    nested,
                    bindings,
                    slice_types,
                    array_types,
                    call_names,
                    parameters,
                    regions,
                )?;
            }
            Statement::ExpectThrow { body, .. } => validate_slice_block(
                body,
                bindings,
                slice_types,
                array_types,
                call_names,
                parameters,
                regions,
            )?,
            Statement::Try {
                body, catch_body, ..
            } => {
                validate_slice_block(
                    body,
                    bindings,
                    slice_types,
                    array_types,
                    call_names,
                    parameters,
                    regions,
                )?;
                validate_slice_block(
                    catch_body,
                    bindings,
                    slice_types,
                    array_types,
                    call_names,
                    parameters,
                    regions,
                )?;
            }
            Statement::If {
                condition,
                then_block,
                else_block,
            } => {
                validate_slice_expression(
                    condition,
                    bindings,
                    slice_types,
                    array_types,
                    call_names,
                    regions,
                )?;
                validate_slice_block(
                    then_block,
                    bindings,
                    slice_types,
                    array_types,
                    call_names,
                    parameters,
                    regions,
                )?;
                validate_slice_block(
                    else_block,
                    bindings,
                    slice_types,
                    array_types,
                    call_names,
                    parameters,
                    regions,
                )?;
            }
            Statement::While {
                condition, body, ..
            } => {
                validate_slice_expression(
                    condition,
                    bindings,
                    slice_types,
                    array_types,
                    call_names,
                    regions,
                )?;
                validate_slice_block(
                    body,
                    bindings,
                    slice_types,
                    array_types,
                    call_names,
                    parameters,
                    regions,
                )?;
            }
            Statement::Assert {
                condition, message, ..
            } => {
                validate_slice_expression(
                    condition,
                    bindings,
                    slice_types,
                    array_types,
                    call_names,
                    regions,
                )?;
                if let Some(message) = message {
                    validate_slice_expression(
                        message,
                        bindings,
                        slice_types,
                        array_types,
                        call_names,
                        regions,
                    )?;
                }
            }
            Statement::FieldUpdate { value, .. } | Statement::FieldSet { value, .. } => {
                validate_slice_expression(
                    value,
                    bindings,
                    slice_types,
                    array_types,
                    call_names,
                    regions,
                )?;
            }
            Statement::Match { subject, arms } => {
                validate_slice_expression(
                    subject,
                    bindings,
                    slice_types,
                    array_types,
                    call_names,
                    regions,
                )?;
                for arm in arms {
                    validate_slice_block(
                        &arm.body,
                        bindings,
                        slice_types,
                        array_types,
                        call_names,
                        parameters,
                        regions,
                    )?;
                }
            }
            Statement::Return(None) | Statement::Break { .. } | Statement::Continue { .. } => {}
        }
    }
    Ok(())
}

fn validate_slice_expression(
    expression: &Expression,
    bindings: &BTreeMap<BindingId, &severian_hir::Binding>,
    slice_types: &BTreeSet<severian_universal::TypeId>,
    array_types: &BTreeSet<severian_universal::TypeId>,
    call_names: &BTreeMap<severian_universal::DefId, &str>,
    regions: &mut BTreeMap<BindingId, SliceRegion>,
) -> Result<(), Diagnostic> {
    if let ExpressionKind::Call {
        callee: severian_hir::Callee::Direct { function, .. },
        arguments,
    } = &expression.kind
    {
        if call_names
            .get(function)
            .is_some_and(|name| name.starts_with("__sev_pointer_set_"))
        {
            if let Some(container) = arguments.first().and_then(direct_container_binding) {
                if let Some(region) = regions.get(&container).copied() {
                    if region.active {
                        if regions.iter().any(|(binding, other)| {
                            *binding != container
                                && other.active
                                && other.owner == region.owner
                                && regions_overlap(region, *other)
                        }) {
                            return Err(Diagnostic::new(
                                "E000302",
                                "overlapping slices cannot coexist when either is written",
                                Some(expression.span),
                            ));
                        }
                    }
                } else if array_types.contains(&bindings[&container].type_id)
                    && regions
                        .values()
                        .any(|region| region.active && region.owner == container)
                {
                    return Err(Diagnostic::new(
                        "E000302",
                        "an owner cannot be written while a slice borrow is active",
                        Some(expression.span),
                    ));
                }
            }
        }
    }
    match &expression.kind {
        ExpressionKind::Aggregate { fields, .. } => {
            for field in fields {
                validate_slice_expression(
                    field,
                    bindings,
                    slice_types,
                    array_types,
                    call_names,
                    regions,
                )?;
            }
        }
        ExpressionKind::Field { object, .. }
        | ExpressionKind::Throw(object)
        | ExpressionKind::Await(object)
        | ExpressionKind::Move(object)
        | ExpressionKind::Convert {
            operand: object, ..
        }
        | ExpressionKind::Borrow {
            operand: object, ..
        }
        | ExpressionKind::Unary {
            operand: object, ..
        } => validate_slice_expression(
            object,
            bindings,
            slice_types,
            array_types,
            call_names,
            regions,
        )?,
        ExpressionKind::Call { arguments, .. } => {
            for argument in arguments {
                validate_slice_expression(
                    argument,
                    bindings,
                    slice_types,
                    array_types,
                    call_names,
                    regions,
                )?;
            }
        }
        ExpressionKind::Async { expression, .. } => validate_slice_expression(
            expression,
            bindings,
            slice_types,
            array_types,
            call_names,
            regions,
        )?,
        ExpressionKind::AsyncFieldUpdate { value, .. } => validate_slice_expression(
            value,
            bindings,
            slice_types,
            array_types,
            call_names,
            regions,
        )?,
        ExpressionKind::Fallback {
            condition,
            value,
            fallback,
        } => {
            for nested in [condition.as_ref(), value.as_ref(), fallback.as_ref()] {
                validate_slice_expression(
                    nested,
                    bindings,
                    slice_types,
                    array_types,
                    call_names,
                    regions,
                )?;
            }
        }
        ExpressionKind::Binary { left, right, .. } => {
            validate_slice_expression(
                left,
                bindings,
                slice_types,
                array_types,
                call_names,
                regions,
            )?;
            validate_slice_expression(
                right,
                bindings,
                slice_types,
                array_types,
                call_names,
                regions,
            )?;
        }
        ExpressionKind::Undefined
        | ExpressionKind::Literal(_)
        | ExpressionKind::Binding(_)
        | ExpressionKind::AddressOf(_)
        | ExpressionKind::Function(_) => {}
    }
    Ok(())
}

fn direct_container_binding(expression: &Expression) -> Option<BindingId> {
    match &expression.kind {
        ExpressionKind::Field { object, index: 0 } => direct_binding(object),
        ExpressionKind::Borrow { operand, .. }
        | ExpressionKind::Move(operand)
        | ExpressionKind::Convert { operand, .. } => direct_container_binding(operand),
        _ => None,
    }
}

fn direct_binding(expression: &Expression) -> Option<BindingId> {
    match &expression.kind {
        ExpressionKind::Binding(binding) => Some(*binding),
        ExpressionKind::Borrow { operand, .. }
        | ExpressionKind::Move(operand)
        | ExpressionKind::Convert { operand, .. } => direct_binding(operand),
        _ => None,
    }
}

fn slice_region(
    expression: &Expression,
    bindings: &BTreeMap<BindingId, &severian_hir::Binding>,
    slice_types: &BTreeSet<severian_universal::TypeId>,
    array_types: &BTreeSet<severian_universal::TypeId>,
    regions: &BTreeMap<BindingId, SliceRegion>,
) -> Option<SliceRegion> {
    match &expression.kind {
        ExpressionKind::Binding(binding) => regions.get(binding).copied(),
        ExpressionKind::Fallback { value, .. } => {
            slice_region(value, bindings, slice_types, array_types, regions)
        }
        ExpressionKind::Borrow { operand, .. }
        | ExpressionKind::Move(operand)
        | ExpressionKind::Convert { operand, .. } => {
            slice_region(operand, bindings, slice_types, array_types, regions)
        }
        ExpressionKind::Aggregate { class, fields }
            if slice_types.contains(class) && fields.len() >= 3 =>
        {
            let owner = region_owner(&fields[0], bindings, slice_types, array_types, regions)?;
            let start = static_integer_value(&fields[1], bindings, &mut BTreeSet::new());
            let length = static_integer_value(&fields[2], bindings, &mut BTreeSet::new());
            Some(SliceRegion {
                owner,
                start,
                end: start.zip(length).map(|(start, length)| start + length),
                active: true,
            })
        }
        _ => None,
    }
}

fn region_owner(
    expression: &Expression,
    bindings: &BTreeMap<BindingId, &severian_hir::Binding>,
    slice_types: &BTreeSet<severian_universal::TypeId>,
    array_types: &BTreeSet<severian_universal::TypeId>,
    regions: &BTreeMap<BindingId, SliceRegion>,
) -> Option<BindingId> {
    let binding = match &expression.kind {
        ExpressionKind::Field { object, index: 0 } => direct_binding(object)?,
        ExpressionKind::Binding(binding) => *binding,
        ExpressionKind::Borrow { operand, .. }
        | ExpressionKind::Move(operand)
        | ExpressionKind::Convert { operand, .. } => {
            return region_owner(operand, bindings, slice_types, array_types, regions)
        }
        _ => return None,
    };
    let ty = bindings.get(&binding)?.type_id;
    if array_types.contains(&ty) {
        Some(binding)
    } else if slice_types.contains(&ty) {
        regions.get(&binding).map(|region| region.owner)
    } else {
        None
    }
}

fn static_integer_value(
    expression: &Expression,
    bindings: &BTreeMap<BindingId, &severian_hir::Binding>,
    visiting: &mut BTreeSet<BindingId>,
) -> Option<i64> {
    match &expression.kind {
        ExpressionKind::Literal(severian_universal::LiteralValue::Integer(value)) => {
            value.parse().ok()
        }
        ExpressionKind::Binding(binding) if visiting.insert(*binding) => {
            static_integer_value(&bindings.get(binding)?.value, bindings, visiting)
        }
        ExpressionKind::Field { object, index } => {
            let binding = direct_binding(object)?;
            let value = &bindings.get(&binding)?.value;
            let aggregate = match &value.kind {
                ExpressionKind::Fallback { value, .. } => value.as_ref(),
                _ => value,
            };
            let ExpressionKind::Aggregate { fields, .. } = &aggregate.kind else {
                return None;
            };
            static_integer_value(fields.get(*index as usize)?, bindings, visiting)
        }
        ExpressionKind::Convert { operand, .. } => {
            static_integer_value(operand, bindings, visiting)
        }
        ExpressionKind::Binary {
            operator,
            left,
            right,
        } => {
            let left = static_integer_value(left, bindings, visiting)?;
            let right = static_integer_value(right, bindings, visiting)?;
            if *operator == severian_universal::BinaryOperator::Add {
                Some(left + right)
            } else if *operator == severian_universal::BinaryOperator::Subtract {
                Some(left - right)
            } else {
                None
            }
        }
        _ => None,
    }
}

fn regions_overlap(left: SliceRegion, right: SliceRegion) -> bool {
    match (left.start, left.end, right.start, right.end) {
        (Some(left_start), Some(left_end), Some(right_start), Some(right_end)) => {
            left_start < right_end && right_start < left_end
        }
        _ => true,
    }
}

fn validate_module(module: &Module) -> Result<(), Diagnostic> {
    let mut declared = BTreeSet::new();
    let bindings = module
        .bindings
        .iter()
        .map(|binding| (binding.id, binding))
        .collect::<BTreeMap<_, _>>();
    if module.initializer.statements.is_empty()
        && module
            .functions
            .iter()
            .all(|function| function.body.is_none())
    {
        for binding in &module.bindings {
            validate_expression(&binding.value, &declared)?;
            declared.insert(binding.id);
        }
        return Ok(());
    }
    for statement in &module.initializer.statements {
        validate_statement(statement, &bindings, &mut declared)?;
    }
    let globals = declared.clone();
    for function in &module.functions {
        let Some(body) = &function.body else {
            continue;
        };
        declared.clone_from(&globals);
        declared.extend(
            function
                .parameters
                .iter()
                .map(|parameter| parameter.binding),
        );
        for statement in &body.statements {
            validate_statement(statement, &bindings, &mut declared)?;
        }
    }
    Ok(())
}

fn validate_statement(
    statement: &Statement,
    bindings: &BTreeMap<BindingId, &severian_hir::Binding>,
    declared: &mut BTreeSet<BindingId>,
) -> Result<(), Diagnostic> {
    match statement {
        Statement::Sequence(block) | Statement::Placement { body: block, .. } => {
            for statement in &block.statements {
                validate_statement(statement, bindings, declared)?;
            }
            Ok(())
        }
        Statement::FieldUpdate { binding, value, .. }
        | Statement::FieldSet { binding, value, .. } => {
            if !declared.contains(binding) {
                return Err(Diagnostic::new(
                    "E000301",
                    "class value updated before it is available",
                    Some(value.span),
                ));
            }
            validate_expression(value, declared)
        }
        Statement::Binding(id) => {
            let binding = bindings
                .get(id)
                .expect("HIR statement references a binding");
            validate_expression(&binding.value, declared)?;
            declared.insert(*id);
            Ok(())
        }
        Statement::Expression(expression) => validate_expression(expression, declared),
        Statement::Return(value) => value
            .as_ref()
            .map_or(Ok(()), |value| validate_expression(value, declared)),
        Statement::Assert {
            condition, message, ..
        } => {
            validate_expression(condition, declared)?;
            if let Some(message) = message {
                validate_expression(message, declared)?;
            }
            Ok(())
        }
        Statement::ExpectThrow { body, .. } => {
            let mut body_declared = declared.clone();
            for statement in &body.statements {
                validate_statement(statement, bindings, &mut body_declared)?;
            }
            Ok(())
        }
        Statement::Try {
            body,
            catch_binding,
            catch_body,
            ..
        } => {
            let mut body_declared = declared.clone();
            for statement in &body.statements {
                validate_statement(statement, bindings, &mut body_declared)?;
            }
            let mut catch_declared = declared.clone();
            catch_declared.insert(*catch_binding);
            for statement in &catch_body.statements {
                validate_statement(statement, bindings, &mut catch_declared)?;
            }
            Ok(())
        }
        Statement::If {
            condition,
            then_block,
            else_block,
        } => {
            validate_expression(condition, declared)?;
            let mut then_declared = declared.clone();
            for statement in &then_block.statements {
                validate_statement(statement, bindings, &mut then_declared)?;
            }
            let mut else_declared = declared.clone();
            for statement in &else_block.statements {
                validate_statement(statement, bindings, &mut else_declared)?;
            }
            Ok(())
        }
        Statement::While {
            condition, body, ..
        } => {
            validate_expression(condition, declared)?;
            let mut body_declared = declared.clone();
            for statement in &body.statements {
                validate_statement(statement, bindings, &mut body_declared)?;
            }
            Ok(())
        }
        Statement::Break { .. } | Statement::Continue { .. } => Ok(()),
        Statement::Match { subject, arms } => {
            validate_expression(subject, declared)?;
            for arm in arms {
                let mut arm_declared = declared.clone();
                if let Some(binding) = arm.binding {
                    arm_declared.insert(binding);
                }
                for statement in &arm.body.statements {
                    validate_statement(statement, bindings, &mut arm_declared)?;
                }
            }
            Ok(())
        }
    }
}

fn validate_expression(
    expression: &Expression,
    declared: &BTreeSet<BindingId>,
) -> Result<(), Diagnostic> {
    match &expression.kind {
        ExpressionKind::Aggregate { fields, .. } => {
            for field in fields {
                validate_expression(field, declared)?;
            }
            Ok(())
        }
        ExpressionKind::Field { object, .. } => validate_expression(object, declared),
        ExpressionKind::Undefined | ExpressionKind::Literal(_) | ExpressionKind::Function(_) => Ok(()),
        ExpressionKind::Binding(id) if declared.contains(id) => Ok(()),
        ExpressionKind::Binding(_) => Err(Diagnostic::new(
            "E000301",
            "value used before it is available",
            Some(expression.span),
        )),
        ExpressionKind::AddressOf(id) if declared.contains(id) => Ok(()),
        ExpressionKind::AddressOf(_) => Err(Diagnostic::new(
            "E000301",
            "value addressed before it is available",
            Some(expression.span),
        )),
        ExpressionKind::Call { arguments, .. } => {
            for argument in arguments {
                validate_expression(argument, declared)?;
            }
            Ok(())
        }
        ExpressionKind::Async { expression, .. } | ExpressionKind::Await(expression) => {
            validate_expression(expression, declared)
        }
        ExpressionKind::AsyncFieldUpdate { binding, value, .. } => {
            if !declared.contains(binding) {
                return Err(Diagnostic::new(
                    "E000301",
                    "class value updated before it is available",
                    Some(expression.span),
                ));
            }
            validate_expression(value, declared)
        }
        ExpressionKind::Fallback {
            condition,
            value,
            fallback,
        } => {
            validate_expression(condition, declared)?;
            validate_expression(value, declared)?;
            validate_expression(fallback, declared)
        }
        ExpressionKind::Throw(error) => validate_expression(error, declared),
        ExpressionKind::Unary { operand, .. }
        | ExpressionKind::Borrow { operand, .. }
        | ExpressionKind::Move(operand)
        | ExpressionKind::Convert { operand, .. } => validate_expression(operand, declared),
        ExpressionKind::Binary { left, right, .. } => {
            validate_expression(left, declared)?;
            validate_expression(right, declared)
        }
    }
}
