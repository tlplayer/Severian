use crate::{
    BasicBlock, Callee, CfgBody, CfgStatement, LocalId, Module, Operand, Place, Rvalue,
    Terminator,
};
use severian_universal::{TypeContext, TypeId};
use std::collections::{BTreeMap, BTreeSet};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum VerifyError {
    InvalidBlock(u32),
    InvalidLocal(u32),
    MissingTerminator(u32),
    BlockArgumentArity(u32),
    BlockArgumentType(u32),
    UseBeforeDefinition { block: u32, local: u32 },
    CallTarget,
    CallArity,
    CallArgumentType,
    InvalidOwnershipState { block: u32, local: u32 },
}

pub(crate) fn verify_structure(module: &Module) -> Result<(), VerifyError> {
    verify_body(&module.initializer_cfg, None)?;
    for function in &module.functions {
        if let Some(body) = &function.cfg {
            verify_body(body, None)?;
        }
    }
    Ok(())
}

pub fn verify(module: &Module, types: &TypeContext) -> Result<(), VerifyError> {
    let signatures = module
        .functions
        .iter()
        .map(|function| {
            (
                function.definition,
                (
                    function
                        .parameters
                        .iter()
                        .map(|value| module.values[value.0 as usize].type_id)
                        .collect::<Vec<_>>(),
                    function.result,
                ),
            )
        })
        .collect::<BTreeMap<_, _>>();
    verify_body(&module.initializer_cfg, Some((types, &signatures)))?;
    for function in &module.functions {
        if let Some(body) = &function.cfg {
            verify_body(body, Some((types, &signatures)))?;
        }
    }
    Ok(())
}

fn verify_body(
    body: &CfgBody,
    calls: Option<(&TypeContext, &BTreeMap<severian_universal::DefId, (Vec<TypeId>, TypeId)>)>,
) -> Result<(), VerifyError> {
    if body.entry.0 as usize >= body.blocks.len() {
        return Err(VerifyError::InvalidBlock(body.entry.0));
    }
    for (index, local) in body.locals.iter().enumerate() {
        if local.id.0 as usize != index {
            return Err(VerifyError::InvalidLocal(local.id.0));
        }
    }
    for (index, block) in body.blocks.iter().enumerate() {
        if block.id.0 as usize != index {
            return Err(VerifyError::InvalidBlock(block.id.0));
        }
        if matches!(&block.terminator, Terminator::Unreachable) && !block.statements.is_empty() {
            return Err(VerifyError::MissingTerminator(block.id.0));
        }
        verify_targets(body, block)?;
    }
    let mut incoming = vec![BTreeSet::new(); body.blocks.len()];
    incoming[body.entry.0 as usize].extend(
        body.locals
            .iter()
            .filter(|local| local.argument)
            .map(|local| local.id),
    );
    let mut changed = true;
    while changed {
        changed = false;
        for block in &body.blocks {
            let mut state = incoming[block.id.0 as usize].clone();
            transfer(body, block, &mut state, calls)?;
            for successor in successors(&block.terminator) {
                let target = &mut incoming[successor.0 as usize];
                let next = if target.is_empty() {
                    state.clone()
                } else {
                    target.intersection(&state).copied().collect()
                };
                if *target != next {
                    *target = next;
                    changed = true;
                }
            }
        }
    }
    Ok(())
}

fn verify_targets(body: &CfgBody, block: &BasicBlock) -> Result<(), VerifyError> {
    for target in successors(&block.terminator) {
        if target.0 as usize >= body.blocks.len() {
            return Err(VerifyError::InvalidBlock(target.0));
        }
    }
    if let Terminator::Goto(target, arguments) = &block.terminator {
        let parameters = &body.blocks[target.0 as usize].parameters;
        if arguments.len() != parameters.len() {
            return Err(VerifyError::BlockArgumentArity(target.0));
        }
        for (argument, parameter) in arguments.iter().zip(parameters) {
            if operand_type(body, argument)? != body.locals[parameter.0 as usize].ty {
                return Err(VerifyError::BlockArgumentType(target.0));
            }
        }
    }
    Ok(())
}

fn transfer(
    body: &CfgBody,
    block: &BasicBlock,
    state: &mut BTreeSet<LocalId>,
    calls: Option<(&TypeContext, &BTreeMap<severian_universal::DefId, (Vec<TypeId>, TypeId)>)>,
) -> Result<(), VerifyError> {
    for statement in &block.statements {
        match statement {
            CfgStatement::Assign(place, value) => {
                verify_rvalue(block.id.0, body, value, state)?;
                verify_place(body, place)?;
                state.insert(place.local);
            }
            CfgStatement::Drop(place) => {
                let local = place.local;
                if !state.remove(&local) {
                    return Err(VerifyError::InvalidOwnershipState {
                        block: block.id.0,
                        local: local.0,
                    });
                }
            }
            CfgStatement::StorageDead(local) => {
                let local = *local;
                if !state.remove(&local) {
                    return Err(VerifyError::InvalidOwnershipState {
                        block: block.id.0,
                        local: local.0,
                    });
                }
            }
            CfgStatement::StorageLive(local) => {
                if local.0 as usize >= body.locals.len() {
                    return Err(VerifyError::InvalidLocal(local.0));
                }
            }
            CfgStatement::Assert { condition, message } => {
                use_operand(block.id.0, body, condition, state)?;
                if let Some(message) = message {
                    use_operand(block.id.0, body, message, state)?;
                }
            }
            CfgStatement::Operation {
                operands, results, ..
            } => {
                for operand in operands {
                    use_operand(block.id.0, body, operand, state)?;
                }
                for result in results {
                    verify_place(body, result)?;
                    state.insert(result.local);
                }
            }
            CfgStatement::Coverage(_) => {}
        }
    }
    if let Terminator::Call {
        callee,
        arguments,
        destination,
        ..
    } = &block.terminator
    {
        for argument in arguments {
            use_operand(block.id.0, body, argument, state)?;
        }
        if let Some((types, signatures)) = calls {
            if let Callee::Direct { function, .. } = callee {
                let Some((parameters, _)) = signatures.get(function) else {
                    return Err(VerifyError::CallTarget);
                };
                if parameters.len() != arguments.len() {
                    return Err(VerifyError::CallArity);
                }
                for (argument, parameter) in arguments.iter().zip(parameters) {
                    if !types.assignable(operand_type(body, argument)?, *parameter) {
                        return Err(VerifyError::CallArgumentType);
                    }
                }
            }
        }
        if let Some(destination) = destination {
            state.insert(destination.local);
        }
    } else {
        for operand in terminator_operands(&block.terminator) {
            use_operand(block.id.0, body, operand, state)?;
        }
    }
    Ok(())
}

fn verify_rvalue(
    block: u32,
    body: &CfgBody,
    value: &Rvalue,
    state: &mut BTreeSet<LocalId>,
) -> Result<(), VerifyError> {
    let operands = match value {
        Rvalue::Use(value) => vec![value],
        Rvalue::Unary { operand, .. } | Rvalue::Convert { operand, .. } => vec![operand],
        Rvalue::Binary { left, right, .. } => vec![left, right],
        Rvalue::Aggregate { fields, .. } => fields.iter().collect(),
        Rvalue::BorrowShared(place) | Rvalue::BorrowExclusive(place) => {
            verify_place(body, place)?;
            if !state.contains(&place.local) {
                return Err(VerifyError::UseBeforeDefinition {
                    block,
                    local: place.local.0,
                });
            }
            Vec::new()
        }
    };
    for operand in operands {
        use_operand(block, body, operand, state)?;
    }
    Ok(())
}

fn use_operand(
    block: u32,
    body: &CfgBody,
    operand: &Operand,
    state: &mut BTreeSet<LocalId>,
) -> Result<(), VerifyError> {
    if let Operand::Copy(place) | Operand::Move(place) = operand {
        verify_place(body, place)?;
        if !state.contains(&place.local) {
            return Err(VerifyError::UseBeforeDefinition {
                block,
                local: place.local.0,
            });
        }
        if matches!(operand, Operand::Move(_)) {
            state.remove(&place.local);
        }
    }
    Ok(())
}

fn verify_place(body: &CfgBody, place: &Place) -> Result<(), VerifyError> {
    if place.local.0 as usize >= body.locals.len() {
        Err(VerifyError::InvalidLocal(place.local.0))
    } else {
        Ok(())
    }
}

fn operand_type(body: &CfgBody, operand: &Operand) -> Result<TypeId, VerifyError> {
    match operand {
        Operand::Copy(place) | Operand::Move(place) => {
            verify_place(body, place)?;
            Ok(body.locals[place.local.0 as usize].ty)
        }
        Operand::Constant { ty, .. } => Ok(*ty),
        Operand::Function(_) => Err(VerifyError::CallArgumentType),
    }
}

fn successors(terminator: &Terminator) -> Vec<crate::BlockId> {
    match terminator {
        Terminator::Goto(target, _) => vec![*target],
        Terminator::Branch {
            then_block,
            else_block,
            ..
        } => vec![*then_block, *else_block],
        Terminator::Switch {
            targets, fallback, ..
        } => targets
            .iter()
            .map(|(_, block)| *block)
            .chain([*fallback])
            .collect(),
        Terminator::Call { target, unwind, .. } => {
            [Some(*target), *unwind].into_iter().flatten().collect()
        }
        Terminator::Return(_) | Terminator::Throw(_) | Terminator::Unreachable => Vec::new(),
    }
}

fn terminator_operands(terminator: &Terminator) -> Vec<&Operand> {
    match terminator {
        Terminator::Goto(_, arguments) => arguments.iter().collect(),
        Terminator::Branch { condition, .. } => vec![condition],
        Terminator::Switch { discriminant, .. } => vec![discriminant],
        Terminator::Return(value) => value.iter().collect(),
        Terminator::Throw(value) => vec![value],
        Terminator::Call { arguments, .. } => arguments.iter().collect(),
        Terminator::Unreachable => Vec::new(),
    }
}
