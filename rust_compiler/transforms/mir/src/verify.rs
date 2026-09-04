use crate::{
    BasicBlock, Callee, CfgBody, CfgStatement, GlobalDecl, LocalId, Module, Operand, Place,
    PlaceBase, Rvalue, Terminator,
};
use severian_universal::{IrContext, RegisteredOperation, TypeContext, TypeId, UniversalContext};
use std::collections::{BTreeMap, BTreeSet};
use std::fmt;

type DefinitionSignatures =
    BTreeMap<(severian_universal::DefId, severian_universal::Substitution), (Vec<TypeId>, TypeId)>;
type InstanceSignatures = BTreeMap<crate::FunctionId, (Vec<TypeId>, TypeId)>;
struct CallSignatures {
    definitions: DefinitionSignatures,
    instances: InstanceSignatures,
}
type CallContext<'a> = (&'a TypeContext, &'a CallSignatures);

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum VerifyError {
    InvalidBlock(u32),
    InvalidLocal(u32),
    InvalidGlobal(u32),
    SourceInfoArity(u32),
    MissingTerminator(u32),
    BlockArgumentArity(u32),
    BlockArgumentType(u32),
    UseBeforeDefinition { block: u32, local: u32 },
    CallTarget,
    CallArity,
    CallArgumentType {
        actual: TypeId,
        expected: TypeId,
        callee: Option<severian_universal::DefId>,
    },
    InvalidOwnershipState { block: u32, local: u32 },
    UnknownOperation(severian_universal::OpId),
    InvalidOperation(String),
}

impl fmt::Display for VerifyError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidBlock(block) => write!(formatter, "invalid basic block {block}"),
            Self::InvalidLocal(local) => write!(formatter, "invalid local {local}"),
            Self::InvalidGlobal(global) => write!(formatter, "invalid global {global}"),
            Self::SourceInfoArity(block) => {
                write!(
                    formatter,
                    "basic block {block} has mismatched source provenance"
                )
            }
            Self::MissingTerminator(block) => {
                write!(formatter, "basic block {block} has no terminator")
            }
            Self::BlockArgumentArity(block) => {
                write!(
                    formatter,
                    "basic block {block} has the wrong argument count"
                )
            }
            Self::BlockArgumentType(block) => {
                write!(
                    formatter,
                    "basic block {block} has an argument type mismatch"
                )
            }
            Self::UseBeforeDefinition { block, local } => {
                write!(
                    formatter,
                    "local {local} is used before definition in block {block}"
                )
            }
            Self::CallTarget => formatter.write_str("call refers to an unknown definition"),
            Self::CallArity => formatter.write_str("call has the wrong argument count"),
            Self::CallArgumentType {
                actual,
                expected,
                callee,
            } => {
                write!(
                    formatter,
                    "call argument type mismatch: {:?} is not assignable to {:?}",
                    actual, expected
                )?;
                if let Some(callee) = callee {
                    write!(formatter, " for {callee:?}")?;
                }
                Ok(())
            }
            Self::InvalidOwnershipState { block, local } => write!(
                formatter,
                "local {local} has an invalid ownership state in block {block}"
            ),
            Self::UnknownOperation(operation) => write!(
                formatter,
                "operation {:032x}:{:032x} is not registered",
                operation.dialect.0, operation.operation.0
            ),
            Self::InvalidOperation(message) => formatter.write_str(message),
        }
    }
}

impl std::error::Error for VerifyError {}

pub(crate) fn verify_structure(module: &Module) -> Result<(), VerifyError> {
    verify_body(&module.initializer, &module.globals, None, None)?;
    for function in &module.functions {
        if let Some(body) = &function.body {
            verify_body(body, &module.globals, None, None)?;
        }
    }
    Ok(())
}

pub fn verify(module: &Module, context: &UniversalContext) -> Result<(), VerifyError> {
    let definitions = module
        .functions
        .iter()
        .map(|function| {
            (
                (function.definition, function.substitution.clone()),
                (function.parameters.clone(), function.result),
            )
        })
        .collect::<BTreeMap<_, _>>();
    let instances = module
        .functions
        .iter()
        .map(|function| (function.id, (function.parameters.clone(), function.result)))
        .collect::<BTreeMap<_, _>>();
    let signatures = CallSignatures {
        definitions,
        instances,
    };
    verify_body(
        &module.initializer,
        &module.globals,
        Some((&context.types, &signatures)),
        Some(context),
    )?;
    for function in &module.functions {
        if let Some(body) = &function.body {
            verify_body(
                body,
                &module.globals,
                Some((&context.types, &signatures)),
                Some(context),
            )?;
        }
    }
    Ok(())
}

fn verify_body(
    body: &CfgBody,
    globals: &[GlobalDecl],
    calls: Option<CallContext<'_>>,
    context: Option<&UniversalContext>,
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
        if block.statement_spans.len() != block.statements.len() {
            return Err(VerifyError::SourceInfoArity(block.id.0));
        }
        if matches!(&block.terminator, Terminator::Unreachable) && !block.statements.is_empty() {
            return Err(VerifyError::MissingTerminator(block.id.0));
        }
        verify_targets(body, globals, block)?;
    }
    // Definite initialization is a forward must-analysis. Non-entry blocks
    // begin at the lattice top and predecessor intersections monotonically
    // remove locals. Beginning at the empty set validates loop bodies before
    // their preheaders have propagated and produces false use-before-definition
    // failures on backedges and call continuation blocks.
    let all_locals = body
        .locals
        .iter()
        .map(|local| local.id)
        .collect::<BTreeSet<_>>();
    let mut incoming = vec![all_locals; body.blocks.len()];
    incoming[body.entry.0 as usize] = body
        .locals
        .iter()
        .filter(|local| local.argument)
        .map(|local| local.id)
        .collect();
    let mut changed = true;
    while changed {
        changed = false;
        for block in &body.blocks {
            let mut state = incoming[block.id.0 as usize].clone();
            transfer_definitions(block, &mut state);
            for successor in successors(&block.terminator) {
                let target = &mut incoming[successor.0 as usize];
                let next = target.intersection(&state).copied().collect();
                if *target != next {
                    *target = next;
                    changed = true;
                }
            }
        }
    }
    for block in &body.blocks {
        let mut state = incoming[block.id.0 as usize].clone();
        transfer(body, globals, block, &mut state, calls, context)?;
    }
    Ok(())
}

fn transfer_definitions(block: &BasicBlock, state: &mut BTreeSet<LocalId>) {
    for statement in &block.statements {
        match statement {
            CfgStatement::Assign(place, _) => {
                if let Some(local) = place.local_id() {
                    state.insert(local);
                }
            }
            CfgStatement::Drop(place) => {
                if let Some(local) = place.local_id() {
                    state.remove(&local);
                }
            }
            CfgStatement::StorageDead(local) => {
                state.remove(local);
            }
            CfgStatement::Operation { results, .. } => {
                for result in results {
                    if let Some(local) = result.local_id() {
                        state.insert(local);
                    }
                }
            }
            CfgStatement::StorageLive(_)
            | CfgStatement::Assert { .. }
            | CfgStatement::Coverage(_) => {}
        }
    }
    let destination = match &block.terminator {
        Terminator::Call {
            destination: Some(destination),
            ..
        }
        | Terminator::Spawn { destination, .. }
        | Terminator::SpawnFieldUpdate { destination, .. } => Some(destination),
        _ => None,
    };
    if let Some(destination) = destination {
        if let Some(local) = destination.local_id() {
            state.insert(local);
        }
    }
}

fn verify_targets(
    body: &CfgBody,
    globals: &[GlobalDecl],
    block: &BasicBlock,
) -> Result<(), VerifyError> {
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
            if operand_type(body, globals, argument)? != body.locals[parameter.0 as usize].ty {
                return Err(VerifyError::BlockArgumentType(target.0));
            }
        }
    }
    Ok(())
}

fn transfer(
    body: &CfgBody,
    globals: &[GlobalDecl],
    block: &BasicBlock,
    state: &mut BTreeSet<LocalId>,
    calls: Option<CallContext<'_>>,
    context: Option<&UniversalContext>,
) -> Result<(), VerifyError> {
    for statement in &block.statements {
        match statement {
            CfgStatement::Assign(place, value) => {
                verify_rvalue(block.id.0, body, globals, value, state)?;
                verify_place(body, globals, place)?;
                if let Some(local) = place.local_id() {
                    state.insert(local);
                }
            }
            CfgStatement::Drop(place) => {
                verify_place(body, globals, place)?;
                let Some(local) = place.local_id() else {
                    continue;
                };
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
            CfgStatement::Assert {
                condition, message, ..
            } => {
                use_operand(block.id.0, body, globals, condition, state)?;
                if let Some(message) = message {
                    use_operand(block.id.0, body, globals, message, state)?;
                }
            }
            CfgStatement::Operation {
                id,
                operands,
                results,
                attributes,
            } => {
                for operand in operands {
                    use_operand(block.id.0, body, globals, operand, state)?;
                }
                for result in results {
                    verify_place(body, globals, result)?;
                    if let Some(local) = result.local_id() {
                        state.insert(local);
                    }
                }
                if let Some(context) = context {
                    if let Some(severian_universal::AttrValue::String(name)) =
                        attributes.get(&severian_universal::MLIR_OPERATION_NAME_ATTRIBUTE)
                    {
                        let Some((dialect, operation)) = name.split_once('.') else {
                            return Err(VerifyError::InvalidOperation(
                                "a direct MLIR operation must use a dialect.operation name".into(),
                            ));
                        };
                        if severian_universal::OpId::named(dialect, operation) != *id {
                            return Err(VerifyError::InvalidOperation(
                                "direct MLIR operation metadata does not match its operation ID"
                                    .into(),
                            ));
                        }
                        continue;
                    }
                    let interface = context
                        .operations
                        .interface(*id)
                        .ok_or(VerifyError::UnknownOperation(*id))?;
                    let operation = RegisteredOperation {
                        id: *id,
                        operands: operands
                            .iter()
                            .map(|operand| operand_type(body, globals, operand))
                            .collect::<Result<Vec<_>, _>>()?,
                        results: results
                            .iter()
                            .map(|place| place_type(body, globals, place))
                            .collect::<Result<Vec<_>, _>>()?,
                        attributes: attributes.clone(),
                    };
                    interface
                        .verify(
                            &operation,
                            &IrContext {
                                types: &context.types,
                                operations: &context.operations,
                            },
                        )
                        .map_err(|diagnostic| VerifyError::InvalidOperation(diagnostic.message))?;
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
            use_operand(block.id.0, body, globals, argument, state)?;
        }
        if let Some((types, signatures)) = calls {
            if let Callee::Direct {
                instance,
                function,
                substitution,
            } = callee
            {
                let signature = instance
                    .and_then(|instance| signatures.instances.get(&instance))
                    .or_else(|| {
                        signatures
                            .definitions
                            .get(&(*function, substitution.clone()))
                    });
                let Some((parameters, _)) = signature else {
                    return Err(VerifyError::CallTarget);
                };
                if parameters.len() != arguments.len() {
                    return Err(VerifyError::CallArity);
                }
                for (argument, parameter) in arguments.iter().zip(parameters) {
                    if !types.assignable(operand_type(body, globals, argument)?, *parameter) {
                        return Err(VerifyError::CallArgumentType {
                            actual: operand_type(body, globals, argument)?,
                            expected: *parameter,
                            callee: Some(*function),
                        });
                    }
                }
            }
        }
        if let Some(destination) = destination {
            verify_place(body, globals, destination)?;
            if let Some(local) = destination.local_id() {
                state.insert(local);
            }
        }
    } else {
        for operand in terminator_operands(&block.terminator) {
            use_operand(block.id.0, body, globals, operand, state)?;
        }
    }
    if let Terminator::SpawnFieldUpdate { place, .. } = &block.terminator {
        verify_place(body, globals, place)?;
    }
    if let Terminator::Spawn { destination, .. }
    | Terminator::SpawnFieldUpdate { destination, .. } = &block.terminator
    {
        verify_place(body, globals, destination)?;
        if let Some(local) = destination.local_id() {
            state.insert(local);
        }
    }
    Ok(())
}

fn verify_rvalue(
    block: u32,
    body: &CfgBody,
    globals: &[GlobalDecl],
    value: &Rvalue,
    state: &mut BTreeSet<LocalId>,
) -> Result<(), VerifyError> {
    let operands = match value {
        Rvalue::Use(value) => vec![value],
        Rvalue::Unary { operand, .. }
        | Rvalue::Convert { operand, .. }
        | Rvalue::Await { task: operand } => vec![operand],
        Rvalue::Binary { left, right, .. } => vec![left, right],
        Rvalue::Aggregate { fields, .. } => fields.iter().collect(),
        Rvalue::BorrowShared(place) | Rvalue::BorrowExclusive(place) | Rvalue::AddressOf(place) => {
            verify_place(body, globals, place)?;
            if let Some(local) = place.local_id().filter(|local| !state.contains(local)) {
                return Err(VerifyError::UseBeforeDefinition {
                    block,
                    local: local.0,
                });
            }
            Vec::new()
        }
    };
    for operand in operands {
        use_operand(block, body, globals, operand, state)?;
    }
    Ok(())
}

fn use_operand(
    block: u32,
    body: &CfgBody,
    globals: &[GlobalDecl],
    operand: &Operand,
    state: &mut BTreeSet<LocalId>,
) -> Result<(), VerifyError> {
    if let Operand::Copy(place) | Operand::Move(place) = operand {
        verify_place(body, globals, place)?;
        let Some(local) = place.local_id() else {
            return Ok(());
        };
        if !state.contains(&local) {
            return Err(VerifyError::UseBeforeDefinition {
                block,
                local: local.0,
            });
        }
        if matches!(operand, Operand::Move(_)) {
            state.remove(&local);
        }
    }
    Ok(())
}

fn verify_place(body: &CfgBody, globals: &[GlobalDecl], place: &Place) -> Result<(), VerifyError> {
    match place.base {
        PlaceBase::Local(local) if local.0 as usize >= body.locals.len() => {
            Err(VerifyError::InvalidLocal(local.0))
        }
        PlaceBase::Global(global) if global.0 as usize >= globals.len() => {
            Err(VerifyError::InvalidGlobal(global.0))
        }
        _ => Ok(()),
    }
}

fn place_type(
    body: &CfgBody,
    globals: &[GlobalDecl],
    place: &Place,
) -> Result<TypeId, VerifyError> {
    verify_place(body, globals, place)?;
    Ok(match place.base {
        PlaceBase::Local(local) => body.locals[local.0 as usize].ty,
        PlaceBase::Global(global) => globals[global.0 as usize].ty,
    })
}

fn operand_type(
    body: &CfgBody,
    globals: &[GlobalDecl],
    operand: &Operand,
) -> Result<TypeId, VerifyError> {
    match operand {
        Operand::Copy(place) | Operand::Move(place) => place_type(body, globals, place),
        Operand::Constant { ty, .. } => Ok(*ty),
        Operand::Function(_) => Err(VerifyError::CallArgumentType {
            actual: TypeId(u32::MAX),
            expected: TypeId(u32::MAX),
            callee: None,
        }),
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
        Terminator::Spawn { target, .. } | Terminator::SpawnFieldUpdate { target, .. } => {
            vec![*target]
        }
        Terminator::Return(_) | Terminator::Throw(_) | Terminator::Unreachable => Vec::new(),
    }
}

fn terminator_operands(terminator: &Terminator) -> Vec<&Operand> {
    match terminator {
        Terminator::Goto(_, arguments) | Terminator::Call { arguments, .. } => {
            arguments.iter().collect()
        }
        Terminator::Branch { condition, .. } => vec![condition],
        Terminator::Switch { discriminant, .. } => vec![discriminant],
        Terminator::Return(value) => value.iter().collect(),
        Terminator::Throw(value) | Terminator::SpawnFieldUpdate { value, .. } => vec![value],
        Terminator::Spawn {
            callee, arguments, ..
        } => {
            let mut operands = arguments.iter().collect::<Vec<_>>();
            match callee {
                Callee::FunctionValue(operand) => operands.push(operand),
                Callee::Method { receiver, .. } => operands.push(receiver),
                Callee::Direct { .. } | Callee::Constructor { .. } | Callee::Intrinsic(_) => {}
            }
            operands
        }
        Terminator::Unreachable => Vec::new(),
    }
}
