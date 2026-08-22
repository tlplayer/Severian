use crate::{BlockId, CfgBody, CfgStatement, LocalId, Operand, Place, Rvalue, Terminator};
use severian_universal::{EffectSet, TyKind, TypeContext};
use std::collections::{BTreeMap, BTreeSet, VecDeque};
use std::fmt;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum LoanKind {
    Shared,
    Exclusive,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct Loan {
    pub place: Place,
    pub kind: LoanKind,
    pub block: BlockId,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub enum OwnershipError {
    UseAfterMove(LocalId),
    DoubleMove(LocalId),
    ConflictingLoan(LocalId),
    EscapingBorrow(LocalId),
    ResourceNotConsumed(LocalId),
    DoubleDrop(LocalId),
}

impl fmt::Display for OwnershipError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UseAfterMove(local) => write!(formatter, "use of moved local {}", local.0),
            Self::DoubleMove(local) => write!(formatter, "local {} is moved twice", local.0),
            Self::ConflictingLoan(local) => {
                write!(formatter, "local {} has conflicting loans", local.0)
            }
            Self::EscapingBorrow(local) => write!(formatter, "borrow of local {} escapes", local.0),
            Self::ResourceNotConsumed(local) => {
                write!(formatter, "resource local {} is not consumed", local.0)
            }
            Self::DoubleDrop(local) => write!(formatter, "local {} is dropped twice", local.0),
        }
    }
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct OwnershipState {
    pub initialized: BTreeSet<LocalId>,
    pub moved: BTreeSet<LocalId>,
    pub loans: BTreeSet<Loan>,
    pub consumed_resources: BTreeSet<LocalId>,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct OwnershipReport {
    pub initialized: BTreeSet<LocalId>,
    pub moved: BTreeSet<LocalId>,
    pub loans: Vec<Loan>,
    pub escaped: BTreeSet<LocalId>,
    pub consumed_resources: BTreeSet<LocalId>,
    pub call_effects: BTreeMap<BlockId, EffectSet>,
    pub block_inputs: BTreeMap<BlockId, OwnershipState>,
    pub block_outputs: BTreeMap<BlockId, OwnershipState>,
}

pub fn analyze_ownership(
    body: &CfgBody,
    types: &TypeContext,
) -> Result<OwnershipReport, Vec<OwnershipError>> {
    let (inputs, outputs, errors, escaped) = solve(body);
    let mut report = OwnershipReport {
        block_inputs: inputs,
        block_outputs: outputs,
        escaped,
        ..OwnershipReport::default()
    };
    for state in report.block_outputs.values() {
        report.initialized.extend(&state.initialized);
        report.moved.extend(&state.moved);
        report.loans.extend(state.loans.iter().cloned());
        report.consumed_resources.extend(&state.consumed_resources);
    }
    report.loans.sort();
    report.loans.dedup();

    let resources = resource_locals(body, types);
    let mut errors = errors;
    for block in &body.blocks {
        if !matches!(
            block.terminator,
            Terminator::Return(_) | Terminator::Throw(_)
        ) {
            continue;
        }
        let Some(state) = report.block_outputs.get(&block.id) else {
            continue;
        };
        for local in &resources {
            if state.initialized.contains(local) && !state.consumed_resources.contains(local) {
                errors.insert(OwnershipError::ResourceNotConsumed(*local));
            }
        }
    }

    if errors.is_empty() {
        Ok(report)
    } else {
        Err(errors.into_iter().collect())
    }
}

pub fn elaborate_drops(body: &mut CfgBody, types: &TypeContext) -> Result<(), Vec<OwnershipError>> {
    let (_, outputs, errors, _) = solve(body);
    if !errors.is_empty() {
        return Err(errors.into_iter().collect());
    }
    let resources = resource_locals(body, types);
    for block in &mut body.blocks {
        if !matches!(
            block.terminator,
            Terminator::Return(_) | Terminator::Throw(_)
        ) {
            continue;
        }
        let Some(state) = outputs.get(&block.id) else {
            continue;
        };
        let already_dropped = block
            .statements
            .iter()
            .filter_map(|statement| match statement {
                CfgStatement::Drop(place) => Some(place.local),
                _ => None,
            })
            .collect::<BTreeSet<_>>();
        for local in resources.iter().rev() {
            if state.initialized.contains(local)
                && !state.consumed_resources.contains(local)
                && !already_dropped.contains(local)
            {
                block
                    .statements
                    .push(CfgStatement::Drop(Place::local(*local)));
            }
        }
    }
    Ok(())
}

fn solve(
    body: &CfgBody,
) -> (
    BTreeMap<BlockId, OwnershipState>,
    BTreeMap<BlockId, OwnershipState>,
    BTreeSet<OwnershipError>,
    BTreeSet<LocalId>,
) {
    let predecessors = predecessors(body);
    let reachable = reachable_blocks(body);
    let entry = OwnershipState {
        initialized: body
            .locals
            .iter()
            .filter(|local| local.argument)
            .map(|local| local.id)
            .collect(),
        ..OwnershipState::default()
    };
    let mut inputs = BTreeMap::new();
    let mut outputs = BTreeMap::new();
    inputs.insert(body.entry, entry);
    let mut queue = VecDeque::from_iter(reachable.iter().copied());
    let mut queued = reachable.clone();
    let mut errors = BTreeSet::new();
    let mut escaped = BTreeSet::new();

    while let Some(block_id) = queue.pop_front() {
        queued.remove(&block_id);
        let input = if block_id == body.entry {
            inputs.get(&block_id).cloned().unwrap_or_default()
        } else {
            join_predecessors(
                predecessors
                    .get(&block_id)
                    .map(Vec::as_slice)
                    .unwrap_or(&[]),
                &outputs,
            )
        };
        if inputs.get(&block_id) != Some(&input) {
            inputs.insert(block_id, input.clone());
        }
        let Some(block) = body.blocks.get(block_id.0 as usize) else {
            continue;
        };
        let mut output = input;
        transfer_block(block, &mut output, &mut errors, &mut escaped);
        if outputs.get(&block_id) == Some(&output) {
            continue;
        }
        outputs.insert(block_id, output);
        for successor in successors(&block.terminator) {
            if reachable.contains(&successor) && queued.insert(successor) {
                queue.push_back(successor);
            }
        }
    }
    (inputs, outputs, errors, escaped)
}

fn join_predecessors(
    predecessors: &[BlockId],
    outputs: &BTreeMap<BlockId, OwnershipState>,
) -> OwnershipState {
    let mut states = predecessors
        .iter()
        .filter_map(|predecessor| outputs.get(predecessor));
    let Some(first) = states.next() else {
        return OwnershipState::default();
    };
    let mut joined = first.clone();
    for state in states {
        joined.initialized = joined
            .initialized
            .intersection(&state.initialized)
            .copied()
            .collect();
        joined.moved.extend(&state.moved);
        joined.loans.extend(state.loans.iter().cloned());
        joined.consumed_resources.extend(&state.consumed_resources);
    }
    joined
}

fn transfer_block(
    block: &crate::BasicBlock,
    state: &mut OwnershipState,
    errors: &mut BTreeSet<OwnershipError>,
    escaped: &mut BTreeSet<LocalId>,
) {
    for statement in &block.statements {
        match statement {
            CfgStatement::Assign(place, value) => {
                inspect_rvalue(value, block.id, state, errors);
                state.initialized.insert(place.local);
                state.moved.remove(&place.local);
                state.consumed_resources.remove(&place.local);
            }
            CfgStatement::Drop(place) => {
                if !state.initialized.remove(&place.local)
                    || !state.consumed_resources.insert(place.local)
                {
                    errors.insert(OwnershipError::DoubleDrop(place.local));
                }
                state.loans.retain(|loan| loan.place.local != place.local);
            }
            CfgStatement::StorageLive(local) => {
                state.initialized.remove(local);
                state.moved.remove(local);
                state.consumed_resources.remove(local);
            }
            CfgStatement::StorageDead(local) => {
                state.initialized.remove(local);
                state.loans.retain(|loan| loan.place.local != *local);
            }
            CfgStatement::Assert { condition, message } => {
                inspect_operand(condition, state, errors);
                if let Some(message) = message {
                    inspect_operand(message, state, errors);
                }
            }
            CfgStatement::Operation {
                operands, results, ..
            } => {
                for operand in operands {
                    inspect_operand(operand, state, errors);
                }
                for result in results {
                    state.initialized.insert(result.local);
                    state.moved.remove(&result.local);
                }
            }
            CfgStatement::Coverage(_) => {}
        }
    }
    for operand in terminator_operands(&block.terminator) {
        inspect_operand(operand, state, errors);
    }
    if let Terminator::Return(Some(Operand::Copy(place))) = &block.terminator {
        if state
            .loans
            .iter()
            .any(|loan| loan.place.local == place.local)
        {
            errors.insert(OwnershipError::EscapingBorrow(place.local));
            escaped.insert(place.local);
        }
    }
}

fn inspect_rvalue(
    value: &Rvalue,
    block: BlockId,
    state: &mut OwnershipState,
    errors: &mut BTreeSet<OwnershipError>,
) {
    match value {
        Rvalue::Use(operand) | Rvalue::Unary { operand, .. } | Rvalue::Convert { operand, .. } => {
            inspect_operand(operand, state, errors)
        }
        Rvalue::Binary { left, right, .. } => {
            inspect_operand(left, state, errors);
            inspect_operand(right, state, errors);
        }
        Rvalue::BorrowShared(place) => {
            add_loan(place, LoanKind::Shared, block, state, errors);
        }
        Rvalue::BorrowExclusive(place) => {
            add_loan(place, LoanKind::Exclusive, block, state, errors);
        }
        Rvalue::Aggregate { fields, .. } => {
            for field in fields {
                inspect_operand(field, state, errors);
            }
        }
    }
}

fn add_loan(
    place: &Place,
    kind: LoanKind,
    block: BlockId,
    state: &mut OwnershipState,
    errors: &mut BTreeSet<OwnershipError>,
) {
    if state.loans.iter().any(|loan| {
        loan.place.local == place.local
            && (loan.kind == LoanKind::Exclusive || kind == LoanKind::Exclusive)
    }) {
        errors.insert(OwnershipError::ConflictingLoan(place.local));
    } else {
        state.loans.insert(Loan {
            place: place.clone(),
            kind,
            block,
        });
    }
}

fn inspect_operand(
    operand: &Operand,
    state: &mut OwnershipState,
    errors: &mut BTreeSet<OwnershipError>,
) {
    let (place, moving) = match operand {
        Operand::Copy(place) => (place, false),
        Operand::Move(place) => (place, true),
        Operand::Constant { .. } | Operand::Function(_) => return,
    };
    if !state.initialized.contains(&place.local) || state.moved.contains(&place.local) {
        errors.insert(OwnershipError::UseAfterMove(place.local));
        return;
    }
    if moving {
        if !state.moved.insert(place.local) {
            errors.insert(OwnershipError::DoubleMove(place.local));
        }
        state.initialized.remove(&place.local);
        state.loans.retain(|loan| loan.place.local != place.local);
    }
}

fn resource_locals(body: &CfgBody, types: &TypeContext) -> Vec<LocalId> {
    body.locals
        .iter()
        .filter(|local| matches!(types.kind(local.ty), Some(TyKind::Resource(_, _))))
        .map(|local| local.id)
        .collect()
}

fn predecessors(body: &CfgBody) -> BTreeMap<BlockId, Vec<BlockId>> {
    let mut result = BTreeMap::<_, Vec<_>>::new();
    for block in &body.blocks {
        for successor in successors(&block.terminator) {
            result.entry(successor).or_default().push(block.id);
        }
    }
    result
}

fn reachable_blocks(body: &CfgBody) -> BTreeSet<BlockId> {
    let mut reachable = BTreeSet::new();
    let mut queue = VecDeque::from([body.entry]);
    while let Some(block) = queue.pop_front() {
        if !reachable.insert(block) {
            continue;
        }
        if let Some(block) = body.blocks.get(block.0 as usize) {
            queue.extend(successors(&block.terminator));
        }
    }
    reachable
}

fn successors(terminator: &Terminator) -> Vec<BlockId> {
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
        Terminator::Call {
            callee, arguments, ..
        } => {
            let mut operands = arguments.iter().collect::<Vec<_>>();
            match callee {
                crate::Callee::FunctionValue(operand) => operands.push(operand),
                crate::Callee::Method { receiver, .. } => operands.push(receiver),
                crate::Callee::Direct { .. }
                | crate::Callee::Constructor { .. }
                | crate::Callee::Intrinsic(_) => {}
            }
            operands
        }
        Terminator::Return(value) => value.iter().collect(),
        Terminator::Throw(value) => vec![value],
        Terminator::Unreachable => Vec::new(),
    }
}
