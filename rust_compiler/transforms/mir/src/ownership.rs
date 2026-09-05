use crate::{BlockId, CfgBody, CfgStatement, LocalId, Operand, Place, Rvalue, Terminator};
use severian_universal::{EffectSet, TypeContext, TypeKind};
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
    pub holder: LocalId,
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

type OwnershipStates = BTreeMap<BlockId, OwnershipState>;
type OwnershipSolution = (
    OwnershipStates,
    OwnershipStates,
    BTreeSet<OwnershipError>,
    BTreeSet<LocalId>,
);

#[derive(Debug, Clone, Default, PartialEq, Eq)]
struct BlockLiveness {
    live_in: BTreeSet<LocalId>,
    live_out: BTreeSet<LocalId>,
    after_statements: Vec<BTreeSet<LocalId>>,
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
    let resources = resource_locals(body, types);
    let resource_set = resources.iter().copied().collect::<BTreeSet<_>>();
    for block in &mut body.blocks {
        let operand = match &mut block.terminator {
            Terminator::Return(Some(operand)) | Terminator::Throw(operand) => Some(operand),
            _ => None,
        };
        if let Some(operand) = operand {
            let transferred = match operand {
                Operand::Copy(place)
                    if place
                        .local_id()
                        .is_some_and(|local| resource_set.contains(&local)) =>
                {
                    Some(place.clone())
                }
                _ => None,
            };
            if let Some(place) = transferred {
                *operand = Operand::Move(place);
            }
        }
    }
    let (_, outputs, errors, _) = solve(body);
    if !errors.is_empty() {
        return Err(errors.into_iter().collect());
    }
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
                CfgStatement::Drop(place) => place.local_id(),
                _ => None,
            })
            .collect::<BTreeSet<_>>();
        let transferred = match &block.terminator {
            Terminator::Return(Some(operand)) | Terminator::Throw(operand) => {
                operand_local(operand)
            }
            _ => None,
        };
        for local in resources.iter().rev() {
            if state.initialized.contains(local)
                && !state.consumed_resources.contains(local)
                && !already_dropped.contains(local)
                && transferred != Some(*local)
            {
                block
                    .statements
                    .push(CfgStatement::Drop(Place::local(*local)));
                block.statement_spans.push(None);
            }
        }
    }
    Ok(())
}

fn solve(body: &CfgBody) -> OwnershipSolution {
    let predecessors = predecessors(body);
    let reachable = reachable_blocks(body);
    let liveness = calculate_liveness(body, &reachable);
    let arguments = body
        .locals
        .iter()
        .filter(|local| local.argument)
        .map(|local| local.id)
        .collect::<BTreeSet<_>>();
    let entry = OwnershipState {
        initialized: body
            .locals
            .iter()
            .filter(|local| local.argument)
            .map(|local| local.id)
            .collect(),
        ..OwnershipState::default()
    };
    // Definite initialization is a must-property. Cyclic non-entry blocks
    // therefore begin at the lattice top and monotonically lose locals as
    // predecessor intersections become known. Starting them at the empty set
    // lets loop backedges alternately add and remove initialization facts.
    let initialized_top = body.locals.iter().map(|local| local.id).collect();
    let top = OwnershipState {
        initialized: initialized_top,
        ..OwnershipState::default()
    };
    let mut inputs = reachable
        .iter()
        .copied()
        .map(|block| (block, top.clone()))
        .collect::<BTreeMap<_, _>>();
    let mut outputs = inputs.clone();
    inputs.insert(body.entry, entry);
    let mut queue = VecDeque::from_iter(reachable.iter().copied());
    let mut queued = reachable.clone();
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
        transfer_block(
            block,
            &mut output,
            &mut BTreeSet::new(),
            &mut BTreeSet::new(),
            &arguments,
            liveness.get(&block_id).expect("reachable block is live"),
        );
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
    // Diagnostics are properties of the fixed-point states. Recording them
    // during iteration reports false errors when a successor is visited before
    // all predecessor outputs are available.
    let mut errors = BTreeSet::new();
    let mut escaped = BTreeSet::new();
    for block_id in &reachable {
        let Some(block) = body.blocks.get(block_id.0 as usize) else {
            continue;
        };
        let mut state = inputs.get(block_id).cloned().unwrap_or_default();
        transfer_block(
            block,
            &mut state,
            &mut errors,
            &mut escaped,
            &arguments,
            liveness.get(block_id).expect("reachable block is live"),
        );
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
    arguments: &BTreeSet<LocalId>,
    liveness: &BlockLiveness,
) {
    state
        .loans
        .retain(|loan| liveness.live_in.contains(&loan.holder));
    for (index, statement) in block.statements.iter().enumerate() {
        match statement {
            CfgStatement::Assign(place, value) => {
                inspect_rvalue(value, place, block.id, state, errors);
                if let Some(local) = place.local_id() {
                    state.initialized.insert(local);
                    state.moved.remove(&local);
                    state.consumed_resources.remove(&local);
                }
            }
            CfgStatement::Drop(place) => {
                let Some(local) = place.local_id() else {
                    continue;
                };
                if !state.initialized.remove(&local) || !state.consumed_resources.insert(local) {
                    errors.insert(OwnershipError::DoubleDrop(local));
                }
                state
                    .loans
                    .retain(|loan| loan.place.local_id() != Some(local) && loan.holder != local);
            }
            CfgStatement::StorageLive(local) => {
                state.initialized.remove(local);
                state.moved.remove(local);
                state.consumed_resources.remove(local);
            }
            CfgStatement::StorageDead(local) => {
                state.initialized.remove(local);
                state
                    .loans
                    .retain(|loan| loan.place.local_id() != Some(*local) && loan.holder != *local);
            }
            CfgStatement::Assert {
                condition, message, ..
            } => {
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
                    if let Some(local) = result.local_id() {
                        state.initialized.insert(local);
                        state.moved.remove(&local);
                    }
                }
            }
            CfgStatement::Coverage(_) => {}
        }
        if let Some(live) = liveness.after_statements.get(index) {
            state.loans.retain(|loan| live.contains(&loan.holder));
        }
    }
    for operand in terminator_operands(&block.terminator) {
        inspect_operand(operand, state, errors);
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
            state.initialized.insert(local);
            state.moved.remove(&local);
            state.consumed_resources.remove(&local);
        }
    }
    // A structured task owns any borrow captured by its arguments until the
    // task result's last use. Ordinary call temporaries simply fall out of the
    // successor's live-in set.
    if let Terminator::Spawn {
        arguments,
        destination,
        ..
    } = &block.terminator
    {
        if let Some(task) = destination.local_id() {
            let argument_holders = arguments
                .iter()
                .filter_map(operand_local)
                .collect::<BTreeSet<_>>();
            let captured = state
                .loans
                .iter()
                .filter(|loan| argument_holders.contains(&loan.holder))
                .cloned()
                .map(|mut loan| {
                    loan.holder = task;
                    loan
                })
                .collect::<Vec<_>>();
            state.loans.extend(captured);
        }
    }
    if let Terminator::Return(Some(Operand::Copy(place))) = &block.terminator {
        if let Some(local) = place.local_id() {
            if let Some(owner) = state
                .loans
                .iter()
                .find(|loan| loan.holder == local)
                .and_then(|loan| loan.place.local_id())
                .filter(|owner| !arguments.contains(owner))
            {
                errors.insert(OwnershipError::EscapingBorrow(owner));
                escaped.insert(owner);
            }
        }
    }
    state
        .loans
        .retain(|loan| liveness.live_out.contains(&loan.holder));
}

fn operand_local(operand: &Operand) -> Option<LocalId> {
    match operand {
        Operand::Copy(place) | Operand::Move(place) => place.local_id(),
        Operand::Constant { .. } | Operand::Function(_) => None,
    }
}

fn inspect_rvalue(
    value: &Rvalue,
    destination: &Place,
    block: BlockId,
    state: &mut OwnershipState,
    errors: &mut BTreeSet<OwnershipError>,
) {
    match value {
        Rvalue::Use(operand) => {
            let propagated = if let (Some(source), Some(holder)) =
                (operand_local(operand), destination.local_id())
            {
                state
                    .loans
                    .iter()
                    .filter(|loan| loan.holder == source)
                    .cloned()
                    .map(|mut loan| {
                        loan.holder = holder;
                        loan
                    })
                    .collect::<Vec<_>>()
            } else {
                Vec::new()
            };
            inspect_operand(operand, state, errors);
            state.loans.extend(propagated);
        }
        Rvalue::Unary { operand, .. }
        | Rvalue::Convert { operand, .. }
        | Rvalue::Await { task: operand } => inspect_operand(operand, state, errors),
        Rvalue::Binary { left, right, .. } => {
            inspect_operand(left, state, errors);
            inspect_operand(right, state, errors);
        }
        Rvalue::BorrowShared(place) => {
            add_loan(place, destination, LoanKind::Shared, block, state, errors);
        }
        Rvalue::BorrowExclusive(place) => {
            add_loan(
                place,
                destination,
                LoanKind::Exclusive,
                block,
                state,
                errors,
            );
        }
        Rvalue::AddressOf(place) => inspect_operand(&Operand::Copy(place.clone()), state, errors),
        Rvalue::Aggregate { fields, .. } | Rvalue::Variant { fields, .. } => {
            for field in fields {
                inspect_operand(field, state, errors);
            }
        }
    }
}

fn add_loan(
    place: &Place,
    holder: &Place,
    kind: LoanKind,
    block: BlockId,
    state: &mut OwnershipState,
    errors: &mut BTreeSet<OwnershipError>,
) {
    let (Some(local), Some(holder)) = (place.local_id(), holder.local_id()) else {
        return;
    };
    if !state.initialized.contains(&local) || state.moved.contains(&local) {
        errors.insert(OwnershipError::UseAfterMove(local));
        return;
    }
    if state.loans.iter().any(|loan| {
        loan.place.local_id() == Some(local)
            && (loan.kind == LoanKind::Exclusive || kind == LoanKind::Exclusive)
    }) {
        errors.insert(OwnershipError::ConflictingLoan(local));
    } else {
        state.loans.insert(Loan {
            place: place.clone(),
            holder,
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
    let Some(local) = place.local_id() else {
        return;
    };
    if !state.initialized.contains(&local) || state.moved.contains(&local) {
        errors.insert(OwnershipError::UseAfterMove(local));
        return;
    }
    if moving {
        if state
            .loans
            .iter()
            .any(|loan| loan.place.local_id() == Some(local))
        {
            errors.insert(OwnershipError::ConflictingLoan(local));
        }
        if !state.moved.insert(local) {
            errors.insert(OwnershipError::DoubleMove(local));
        }
        state.initialized.remove(&local);
        state
            .loans
            .retain(|loan| loan.place.local_id() != Some(local));
    }
}

fn resource_locals(body: &CfgBody, types: &TypeContext) -> Vec<LocalId> {
    body.locals
        .iter()
        .filter(|local| {
            matches!(
                types.kind(local.ty),
                Some(TypeKind::Resource(_, _) | TypeKind::Tensor { .. })
            )
        })
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

fn calculate_liveness(
    body: &CfgBody,
    reachable: &BTreeSet<BlockId>,
) -> BTreeMap<BlockId, BlockLiveness> {
    let mut result = reachable
        .iter()
        .copied()
        .map(|block| (block, BlockLiveness::default()))
        .collect::<BTreeMap<_, _>>();
    loop {
        let previous = result.clone();
        for block in body
            .blocks
            .iter()
            .rev()
            .filter(|block| reachable.contains(&block.id))
        {
            let mut live_out = BTreeSet::new();
            for successor in successors(&block.terminator) {
                if let Some(successor) = previous.get(&successor) {
                    live_out.extend(&successor.live_in);
                }
            }
            let mut live = live_out.clone();
            apply_terminator_liveness(&block.terminator, &mut live);
            let mut after_statements = vec![BTreeSet::new(); block.statements.len()];
            for (index, statement) in block.statements.iter().enumerate().rev() {
                after_statements[index] = live.clone();
                apply_statement_liveness(statement, &mut live);
            }
            result.insert(
                block.id,
                BlockLiveness {
                    live_in: live,
                    live_out,
                    after_statements,
                },
            );
        }
        if result == previous {
            return result;
        }
    }
}

fn apply_statement_liveness(statement: &CfgStatement, live: &mut BTreeSet<LocalId>) {
    match statement {
        CfgStatement::Assign(destination, value) => {
            define_place(destination, live);
            use_rvalue(value, live);
            if !destination.projection.is_empty() {
                use_place(destination, live);
            }
        }
        CfgStatement::Drop(place) => {
            define_place(place, live);
            use_place(place, live);
        }
        CfgStatement::StorageLive(local) | CfgStatement::StorageDead(local) => {
            live.remove(local);
        }
        CfgStatement::Assert {
            condition, message, ..
        } => {
            use_operand(condition, live);
            if let Some(message) = message {
                use_operand(message, live);
            }
        }
        CfgStatement::Operation {
            operands, results, ..
        } => {
            for result in results {
                define_place(result, live);
            }
            for operand in operands {
                use_operand(operand, live);
            }
        }
        CfgStatement::Coverage(_) => {}
    }
}

fn apply_terminator_liveness(terminator: &Terminator, live: &mut BTreeSet<LocalId>) {
    match terminator {
        Terminator::Call {
            destination: Some(destination),
            ..
        }
        | Terminator::Spawn { destination, .. }
        | Terminator::SpawnFieldUpdate { destination, .. } => define_place(destination, live),
        _ => {}
    }
    for operand in terminator_operands(terminator) {
        use_operand(operand, live);
    }
}

fn use_rvalue(value: &Rvalue, live: &mut BTreeSet<LocalId>) {
    match value {
        Rvalue::Use(operand)
        | Rvalue::Unary { operand, .. }
        | Rvalue::Convert { operand, .. }
        | Rvalue::Await { task: operand } => use_operand(operand, live),
        Rvalue::Binary { left, right, .. } => {
            use_operand(left, live);
            use_operand(right, live);
        }
        Rvalue::BorrowShared(place) | Rvalue::BorrowExclusive(place) | Rvalue::AddressOf(place) => {
            use_place(place, live)
        }
        Rvalue::Aggregate { fields, .. } | Rvalue::Variant { fields, .. } => {
            for field in fields {
                use_operand(field, live);
            }
        }
    }
}

fn use_operand(operand: &Operand, live: &mut BTreeSet<LocalId>) {
    if let Operand::Copy(place) | Operand::Move(place) = operand {
        use_place(place, live);
    }
}

fn use_place(place: &Place, live: &mut BTreeSet<LocalId>) {
    if let Some(local) = place.local_id() {
        live.insert(local);
    }
    for projection in &place.projection {
        if let crate::Projection::Index(local) = projection {
            live.insert(*local);
        }
    }
}

fn define_place(place: &Place, live: &mut BTreeSet<LocalId>) {
    if place.projection.is_empty() {
        if let Some(local) = place.local_id() {
            live.remove(&local);
        }
    }
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
        Terminator::Spawn { target, .. } | Terminator::SpawnFieldUpdate { target, .. } => {
            vec![*target]
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
        Terminator::Spawn {
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
        Terminator::SpawnFieldUpdate { value, .. } | Terminator::Throw(value) => vec![value],
        Terminator::Return(value) => value.iter().collect(),
        Terminator::Unreachable => Vec::new(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{BasicBlock, Callee, LocalDecl};
    use severian_universal::{DeclarationId, DefId, LiteralValue, TypeId};

    fn local(id: u32, argument: bool) -> LocalDecl {
        LocalDecl {
            id: LocalId(id),
            ty: TypeId(0),
            mutable: true,
            argument,
            borrowed: false,
            span: None,
        }
    }

    fn initialize(local: u32) -> CfgStatement {
        CfgStatement::Assign(
            Place::local(LocalId(local)),
            Rvalue::Use(Operand::Constant {
                value: LiteralValue::Integer("1".into()),
                ty: TypeId(0),
            }),
        )
    }

    #[test]
    fn direct_call_borrows_end_at_the_call_boundary() {
        let source = LocalId(0);
        let shared = LocalId(1);
        let exclusive = LocalId(2);
        let body = CfgBody {
            entry: BlockId(0),
            blocks: vec![
                BasicBlock {
                    id: BlockId(0),
                    execution: None,
                    parameters: Vec::new(),
                    statements: vec![CfgStatement::Assign(
                        Place::local(shared),
                        Rvalue::BorrowShared(Place::local(source)),
                    )],
                    statement_spans: vec![None],
                    terminator: Terminator::Call {
                        callee: Callee::Direct {
                            instance: None,
                            function: DefId {
                                package: 0,
                                module: 0,
                                declaration: DeclarationId(0),
                            },
                            substitution: Default::default(),
                        },
                        arguments: vec![Operand::Copy(Place::local(shared))],
                        destination: None,
                        target: BlockId(1),
                        unwind: None,
                    },
                    terminator_span: None,
                },
                BasicBlock {
                    id: BlockId(1),
                    execution: None,
                    parameters: Vec::new(),
                    statements: vec![CfgStatement::Assign(
                        Place::local(exclusive),
                        Rvalue::BorrowExclusive(Place::local(source)),
                    )],
                    statement_spans: vec![None],
                    terminator: Terminator::Return(None),
                    terminator_span: None,
                },
            ],
            locals: vec![local(0, true), local(1, false), local(2, false)],
            return_type: TypeId(0),
        };

        let (_, _, errors, _) = solve(&body);
        assert!(errors.is_empty(), "{errors:?}");
    }

    #[test]
    fn copied_borrow_holder_keeps_the_loan_active() {
        let body = CfgBody {
            entry: BlockId(0),
            blocks: vec![BasicBlock {
                id: BlockId(0),
                execution: None,
                parameters: Vec::new(),
                statements: vec![
                    CfgStatement::Assign(
                        Place::local(LocalId(1)),
                        Rvalue::BorrowShared(Place::local(LocalId(0))),
                    ),
                    CfgStatement::Assign(
                        Place::local(LocalId(2)),
                        Rvalue::Use(Operand::Copy(Place::local(LocalId(1)))),
                    ),
                    CfgStatement::Assign(
                        Place::local(LocalId(3)),
                        Rvalue::BorrowExclusive(Place::local(LocalId(0))),
                    ),
                ],
                statement_spans: vec![None; 3],
                terminator: Terminator::Return(Some(Operand::Copy(Place::local(LocalId(2))))),
                terminator_span: None,
            }],
            locals: vec![
                local(0, true),
                local(1, false),
                local(2, false),
                local(3, false),
            ],
            return_type: TypeId(0),
        };

        let (_, _, errors, _) = solve(&body);
        assert!(errors.contains(&OwnershipError::ConflictingLoan(LocalId(0))));
    }

    #[test]
    fn retained_borrow_ends_after_its_last_use() {
        let body = CfgBody {
            entry: BlockId(0),
            blocks: vec![BasicBlock {
                id: BlockId(0),
                execution: None,
                parameters: Vec::new(),
                statements: vec![
                    CfgStatement::Assign(
                        Place::local(LocalId(1)),
                        Rvalue::BorrowShared(Place::local(LocalId(0))),
                    ),
                    CfgStatement::Assert {
                        condition: Operand::Copy(Place::local(LocalId(1))),
                        message: None,
                        origin: crate::AssertionOrigin {
                            statement_start: 0,
                            condition_start: 0,
                            condition_end: 0,
                            location: None,
                        },
                    },
                    CfgStatement::Assign(
                        Place::local(LocalId(2)),
                        Rvalue::BorrowExclusive(Place::local(LocalId(0))),
                    ),
                ],
                statement_spans: vec![None; 3],
                terminator: Terminator::Return(Some(Operand::Copy(Place::local(LocalId(2))))),
                terminator_span: None,
            }],
            locals: vec![local(0, true), local(1, false), local(2, false)],
            return_type: TypeId(0),
        };

        let (_, _, errors, _) = solve(&body);
        assert!(errors.is_empty(), "{errors:?}");
    }

    #[test]
    fn borrowed_local_cannot_escape_but_borrowed_argument_can() {
        let body = |argument| CfgBody {
            entry: BlockId(0),
            blocks: vec![BasicBlock {
                id: BlockId(0),
                execution: None,
                parameters: Vec::new(),
                statements: if argument {
                    vec![CfgStatement::Assign(
                        Place::local(LocalId(1)),
                        Rvalue::BorrowShared(Place::local(LocalId(0))),
                    )]
                } else {
                    vec![
                        initialize(0),
                        CfgStatement::Assign(
                            Place::local(LocalId(1)),
                            Rvalue::BorrowShared(Place::local(LocalId(0))),
                        ),
                    ]
                },
                statement_spans: vec![None; if argument { 1 } else { 2 }],
                terminator: Terminator::Return(Some(Operand::Copy(Place::local(LocalId(1))))),
                terminator_span: None,
            }],
            locals: vec![local(0, argument), local(1, false)],
            return_type: TypeId(0),
        };

        let (_, _, local_errors, _) = solve(&body(false));
        assert!(local_errors.contains(&OwnershipError::EscapingBorrow(LocalId(0))));

        let (_, _, argument_errors, _) = solve(&body(true));
        assert!(argument_errors.is_empty(), "{argument_errors:?}");
    }
}
