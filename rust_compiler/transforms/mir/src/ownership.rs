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
    ConsumeView(LocalId),
    UninitializedUse(LocalId),
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
            Self::UninitializedUse(local) => write!(formatter, "use of uninitialized field in local {}", local.0),
            Self::ConsumeView(local) => write!(formatter, "cannot consume or release view local {}; transfer ownership explicitly", local.0),
        }
    }
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct OwnershipState {
    pub initialized: BTreeSet<LocalId>,
    pub moved: BTreeSet<LocalId>,
    pub moved_fields: BTreeSet<Place>,
    pub uninitialized_fields: BTreeSet<Place>,
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

/// A checked body is borrowed immutably for the lifetime of its ownership plan.
/// Lowering cannot mutate the graph while using this proof. Any transformation
/// must finish first and obtain a fresh plan from this authority.
pub struct OwnershipPlan<'a> {
    body: &'a CfgBody,
    types: &'a TypeContext,
    report: OwnershipReport,
}

impl<'a> OwnershipPlan<'a> {
    pub fn body(&self) -> &'a CfgBody { self.body }
    pub fn types(&self) -> &'a TypeContext { self.types }
    pub fn report(&self) -> &OwnershipReport { &self.report }
}

pub fn ownership_plan<'a>(body: &'a CfgBody, types: &'a TypeContext) -> Result<OwnershipPlan<'a>, Vec<OwnershipError>> {
    analyze_ownership(body, types).map(|report| OwnershipPlan { body, types, report })
}

fn view_consumption_errors(body: &CfgBody, types: &TypeContext) -> BTreeSet<OwnershipError> {
    let views = class_borrowed_locals(body, types);
    let mut errors = BTreeSet::new();
    for block in &body.blocks {
        for statement in &block.statements {
            let operands: Vec<&Operand> = match statement {
                CfgStatement::Assign(_, Rvalue::Use(operand) | Rvalue::Unary { operand, .. } | Rvalue::Convert { operand, .. }) => vec![operand],
                CfgStatement::Assign(_, Rvalue::Binary { left, right, .. }) => vec![left, right],
                CfgStatement::Assign(_, Rvalue::Aggregate { fields, .. } | Rvalue::Variant { fields, .. }) => fields.iter().collect(),
                CfgStatement::Operation { operands, .. } => operands.iter().collect(),
                _ => vec![],
            };
            for operand in operands {
                if let Operand::Move(place) = operand {
                    if place.local_id().is_some_and(|local| views.contains(&local)) {
                        errors.insert(OwnershipError::ConsumeView(place.local_id().unwrap()));
                    }
                }
            }
            if let CfgStatement::Drop(place) = statement {
                if place.projection.is_empty() && place.local_id().is_some_and(|local| views.contains(&local)) {
                    errors.insert(OwnershipError::ConsumeView(place.local_id().unwrap()));
                }
            }
        }
        for operand in terminator_operands(&block.terminator) {
            if let Operand::Move(place) = operand {
                if place.local_id().is_some_and(|local| views.contains(&local)) {
                    errors.insert(OwnershipError::ConsumeView(place.local_id().unwrap()));
                }
            }
        }
    }
    errors
}

pub fn analyze_ownership(
    body: &CfgBody,
    types: &TypeContext,
) -> Result<OwnershipReport, Vec<OwnershipError>> {
    let (inputs, outputs, errors, escaped) = solve(body, types);
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
    errors.extend(view_consumption_errors(body, types));
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
                let mut remaining = Vec::new();
                remaining_field_drops(Place::local(*local), body.locals[local.0 as usize].ty,
                    &state.moved_fields.union(&state.uninitialized_fields).cloned().collect(), types, &mut remaining);
                if !remaining.is_empty() { errors.insert(OwnershipError::ResourceNotConsumed(*local)); }
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
    // Reject invalid source lifetimes before emitting any cleanup operations.
    // Cleanup cannot repair a borrow by deleting the loan that forbids a free.
    let (_, _, mut errors, _) = solve(body, types);
    errors.extend(view_consumption_errors(body, types));
    if !errors.is_empty() { return Err(errors.into_iter().collect()); }
    prepare_class_transfers(body, types);
    retain_storage_copies(body, types);
    let resources = resource_locals(body, types);
    let resource_set = resources.iter().copied().collect::<BTreeSet<_>>();
    elaborate_storage_destruction(body, &resource_set, types);
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
    let (_, outputs, errors, _) = solve(body, types);
    if !errors.is_empty() {
        if std::env::var_os("SEVERIAN_OWNERSHIP_TRACE").is_some() { eprintln!("{body:#?}"); }
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
        let transferred = match &block.terminator {
            Terminator::Return(Some(operand)) | Terminator::Throw(operand) => {
                operand_local(operand)
            }
            _ => None,
        };
        // A previous drop in this block may have released an older assignment.
        // The final state determines whether the current value still needs cleanup.
        for local in resources.iter().rev() {
            if state.initialized.contains(local)
                && !state.consumed_resources.contains(local)
                && transferred != Some(*local)
            {
                let mut drops = Vec::new();
                remaining_field_drops(Place::local(*local), body.locals[local.0 as usize].ty,
                    &state.moved_fields.union(&state.uninitialized_fields).cloned().collect(), types, &mut drops);
                for place in drops {
                    block.statements.push(CfgStatement::Drop(place));
                    block.statement_spans.push(body.locals[local.0 as usize].span);
                }
            }
        }
    }
    if types.resolve_name("string").and_then(|ty| types.destruction(ty)).is_none() {
        elaborate_temporary_strings(body, types);
    }
    Ok(())
}

fn elaborate_storage_destruction(body: &mut CfgBody, resources: &BTreeSet<LocalId>, types: &TypeContext) {
    let (inputs, _, _, _) = solve(body, types);
    for block in &mut body.blocks {
        let Some(mut state) = inputs.get(&block.id).cloned() else { continue; };
        let original = std::mem::take(&mut block.statements);
        let spans = std::mem::take(&mut block.statement_spans);
        for (index, statement) in original.into_iter().enumerate() {
            let span = spans.get(index).copied().flatten();
            if let CfgStatement::Assign(place, value) = &statement {
                if !place.projection.is_empty() && place.local_id().is_some_and(|local| state.initialized.contains(&local))
                    && !state.moved_fields.iter().chain(&state.uninitialized_fields).any(|missing| place_prefix(missing, place))
                    && !matches!(value, Rvalue::Use(Operand::Copy(source) | Operand::Move(source)) if source == place)
                {
                    let mut ty = body.locals[place.local_id().unwrap().0 as usize].ty;
                    let mut known = true;
                    for projection in &place.projection {
                        if let crate::Projection::Field(index) = projection {
                            if let Some(field) = types.destruction_fields(ty).get(*index as usize) { ty = *field; }
                            else { known = false; break; }
                        } else { known = false; break; }
                    }
                    if known && types.destruction(ty).is_some() {
                        block.statements.push(CfgStatement::Drop(place.clone()));
                        block.statement_spans.push(span);
                    }
                }
            }
            let old = match &statement {
                CfgStatement::StorageDead(local) => Some(*local),
                CfgStatement::Assign(place, value) if place.projection.is_empty() => {
                    let same = matches!(value, Rvalue::Use(Operand::Copy(source) | Operand::Move(source)) if source == place);
                    if same { None } else { place.local_id() }
                }
                _ => None,
            };
            if let Some(local) = old.filter(|local| resources.contains(local) && state.initialized.contains(local)) {
                let mut drops = Vec::new();
                remaining_field_drops(Place::local(local), body.locals[local.0 as usize].ty,
                    &state.moved_fields.union(&state.uninitialized_fields).cloned().collect(), types, &mut drops);
                for place in drops {
                    block.statements.push(CfgStatement::Drop(place));
                    block.statement_spans.push(span);
                }
                state.initialized.remove(&local);
                state.consumed_resources.insert(local);
            }
            match &statement {
                CfgStatement::Assign(place, value) => {
                    let pending = construction_fields(place, value, types, &state);
                    inspect_rvalue(value, place, block.id, &mut state, &mut BTreeSet::new());
                    state.uninitialized_fields.retain(|field| !place_prefix(place, field));
                    state.uninitialized_fields.extend(pending);
                    if let Some(local) = place.local_id() {
                        state.moved_fields.retain(|missing| !place_prefix(place, missing));
                        state.initialized.insert(local);
                        state.moved.remove(&local);
                        state.consumed_resources.remove(&local);
                    }
                }
                CfgStatement::StorageLive(local) | CfgStatement::StorageDead(local) => {
                    state.initialized.remove(local);
                }
                CfgStatement::Drop(place) => {
                    if let Some(local) = place.local_id() {
                        if place.projection.is_empty() { state.initialized.remove(&local); }
                        else { state.moved_fields.insert(place.clone()); }
                    }
                }
                _ => {}
            }
            block.statements.push(statement);
            block.statement_spans.push(span);
        }
    }
}

fn construction_fields(destination: &Place, value: &Rvalue, types: &TypeContext, state: &OwnershipState) -> Vec<Place> {
    match value {
        Rvalue::Aggregate { type_id, fields } if fields.is_empty() => {
            types.destruction_fields(*type_id).iter().enumerate().map(|(index, _)| {
                let mut field = destination.clone();
                field.projection.push(crate::Projection::Field(index as u32));
                field
            }).collect()
        }
        Rvalue::Use(Operand::Copy(source) | Operand::Move(source)) => state.uninitialized_fields.iter()
            .filter(|field| place_prefix(source, field))
            .map(|field| {
                let mut target = destination.clone();
                target.projection.extend_from_slice(&field.projection[source.projection.len()..]);
                target
            }).collect(),
        _ => vec![],
    }
}

fn place_prefix(prefix: &Place, place: &Place) -> bool {
    prefix.base == place.base && place.projection.starts_with(&prefix.projection)
}

fn remaining_field_drops(place: Place, ty: severian_universal::TypeId, moved: &BTreeSet<Place>, types: &TypeContext, drops: &mut Vec<Place>) {
    if moved.iter().any(|missing| place_prefix(missing, &place)) { return; }
    if !moved.iter().any(|missing| place_prefix(&place, missing)) {
        if types.destruction(ty).is_some() { drops.push(place); }
        return;
    }
    for (index, field) in types.destruction_fields(ty).iter().enumerate().rev() {
        let mut member = place.clone();
        member.projection.push(crate::Projection::Field(index as u32));
        remaining_field_drops(member, *field, moved, types, drops);
    }
}

fn rebound_storage_arguments(body: &CfgBody, types: &TypeContext) -> BTreeSet<LocalId> {
    let mut rebound = BTreeSet::new();
    for block in &body.blocks {
        let destinations = block.statements.iter().filter_map(|statement| match statement {
            CfgStatement::Assign(place, _) => Some(place),
            _ => None,
        }).chain(match &block.terminator {
            Terminator::Call { destination, .. } => destination.as_ref(),
            _ => None,
        });
        for place in destinations {
            if !place.projection.is_empty() { continue; }
            if let Some(local) = place.local_id() {
                let declaration = &body.locals[local.0 as usize];
                if declaration.argument && types.retention(declaration.ty).is_some() {
                    rebound.insert(local);
                }
            }
        }
    }
    rebound
}

fn class_borrowed_locals(body: &CfgBody, types: &TypeContext) -> BTreeSet<LocalId> {
    let rebound = rebound_storage_arguments(body, types);
    let mut borrowed = body.locals.iter().filter(|local| !rebound.contains(&local.id) && (local.borrowed || (local.argument && types.retention(local.ty).is_some()))).map(|local| local.id).collect::<BTreeSet<_>>();
    for block in &body.blocks {
        if let Terminator::Call { callee: crate::Callee::Direct { function, .. }, destination: Some(place), .. } = &block.terminator {
            if types.borrowed_result(*function) {
                if let Some(local) = place.local_id() {
                    if types.retention(body.locals[local.0 as usize].ty).is_none() { borrowed.insert(local); }
                }
            }
        }
    }
    loop {
        let before = borrowed.len();
        for block in &body.blocks {
            for statement in &block.statements {
                let CfgStatement::Assign(destination, value) = statement else { continue; };
                let Some(local) = destination.local_id() else { continue; };
                if types.destruction(body.locals[local.0 as usize].ty).is_none() { continue; }
                let is_borrowed = match value {
                    Rvalue::BorrowShared(_) | Rvalue::BorrowExclusive(_) => true,
                    Rvalue::Use(Operand::Copy(source)) => types.retention(body.locals[local.0 as usize].ty).is_none()
                        && (!source.projection.is_empty() || source.local_id().is_none_or(|id| borrowed.contains(&id))),
                    _ => false,
                };
                if is_borrowed { borrowed.insert(local); }
            }
        }
        if before == borrowed.len() { break; }
    }
    borrowed
}

fn copied_storage_type(body: &CfgBody, place: &Place, types: &TypeContext) -> Option<severian_universal::TypeId> {
    let mut ty = body.locals.get(place.local_id()?.0 as usize)?.ty;
    for projection in &place.projection {
        let crate::Projection::Field(index) = projection else { return None; };
        ty = *types.destruction_fields(ty).get(*index as usize)?;
    }
    types.retention(ty).map(|_| ty)
}

fn retain_storage_copies(body: &mut CfgBody, types: &TypeContext) {
    // A rebound parameter owns its local value, including the initial borrowed
    // argument. Otherwise assigning a temporary leaves a dangling borrowed
    // alias as soon as that temporary is destroyed.
    for local in rebound_storage_arguments(body, types) {
        let entry = &mut body.blocks[body.entry.0 as usize];
        entry.statements.insert(0, CfgStatement::Retain(Place::local(local)));
        entry.statement_spans.insert(0, None);
    }
    let borrowed = class_borrowed_locals(body, types);
    let snapshot = body.clone();
    let mut borrowed_returns = Vec::new();
    for block in &snapshot.blocks {
        if let Terminator::Call { callee: crate::Callee::Direct { function, .. }, destination: Some(place), target, .. } = &block.terminator {
            if types.borrowed_result(*function) && copied_storage_type(&snapshot, place, types).is_some() {
                borrowed_returns.push((*target, place.clone()));
            }
        }
    }
    for (target, place) in borrowed_returns {
        let block = &mut body.blocks[target.0 as usize];
        block.statements.insert(0, CfgStatement::Retain(place));
        block.statement_spans.insert(0, None);
    }
    let (construction_inputs, _, _, _) = solve(&snapshot, types);
    for block in &mut body.blocks {
        let mut construction = construction_inputs.get(&block.id).cloned().unwrap_or_default();
        let statements = std::mem::take(&mut block.statements);
        let spans = std::mem::take(&mut block.statement_spans);
        for (index, statement) in statements.into_iter().enumerate() {
            let span = spans.get(index).copied().flatten();
            if let CfgStatement::Assign(destination, value) = &statement {
                let copies: Vec<&Operand> = match value {
                    // Borrowing a record does not borrow a new value assigned
                    // into one of its owning fields.
                    Rvalue::Use(operand) if !destination.projection.is_empty()
                        || destination.local_id().is_none_or(|id| !borrowed.contains(&id)) => vec![operand],
                    Rvalue::Aggregate { fields, .. } | Rvalue::Variant { fields, .. } => fields.iter().collect(),
                    _ => vec![],
                };
                for operand in copies {
                    if let Operand::Copy(place) = operand {
                        if place.local_id().is_none() || copied_storage_type(&snapshot, place, types).is_some() {
                            let mut retained = Vec::new();
                            if let Some(ty) = copied_storage_type(&snapshot, place, types) {
                                remaining_field_drops(place.clone(), ty, &construction.uninitialized_fields, types, &mut retained);
                            } else if !construction.uninitialized_fields.iter().any(|field| place_prefix(place, field)) {
                                retained.push(place.clone());
                            }
                            for place in retained {
                                block.statements.push(CfgStatement::Retain(place));
                                block.statement_spans.push(span);
                            }
                        }
                    }
                }
            }
            if let CfgStatement::Assign(place, value) = &statement {
                let pending = construction_fields(place, value, types, &construction);
                construction.uninitialized_fields.retain(|field| !place_prefix(place, field));
                construction.uninitialized_fields.extend(pending);
            }
            block.statements.push(statement);
            block.statement_spans.push(span);
        }
        if let Terminator::Return(Some(Operand::Copy(place))) | Terminator::Throw(Operand::Copy(place)) = &block.terminator {
            if place.local_id().is_some_and(|id| borrowed.contains(&id)) && copied_storage_type(&snapshot, place, types).is_some() {
                block.statements.push(CfgStatement::Retain(place.clone()));
                block.statement_spans.push(None);
            }
        }
    }
}

fn prepare_class_transfers(body: &mut CfgBody, types: &TypeContext) {
    let borrowed = class_borrowed_locals(body, types);
    let owned = body.locals.iter().filter(|local| !borrowed.contains(&local.id) && types.destruction(local.ty).is_some() && types.retention(local.ty).is_none())
        .map(|local| local.id).collect::<BTreeSet<_>>();
    let transfer = |operand: &mut Operand| {
        if let Operand::Copy(place) = operand {
            if place.projection.is_empty() && place.local_id().is_some_and(|id| owned.contains(&id)) {
                *operand = Operand::Move(place.clone());
            }
        }
    };
    for block in &mut body.blocks {
        if let Terminator::Call { callee: crate::Callee::Direct { function, .. }, arguments, .. } = &mut block.terminator {
            for (index, argument) in arguments.iter_mut().enumerate() {
                if types.transfers_argument(*function, index) {
                    if let Operand::Copy(place) = argument { *argument = Operand::Move(place.clone()); }
                }
            }
        }
        for statement in &mut block.statements {
            if let CfgStatement::Assign(destination, value) = statement {
                match value {
                    Rvalue::Use(operand) if destination.local_id().is_some_and(|id| owned.contains(&id)) => transfer(operand),
                    Rvalue::Aggregate { fields, .. } | Rvalue::Variant { fields, .. } => {
                        for field in fields { transfer(field); }
                    }
                    _ => {}
                }
            }
        }
    }
}

/// Reclaim allocation-producing string intermediates once their last borrowing
/// operation finishes. Values transferred into storage or calls remain owned by
/// that destination; they must not be freed by a temporary-lifetime shortcut.
fn elaborate_temporary_strings(body: &mut CfgBody, types: &TypeContext) {
    let mut candidates = BTreeMap::new();
    let mut definitions = BTreeMap::<LocalId, usize>::new();
    for block in &body.blocks {
        for (index, statement) in block.statements.iter().enumerate() {
            if let CfgStatement::Assign(place, value) = statement {
                if let Some(local) = place.local_id() {
                    *definitions.entry(local).or_default() += 1;
                    if place.projection.is_empty()
                        && !body.locals[local.0 as usize].argument
                        && !body.locals[local.0 as usize].borrowed
                        && matches!(value, Rvalue::Binary { operator, .. } if *operator == severian_universal::BinaryOperator::Add)
                        && types
                            .primitive(body.locals[local.0 as usize].ty)
                            .is_some_and(|primitive| {
                                primitive.representation
                                    == severian_universal::PrimitiveRepresentation::String
                            })
                    {
                        candidates.insert(local, (block.id, index, index));
                    }
                }
            }
        }
    }
    candidates.retain(|local, _| definitions.get(local) == Some(&1));
    for block in &body.blocks {
        for (index, statement) in block.statements.iter().enumerate() {
            let mut used = BTreeSet::new();
            apply_statement_liveness(statement, &mut used);
            // These operations borrow their input only for the operation itself.
            let borrows = match statement {
                CfgStatement::Assign(_, Rvalue::Binary { left, right, .. }) =>
                    !matches!(left, Operand::Move(_)) && !matches!(right, Operand::Move(_)),
                CfgStatement::Assert { condition, message, .. } =>
                    !matches!(condition, Operand::Move(_)) && !matches!(message, Some(Operand::Move(_))),
                _ => false,
            };
            for local in used {
                if let Some((owner, definition, last_use)) = candidates.get_mut(&local) {
                    if borrows && *owner == block.id && *definition < index {
                        *last_use = index;
                    } else {
                        candidates.remove(&local);
                    }
                }
            }
            if let CfgStatement::Drop(place) = statement {
                if let Some(local) = place.local_id() {
                    candidates.remove(&local);
                }
            }
        }
        let mut used = BTreeSet::new();
        apply_terminator_liveness(&block.terminator, &mut used);
        for local in used {
            candidates.remove(&local);
        }
    }
    for block in &mut body.blocks {
        let mut releases = candidates
            .iter()
            .filter_map(|(local, (owner, _, last))| {
                (*owner == block.id).then_some((*last + 1, *local))
            })
            .collect::<Vec<_>>();
        releases.sort_by_key(|(index, local)| std::cmp::Reverse((*index, *local)));
        for (index, local) in releases {
            let span = block.statement_spans.get(index - 1).copied().flatten();
            block
                .statements
                .insert(index, CfgStatement::Drop(Place::local(local)));
            block.statement_spans.insert(index, span);
        }
    }
}

fn solve(body: &CfgBody, types: &TypeContext) -> OwnershipSolution { solve_detailed(body, types).0 }

pub fn ownership_error_span(body: &CfgBody, types: &TypeContext, error: &OwnershipError) -> Option<severian_source::Span> {
    if let Some(span) = solve_detailed(body, types).1.get(error) { return Some(*span); }
    let local = match error {
        OwnershipError::UseAfterMove(local) | OwnershipError::DoubleMove(local)
        | OwnershipError::ConflictingLoan(local) | OwnershipError::EscapingBorrow(local)
        | OwnershipError::ResourceNotConsumed(local) | OwnershipError::DoubleDrop(local)
        | OwnershipError::ConsumeView(local) | OwnershipError::UninitializedUse(local) => *local,
    };
    if matches!(error, OwnershipError::ConsumeView(_)) {
        for block in &body.blocks {
            for (index, statement) in block.statements.iter().enumerate() {
                if matches!(statement, CfgStatement::Drop(place) if place.local_id() == Some(local)) {
                    if let Some(span) = block.statement_spans.get(index).copied().flatten() { return Some(span); }
                }
            }
            if terminator_operands(&block.terminator).iter().any(|operand| matches!(operand, Operand::Move(place) if place.local_id() == Some(local))) {
                if block.terminator_span.is_some() { return block.terminator_span; }
            }
        }
    }
    body.locals.get(local.0 as usize).and_then(|local| local.span)
}

fn solve_detailed(body: &CfgBody, types: &TypeContext) -> (OwnershipSolution, BTreeMap<OwnershipError, severian_source::Span>) {
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
    // An unvisited predecessor denotes the lattice top: intersecting with it
    // changes nothing. Materialize states only when a reachable edge arrives,
    // instead of cloning every local into every block twice.
    let mut inputs = BTreeMap::from([(body.entry, entry)]);
    let mut outputs = BTreeMap::new();
    let mut queue = VecDeque::from([body.entry]);
    let mut queued = BTreeSet::from([body.entry]);
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
            types,
            &mut BTreeMap::new(),
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
    let mut locations = BTreeMap::new();
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
            types,
            &mut locations,
        );
    }
    ((inputs, outputs, errors, escaped), locations)
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
        joined.moved_fields.extend(state.moved_fields.iter().cloned());
        joined.uninitialized_fields.extend(state.uninitialized_fields.iter().cloned());
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
    types: &TypeContext,
    locations: &mut BTreeMap<OwnershipError, severian_source::Span>,
) {
    state
        .loans
        .retain(|loan| liveness.live_in.contains(&loan.holder));
    for (index, statement) in block.statements.iter().enumerate() {
        let before = errors.clone();
        match statement {
            CfgStatement::Assign(place, value) => {
                inspect_write(place, state, errors);
                let pending = construction_fields(place, value, types, state);
                inspect_rvalue(value, place, block.id, state, errors);
                state.uninitialized_fields.retain(|field| !place_prefix(place, field));
                state.uninitialized_fields.extend(pending);
                if let Some(local) = place.local_id() {
                    state.moved_fields.retain(|missing| !place_prefix(place, missing));
                    state.initialized.insert(local);
                    state.moved.remove(&local);
                    state.consumed_resources.remove(&local);
                }
            }
            CfgStatement::Retain(place) => inspect_operand(&Operand::Copy(place.clone()), state, errors),
            CfgStatement::Drop(place) => {
                let Some(local) = place.local_id() else {
                    continue;
                };
                if !place.projection.is_empty() {
                    inspect_operand(&Operand::Move(place.clone()), state, errors);
                    continue;
                }
                if state.loans.iter().any(|loan| loan.place.local_id() == Some(local) && loan.holder != local) {
                    errors.insert(OwnershipError::ConflictingLoan(local));
                }
                if state.uninitialized_fields.iter().any(|field| place_prefix(place, field)) {
                    errors.insert(OwnershipError::UninitializedUse(local));
                }
                if !state.initialized.remove(&local) || !state.consumed_resources.insert(local) {
                    errors.insert(OwnershipError::DoubleDrop(local));
                }
                state
                    .loans
                    .retain(|loan| loan.place.local_id() != Some(local) && loan.holder != local);
            }
            CfgStatement::StorageLive(local) => {
                state.initialized.remove(local);
                state.uninitialized_fields.retain(|field| field.local_id() != Some(*local));
                state.moved.remove(local);
                state.consumed_resources.remove(local);
            }
            CfgStatement::StorageDead(local) => {
                if state.loans.iter().any(|loan| loan.place.local_id() == Some(*local) && loan.holder != *local) {
                    errors.insert(OwnershipError::ConflictingLoan(*local));
                }
                state.initialized.remove(local);
                state.uninitialized_fields.retain(|field| field.local_id() != Some(*local));
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
        if let Some(span) = block.statement_spans.get(index).copied().flatten() {
            for error in errors.difference(&before) { locations.entry(error.clone()).or_insert(span); }
        }
        if let Some(live) = liveness.after_statements.get(index) {
            state.loans.retain(|loan| live.contains(&loan.holder));
        }
    }
    let before = errors.clone();
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
    if let Some(span) = block.terminator_span {
        for error in errors.difference(&before) { locations.entry(error.clone()).or_insert(span); }
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
            let construction = matches!(operand, Operand::Copy(place) if place.projection.is_empty()
                && state.uninitialized_fields.iter().any(|field| place_prefix(place, field)));
            if !construction { inspect_operand(operand, state, errors); }
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

fn inspect_write(place: &Place, state: &OwnershipState, errors: &mut BTreeSet<OwnershipError>) {
    let Some(local) = place.local_id() else { return; };
    for loan in &state.loans {
        if loan.place.local_id() == Some(local)
            && (place_prefix(place, &loan.place) || place_prefix(&loan.place, place))
            && loan.holder != local {
            errors.insert(OwnershipError::ConflictingLoan(local));
        }
        if !place.projection.is_empty() && loan.holder == local && loan.kind == LoanKind::Shared {
            errors.insert(OwnershipError::ConflictingLoan(local));
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
    if state.uninitialized_fields.iter().any(|missing| place_prefix(missing, place) || place_prefix(place, missing)) {
        errors.insert(OwnershipError::UninitializedUse(local));
        return;
    }
    if state.moved_fields.iter().any(|missing| place_prefix(missing, place) || place_prefix(place, missing)) {
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
    if state.uninitialized_fields.iter().any(|missing| place_prefix(missing, place) || place_prefix(place, missing)) {
        errors.insert(OwnershipError::UninitializedUse(local));
        return;
    }
    if state.moved_fields.iter().any(|missing| place_prefix(missing, place) || place_prefix(place, missing)) {
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
        if !place.projection.is_empty() {
            state.moved_fields.insert(place.clone());
            return;
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
    let borrowed = class_borrowed_locals(body, types);
    body.locals
        .iter()
        .filter(|local| {
            !borrowed.contains(&local.id) && (types.destruction(local.ty).is_some() || matches!(
                types.kind(local.ty),
                Some(TypeKind::Resource(_, _) | TypeKind::Tensor { .. })
            ))
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
        let mut changed = false;
        for block in body
            .blocks
            .iter()
            .rev()
            .filter(|block| reachable.contains(&block.id))
        {
            let mut live_out = BTreeSet::new();
            for successor in successors(&block.terminator) {
                if let Some(successor) = result.get(&successor) {
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
            let next = BlockLiveness { live_in: live, live_out, after_statements };
            if result.get(&block.id) != Some(&next) {
                result.insert(block.id, next);
                changed = true;
            }
        }
        if !changed {
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
        CfgStatement::Retain(place) => use_place(place, live),
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

    fn borrowed_owner_body(invalidation: CfgStatement) -> CfgBody {
        let span = severian_source::Span::new(severian_source::SourceId(7), 20, 30);
        CfgBody {
            entry: BlockId(0),
            locals: vec![local(0, true), local(1, false)],
            return_type: TypeId(0),
            blocks: vec![BasicBlock {
                id: BlockId(0), execution: None, parameters: vec![],
                statements: vec![
                    CfgStatement::Assign(Place::local(LocalId(1)), Rvalue::BorrowShared(Place::local(LocalId(0)))),
                    invalidation,
                    CfgStatement::Retain(Place::local(LocalId(1))),
                ],
                statement_spans: vec![None, Some(span), None],
                terminator: Terminator::Return(None), terminator_span: None,
            }],
        }
    }

    #[test]
    fn live_views_forbid_owner_release_reassignment_and_scope_end() {
        for invalidation in [CfgStatement::Drop(Place::local(LocalId(0))), initialize(0), CfgStatement::StorageDead(LocalId(0))] {
            let body = borrowed_owner_body(invalidation);
            let error = OwnershipError::ConflictingLoan(LocalId(0));
            assert!(analyze_ownership(&body, &TypeContext::default()).unwrap_err().contains(&error));
            assert_eq!(ownership_error_span(&body, &TypeContext::default(), &error), body.blocks[0].statement_spans[1]);
        }
    }

    #[test]
    fn a_view_never_acquires_release_authority_from_drop_elaboration() {
        let mut body = borrowed_owner_body(CfgStatement::Drop(Place::local(LocalId(0))));
        body.locals[0].borrowed = true;
        let before = body.clone();
        assert!(elaborate_drops(&mut body, &TypeContext::default()).unwrap_err().contains(&OwnershipError::ConsumeView(LocalId(0))));
        assert_eq!(body, before, "an invalid source lifetime must not be rewritten");
    }

    #[test]
    fn lowering_plan_expires_when_the_body_is_changed() {
        let mut body = borrowed_owner_body(CfgStatement::Drop(Place::local(LocalId(0))));
        body.blocks[0].statements.swap(1, 2); // view's last use precedes release
        let types = TypeContext::default();
        assert!(ownership_plan(&body, &types).is_ok());
        body.blocks[0].statements.swap(1, 2);
        assert!(ownership_plan(&body, &types).is_err());
    }

    #[test]
    fn partial_construction_releases_only_initialized_fields_and_cannot_escape() {
        let mut types = TypeContext::default();
        let owner_drop = DefId { package: 0, module: 0, declaration: DeclarationId(100) };
        let field_drop = DefId { package: 0, module: 0, declaration: DeclarationId(101) };
        types.register_destruction(TypeId(0), owner_drop);
        types.register_destruction(TypeId(1), field_drop);
        types.register_destruction_fields(TypeId(0), vec![TypeId(1), TypeId(1)], false);
        let first = Place { base: crate::PlaceBase::Local(LocalId(0)), projection: vec![crate::Projection::Field(0)] };
        let span = severian_source::Span::new(severian_source::SourceId(3), 40, 50);
        let mut body = CfgBody {
            entry: BlockId(0), locals: vec![local(0, false)], return_type: TypeId(0),
            blocks: vec![BasicBlock {
                id: BlockId(0), execution: None, parameters: vec![],
                statements: vec![
                    CfgStatement::Assign(Place::local(LocalId(0)), Rvalue::Aggregate { type_id: TypeId(0), fields: vec![] }),
                    CfgStatement::Assign(first.clone(), Rvalue::Use(Operand::Constant { value: LiteralValue::String("initialized".into()), ty: TypeId(1) })),
                ],
                statement_spans: vec![None, None],
                terminator: Terminator::Return(Some(Operand::Copy(Place::local(LocalId(0))))), terminator_span: Some(span),
            }],
        };
        let error = OwnershipError::UninitializedUse(LocalId(0));
        assert!(elaborate_drops(&mut body, &types).unwrap_err().contains(&error));
        assert_eq!(ownership_error_span(&body, &types, &error), Some(span));
        body.blocks[0].terminator = Terminator::Return(None);
        elaborate_drops(&mut body, &types).unwrap();
        let drops = body.blocks[0].statements.iter().filter_map(|statement| match statement {
            CfgStatement::Drop(place) => Some(place.clone()), _ => None,
        }).collect::<Vec<_>>();
        assert_eq!(drops, vec![first]);
        analyze_ownership(&body, &types).unwrap();
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

        let (_, _, errors, _) = solve(&body, &TypeContext::default());
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

        let (_, _, errors, _) = solve(&body, &TypeContext::default());
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

        let (_, _, errors, _) = solve(&body, &TypeContext::default());
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

        let (_, _, local_errors, _) = solve(&body(false), &TypeContext::default());
        assert!(local_errors.contains(&OwnershipError::EscapingBorrow(LocalId(0))));

        let (_, _, argument_errors, _) = solve(&body(true), &TypeContext::default());
        assert!(argument_errors.is_empty(), "{argument_errors:?}");
    }
}

#[cfg(test)]
mod temporary_string_tests {
    use super::*;
    use severian_universal::{
        BinaryOperator, LiteralValue, PrimitiveCategory, PrimitiveRepresentation,
        TypeContextBuilder,
    };

    fn example() -> (CfgBody, TypeContext) {
        let mut types = TypeContextBuilder::new();
        let string = types.register_declaration("core.string", "string").unwrap();
        types
            .define_primitive(
                string,
                PrimitiveCategory::Text,
                PrimitiveRepresentation::String,
                true,
            )
            .unwrap();
        let constant = || Operand::Constant {
            value: LiteralValue::String("x".into()),
            ty: string,
        };
        let local = |index| Place::local(LocalId(index));
        let statements = vec![
            CfgStatement::Assign(
                local(0),
                Rvalue::Binary {
                    operator: BinaryOperator::Add,
                    left: constant(),
                    right: constant(),
                },
            ),
            CfgStatement::Assign(
                local(1),
                Rvalue::Binary {
                    operator: BinaryOperator::Add,
                    left: Operand::Copy(local(0)),
                    right: constant(),
                },
            ),
            CfgStatement::Assign(local(2), Rvalue::Use(Operand::Copy(local(1)))),
        ];
        let body = CfgBody {
            entry: BlockId(0),
            locals: (0..3)
                .map(|index| crate::LocalDecl {
                    id: LocalId(index),
                    ty: string,
                    mutable: false,
                    argument: false,
                    borrowed: false,
                    span: None,
                })
                .collect(),
            blocks: vec![crate::BasicBlock {
                id: BlockId(0),
                execution: None,
                parameters: vec![],
                statement_spans: vec![None; statements.len()],
                statements,
                terminator: Terminator::Return(Some(Operand::Copy(local(2)))),
                terminator_span: None,
            }],
            return_type: string,
        };
        (body, types.build())
    }

    #[test]
    fn an_explicitly_consumed_operand_is_not_released_again() {
        let (mut body, types) = example();
        if let CfgStatement::Assign(_, Rvalue::Binary { left, .. }) = &mut body.blocks[0].statements[1] {
            *left = Operand::Move(Place::local(LocalId(0)));
        }
        elaborate_drops(&mut body, &types).unwrap();
        assert_eq!(body.blocks[0].statements.len(), 3);
        analyze_ownership(&body, &types).unwrap();
    }

    #[test]
    fn releases_intermediate_after_borrow_and_preserves_transferred_storage() {
        let (mut body, types) = example();
        elaborate_temporary_strings(&mut body, &types);
        assert_eq!(
            body.blocks[0].statements[2],
            CfgStatement::Drop(Place::local(LocalId(0)))
        );
        assert_eq!(body.blocks[0].statements.len(), 4);
        assert!(analyze_ownership(&body, &types).is_ok());
        let once = body.clone();
        elaborate_temporary_strings(&mut body, &types);
        assert_eq!(body, once);
    }

    #[test]
    fn a_returned_or_borrowed_slot_is_not_released() {
        let (mut body, types) = example();
        body.locals[0].borrowed = true;
        let before = body.clone();
        elaborate_temporary_strings(&mut body, &types);
        assert_eq!(body, before);
        body.locals[0].borrowed = false;
        body.blocks[0].terminator =
            Terminator::Return(Some(Operand::Copy(Place::local(LocalId(0)))));
        let before = body.clone();
        elaborate_temporary_strings(&mut body, &types);
        assert_eq!(body, before);
    }
}
