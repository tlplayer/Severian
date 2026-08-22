use crate::{CfgBody, CfgStatement, LocalId, Operand, Place, Rvalue, Terminator};
use severian_universal::{EffectSet, TyKind, TypeContext};
use std::collections::{BTreeMap, BTreeSet};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LoanKind {
    Shared,
    Exclusive,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Loan {
    pub place: Place,
    pub kind: LoanKind,
    pub block: crate::BlockId,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum OwnershipError {
    UseAfterMove(LocalId),
    DoubleMove(LocalId),
    ConflictingLoan(LocalId),
    EscapingBorrow(LocalId),
    ResourceNotConsumed(LocalId),
    DoubleDrop(LocalId),
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct OwnershipReport {
    pub initialized: BTreeSet<LocalId>,
    pub moved: BTreeSet<LocalId>,
    pub loans: Vec<Loan>,
    pub escaped: BTreeSet<LocalId>,
    pub consumed_resources: BTreeSet<LocalId>,
    pub call_effects: BTreeMap<crate::BlockId, EffectSet>,
}

pub fn analyze_ownership(
    body: &CfgBody,
    types: &TypeContext,
) -> Result<OwnershipReport, Vec<OwnershipError>> {
    let mut report = OwnershipReport::default();
    let mut errors = Vec::new();
    report.initialized.extend(
        body.locals
            .iter()
            .filter(|local| local.argument)
            .map(|local| local.id),
    );
    for block in &body.blocks {
        for statement in &block.statements {
            match statement {
                CfgStatement::Assign(place, value) => {
                    inspect_rvalue(value, block.id, &mut report, &mut errors);
                    report.initialized.insert(place.local);
                    report.moved.remove(&place.local);
                }
                CfgStatement::Drop(place) => {
                    if !report.initialized.remove(&place.local)
                        || !report.consumed_resources.insert(place.local)
                    {
                        errors.push(OwnershipError::DoubleDrop(place.local));
                    }
                }
                CfgStatement::StorageDead(local) => {
                    report.initialized.remove(local);
                    report.loans.retain(|loan| loan.place.local != *local);
                }
                CfgStatement::Assert { condition, message } => {
                    inspect_operand(condition, &mut report, &mut errors);
                    if let Some(message) = message {
                        inspect_operand(message, &mut report, &mut errors);
                    }
                }
                CfgStatement::Operation {
                    operands, results, ..
                } => {
                    for operand in operands {
                        inspect_operand(operand, &mut report, &mut errors);
                    }
                    report
                        .initialized
                        .extend(results.iter().map(|place| place.local));
                }
                CfgStatement::StorageLive(_) | CfgStatement::Coverage(_) => {}
            }
        }
        for operand in terminator_operands(&block.terminator) {
            inspect_operand(operand, &mut report, &mut errors);
        }
        if let Terminator::Return(Some(Operand::Copy(place) | Operand::Move(place))) =
            &block.terminator
        {
            if report
                .loans
                .iter()
                .any(|loan| loan.place.local == place.local)
            {
                errors.push(OwnershipError::EscapingBorrow(place.local));
                report.escaped.insert(place.local);
            }
        }
    }
    for local in &body.locals {
        if matches!(types.kind(local.ty), Some(TyKind::Resource(_, _)))
            && report.initialized.contains(&local.id)
            && !report.consumed_resources.contains(&local.id)
        {
            errors.push(OwnershipError::ResourceNotConsumed(local.id));
        }
    }
    if errors.is_empty() {
        Ok(report)
    } else {
        Err(errors)
    }
}

pub fn elaborate_drops(body: &mut CfgBody, types: &TypeContext) {
    let resources = body
        .locals
        .iter()
        .filter(|local| matches!(types.kind(local.ty), Some(TyKind::Resource(_, _))))
        .map(|local| local.id)
        .collect::<Vec<_>>();
    for block in &mut body.blocks {
        if matches!(
            &block.terminator,
            Terminator::Return(_) | Terminator::Throw(_)
        ) {
            let already_dropped = block
                .statements
                .iter()
                .filter_map(|statement| match statement {
                    CfgStatement::Drop(place) => Some(place.local),
                    _ => None,
                })
                .collect::<BTreeSet<_>>();
            for local in resources.iter().rev() {
                if !already_dropped.contains(local) {
                    block
                        .statements
                        .push(CfgStatement::Drop(Place::local(*local)));
                }
            }
        }
    }
}

fn inspect_rvalue(
    value: &Rvalue,
    block: crate::BlockId,
    report: &mut OwnershipReport,
    errors: &mut Vec<OwnershipError>,
) {
    match value {
        Rvalue::Use(operand) | Rvalue::Unary { operand, .. } | Rvalue::Convert { operand, .. } => {
            inspect_operand(operand, report, errors)
        }
        Rvalue::Binary { left, right, .. } => {
            inspect_operand(left, report, errors);
            inspect_operand(right, report, errors);
        }
        Rvalue::BorrowShared(place) => add_loan(place, LoanKind::Shared, block, report, errors),
        Rvalue::BorrowExclusive(place) => {
            add_loan(place, LoanKind::Exclusive, block, report, errors)
        }
        Rvalue::Aggregate { fields, .. } => {
            for field in fields {
                inspect_operand(field, report, errors);
            }
        }
    }
}

fn add_loan(
    place: &Place,
    kind: LoanKind,
    block: crate::BlockId,
    report: &mut OwnershipReport,
    errors: &mut Vec<OwnershipError>,
) {
    let conflicts = report.loans.iter().any(|loan| {
        loan.place.local == place.local
            && (loan.kind == LoanKind::Exclusive || kind == LoanKind::Exclusive)
    });
    if conflicts {
        errors.push(OwnershipError::ConflictingLoan(place.local));
    } else {
        report.loans.push(Loan {
            place: place.clone(),
            kind,
            block,
        });
    }
}

fn inspect_operand(
    operand: &Operand,
    report: &mut OwnershipReport,
    errors: &mut Vec<OwnershipError>,
) {
    if let Operand::Copy(place) | Operand::Move(place) = operand {
        if !report.initialized.contains(&place.local) || report.moved.contains(&place.local) {
            errors.push(OwnershipError::UseAfterMove(place.local));
        }
        if matches!(operand, Operand::Move(_)) && !report.moved.insert(place.local) {
            errors.push(OwnershipError::DoubleMove(place.local));
        }
    }
}

fn terminator_operands(terminator: &Terminator) -> Vec<&Operand> {
    match terminator {
        Terminator::Goto(_, arguments) => arguments.iter().collect(),
        Terminator::Branch { condition, .. } => vec![condition],
        Terminator::Switch { discriminant, .. } => vec![discriminant],
        Terminator::Call { arguments, .. } => arguments.iter().collect(),
        Terminator::Return(value) => value.iter().collect(),
        Terminator::Throw(value) => vec![value],
        Terminator::Unreachable => Vec::new(),
    }
}
