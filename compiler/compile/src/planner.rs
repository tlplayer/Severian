use crate::{
    CompileError, CompilePlan, CompileRegion, EffectSet, PlanSegment, PlannedBlock,
    PlannedFunction, StandardRegion,
};
use severian_artifact::CompiledRegionId;
use severian_mir::{Block, Module, Operation, Value, ValueId};
use severian_universal::{BinaryOperator, CompileRoute, CompilerId, TypeContext};
use std::collections::{BTreeMap, BTreeSet};

pub fn plan(module: &Module, types: &TypeContext) -> Result<CompilePlan, CompileError> {
    let values = module
        .values
        .iter()
        .map(|value| (value.id, *value))
        .collect::<BTreeMap<_, _>>();
    let mut next_region = 0u32;
    let initializer = plan_block(
        &module.initializer,
        &values,
        &module.globals.iter().copied().collect(),
        types,
        &mut next_region,
    )?;
    let functions = module
        .functions
        .iter()
        .map(|function| {
            Ok(PlannedFunction {
                declaration: function.clone(),
                body: function
                    .body
                    .as_ref()
                    .map(|body| {
                        plan_block(body, &values, &BTreeSet::new(), types, &mut next_region)
                    })
                    .transpose()?,
            })
        })
        .collect::<Result<Vec<_>, CompileError>>()?;
    Ok(CompilePlan {
        source: module.clone(),
        initializer,
        functions,
    })
}

fn plan_block(
    block: &Block,
    values: &BTreeMap<ValueId, Value>,
    retained: &BTreeSet<ValueId>,
    types: &TypeContext,
    next_region: &mut u32,
) -> Result<PlannedBlock, CompileError> {
    let routes = block
        .operations
        .iter()
        .enumerate()
        .map(|(index, operation)| operation_route(index, operation, values, types))
        .collect::<Result<Vec<_>, _>>()?;
    let mut segments = Vec::new();
    let mut start = 0usize;
    while start < block.operations.len() {
        let route = routes[start];
        let mut end = start + 1;
        while end < block.operations.len() && routes[end] == route {
            end += 1;
        }
        let operations = block.operations[start..end].to_vec();
        match route {
            CompileRoute::Standard => {
                segments.push(PlanSegment::Standard(StandardRegion { operations }));
            }
            CompileRoute::Compiler(compiler) => {
                segments.push(PlanSegment::Compiler(build_region(
                    values,
                    CompiledRegionId::new(*next_region),
                    compiler,
                    operations,
                    &block.operations[end..],
                    retained,
                )?));
                *next_region += 1;
            }
        }
        start = end;
    }
    Ok(PlannedBlock { segments })
}

fn operation_route(
    index: usize,
    operation: &Operation,
    values: &BTreeMap<ValueId, Value>,
    types: &TypeContext,
) -> Result<CompileRoute, CompileError> {
    if matches!(operation, Operation::CompiledRegionCall { .. }) {
        return Err(CompileError::PlannerGeneratedOperation(index));
    }
    if matches!(
        operation,
        Operation::Return { .. }
            | Operation::Assert { .. }
            | Operation::If { .. }
            | Operation::Match { .. }
    ) {
        return Ok(CompileRoute::Standard);
    }
    let mut compilers = BTreeSet::new();
    for value_id in operation_inputs(operation)
        .into_iter()
        .chain(operation_outputs(operation))
    {
        let value = values
            .get(&value_id)
            .ok_or(CompileError::MissingValue(value_id.0))?;
        match types
            .compile_route(value.type_id)
            .map_err(|error| CompileError::Type(value.type_id, error.to_string()))?
        {
            CompileRoute::Standard => {}
            CompileRoute::Compiler(compiler) => {
                compilers.insert(compiler);
            }
        }
    }
    match compilers.len() {
        0 => Ok(CompileRoute::Standard),
        1 => Ok(CompileRoute::Compiler(
            *compilers.first().expect("one compiler exists"),
        )),
        _ => Err(CompileError::ConflictingCompilers {
            operation: index,
            compilers: compilers.into_iter().collect(),
        }),
    }
}

fn build_region(
    values: &BTreeMap<ValueId, Value>,
    id: CompiledRegionId,
    compiler: CompilerId,
    operations: Vec<Operation>,
    following: &[Operation],
    retained: &BTreeSet<ValueId>,
) -> Result<CompileRegion, CompileError> {
    let produced = operations
        .iter()
        .flat_map(operation_outputs)
        .collect::<BTreeSet<_>>();
    let mut inputs = Vec::new();
    let mut seen_inputs = BTreeSet::new();
    for input in operations.iter().flat_map(operation_inputs) {
        if !produced.contains(&input) && seen_inputs.insert(input) {
            inputs.push(value(values, input)?);
        }
    }
    let live_after = following
        .iter()
        .flat_map(operation_inputs)
        .chain(retained.iter().copied())
        .collect::<BTreeSet<_>>();
    let outputs = produced
        .into_iter()
        .filter(|output| live_after.contains(output))
        .map(|output| value(values, output))
        .collect::<Result<Vec<_>, _>>()?;
    let effects = operations
        .iter()
        .fold(EffectSet::default(), |mut effects, operation| {
            if matches!(
                operation,
                Operation::Binary {
                    operator: BinaryOperator::Divide
                        | BinaryOperator::Remainder
                        | BinaryOperator::Power,
                    ..
                }
            ) {
                effects.may_trap = true;
            }
            if matches!(operation, Operation::Call { .. }) {
                effects.reads_memory = true;
                effects.writes_memory = true;
                effects.may_trap = true;
            }
            effects
        });
    Ok(CompileRegion {
        id,
        compiler,
        operations,
        inputs,
        outputs,
        effects,
    })
}

fn value(values: &BTreeMap<ValueId, Value>, id: ValueId) -> Result<Value, CompileError> {
    values
        .get(&id)
        .copied()
        .ok_or(CompileError::MissingValue(id.0))
}

fn operation_inputs(operation: &Operation) -> Vec<ValueId> {
    match operation {
        Operation::Constant { .. } => Vec::new(),
        Operation::Unary { operand, .. } => vec![*operand],
        Operation::Binary { left, right, .. } => vec![*left, *right],
        Operation::Call { arguments, .. } => arguments.clone(),
        Operation::Return { value } => value.iter().copied().collect(),
        Operation::Assert {
            condition, message, ..
        } => std::iter::once(*condition)
            .chain(message.iter().copied())
            .collect(),
        Operation::If {
            condition,
            then_block,
            else_block,
        } => std::iter::once(*condition)
            .chain(then_block.operations.iter().flat_map(operation_inputs))
            .chain(else_block.operations.iter().flat_map(operation_inputs))
            .collect(),
        Operation::Match { subject, arms } => std::iter::once(*subject)
            .chain(
                arms.iter()
                    .flat_map(|arm| arm.body.operations.iter().flat_map(operation_inputs)),
            )
            .collect(),
        Operation::CompiledRegionCall { inputs, .. } => inputs.clone(),
    }
}

fn operation_outputs(operation: &Operation) -> Vec<ValueId> {
    match operation {
        Operation::Constant { result, .. }
        | Operation::Unary { result, .. }
        | Operation::Binary { result, .. }
        | Operation::Call { result, .. } => vec![*result],
        Operation::Return { .. } | Operation::Assert { .. } => Vec::new(),
        Operation::If {
            then_block,
            else_block,
            ..
        } => then_block
            .operations
            .iter()
            .flat_map(operation_outputs)
            .chain(else_block.operations.iter().flat_map(operation_outputs))
            .collect(),
        Operation::Match { arms, .. } => arms
            .iter()
            .flat_map(|arm| arm.body.operations.iter().flat_map(operation_outputs))
            .collect(),
        Operation::CompiledRegionCall { outputs, .. } => outputs.clone(),
    }
}
