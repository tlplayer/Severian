use crate::{CompileError, CompilePlan, CompileRegion, EffectSet, PlanSegment, StandardRegion};
use severian_artifact::CompiledRegionId;
use severian_mir::{Module, Operation, Value, ValueId};
use severian_universal::{BinaryOperator, CompileRoute, CompilerId, TypeContext};
use std::collections::{BTreeMap, BTreeSet};

pub fn plan(module: &Module, types: &TypeContext) -> Result<CompilePlan, CompileError> {
    let values = module
        .values
        .iter()
        .map(|value| (value.id, *value))
        .collect::<BTreeMap<_, _>>();
    let routes = module
        .operations
        .iter()
        .enumerate()
        .map(|(index, operation)| operation_route(index, operation, &values, types))
        .collect::<Result<Vec<_>, _>>()?;
    let mut segments = Vec::new();
    let mut next_region = 0u32;
    let mut start = 0usize;
    while start < module.operations.len() {
        let route = routes[start];
        let mut end = start + 1;
        while end < module.operations.len() && routes[end] == route {
            end += 1;
        }
        let operations = module.operations[start..end].to_vec();
        match route {
            CompileRoute::Standard => {
                segments.push(PlanSegment::Standard(StandardRegion { operations }));
            }
            CompileRoute::Compiler(compiler) => {
                segments.push(PlanSegment::Compiler(build_region(
                    module,
                    &values,
                    CompiledRegionId::new(next_region),
                    compiler,
                    start,
                    end,
                    operations,
                )?));
                next_region += 1;
            }
        }
        start = end;
    }
    Ok(CompilePlan {
        source: module.clone(),
        segments,
    })
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
    module: &Module,
    values: &BTreeMap<ValueId, Value>,
    id: CompiledRegionId,
    compiler: CompilerId,
    start: usize,
    end: usize,
    operations: Vec<Operation>,
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
    let bound = module
        .bindings
        .iter()
        .map(|(_, value)| *value)
        .collect::<BTreeSet<_>>();
    let mut outputs = Vec::new();
    for output in produced {
        let used_outside = module
            .operations
            .iter()
            .enumerate()
            .any(|(index, operation)| {
                (index < start || index >= end) && operation_inputs(operation).contains(&output)
            });
        if used_outside || bound.contains(&output) {
            outputs.push(value(values, output)?);
        }
    }
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
        Operation::CompiledRegionCall { inputs, .. } => inputs.clone(),
    }
}

fn operation_outputs(operation: &Operation) -> Vec<ValueId> {
    match operation {
        Operation::Constant { result, .. }
        | Operation::Unary { result, .. }
        | Operation::Binary { result, .. } => vec![*result],
        Operation::CompiledRegionCall { outputs, .. } => outputs.clone(),
    }
}
