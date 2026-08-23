use crate::{
    model::resume_block, CompileError, CompilePlan, CompileRegion, EffectSet, PlanSegment,
    PlannedBlock, PlannedFunction, StandardRegion,
};
use severian_artifact::CompiledRegionId;
use severian_mir::{Block, Module, Operation, Value, ValueId};
use severian_universal::{BinaryOperator, CompileRoute, CompilerId, TypeContext};
use std::collections::{BTreeMap, BTreeSet};

pub fn plan(module: &Module, types: &TypeContext) -> Result<CompilePlan, CompileError> {
    let class_types = module
        .classes
        .iter()
        .map(|class| class.id)
        .collect::<BTreeSet<_>>();
    let values = module
        .values
        .iter()
        .map(|value| (value.id, *value))
        .collect::<BTreeMap<_, _>>();
    let mut next_region = 0u32;
    let mut nested_regions = Vec::new();
    let initializer = plan_block(
        &module.initializer,
        &values,
        &module.globals.iter().copied().collect(),
        types,
        &class_types,
        &mut next_region,
        &mut nested_regions,
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
                        plan_block(
                            body,
                            &values,
                            &BTreeSet::new(),
                            types,
                            &class_types,
                            &mut next_region,
                            &mut nested_regions,
                        )
                    })
                    .transpose()?,
            })
        })
        .collect::<Result<Vec<_>, CompileError>>()?;
    Ok(CompilePlan {
        source: module.clone(),
        initializer,
        functions,
        nested_regions,
    })
}

fn plan_block(
    block: &Block,
    values: &BTreeMap<ValueId, Value>,
    retained: &BTreeSet<ValueId>,
    types: &TypeContext,
    class_types: &BTreeSet<severian_universal::TypeId>,
    next_region: &mut u32,
    nested_regions: &mut Vec<CompileRegion>,
) -> Result<PlannedBlock, CompileError> {
    let mut operations = Vec::with_capacity(block.operations.len());
    for (index, operation) in block.operations.iter().enumerate() {
        let nested_retained = block.operations[index + 1..]
            .iter()
            .flat_map(operation_inputs)
            .chain(retained.iter().copied())
            .collect::<BTreeSet<_>>();
        operations.push(rewrite_nested_control_flow(
            operation,
            values,
            &nested_retained,
            types,
            class_types,
            next_region,
            nested_regions,
        )?);
    }
    let routes = operations
        .iter()
        .enumerate()
        .map(|(index, operation)| operation_route(index, operation, values, types, class_types))
        .collect::<Result<Vec<_>, _>>()?;
    let mut segments = Vec::new();
    let mut start = 0usize;
    while start < operations.len() {
        let route = routes[start];
        let mut end = start + 1;
        while end < operations.len() && routes[end] == route {
            end += 1;
        }
        let region_operations = operations[start..end].to_vec();
        match route {
            CompileRoute::Standard => {
                segments.push(PlanSegment::Standard(StandardRegion {
                    operations: region_operations,
                }));
            }
            CompileRoute::Compiler(compiler) => {
                segments.push(PlanSegment::Compiler(build_region(
                    values,
                    CompiledRegionId::new(*next_region),
                    compiler,
                    region_operations,
                    &operations[end..],
                    retained,
                )?));
                *next_region += 1;
            }
        }
        start = end;
    }
    Ok(PlannedBlock { segments })
}

fn rewrite_nested_control_flow(
    operation: &Operation,
    values: &BTreeMap<ValueId, Value>,
    retained: &BTreeSet<ValueId>,
    types: &TypeContext,
    class_types: &BTreeSet<severian_universal::TypeId>,
    next_region: &mut u32,
    nested_regions: &mut Vec<CompileRegion>,
) -> Result<Operation, CompileError> {
    Ok(match operation {
        Operation::If {
            condition,
            then_block,
            else_block,
        } => Operation::If {
            condition: *condition,
            then_block: plan_nested_block(
                then_block,
                values,
                retained,
                types,
                class_types,
                next_region,
                nested_regions,
            )?,
            else_block: plan_nested_block(
                else_block,
                values,
                retained,
                types,
                class_types,
                next_region,
                nested_regions,
            )?,
        },
        Operation::Match { subject, arms } => Operation::Match {
            subject: *subject,
            arms: arms
                .iter()
                .map(|arm| {
                    let mut arm = arm.clone();
                    arm.body = plan_nested_block(
                        &arm.body,
                        values,
                        retained,
                        types,
                        class_types,
                        next_region,
                        nested_regions,
                    )?;
                    Ok(arm)
                })
                .collect::<Result<Vec<_>, CompileError>>()?,
        },
        Operation::While {
            condition_block,
            condition,
            body,
        } => Operation::While {
            condition_block: plan_nested_block(
                condition_block,
                values,
                retained,
                types,
                class_types,
                next_region,
                nested_regions,
            )?,
            condition: *condition,
            body: plan_nested_block(
                body,
                values,
                retained,
                types,
                class_types,
                next_region,
                nested_regions,
            )?,
        },
        operation => operation.clone(),
    })
}

fn plan_nested_block(
    block: &Block,
    values: &BTreeMap<ValueId, Value>,
    retained: &BTreeSet<ValueId>,
    types: &TypeContext,
    class_types: &BTreeSet<severian_universal::TypeId>,
    next_region: &mut u32,
    nested_regions: &mut Vec<CompileRegion>,
) -> Result<Block, CompileError> {
    let planned = plan_block(
        block,
        values,
        retained,
        types,
        class_types,
        next_region,
        nested_regions,
    )?;
    nested_regions.extend(planned.segments.iter().filter_map(|segment| match segment {
        PlanSegment::Compiler(region) => Some(region.clone()),
        PlanSegment::Standard(_) => None,
    }));
    Ok(resume_block(&planned))
}

fn operation_route(
    index: usize,
    operation: &Operation,
    values: &BTreeMap<ValueId, Value>,
    types: &TypeContext,
    class_types: &BTreeSet<severian_universal::TypeId>,
) -> Result<CompileRoute, CompileError> {
    if matches!(operation, Operation::CompiledRegionCall { .. }) {
        return Err(CompileError::PlannerGeneratedOperation(index));
    }
    if matches!(
        operation,
        Operation::Coverage { .. }
            | Operation::Return { .. }
            | Operation::Assert { .. }
            | Operation::If { .. }
            | Operation::Match { .. }
            | Operation::While { .. }
            | Operation::Break
            | Operation::Continue
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
        if class_types.contains(&value.type_id) {
            continue;
        }
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
        Operation::Coverage { .. }
        | Operation::Constant { .. }
        | Operation::Break
        | Operation::Continue => Vec::new(),
        Operation::Aggregate { fields, .. } => fields.clone(),
        Operation::FieldGet { object, .. } => vec![*object],
        Operation::FieldSet { object, value, .. } => vec![*object, *value],
        Operation::Assign { target, value } => vec![*target, *value],
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
        Operation::While {
            condition_block,
            condition,
            body,
        } => std::iter::once(*condition)
            .chain(condition_block.operations.iter().flat_map(operation_inputs))
            .chain(body.operations.iter().flat_map(operation_inputs))
            .collect(),
        Operation::CompiledRegionCall { inputs, .. } => inputs.clone(),
    }
}

fn operation_outputs(operation: &Operation) -> Vec<ValueId> {
    match operation {
        Operation::Coverage { .. } | Operation::Break | Operation::Continue => Vec::new(),
        Operation::Constant { result, .. }
        | Operation::Unary { result, .. }
        | Operation::Binary { result, .. }
        | Operation::Aggregate { result, .. }
        | Operation::FieldGet { result, .. }
        | Operation::FieldSet { result, .. }
        | Operation::Call { result, .. } => vec![*result],
        Operation::Assign { target, .. } => vec![*target],
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
        Operation::While {
            condition_block,
            body,
            ..
        } => condition_block
            .operations
            .iter()
            .flat_map(operation_outputs)
            .chain(body.operations.iter().flat_map(operation_outputs))
            .collect(),
        Operation::CompiledRegionCall { outputs, .. } => outputs.clone(),
    }
}
