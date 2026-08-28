#![allow(dead_code)]

use crate::{
    model::resume_block, CompileError, CompilePlan, CompileRegion, EffectSet, PlanSegment,
    PlannedBlock, PlannedFunction, StandardRegion,
};
use severian_artifact::CompiledRegionId;
use severian_mir::{Block, Module, Operation, Value, ValueId};
use severian_universal::{BinaryOperator, CompileRoute, CompilerId, TypeContext};
use std::collections::{BTreeMap, BTreeSet};

pub fn plan(module: &Module, types: &TypeContext) -> Result<CompilePlan, CompileError> {
    let mut source = module.clone();
    let mut nested_regions = Vec::new();
    let mut next_region = 0u32;
    extract_cfg_compile_operations(
        &mut source.initializer,
        &source.globals,
        types,
        &mut next_region,
        &mut nested_regions,
    )?;
    for function in &mut source.functions {
        if let Some(body) = &mut function.body {
            extract_cfg_compile_operations(
                body,
                &source.globals,
                types,
                &mut next_region,
                &mut nested_regions,
            )?;
        }
    }
    let initializer = PlannedBlock {
        segments: vec![PlanSegment::Standard(StandardRegion {
            operations: Vec::new(),
        })],
    };
    let functions = source
        .functions
        .iter()
        .map(|function| PlannedFunction {
            declaration: function.clone(),
            body: function.body.as_ref().map(|_| PlannedBlock {
                segments: vec![PlanSegment::Standard(StandardRegion {
                    operations: Vec::new(),
                })],
            }),
        })
        .collect();
    Ok(CompilePlan {
        source,
        initializer,
        functions,
        nested_regions,
    })
}

fn extract_cfg_compile_operations(
    body: &mut severian_mir::CfgBody,
    globals: &[severian_mir::GlobalDecl],
    types: &TypeContext,
    next_region: &mut u32,
    regions: &mut Vec<CompileRegion>,
) -> Result<(), CompileError> {
    let locals = body.locals.clone();
    for block in &mut body.blocks {
        let placement = block.execution;
        // A constant whose type owns this compile route is itself a compile
        // operation. Normalize it before finding maximal same-compiler runs.
        for statement in &mut block.statements {
            let routed_assignment = match statement {
                severian_mir::CfgStatement::Assign(
                    place,
                    severian_mir::Rvalue::Use(severian_mir::Operand::Constant { .. }),
                ) => {
                    let ty = cfg_place_type(&locals, globals, place)?;
                    // Raw pointers use stable structural TypeIds rather than
                    // entries in TypeContext. They are always handled by the
                    // native pipeline, even when initialized from `None`.
                    if severian_universal::is_raw_pointer_type(ty) {
                        continue;
                    }
                    match types
                        .compile_route(ty)
                        .map_err(|error| CompileError::Type(ty, error.to_string()))?
                    {
                        CompileRoute::Compiler(compiler) => Some((place.clone(), compiler)),
                        CompileRoute::Standard => None,
                    }
                }
                _ => None,
            };
            if let Some((place, compiler)) = routed_assignment {
                *statement = severian_mir::CfgStatement::Operation {
                    id: severian_universal::OpId::named("compile", "materialize"),
                    operands: Vec::new(),
                    results: vec![place],
                    attributes: severian_universal::Attrs::from([(
                        severian_universal::COMPILE_TYPE_ATTRIBUTE,
                        severian_universal::AttrValue::Compiler(compiler),
                    )]),
                };
            }
        }

        let old_statements = std::mem::take(&mut block.statements);
        let old_spans = std::mem::take(&mut block.statement_spans);
        let mut statements = Vec::with_capacity(old_statements.len());
        let mut spans = Vec::with_capacity(old_spans.len());
        let mut start = 0usize;
        while start < old_statements.len() {
            let Some(compiler) = cfg_statement_compiler(&old_statements[start]) else {
                statements.push(old_statements[start].clone());
                spans.push(old_spans.get(start).copied().flatten());
                start += 1;
                continue;
            };
            let mut end = start + 1;
            while end < old_statements.len()
                && cfg_statement_compiler(&old_statements[end]) == Some(compiler)
            {
                end += 1;
            }

            let region_id = CompiledRegionId::new(*next_region);
            *next_region += 1;
            let mut external_operands = Vec::new();
            let mut external_slots = Vec::new();
            let mut input_types = Vec::new();
            let mut slot_types = Vec::new();
            let mut place_slots = BTreeMap::new();
            let mut final_results = BTreeMap::new();
            let mut compile_operations = Vec::with_capacity(end - start);

            for statement in &old_statements[start..end] {
                let severian_mir::CfgStatement::Operation {
                    id,
                    operands,
                    results,
                    attributes,
                } = statement
                else {
                    unreachable!("same-compiler run contains an operation")
                };
                let mut operand_types = Vec::with_capacity(operands.len());
                let mut operand_slots = Vec::with_capacity(operands.len());
                for operand in operands {
                    let ty = cfg_operand_type(&locals, globals, operand)?;
                    let existing = match operand {
                        severian_mir::Operand::Copy(place) | severian_mir::Operand::Move(place) => {
                            place_slots.get(place).copied()
                        }
                        severian_mir::Operand::Constant { .. } => None,
                        severian_mir::Operand::Function(_) => {
                            return Err(CompileError::InvalidArtifact(
                                "CompileOp cannot consume a function value".into(),
                            ))
                        }
                    };
                    let slot = if let Some(slot) = existing {
                        slot
                    } else {
                        let slot = u32::try_from(slot_types.len()).map_err(|_| {
                            CompileError::InvalidArtifact(
                                "compiled region has too many values".into(),
                            )
                        })?;
                        slot_types.push(ty);
                        input_types.push(ty);
                        external_operands.push(operand.clone());
                        external_slots.push(slot);
                        if let severian_mir::Operand::Copy(place)
                        | severian_mir::Operand::Move(place) = operand
                        {
                            place_slots.insert(place.clone(), slot);
                        }
                        slot
                    };
                    operand_types.push(ty);
                    operand_slots.push(slot);
                }

                let mut result_types = Vec::with_capacity(results.len());
                let mut result_slots = Vec::with_capacity(results.len());
                for place in results {
                    let ty = cfg_place_type(&locals, globals, place)?;
                    let slot = u32::try_from(slot_types.len()).map_err(|_| {
                        CompileError::InvalidArtifact("compiled region has too many values".into())
                    })?;
                    slot_types.push(ty);
                    place_slots.insert(place.clone(), slot);
                    final_results.insert(place.clone(), (slot, ty));
                    result_types.push(ty);
                    result_slots.push(slot);
                }
                compile_operations.push(crate::CompileOperation {
                    id: *id,
                    operands: operand_types,
                    results: result_types,
                    operand_slots,
                    result_slots,
                    attributes: attributes.clone(),
                });
            }

            // Discovery interleaves external operands with produced values,
            // but the region ABI reserves the dense prefix for inputs. Remap
            // once after discovery so rank, dtype, and operation count never
            // affect slot identity.
            let mut slot_remap = BTreeMap::new();
            for (new, old) in external_slots.into_iter().enumerate() {
                slot_remap.insert(
                    old,
                    u32::try_from(new).map_err(|_| {
                        CompileError::InvalidArtifact("compiled region has too many inputs".into())
                    })?,
                );
            }
            let mut next_slot = u32::try_from(input_types.len()).map_err(|_| {
                CompileError::InvalidArtifact("compiled region has too many inputs".into())
            })?;
            for operation in &compile_operations {
                for old in &operation.result_slots {
                    slot_remap.entry(*old).or_insert_with(|| {
                        let slot = next_slot;
                        next_slot += 1;
                        slot
                    });
                }
            }
            for operation in &mut compile_operations {
                for slot in operation
                    .operand_slots
                    .iter_mut()
                    .chain(&mut operation.result_slots)
                {
                    *slot = slot_remap[slot];
                }
            }
            for (slot, _) in final_results.values_mut() {
                *slot = slot_remap[slot];
            }

            let (result_places, output_slots, output_types) = final_results.into_iter().fold(
                (Vec::new(), Vec::new(), Vec::new()),
                |(mut places, mut slots, mut types), (place, (slot, ty))| {
                    places.push(place);
                    slots.push(slot);
                    types.push(ty);
                    (places, slots, types)
                },
            );
            let mut wrapper_attributes = compile_operations[0].attributes.clone();
            wrapper_attributes.insert(
                severian_universal::COMPILED_ARTIFACT_ATTRIBUTE,
                severian_universal::AttrValue::Integer(i128::from(region_id.index())),
            );
            statements.push(severian_mir::CfgStatement::Operation {
                id: compile_operations[0].id,
                operands: external_operands,
                results: result_places,
                attributes: wrapper_attributes,
            });
            spans.push(old_spans.get(start).copied().flatten());
            let mut region = CompileRegion {
                id: region_id,
                compiler,
                operations: Vec::new(),
                compile_operations,
                output_slots,
                inputs: input_types
                    .into_iter()
                    .enumerate()
                    .map(|(index, type_id)| Value {
                        id: ValueId(index as u32),
                        type_id,
                    })
                    .collect(),
                outputs: output_types
                    .into_iter()
                    .enumerate()
                    .map(|(index, type_id)| Value {
                        id: ValueId(index as u32),
                        type_id,
                    })
                    .collect(),
                value_contracts: Vec::new(),
                effects: EffectSet {
                    reads_memory: true,
                    writes_memory: true,
                    may_trap: true,
                },
                placement,
            };
            region
                .rebuild_value_contracts(types)
                .map_err(CompileError::InvalidArtifact)?;
            regions.push(region);
            start = end;
        }
        block.statements = statements;
        block.statement_spans = spans;
    }
    Ok(())
}

fn cfg_statement_compiler(statement: &severian_mir::CfgStatement) -> Option<CompilerId> {
    let severian_mir::CfgStatement::Operation { attributes, .. } = statement else {
        return None;
    };
    match attributes.get(&severian_universal::COMPILE_TYPE_ATTRIBUTE) {
        Some(severian_universal::AttrValue::Compiler(compiler)) => Some(*compiler),
        _ => None,
    }
}

fn cfg_operand_type(
    locals: &[severian_mir::LocalDecl],
    globals: &[severian_mir::GlobalDecl],
    operand: &severian_mir::Operand,
) -> Result<severian_universal::TypeId, CompileError> {
    match operand {
        severian_mir::Operand::Copy(place) | severian_mir::Operand::Move(place) => {
            cfg_place_type(locals, globals, place)
        }
        severian_mir::Operand::Constant { ty, .. } => Ok(*ty),
        severian_mir::Operand::Function(_) => Err(CompileError::InvalidArtifact(
            "CompileOp cannot consume a function value".into(),
        )),
    }
}

fn cfg_place_type(
    locals: &[severian_mir::LocalDecl],
    globals: &[severian_mir::GlobalDecl],
    place: &severian_mir::Place,
) -> Result<severian_universal::TypeId, CompileError> {
    if !place.projection.is_empty() {
        return Err(CompileError::InvalidArtifact(
            "projected CompileOp values are not supported".into(),
        ));
    }
    match place.base {
        severian_mir::PlaceBase::Local(local) => locals
            .get(local.0 as usize)
            .map(|local| local.ty)
            .ok_or(CompileError::MissingValue(local.0)),
        severian_mir::PlaceBase::Global(global) => globals
            .get(global.0 as usize)
            .map(|global| global.ty)
            .ok_or(CompileError::MissingValue(global.0)),
    }
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
        if class_types.contains(&value.type_id)
            || severian_universal::is_raw_pointer_type(value.type_id)
        {
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
        compile_operations: Vec::new(),
        output_slots: Vec::new(),
        inputs,
        outputs,
        value_contracts: Vec::new(),
        effects,
        placement: None,
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
        Operation::Call { arguments, .. } | Operation::Spawn { arguments, .. } => arguments.clone(),
        Operation::Await { task, .. } => vec![*task],
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
        Operation::Coverage { .. }
        | Operation::Break
        | Operation::Continue
        | Operation::Return { .. }
        | Operation::Assert { .. } => Vec::new(),
        Operation::Constant { result, .. }
        | Operation::Unary { result, .. }
        | Operation::Binary { result, .. }
        | Operation::Aggregate { result, .. }
        | Operation::FieldGet { result, .. }
        | Operation::FieldSet { result, .. }
        | Operation::Call { result, .. }
        | Operation::Spawn { result, .. }
        | Operation::Await { result, .. } => vec![*result],
        Operation::Assign { target, .. } => vec![*target],
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
