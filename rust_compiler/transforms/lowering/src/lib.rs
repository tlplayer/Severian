#![forbid(unsafe_code)]

use severian_lir::{
    BinaryOperation, Block as LirBlock, Constant, Function as LirFunction, FunctionId,
    FunctionLinkage, LoweredFloatFormat, LoweredTensorDimension, LoweredTensorElement,
    LoweredTensorShape, LoweredType, Module as LirModule, Operation as LirOperation,
    UnaryOperation, Value, ValueId,
};
use severian_mir::Module as MirModule;
use severian_target::TargetSpec;
use severian_universal::{
    BinaryOperator, FloatFormat, IntegerWidth, LiteralValue, PrimitiveRepresentation,
    TensorDimension, TensorShape, TypeContext, TypeId, UnaryOperator,
};
use std::{collections::BTreeSet, fmt};

mod cfg_lowering_entry {
    use super::*;

    pub fn lower(
        mir: &MirModule,
        types: &TypeContext,
        target: &TargetSpec,
    ) -> Result<LirModule, LoweringError> {
        let mut context = CfgLowering {
            mir,
            types,
            target,
            values: Vec::new(),
            task_locals: BTreeSet::new(),
        };
        let mut initializer_cfg = context.lower_cfg_body(&mir.initializer)?;
        initializer_cfg.return_type = LoweredType::Unit;
        let functions = mir
            .functions
            .iter()
            .map(|function| {
                Ok(LirFunction {
                    id: FunctionId(function.id.0),
                    name: function.name.clone(),
                    parameters: Vec::new(),
                    result: context.lower_mir_type(function.result)?,
                    body: None,
                    linkage: match &function.call_type {
                        severian_mir::CallType::Severian => FunctionLinkage::Internal,
                        severian_mir::CallType::External(call) => FunctionLinkage::External {
                            symbol: call.symbol.0.clone(),
                        },
                    },
                    parameter_types: function
                        .parameters
                        .iter()
                        .map(|ty| context.lower_mir_type(*ty))
                        .collect::<Result<Vec<_>, _>>()?,
                    cfg: function
                        .body
                        .as_ref()
                        .map(|body| context.lower_cfg_body(body))
                        .transpose()?,
                })
            })
            .collect::<Result<Vec<_>, LoweringError>>()?;
        let storage_globals = mir
            .globals
            .iter()
            .map(|global| {
                Ok(severian_lir::GlobalDecl {
                    id: severian_lir::GlobalId(global.id.0),
                    ty: context.lower_mir_type(global.ty)?,
                    mutable: global.mutable,
                    span: global.span,
                })
            })
            .collect::<Result<Vec<_>, LoweringError>>()?;
        let classes = mir
            .classes
            .iter()
            .enumerate()
            .map(|(id, declaration)| {
                Ok(severian_lir::ClassDeclaration {
                    id: id as u32,
                    name: declaration.name.clone(),
                    fields: declaration
                        .fields
                        .iter()
                        .map(|field| {
                            Ok(severian_lir::ClassFieldDeclaration {
                                name: field.name.clone(),
                                ty: context.lower_mir_type(field.ty)?,
                            })
                        })
                        .collect::<Result<Vec<_>, LoweringError>>()?,
                })
            })
            .collect::<Result<Vec<_>, LoweringError>>()?;
        let traits = mir
            .traits
            .iter()
            .map(|declaration| {
                Ok(severian_lir::TraitDeclaration {
                    id: severian_lir::TraitId {
                        package: declaration.definition.package,
                        module: declaration.definition.module,
                        declaration: declaration.definition.declaration.0,
                    },
                    name: declaration.name.clone(),
                    methods: declaration
                        .methods
                        .iter()
                        .map(|method| {
                            Ok(severian_lir::TraitMethodDeclaration {
                                name: method.name.clone(),
                                parameters: method
                                    .parameters
                                    .iter()
                                    .map(|parameter| match parameter {
                                        severian_mir::TraitType::SelfType => {
                                            Ok(severian_lir::TraitType::SelfType)
                                        }
                                        severian_mir::TraitType::Concrete(ty) => context
                                            .lower_mir_type(*ty)
                                            .map(severian_lir::TraitType::Concrete),
                                        severian_mir::TraitType::Symbolic(name) => {
                                            Ok(severian_lir::TraitType::Symbolic(name.clone()))
                                        }
                                    })
                                    .collect::<Result<Vec<_>, LoweringError>>()?,
                                result: match &method.result {
                                    severian_mir::TraitType::SelfType => {
                                        severian_lir::TraitType::SelfType
                                    }
                                    severian_mir::TraitType::Concrete(ty) => {
                                        severian_lir::TraitType::Concrete(
                                            context.lower_mir_type(*ty)?,
                                        )
                                    }
                                    severian_mir::TraitType::Symbolic(name) => {
                                        severian_lir::TraitType::Symbolic(name.clone())
                                    }
                                },
                            })
                        })
                        .collect::<Result<Vec<_>, LoweringError>>()?,
                })
            })
            .collect::<Result<Vec<_>, LoweringError>>()?;
        let uses_gpu = cfg_uses_gpu(&initializer_cfg)
            || functions
                .iter()
                .filter_map(|function| function.cfg.as_ref())
                .any(cfg_uses_gpu);
        Ok(LirModule {
            values: context.values,
            globals: Vec::new(),
            initializer: LirBlock::default(),
            functions,
            entry: mir.entry.map(|entry| FunctionId(entry.0)),
            traits,
            classes,
            storage_globals,
            initializer_cfg: Some(initializer_cfg),
            gpu_architecture: uses_gpu
                .then(|| context.target.rocm_device())
                .flatten()
                .map(|device| device.architecture.clone()),
        })
    }
}

pub use cfg_lowering_entry::lower;

struct CfgLowering<'a> {
    mir: &'a MirModule,
    types: &'a TypeContext,
    target: &'a TargetSpec,
    values: Vec<Value>,
    task_locals: BTreeSet<severian_mir::LocalId>,
}

impl CfgLowering<'_> {
    fn lower_cfg_body(
        &mut self,
        body: &severian_mir::CfgBody,
    ) -> Result<severian_lir::CfgBody, LoweringError> {
        self.task_locals = task_locals(body);
        let mut scoped_spawns = Vec::new();
        let locals = body
            .locals
            .iter()
            .map(|local| {
                Ok(severian_lir::LocalDecl {
                    id: severian_lir::LocalId(local.id.0),
                    ty: if self.task_locals.contains(&local.id) {
                        self.lower_mir_type(local.ty)?
                            .task()
                            .expect("task results cannot themselves be tasks")
                    } else {
                        self.lower_mir_type(local.ty)?
                    },
                    mutable: local.mutable,
                    argument: local.argument,
                    span: local.span,
                })
            })
            .collect::<Result<Vec<_>, LoweringError>>()?;
        let mut blocks = Vec::with_capacity(body.blocks.len());
        for block in &body.blocks {
            let mut operations = Vec::new();
            let mut operation_spans = Vec::new();
            for (index, statement) in block.statements.iter().enumerate() {
                let start = operations.len();
                self.lower_statement(body, statement, &mut operations)?;
                operation_spans.extend(std::iter::repeat_n(
                    block.statement_spans.get(index).copied().flatten(),
                    operations.len() - start,
                ));
            }
            let start = operations.len();
            let terminator = self.lower_terminator(body, &block.terminator, &mut operations)?;
            operation_spans.extend(std::iter::repeat_n(
                block.terminator_span,
                operations.len() - start,
            ));
            if !matches!(
                &block.terminator,
                severian_mir::Terminator::Spawn {
                    owner: severian_mir::TaskOwner::Runtime,
                    ..
                } | severian_mir::Terminator::SpawnFieldUpdate {
                    owner: severian_mir::TaskOwner::Runtime,
                    ..
                }
            ) {
                if let Some(result) =
                    operations[start..]
                        .iter()
                        .find_map(|operation| match operation {
                            LirOperation::Spawn { result, .. }
                            | LirOperation::SpawnFieldUpdate { result, .. } => Some(result),
                            _ => None,
                        })
                {
                    scoped_spawns.push((block.id, *result));
                }
            }
            blocks.push(severian_lir::BasicBlock {
                id: severian_lir::BlockId(block.id.0),
                execution: block.execution,
                operations,
                operation_spans,
                terminator,
                terminator_span: block.terminator_span,
            });
        }
        let dominators = cfg_dominators(body);
        let mut scoped_awaits = vec![Vec::new(); blocks.len()];
        for (spawn, task) in scoped_spawns {
            for source in &body.blocks {
                if !dominators[source.id.0 as usize].contains(&spawn) {
                    continue;
                }
                let successors = cfg_successors(&source.terminator);
                let leaves_spawn_region = matches!(
                    source.terminator,
                    severian_mir::Terminator::Return(_) | severian_mir::Terminator::Throw(_)
                ) || successors
                    .iter()
                    .any(|successor| !dominators[successor.0 as usize].contains(&spawn));
                if leaves_spawn_region {
                    scoped_awaits[source.id.0 as usize].push(task);
                }
            }
        }
        for (block, tasks) in blocks.iter_mut().zip(scoped_awaits) {
            for task in tasks {
                let result_type = self.value_type(task).task_result().ok_or_else(|| {
                    LoweringError::UnsupportedCfgOperation(
                        "scoped spawn result is not a task".into(),
                    )
                })?;
                let result = self.new_value(result_type);
                block.operations.push(LirOperation::Await { task, result });
                block.operation_spans.push(block.terminator_span);
            }
        }
        retain_supported_gpu_regions(&mut blocks, self.target.rocm_device().is_some());
        Ok(severian_lir::CfgBody {
            entry: severian_lir::BlockId(body.entry.0),
            blocks,
            locals,
            return_type: self.lower_mir_type(body.return_type)?,
        })
    }

    fn lower_statement(
        &mut self,
        body: &severian_mir::CfgBody,
        statement: &severian_mir::CfgStatement,
        operations: &mut Vec<LirOperation>,
    ) -> Result<(), LoweringError> {
        match statement {
            severian_mir::CfgStatement::Assign(place, rvalue) => {
                let value = self.lower_rvalue(body, rvalue, operations)?;
                operations.push(LirOperation::Store {
                    place: self.lower_place(place),
                    value,
                });
            }
            severian_mir::CfgStatement::Drop(place) => {
                let ty = self.place_type(body, place)?;
                if matches!(ty, LoweredType::Tensor { .. }) && place.projection.is_empty() {
                    let address = self.new_value(LoweredType::Bytes);
                    operations.push(LirOperation::AddressOf {
                        place: self.lower_place(place),
                        result: address,
                    });
                    operations.push(LirOperation::RuntimeCall {
                        symbol: "__sev_tensor_local_release".into(),
                        arguments: vec![address],
                        result: None,
                    });
                } else if ty == LoweredType::String {
                    let value = self.load_place(body, place, operations)?;
                    operations.push(LirOperation::RuntimeCall {
                        symbol: "__sev_string_release".into(),
                        arguments: vec![value],
                        result: None,
                    });
                }
            }
            severian_mir::CfgStatement::StorageLive(_)
            | severian_mir::CfgStatement::StorageDead(_) => {}
            severian_mir::CfgStatement::Assert {
                condition,
                message,
                origin,
            } => {
                let condition = self.lower_operand(body, condition, operations)?;
                let message = message
                    .as_ref()
                    .map(|message| self.lower_operand(body, message, operations))
                    .transpose()?;
                operations.push(LirOperation::Assert {
                    condition,
                    message,
                    location: origin.location.as_ref().map(|location| {
                        severian_lir::AssertionLocation {
                            file: location.file.clone(),
                            line: location.line,
                            column: location.column,
                            expression: location.expression.clone(),
                        }
                    }),
                });
            }
            severian_mir::CfgStatement::Coverage(point) => {
                operations.push(LirOperation::Coverage {
                    key: point.key.clone().unwrap_or_else(|| {
                        format!("pending:{}:{}", point.span_start, point.ordinal)
                    }),
                });
            }
            severian_mir::CfgStatement::Operation {
                operands,
                results,
                attributes,
                ..
            } => {
                if let Some(severian_universal::AttrValue::String(mnemonic)) =
                    attributes.get(&severian_universal::MLIR_OPERATION_NAME_ATTRIBUTE)
                {
                    let inputs = operands
                        .iter()
                        .map(|operand| self.lower_operand(body, operand, operations))
                        .collect::<Result<Vec<_>, _>>()?;
                    let [place] = results.as_slice() else {
                        return Err(LoweringError::UnsupportedCfgOperation(
                            "direct MLIR expressions require exactly one result".into(),
                        ));
                    };
                    let result = self.new_value(self.place_type(body, place)?);
                    let parameters = attributes
                        .get(&severian_universal::MLIR_OPERATION_PARAMETERS_ATTRIBUTE)
                        .and_then(|value| match value {
                            severian_universal::AttrValue::String(value) => Some(value.clone()),
                            _ => None,
                        });
                    operations.push(LirOperation::Mlir {
                        mnemonic: mnemonic.clone(),
                        parameters,
                        operands: inputs,
                        result,
                    });
                    operations.push(LirOperation::Store {
                        place: self.lower_place(place),
                        value: result,
                    });
                    return Ok(());
                }
                let Some(severian_universal::AttrValue::Integer(artifact)) =
                    attributes.get(&severian_universal::COMPILED_ARTIFACT_ATTRIBUTE)
                else {
                    return Err(LoweringError::UnsupportedCfgOperation(
                        "registered operation did not pass through CompileType planning".into(),
                    ));
                };
                let artifact = u32::try_from(*artifact).map_err(|_| {
                    LoweringError::UnsupportedCfgOperation(
                        "compiled artifact identity is out of range".into(),
                    )
                })?;
                let inputs = operands
                    .iter()
                    .map(|operand| self.lower_operand(body, operand, operations))
                    .collect::<Result<Vec<_>, _>>()?;
                let outputs = results
                    .iter()
                    .map(|place| self.place_type(body, place).map(|ty| self.new_value(ty)))
                    .collect::<Result<Vec<_>, _>>()?;
                operations.push(LirOperation::ArtifactCall {
                    artifact: severian_artifact::ArtifactId::for_region(
                        severian_artifact::CompiledRegionId::new(artifact),
                    ),
                    inputs,
                    outputs: outputs.clone(),
                });
                for (place, value) in results.iter().zip(outputs) {
                    operations.push(LirOperation::Store {
                        place: self.lower_place(place),
                        value,
                    });
                }
            }
        }
        Ok(())
    }

    fn lower_rvalue(
        &mut self,
        body: &severian_mir::CfgBody,
        rvalue: &severian_mir::Rvalue,
        operations: &mut Vec<LirOperation>,
    ) -> Result<ValueId, LoweringError> {
        match rvalue {
            severian_mir::Rvalue::Use(operand) => self.lower_operand(body, operand, operations),
            severian_mir::Rvalue::Convert {
                operand,
                conversion,
            } => {
                let operand = self.lower_operand(body, operand, operations)?;
                let result = self.new_value(self.lower_mir_type(conversion.to)?);
                operations.push(LirOperation::Convert {
                    operand,
                    result,
                    kind: conversion.kind,
                });
                Ok(result)
            }
            severian_mir::Rvalue::BorrowShared(place)
            | severian_mir::Rvalue::BorrowExclusive(place) => {
                self.load_place(body, place, operations)
            }
            severian_mir::Rvalue::AddressOf(place) => {
                let result = self.new_value(LoweredType::Bytes);
                operations.push(LirOperation::AddressOf {
                    place: self.lower_place(place),
                    result,
                });
                Ok(result)
            }
            severian_mir::Rvalue::Unary { operator, operand } => {
                let operand = self.lower_operand(body, operand, operations)?;
                let result = self.new_value(self.value_type(operand));
                operations.push(LirOperation::Unary {
                    operator: lower_unary(*operator),
                    operand,
                    result,
                });
                Ok(result)
            }
            severian_mir::Rvalue::Binary {
                operator,
                left,
                right,
            } => {
                let left = self.lower_operand(body, left, operations)?;
                let right = self.lower_operand(body, right, operations)?;
                let left_type = self.value_type(left);
                let result_type = match *operator {
                    BinaryOperator::Equal
                    | BinaryOperator::NotEqual
                    | BinaryOperator::Less
                    | BinaryOperator::LessEqual
                    | BinaryOperator::Greater
                    | BinaryOperator::GreaterEqual
                    | BinaryOperator::Contains => self.lower_type_named("bool")?,
                    _ => left_type.clone(),
                };
                let result = self.new_value(result_type);
                if left_type == LoweredType::String {
                    match *operator {
                        BinaryOperator::Add => operations.push(LirOperation::RuntimeCall {
                            symbol: "__sev_string_concat".into(),
                            arguments: vec![left, right],
                            result: Some(result),
                        }),
                        BinaryOperator::Equal
                        | BinaryOperator::NotEqual
                        | BinaryOperator::Less
                        | BinaryOperator::LessEqual
                        | BinaryOperator::Greater
                        | BinaryOperator::GreaterEqual => {
                            let comparison_type = self.lower_type_named("i32")?;
                            let comparison = self.new_value(comparison_type.clone());
                            let zero = self.new_value(comparison_type);
                            operations.push(LirOperation::RuntimeCall {
                                symbol: "__sev_string_compare".into(),
                                arguments: vec![left, right],
                                result: Some(comparison),
                            });
                            operations.push(LirOperation::Constant {
                                value: Constant::Integer("0".into()),
                                result: zero,
                            });
                            operations.push(LirOperation::Binary {
                                operator: lower_binary(*operator),
                                left: comparison,
                                right: zero,
                                result,
                            });
                        }
                        _ => {
                            return Err(LoweringError::UnsupportedStringOperation(*operator));
                        }
                    }
                } else if matches!(left_type, LoweredType::Float { .. })
                    && *operator == BinaryOperator::FloorDivide
                {
                    let quotient = self.new_value(left_type);
                    operations.push(LirOperation::Binary {
                        operator: BinaryOperator::Divide,
                        left,
                        right,
                        result: quotient,
                    });
                    operations.push(LirOperation::Mlir {
                        mnemonic: "math.floor".into(),
                        parameters: None,
                        operands: vec![quotient],
                        result,
                    });
                } else if matches!(left_type, LoweredType::Float { .. })
                    && *operator == BinaryOperator::Remainder
                {
                    let quotient = self.new_value(left_type.clone());
                    let floored = self.new_value(left_type.clone());
                    let product = self.new_value(left_type);
                    operations.push(LirOperation::Binary {
                        operator: BinaryOperator::Divide,
                        left,
                        right,
                        result: quotient,
                    });
                    operations.push(LirOperation::Mlir {
                        mnemonic: "math.floor".into(),
                        parameters: None,
                        operands: vec![quotient],
                        result: floored,
                    });
                    operations.push(LirOperation::Binary {
                        operator: BinaryOperator::Multiply,
                        left: floored,
                        right,
                        result: product,
                    });
                    operations.push(LirOperation::Binary {
                        operator: BinaryOperator::Subtract,
                        left,
                        right: product,
                        result,
                    });
                } else {
                    operations.push(LirOperation::Binary {
                        operator: lower_binary(*operator),
                        left,
                        right,
                        result,
                    });
                }
                Ok(result)
            }
            severian_mir::Rvalue::Aggregate { type_id, fields } => {
                let fields = fields
                    .iter()
                    .map(|field| self.lower_operand(body, field, operations))
                    .collect::<Result<Vec<_>, _>>()?;
                let result = self.new_value(self.lower_mir_type(*type_id)?);
                let class =
                    self.mir
                        .classes
                        .iter()
                        .position(|class| class.id == *type_id)
                        .ok_or(LoweringError::NotPrimitive(*type_id))? as u32;
                operations.push(LirOperation::Aggregate {
                    class,
                    fields,
                    result,
                });
                Ok(result)
            }
            severian_mir::Rvalue::Await { task } => {
                let task = self.lower_operand(body, task, operations)?;
                let result_type = self.value_type(task).task_result().ok_or_else(|| {
                    LoweringError::UnsupportedCfgOperation("await operand is not a task".into())
                })?;
                let result = self.new_value(result_type);
                operations.push(LirOperation::Await { task, result });
                Ok(result)
            }
        }
    }

    fn lower_operand(
        &mut self,
        body: &severian_mir::CfgBody,
        operand: &severian_mir::Operand,
        operations: &mut Vec<LirOperation>,
    ) -> Result<ValueId, LoweringError> {
        match operand {
            severian_mir::Operand::Copy(place) | severian_mir::Operand::Move(place) => {
                self.load_place(body, place, operations)
            }
            severian_mir::Operand::Constant { value, ty } => {
                let result = self.new_value(self.lower_mir_type(*ty)?);
                operations.push(LirOperation::Constant {
                    value: lower_constant(value),
                    result,
                });
                Ok(result)
            }
            severian_mir::Operand::Function(definition) => Err(
                LoweringError::UnsupportedCfgOperation(format!("function value {definition:?}")),
            ),
        }
    }

    fn load_place(
        &mut self,
        body: &severian_mir::CfgBody,
        place: &severian_mir::Place,
        operations: &mut Vec<LirOperation>,
    ) -> Result<ValueId, LoweringError> {
        let result = self.new_value(self.place_type(body, place)?);
        operations.push(LirOperation::Load {
            place: self.lower_place(place),
            result,
        });
        Ok(result)
    }

    fn lower_terminator(
        &mut self,
        body: &severian_mir::CfgBody,
        terminator: &severian_mir::Terminator,
        operations: &mut Vec<LirOperation>,
    ) -> Result<severian_lir::Terminator, LoweringError> {
        match terminator {
            severian_mir::Terminator::Goto(target, arguments) => {
                for (argument, parameter) in arguments
                    .iter()
                    .zip(&body.blocks[target.0 as usize].parameters)
                {
                    let value = self.lower_operand(body, argument, operations)?;
                    operations.push(LirOperation::Store {
                        place: severian_lir::Place {
                            base: severian_lir::PlaceBase::Local(severian_lir::LocalId(
                                parameter.0,
                            )),
                            projection: Vec::new(),
                        },
                        value,
                    });
                }
                Ok(severian_lir::Terminator::Goto(severian_lir::BlockId(
                    target.0,
                )))
            }
            severian_mir::Terminator::Branch {
                condition,
                then_block,
                else_block,
            } => Ok(severian_lir::Terminator::Branch {
                condition: self.lower_operand(body, condition, operations)?,
                then_block: severian_lir::BlockId(then_block.0),
                else_block: severian_lir::BlockId(else_block.0),
            }),
            severian_mir::Terminator::Call {
                callee,
                arguments,
                destination,
                target,
                ..
            } => {
                let function = self.resolve_callee(callee)?;
                let mut lowered_arguments = Vec::new();
                if let severian_mir::Callee::Method { receiver, .. } = callee {
                    lowered_arguments.push(self.lower_operand(body, receiver, operations)?);
                }
                lowered_arguments.extend(
                    arguments
                        .iter()
                        .map(|argument| self.lower_operand(body, argument, operations))
                        .collect::<Result<Vec<_>, _>>()?,
                );
                if let Some(external) = self
                    .mir
                    .functions
                    .iter()
                    .find(|candidate| candidate.id.0 == function.0)
                    .and_then(|candidate| match &candidate.call_type {
                        severian_mir::CallType::External(call)
                            if call.interface.0 == "native-runtime"
                                && call.symbol.0.contains("_aggregate") =>
                        {
                            Some((call.symbol.0.clone(), candidate.result))
                        }
                        _ => None,
                    })
                {
                    let result_type = self.lower_mir_type(external.1)?;
                    let result =
                        (result_type != LoweredType::Unit).then(|| self.new_value(result_type));
                    operations.push(LirOperation::RuntimeCall {
                        symbol: external.0,
                        arguments: lowered_arguments,
                        result,
                    });
                    if let (Some(result), Some(destination)) = (result, destination) {
                        operations.push(LirOperation::Store {
                            place: self.lower_place(destination),
                            value: result,
                        });
                    }
                    return Ok(severian_lir::Terminator::Goto(severian_lir::BlockId(
                        target.0,
                    )));
                }
                Ok(severian_lir::Terminator::Call {
                    function,
                    arguments: lowered_arguments,
                    destination: destination.as_ref().map(|place| self.lower_place(place)),
                    target: severian_lir::BlockId(target.0),
                })
            }
            severian_mir::Terminator::Spawn {
                callee,
                arguments,
                destination,
                target,
                owner,
                locked,
            } => {
                let function = self.resolve_callee(callee)?;
                let arguments = arguments
                    .iter()
                    .map(|argument| self.lower_operand(body, argument, operations))
                    .collect::<Result<Vec<_>, _>>()?;
                let result_type = self.place_type(body, destination)?;
                let result = self.new_value(result_type);
                operations.push(LirOperation::Spawn {
                    function,
                    arguments,
                    result,
                    owner: match owner {
                        severian_mir::TaskOwner::SelfScope => severian_lir::TaskOwner::SelfScope,
                        severian_mir::TaskOwner::Runtime => severian_lir::TaskOwner::Runtime,
                        severian_mir::TaskOwner::Inferred => severian_lir::TaskOwner::Inferred,
                    },
                    locked: *locked,
                });
                operations.push(LirOperation::Store {
                    place: self.lower_place(destination),
                    value: result,
                });
                Ok(severian_lir::Terminator::Goto(severian_lir::BlockId(
                    target.0,
                )))
            }
            severian_mir::Terminator::SpawnFieldUpdate {
                place,
                operator,
                value,
                destination,
                target,
                owner,
                locked,
            } => {
                let value = self.lower_operand(body, value, operations)?;
                let result = self.new_value(self.place_type(body, destination)?);
                operations.push(LirOperation::SpawnFieldUpdate {
                    place: self.lower_place(place),
                    operator: lower_binary(*operator),
                    value,
                    result,
                    owner: match owner {
                        severian_mir::TaskOwner::SelfScope => severian_lir::TaskOwner::SelfScope,
                        severian_mir::TaskOwner::Runtime => severian_lir::TaskOwner::Runtime,
                        severian_mir::TaskOwner::Inferred => severian_lir::TaskOwner::Inferred,
                    },
                    locked: *locked,
                });
                operations.push(LirOperation::Store {
                    place: self.lower_place(destination),
                    value: result,
                });
                Ok(severian_lir::Terminator::Goto(severian_lir::BlockId(
                    target.0,
                )))
            }
            severian_mir::Terminator::Return(value) => Ok(severian_lir::Terminator::Return(
                value
                    .as_ref()
                    .map(|value| self.lower_operand(body, value, operations))
                    .transpose()?,
            )),
            severian_mir::Terminator::Switch {
                discriminant,
                targets,
                fallback,
            } => {
                let discriminant = self.lower_operand(body, discriminant, operations)?;
                let mut lowered_targets = Vec::new();
                for (case, target) in targets {
                    match case {
                        severian_mir::Case::Type(ty)
                            if self.lower_mir_type(*ty)? == self.value_type(discriminant) =>
                        {
                            return Ok(severian_lir::Terminator::Goto(severian_lir::BlockId(
                                target.0,
                            )));
                        }
                        severian_mir::Case::Type(_) => {}
                        severian_mir::Case::Integer(value) => lowered_targets.push((
                            severian_lir::Case::Integer(*value),
                            severian_lir::BlockId(target.0),
                        )),
                        severian_mir::Case::Boolean(value) => lowered_targets.push((
                            severian_lir::Case::Boolean(*value),
                            severian_lir::BlockId(target.0),
                        )),
                        severian_mir::Case::Variant(value) => lowered_targets.push((
                            severian_lir::Case::Variant(*value),
                            severian_lir::BlockId(target.0),
                        )),
                    }
                }
                Ok(severian_lir::Terminator::Switch {
                    discriminant,
                    targets: lowered_targets,
                    fallback: severian_lir::BlockId(fallback.0),
                })
            }
            severian_mir::Terminator::Throw(value) => {
                let error = self.lower_operand(body, value, operations)?;
                operations.push(severian_lir::Operation::RuntimeCall {
                    symbol: "__sev_throw".into(),
                    arguments: vec![error],
                    result: None,
                });
                Ok(severian_lir::Terminator::Unreachable)
            }
            severian_mir::Terminator::Unreachable => Ok(severian_lir::Terminator::Unreachable),
        }
    }

    fn resolve_callee(&self, callee: &severian_mir::Callee) -> Result<FunctionId, LoweringError> {
        if let severian_mir::Callee::Direct {
            instance: Some(instance),
            ..
        } = callee
        {
            return self
                .mir
                .functions
                .iter()
                .find(|function| function.id == *instance)
                .map(|function| FunctionId(function.id.0))
                .ok_or_else(|| {
                    LoweringError::UnsupportedCfgOperation(format!("callee {callee:?}"))
                });
        }
        let (definition, substitution) = match callee {
            severian_mir::Callee::Direct {
                function,
                substitution,
                ..
            } => (*function, substitution),
            severian_mir::Callee::Method {
                implementation,
                substitution,
                ..
            } => (*implementation, substitution),
            _ => {
                return Err(LoweringError::UnsupportedCfgOperation(format!(
                    "callee {callee:?}"
                )));
            }
        };
        self.mir
            .functions
            .iter()
            .find(|function| {
                function.definition == definition && function.substitution == *substitution
            })
            .map(|function| FunctionId(function.id.0))
            .ok_or_else(|| LoweringError::UnsupportedCfgOperation(format!("callee {callee:?}")))
    }

    fn lower_place(&self, place: &severian_mir::Place) -> severian_lir::Place {
        severian_lir::Place {
            base: match place.base {
                severian_mir::PlaceBase::Local(local) => {
                    severian_lir::PlaceBase::Local(severian_lir::LocalId(local.0))
                }
                severian_mir::PlaceBase::Global(global) => {
                    severian_lir::PlaceBase::Global(severian_lir::GlobalId(global.0))
                }
            },
            projection: place
                .projection
                .iter()
                .map(|projection| match projection {
                    severian_mir::Projection::Field(field) => {
                        severian_lir::Projection::Field(*field)
                    }
                    severian_mir::Projection::Index(local) => {
                        severian_lir::Projection::Index(severian_lir::LocalId(local.0))
                    }
                    severian_mir::Projection::Dereference => severian_lir::Projection::Dereference,
                    severian_mir::Projection::Downcast(variant) => {
                        severian_lir::Projection::Downcast(*variant)
                    }
                })
                .collect(),
        }
    }

    fn place_type(
        &self,
        body: &severian_mir::CfgBody,
        place: &severian_mir::Place,
    ) -> Result<LoweredType, LoweringError> {
        let mut ty = match place.base {
            severian_mir::PlaceBase::Local(local) => {
                body.locals
                    .get(local.0 as usize)
                    .ok_or(LoweringError::UnknownLocal(local.0))?
                    .ty
            }
            severian_mir::PlaceBase::Global(global) => {
                self.mir
                    .globals
                    .get(global.0 as usize)
                    .ok_or(LoweringError::UnknownGlobal(global.0))?
                    .ty
            }
        };
        for projection in &place.projection {
            if let severian_mir::Projection::Field(field) = projection {
                ty = self
                    .mir
                    .classes
                    .iter()
                    .find(|class| class.id == ty)
                    .and_then(|class| class.fields.get(*field as usize))
                    .ok_or(LoweringError::InvalidProjection)?
                    .ty;
            }
        }
        let lowered = self.lower_mir_type(ty)?;
        if place.projection.is_empty()
            && matches!(place.base, severian_mir::PlaceBase::Local(local) if self.task_locals.contains(&local))
        {
            Ok(lowered
                .task()
                .expect("task results cannot themselves be tasks"))
        } else {
            Ok(lowered)
        }
    }

    fn lower_mir_type(&self, type_id: TypeId) -> Result<LoweredType, LoweringError> {
        if severian_universal::is_raw_pointer_type(type_id) {
            return Ok(LoweredType::Bytes);
        }
        if let Some(tensor) = self.types.tensor(type_id) {
            return lower_tensor_type(&tensor, self.types, self.target);
        }
        if let Some(id) = self
            .mir
            .classes
            .iter()
            .position(|class| class.id == type_id)
        {
            Ok(LoweredType::Aggregate(id as u32))
        } else {
            lower_type(type_id, self.types, self.target)
        }
    }

    fn lower_type_named(&self, name: &str) -> Result<LoweredType, LoweringError> {
        let ty = self
            .types
            .resolve_name(name)
            .ok_or_else(|| LoweringError::UnknownTypeName(name.into()))?;
        self.lower_mir_type(ty)
    }

    fn new_value(&mut self, ty: LoweredType) -> ValueId {
        let id = ValueId(self.values.len() as u32);
        self.values.push(Value { id, ty });
        id
    }

    fn value_type(&self, value: ValueId) -> LoweredType {
        self.values[value.0 as usize].ty.clone()
    }
}

fn cfg_uses_gpu(body: &severian_lir::CfgBody) -> bool {
    body.blocks
        .iter()
        .any(|block| block.execution == Some(severian_universal::ExecutionPlacement::Gpu))
}

/// Keeps a placed CFG region on the device only when every block can be
/// represented without calling back into the native host runtime. Placement is
/// a portable request, so an unavailable or incompatible device route executes
/// the complete region on its ordinary host path.
fn retain_supported_gpu_regions(blocks: &mut [severian_lir::BasicBlock], rocm_available: bool) {
    use severian_universal::ExecutionPlacement;

    let gpu_blocks = blocks
        .iter()
        .filter(|block| block.execution == Some(ExecutionPlacement::Gpu))
        .map(|block| block.id)
        .collect::<BTreeSet<_>>();
    if gpu_blocks.is_empty() {
        return;
    }
    if !rocm_available {
        for block in blocks {
            if block.execution == Some(ExecutionPlacement::Gpu) {
                block.execution = None;
            }
        }
        return;
    }

    let mut adjacency =
        std::collections::BTreeMap::<severian_lir::BlockId, BTreeSet<severian_lir::BlockId>>::new();
    for block in blocks.iter() {
        if !gpu_blocks.contains(&block.id) {
            continue;
        }
        for successor in lir_cfg_successors(&block.terminator) {
            if gpu_blocks.contains(&successor) {
                adjacency.entry(block.id).or_default().insert(successor);
                adjacency.entry(successor).or_default().insert(block.id);
            }
        }
    }

    let compatible = blocks
        .iter()
        .map(|block| (block.id, gpu_block_is_device_compatible(block)))
        .collect::<std::collections::BTreeMap<_, _>>();
    let mut visited = BTreeSet::new();
    let mut fallback = BTreeSet::new();
    for seed in &gpu_blocks {
        if visited.contains(seed) {
            continue;
        }
        let mut pending = vec![*seed];
        let mut component = BTreeSet::new();
        let mut supported = true;
        while let Some(id) = pending.pop() {
            if !visited.insert(id) {
                continue;
            }
            component.insert(id);
            supported &= compatible.get(&id).copied().unwrap_or(false);
            pending.extend(
                adjacency
                    .get(&id)
                    .into_iter()
                    .flat_map(|neighbors| neighbors.iter().copied()),
            );
        }
        if !supported {
            fallback.extend(component);
        }
    }

    for block in blocks {
        if fallback.contains(&block.id) {
            block.execution = None;
        }
    }
}

fn gpu_block_is_device_compatible(block: &severian_lir::BasicBlock) -> bool {
    block
        .operations
        .iter()
        .all(gpu_operation_is_device_compatible)
        && matches!(
            block.terminator,
            severian_lir::Terminator::Goto(_) | severian_lir::Terminator::Branch { .. }
        )
}

fn gpu_operation_is_device_compatible(operation: &LirOperation) -> bool {
    matches!(
        operation,
        LirOperation::Constant {
            value: Constant::Integer(_)
                | Constant::Float(_)
                | Constant::Boolean(_)
                | Constant::None
                | Constant::Unit,
            ..
        } | LirOperation::Unary { .. }
            | LirOperation::Convert { .. }
            | LirOperation::Binary { .. }
            | LirOperation::Aggregate { .. }
            | LirOperation::FieldGet { .. }
            | LirOperation::FieldSet { .. }
            | LirOperation::Load { .. }
            | LirOperation::AddressOf { .. }
            | LirOperation::Store { .. }
    )
}

fn lir_cfg_successors(terminator: &severian_lir::Terminator) -> Vec<severian_lir::BlockId> {
    match terminator {
        severian_lir::Terminator::Goto(target) | severian_lir::Terminator::Call { target, .. } => {
            vec![*target]
        }
        severian_lir::Terminator::Branch {
            then_block,
            else_block,
            ..
        } => vec![*then_block, *else_block],
        severian_lir::Terminator::Switch {
            targets, fallback, ..
        } => targets
            .iter()
            .map(|(_, target)| *target)
            .chain(std::iter::once(*fallback))
            .collect(),
        severian_lir::Terminator::Return(_)
        | severian_lir::Terminator::Throw(_)
        | severian_lir::Terminator::Unreachable => Vec::new(),
    }
}

fn cfg_dominators(body: &severian_mir::CfgBody) -> Vec<BTreeSet<severian_mir::BlockId>> {
    let all = body
        .blocks
        .iter()
        .map(|block| block.id)
        .collect::<BTreeSet<_>>();
    let mut predecessors = vec![Vec::new(); body.blocks.len()];
    for block in &body.blocks {
        for successor in cfg_successors(&block.terminator) {
            predecessors[successor.0 as usize].push(block.id);
        }
    }
    let mut dominators = body
        .blocks
        .iter()
        .map(|block| {
            if block.id == body.entry {
                BTreeSet::from([body.entry])
            } else {
                all.clone()
            }
        })
        .collect::<Vec<_>>();
    loop {
        let mut changed = false;
        for block in &body.blocks {
            if block.id == body.entry {
                continue;
            }
            let mut updated =
                if let Some((first, rest)) = predecessors[block.id.0 as usize].split_first() {
                    let mut intersection = dominators[first.0 as usize].clone();
                    for predecessor in rest {
                        intersection = intersection
                            .intersection(&dominators[predecessor.0 as usize])
                            .copied()
                            .collect();
                    }
                    intersection
                } else {
                    BTreeSet::new()
                };
            updated.insert(block.id);
            if updated != dominators[block.id.0 as usize] {
                dominators[block.id.0 as usize] = updated;
                changed = true;
            }
        }
        if !changed {
            return dominators;
        }
    }
}

fn cfg_successors(terminator: &severian_mir::Terminator) -> Vec<severian_mir::BlockId> {
    match terminator {
        severian_mir::Terminator::Goto(target, _)
        | severian_mir::Terminator::Call { target, .. }
        | severian_mir::Terminator::Spawn { target, .. }
        | severian_mir::Terminator::SpawnFieldUpdate { target, .. } => vec![*target],
        severian_mir::Terminator::Branch {
            then_block,
            else_block,
            ..
        } => vec![*then_block, *else_block],
        severian_mir::Terminator::Switch {
            targets, fallback, ..
        } => targets
            .iter()
            .map(|(_, target)| *target)
            .chain(std::iter::once(*fallback))
            .collect(),
        severian_mir::Terminator::Return(_)
        | severian_mir::Terminator::Throw(_)
        | severian_mir::Terminator::Unreachable => Vec::new(),
    }
}

fn task_locals(body: &severian_mir::CfgBody) -> BTreeSet<severian_mir::LocalId> {
    let mut tasks = body
        .blocks
        .iter()
        .filter_map(|block| match &block.terminator {
            severian_mir::Terminator::Spawn { destination, .. }
            | severian_mir::Terminator::SpawnFieldUpdate { destination, .. } => {
                destination.local_id()
            }
            _ => None,
        })
        .collect::<BTreeSet<_>>();
    let mut changed = true;
    while changed {
        changed = false;
        for statement in body.blocks.iter().flat_map(|block| &block.statements) {
            let severian_mir::CfgStatement::Assign(destination, severian_mir::Rvalue::Use(source)) =
                statement
            else {
                continue;
            };
            let (
                Some(destination),
                severian_mir::Operand::Copy(source) | severian_mir::Operand::Move(source),
            ) = (destination.local_id(), source)
            else {
                continue;
            };
            if source
                .local_id()
                .is_some_and(|source| tasks.contains(&source))
            {
                changed |= tasks.insert(destination);
            }
        }
    }
    tasks
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LoweringError {
    NotPrimitive(TypeId),
    UnknownLocal(u32),
    UnknownGlobal(u32),
    UnknownTypeName(String),
    InvalidProjection,
    UnsupportedStringOperation(BinaryOperator),
    UnsupportedCfgOperation(String),
}

impl fmt::Display for LoweringError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{self:?}")
    }
}

impl std::error::Error for LoweringError {}

fn lower_constant(value: &LiteralValue) -> Constant {
    match value {
        LiteralValue::Integer(value) => Constant::Integer(value.clone()),
        LiteralValue::Float(value) => Constant::Float(value.clone()),
        LiteralValue::Boolean(value) => Constant::Boolean(*value),
        LiteralValue::Character(value) => Constant::Integer(u32::from(*value).to_string()),
        LiteralValue::String(value) => Constant::String(value.clone()),
        LiteralValue::Bytes(value) => Constant::Bytes(value.clone()),
        LiteralValue::None => Constant::None,
        LiteralValue::Unit => Constant::Unit,
    }
}

fn lower_unary(operator: UnaryOperator) -> UnaryOperation {
    match operator {
        UnaryOperator::Positive => UnaryOperation::Positive,
        UnaryOperator::Negative => UnaryOperation::Negative,
        UnaryOperator::Not => UnaryOperation::Not,
    }
}

fn lower_binary(operator: BinaryOperator) -> BinaryOperation {
    operator
}

fn lower_type(
    id: TypeId,
    types: &TypeContext,
    target: &TargetSpec,
) -> Result<LoweredType, LoweringError> {
    let primitive = types.primitive(id).ok_or(LoweringError::NotPrimitive(id))?;
    Ok(match primitive.representation {
        PrimitiveRepresentation::Integer { bits, signed } => LoweredType::Integer {
            bits: match bits {
                IntegerWidth::Fixed(bits) => bits,
                IntegerWidth::Machine => target.machine_integer_bits(),
            },
            signed,
        },
        PrimitiveRepresentation::PointerInteger { signed } => LoweredType::Integer {
            bits: target.pointer_bits(),
            signed,
        },
        PrimitiveRepresentation::Float { format } => LoweredType::Float {
            format: match format {
                FloatFormat::Float8E4M3Fn => LoweredFloatFormat::Float8E4M3Fn,
                FloatFormat::Float8E5M2 => LoweredFloatFormat::Float8E5M2,
                FloatFormat::Ieee(bits) => LoweredFloatFormat::Ieee(bits),
                FloatFormat::BrainFloat16 => LoweredFloatFormat::BrainFloat16,
                FloatFormat::Machine => LoweredFloatFormat::Ieee(target.machine_float_bits()),
            },
        },
        PrimitiveRepresentation::Boolean => LoweredType::Boolean,
        PrimitiveRepresentation::Character => LoweredType::Integer {
            bits: 32,
            signed: false,
        },
        PrimitiveRepresentation::String => LoweredType::String,
        PrimitiveRepresentation::Bytes => LoweredType::Bytes,
        PrimitiveRepresentation::None => LoweredType::None,
        PrimitiveRepresentation::Unit => LoweredType::Unit,
        PrimitiveRepresentation::Arguments => LoweredType::Arguments,
    })
}

fn lower_tensor_type(
    tensor: &severian_universal::TensorType,
    types: &TypeContext,
    target: &TargetSpec,
) -> Result<LoweredType, LoweringError> {
    let scalar = lower_type(tensor.element, types, target)?;
    let element = match scalar {
        LoweredType::Integer { bits, signed } => LoweredTensorElement::Integer { bits, signed },
        LoweredType::Float { format } => LoweredTensorElement::Float { format },
        LoweredType::Boolean => LoweredTensorElement::Boolean,
        _ => return Err(LoweringError::NotPrimitive(tensor.element)),
    };
    let shape = match &tensor.shape {
        TensorShape::Unranked => LoweredTensorShape::Unranked,
        TensorShape::Ranked(dimensions) => LoweredTensorShape::Ranked(
            dimensions
                .iter()
                .map(|dimension| match dimension {
                    TensorDimension::Dynamic => LoweredTensorDimension::Dynamic,
                    TensorDimension::Known(value) => LoweredTensorDimension::Known(*value),
                })
                .collect(),
        ),
    };
    Ok(LoweredType::Tensor { element, shape })
}

#[cfg(test)]
mod cfg_tests {
    use super::*;
    use severian_target::{Device, DeviceKind, FeatureSet};
    use severian_universal::{PrimitiveCategory, TypeContextBuilder};

    #[test]
    fn generic_gpu_cfg_blocks_are_selected_by_the_discovered_device() {
        let mut types = TypeContextBuilder::new();
        let unit = types.register_declaration("core.unit", "unit").unwrap();
        types
            .define_primitive(
                unit,
                PrimitiveCategory::Unit,
                PrimitiveRepresentation::Unit,
                true,
            )
            .unwrap();
        let types = types.build();
        let block = |id, execution, terminator| severian_mir::BasicBlock {
            id: severian_mir::BlockId(id),
            execution,
            parameters: Vec::new(),
            statements: Vec::new(),
            statement_spans: Vec::new(),
            terminator,
            terminator_span: None,
        };
        let mir = severian_mir::Module {
            initializer: severian_mir::CfgBody {
                entry: severian_mir::BlockId(0),
                blocks: vec![
                    block(
                        0,
                        None,
                        severian_mir::Terminator::Goto(severian_mir::BlockId(1), Vec::new()),
                    ),
                    block(
                        1,
                        Some(severian_universal::ExecutionPlacement::Gpu),
                        severian_mir::Terminator::Goto(severian_mir::BlockId(2), Vec::new()),
                    ),
                    block(2, None, severian_mir::Terminator::Return(None)),
                ],
                locals: Vec::new(),
                return_type: unit,
            },
            ..severian_mir::Module::default()
        };
        let mut target = TargetSpec::new("x86_64-unknown-linux");
        target.devices.push(Device {
            name: "gpu0".into(),
            kind: DeviceKind::Gpu,
            architecture: "gfx1100".into(),
            features: FeatureSet::from_names(["vendor.amd", "driver.rocm"]),
        });

        let lir = lower(&mir, &types, &target).unwrap();
        assert_eq!(lir.gpu_architecture.as_deref(), Some("gfx1100"));
        assert_eq!(
            lir.initializer_cfg.as_ref().unwrap().blocks[1].execution,
            Some(severian_universal::ExecutionPlacement::Gpu)
        );
    }

    #[test]
    fn gpu_cfg_regions_that_call_the_host_runtime_fall_back_as_a_unit() {
        use severian_lir::{BasicBlock, BlockId, Operation, Terminator};
        use severian_universal::ExecutionPlacement;

        let block = |id, execution, operations: Vec<Operation>, terminator| BasicBlock {
            id: BlockId(id),
            execution,
            operation_spans: vec![None; operations.len()],
            operations,
            terminator,
            terminator_span: None,
        };
        let mut blocks = vec![
            block(0, None, Vec::new(), Terminator::Goto(BlockId(1))),
            block(
                1,
                Some(ExecutionPlacement::Gpu),
                vec![Operation::Constant {
                    value: Constant::Integer("1".into()),
                    result: ValueId(0),
                }],
                Terminator::Goto(BlockId(2)),
            ),
            block(
                2,
                Some(ExecutionPlacement::Gpu),
                vec![Operation::RuntimeCall {
                    symbol: "__sev_list_len".into(),
                    arguments: vec![ValueId(1)],
                    result: Some(ValueId(2)),
                }],
                Terminator::Goto(BlockId(3)),
            ),
            block(3, None, Vec::new(), Terminator::Return(None)),
        ];

        retain_supported_gpu_regions(&mut blocks, true);

        assert_eq!(blocks[1].execution, None);
        assert_eq!(blocks[2].execution, None);
        assert!(!cfg_uses_gpu(&severian_lir::CfgBody {
            entry: BlockId(0),
            blocks,
            locals: Vec::new(),
            return_type: LoweredType::Unit,
        }));
    }

    #[test]
    fn uncaught_throws_terminate_through_the_runtime() {
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
        let types = types.build();
        let mir = severian_mir::Module {
            initializer: severian_mir::CfgBody {
                entry: severian_mir::BlockId(0),
                blocks: vec![severian_mir::BasicBlock {
                    id: severian_mir::BlockId(0),
                    execution: None,
                    parameters: Vec::new(),
                    statements: Vec::new(),
                    statement_spans: Vec::new(),
                    terminator: severian_mir::Terminator::Throw(severian_mir::Operand::Constant {
                        value: LiteralValue::String("expected failure".into()),
                        ty: string,
                    }),
                    terminator_span: None,
                }],
                locals: Vec::new(),
                return_type: string,
            },
            ..severian_mir::Module::default()
        };
        let lir = lower(&mir, &types, &TargetSpec::host()).unwrap();
        let block = &lir.initializer_cfg.as_ref().unwrap().blocks[0];
        assert!(matches!(
            block.operations.last(),
            Some(LirOperation::RuntimeCall {
                symbol,
                result: None,
                ..
            }) if symbol == "__sev_throw"
        ));
        assert_eq!(block.terminator, severian_lir::Terminator::Unreachable);
    }
}

#[cfg(any())]
mod legacy_structured_lowering {
    use super::*;
    use severian_mir::{Block as MirBlock, Operation as MirOperation};

    pub fn lower(
        mir: &MirModule,
        types: &TypeContext,
        target: &TargetSpec,
    ) -> Result<LirModule, LoweringError> {
        let mut values = mir
            .values
            .iter()
            .map(|value| {
                Ok(Value {
                    id: ValueId(value.id.0),
                    ty: lower_mir_type(value.type_id, mir, types, target)?,
                })
            })
            .collect::<Result<Vec<_>, LoweringError>>()?;
        let initializer = lower_block(&mir.initializer, mir, types, target, &mut values)?;
        let mut functions = Vec::new();
        for function in &mir.functions {
            functions.push(LirFunction {
                id: FunctionId(function.id.0),
                name: function.name.clone(),
                parameters: function
                    .parameters
                    .iter()
                    .map(|value| ValueId(value.0))
                    .collect(),
                result: lower_mir_type(function.result, mir, types, target)?,
                body: function
                    .body
                    .as_ref()
                    .map(|body| lower_block(body, mir, types, target, &mut values))
                    .transpose()?,
                linkage: match &function.call_type {
                    severian_mir::CallType::Severian => FunctionLinkage::Internal,
                    severian_mir::CallType::External(call) => FunctionLinkage::External {
                        symbol: call.symbol.0.clone(),
                    },
                },
            });
        }
        Ok(LirModule {
            values,
            globals: mir.globals.iter().map(|value| ValueId(value.0)).collect(),
            initializer,
            functions,
            entry: mir.entry.map(|entry| FunctionId(entry.0)),
            traits: mir
                .traits
                .iter()
                .map(|declaration| {
                    Ok(severian_lir::TraitDeclaration {
                        id: severian_lir::TraitId {
                            package: declaration.definition.package,
                            module: declaration.definition.module,
                            declaration: declaration.definition.declaration.0,
                        },
                        name: declaration.name.clone(),
                        methods: declaration
                            .methods
                            .iter()
                            .map(|method| {
                                Ok(severian_lir::TraitMethodDeclaration {
                                    name: method.name.clone(),
                                    parameters: method
                                        .parameters
                                        .iter()
                                        .map(|parameter| match parameter {
                                            severian_mir::TraitType::SelfType => {
                                                Ok(severian_lir::TraitType::SelfType)
                                            }
                                            severian_mir::TraitType::Concrete(ty) => {
                                                lower_type(*ty, types, target)
                                                    .map(severian_lir::TraitType::Concrete)
                                            }
                                            severian_mir::TraitType::Symbolic(name) => {
                                                Ok(severian_lir::TraitType::Symbolic(name.clone()))
                                            }
                                        })
                                        .collect::<Result<Vec<_>, _>>()?,
                                    result: match &method.result {
                                        severian_mir::TraitType::SelfType => {
                                            severian_lir::TraitType::SelfType
                                        }
                                        severian_mir::TraitType::Concrete(ty) => {
                                            severian_lir::TraitType::Concrete(lower_type(
                                                *ty, types, target,
                                            )?)
                                        }
                                        severian_mir::TraitType::Symbolic(name) => {
                                            severian_lir::TraitType::Symbolic(name.clone())
                                        }
                                    },
                                })
                            })
                            .collect::<Result<Vec<_>, LoweringError>>()?,
                    })
                })
                .collect::<Result<Vec<_>, LoweringError>>()?,
            classes: mir
                .classes
                .iter()
                .enumerate()
                .map(|(id, declaration)| {
                    Ok(severian_lir::ClassDeclaration {
                        id: id as u32,
                        name: declaration.name.clone(),
                        fields: declaration
                            .fields
                            .iter()
                            .map(|field| {
                                Ok(severian_lir::ClassFieldDeclaration {
                                    name: field.name.clone(),
                                    ty: lower_mir_type(field.ty, mir, types, target)?,
                                })
                            })
                            .collect::<Result<Vec<_>, LoweringError>>()?,
                    })
                })
                .collect::<Result<Vec<_>, LoweringError>>()?,
            gpu_architecture: target
                .rocm_device()
                .map(|device| device.architecture.clone()),
        })
    }

    fn lower_block(
        block: &MirBlock,
        module: &MirModule,
        types: &TypeContext,
        target: &TargetSpec,
        values: &mut Vec<Value>,
    ) -> Result<LirBlock, LoweringError> {
        let mut operations = Vec::new();
        let mut owned_strings = Vec::new();
        let mut scoped_tasks = Vec::new();
        for operation in &block.operations {
            let operation = match operation {
                MirOperation::Coverage { point } => LirOperation::Coverage {
                    key: point
                        .key
                        .clone()
                        .expect("source attachment assigns every coverage key"),
                },
                MirOperation::Constant { value, result } => LirOperation::Constant {
                    value: lower_constant(value),
                    result: ValueId(result.0),
                },
                MirOperation::Unary {
                    operator,
                    operand,
                    result,
                } => LirOperation::Unary {
                    operator: lower_unary(*operator),
                    operand: ValueId(operand.0),
                    result: ValueId(result.0),
                },
                MirOperation::Binary {
                    operator,
                    left,
                    right,
                    result,
                } => {
                    let left_type = mir_value_type(module, *left)?;
                    if lower_type(left_type, types, target)? == LoweredType::String {
                        match operator {
                            BinaryOperator::Add => {
                                let result = ValueId(result.0);
                                owned_strings.push(result);
                                LirOperation::RuntimeCall {
                                    symbol: "__sev_string_concat".into(),
                                    arguments: vec![ValueId(left.0), ValueId(right.0)],
                                    result: Some(result),
                                }
                            }
                            BinaryOperator::Equal
                            | BinaryOperator::NotEqual
                            | BinaryOperator::Less
                            | BinaryOperator::LessEqual
                            | BinaryOperator::Greater
                            | BinaryOperator::GreaterEqual => {
                                let comparison = new_value(
                                    values,
                                    LoweredType::Integer {
                                        bits: 32,
                                        signed: true,
                                    },
                                );
                                let zero = new_value(
                                    values,
                                    LoweredType::Integer {
                                        bits: 32,
                                        signed: true,
                                    },
                                );
                                operations.push(LirOperation::RuntimeCall {
                                    symbol: "__sev_string_compare".into(),
                                    arguments: vec![ValueId(left.0), ValueId(right.0)],
                                    result: Some(comparison),
                                });
                                operations.push(LirOperation::Constant {
                                    value: Constant::Integer("0".into()),
                                    result: zero,
                                });
                                LirOperation::Binary {
                                    operator: lower_binary(*operator),
                                    left: comparison,
                                    right: zero,
                                    result: ValueId(result.0),
                                }
                            }
                            _ => {
                                return Err(LoweringError::UnsupportedStringOperation(*operator));
                            }
                        }
                    } else {
                        LirOperation::Binary {
                            operator: lower_binary(*operator),
                            left: ValueId(left.0),
                            right: ValueId(right.0),
                            result: ValueId(result.0),
                        }
                    }
                }
                MirOperation::Aggregate {
                    class,
                    fields,
                    result,
                } => LirOperation::Aggregate {
                    class: module
                        .classes
                        .iter()
                        .position(|known| known.id == *class)
                        .ok_or(LoweringError::NotPrimitive(*class))?
                        as u32,
                    fields: fields.iter().map(|value| ValueId(value.0)).collect(),
                    result: ValueId(result.0),
                },
                MirOperation::FieldGet {
                    object,
                    field,
                    result,
                } => LirOperation::FieldGet {
                    object: ValueId(object.0),
                    field: *field,
                    result: ValueId(result.0),
                },
                MirOperation::FieldSet {
                    object,
                    field,
                    value,
                    result,
                } => LirOperation::FieldSet {
                    object: ValueId(object.0),
                    field: *field,
                    value: ValueId(value.0),
                    result: ValueId(result.0),
                },
                MirOperation::Assign { target, value } => LirOperation::Assign {
                    target: ValueId(target.0),
                    value: ValueId(value.0),
                },
                MirOperation::Call {
                    function,
                    arguments,
                    result,
                } => LirOperation::Call {
                    function: FunctionId(function.0),
                    arguments: arguments.iter().map(|value| ValueId(value.0)).collect(),
                    result: ValueId(result.0),
                },
                MirOperation::Spawn {
                    function,
                    arguments,
                    result,
                    owner,
                    locked,
                } => LirOperation::Spawn {
                    function: FunctionId(function.0),
                    arguments: arguments.iter().map(|value| ValueId(value.0)).collect(),
                    result: ValueId(result.0),
                    owner: match owner {
                        severian_mir::TaskOwner::SelfScope => severian_lir::TaskOwner::SelfScope,
                        severian_mir::TaskOwner::Runtime => severian_lir::TaskOwner::Runtime,
                        severian_mir::TaskOwner::Inferred => severian_lir::TaskOwner::Inferred,
                    },
                    locked: *locked,
                },
                MirOperation::Await { task, result } => LirOperation::Await {
                    task: ValueId(task.0),
                    result: ValueId(result.0),
                },
                MirOperation::Return { value } => LirOperation::Return {
                    value: value.map(|value| ValueId(value.0)),
                },
                MirOperation::Assert {
                    condition,
                    message,
                    origin,
                } => LirOperation::Assert {
                    condition: ValueId(condition.0),
                    message: message.map(|message| ValueId(message.0)),
                    location: origin.location.as_ref().map(|location| {
                        severian_lir::AssertionLocation {
                            file: location.file.clone(),
                            line: location.line,
                            column: location.column,
                            expression: location.expression.clone(),
                        }
                    }),
                },
                MirOperation::If {
                    condition,
                    then_block,
                    else_block,
                } => LirOperation::If {
                    condition: ValueId(condition.0),
                    then_block: lower_block(then_block, module, types, target, values)?,
                    else_block: lower_block(else_block, module, types, target, values)?,
                },
                MirOperation::While {
                    condition_block,
                    condition,
                    body,
                } => LirOperation::While {
                    condition_block: lower_block(condition_block, module, types, target, values)?,
                    condition: ValueId(condition.0),
                    body: lower_block(body, module, types, target, values)?,
                },
                MirOperation::Break => LirOperation::Break,
                MirOperation::Continue => LirOperation::Continue,
                MirOperation::Match { subject, arms } => {
                    let subject_type = module
                        .values
                        .iter()
                        .find(|value| value.id == *subject)
                        .ok_or(LoweringError::UnknownValue(*subject))?
                        .type_id;
                    let arm = arms
                        .iter()
                        .find(|arm| arm.type_id == Some(subject_type))
                        .or_else(|| arms.iter().find(|arm| arm.type_id.is_none()))
                        .ok_or(LoweringError::NonExhaustiveMatch(subject_type))?;
                    operations
                        .extend(lower_block(&arm.body, module, types, target, values)?.operations);
                    continue;
                }
                MirOperation::CompiledRegionCall {
                    artifact,
                    inputs,
                    outputs,
                } => LirOperation::ArtifactCall {
                    artifact: *artifact,
                    inputs: inputs.iter().map(|value| ValueId(value.0)).collect(),
                    outputs: outputs.iter().map(|value| ValueId(value.0)).collect(),
                },
            };
            match &operation {
                LirOperation::Spawn { result, owner, .. }
                    if *owner != severian_lir::TaskOwner::Runtime =>
                {
                    scoped_tasks.push(*result);
                }
                LirOperation::Await { task, .. } => {
                    scoped_tasks.retain(|known| known != task);
                }
                LirOperation::Return { .. } => {
                    for task in scoped_tasks.drain(..) {
                        let ty = values
                            .get(task.0 as usize)
                            .expect("task result references a lowered value")
                            .ty;
                        let result = new_value(values, ty);
                        operations.push(LirOperation::Await { task, result });
                    }
                }
                _ => {}
            }
            operations.push(operation);
        }
        for task in scoped_tasks {
            let ty = values
                .get(task.0 as usize)
                .expect("task result references a lowered value")
                .ty;
            let result = new_value(values, ty);
            operations.push(LirOperation::Await { task, result });
        }
        insert_owned_string_releases(&mut operations, &owned_strings, module)?;
        Ok(LirBlock { operations })
    }

    fn mir_value_type(
        module: &MirModule,
        value: severian_mir::ValueId,
    ) -> Result<TypeId, LoweringError> {
        module
            .values
            .iter()
            .find(|known| known.id == value)
            .map(|known| known.type_id)
            .ok_or(LoweringError::UnknownValue(value))
    }

    fn lower_mir_type(
        type_id: TypeId,
        module: &MirModule,
        types: &TypeContext,
        target: &TargetSpec,
    ) -> Result<LoweredType, LoweringError> {
        if let Some(tensor) = types.tensor(type_id) {
            lower_tensor_type(&tensor, types, target)
        } else if let Some(id) = module.classes.iter().position(|class| class.id == type_id) {
            Ok(LoweredType::Aggregate(id as u32))
        } else {
            lower_type(type_id, types, target)
        }
    }

    fn new_value(values: &mut Vec<Value>, ty: LoweredType) -> ValueId {
        let id = ValueId(values.len() as u32);
        values.push(Value { id, ty });
        id
    }

    fn insert_owned_string_releases(
        operations: &mut Vec<LirOperation>,
        owned: &[ValueId],
        module: &MirModule,
    ) -> Result<(), LoweringError> {
        let mut releases = Vec::new();
        for value in owned {
            if module.globals.iter().any(|global| global.0 == value.0)
                || operations
                    .iter()
                    .any(|operation| returns_value(operation, *value))
            {
                return Err(LoweringError::OwnedStringEscapes(*value));
            }
            // Storing the allocation in mutable or aggregate storage transfers its
            // lifetime to that storage. Releasing it after the store leaves a
            // dangling value for subsequent reads. Storage destruction will
            // become the corresponding release point once destructors are
            // represented in LIR; until then, retain the allocation.
            if operations.iter().any(|operation| match operation {
                LirOperation::Aggregate { fields, .. } => fields.contains(value),
                LirOperation::FieldSet {
                    value: field_value, ..
                } => field_value == value,
                LirOperation::Assign {
                    value: assigned, ..
                } => assigned == value,
                _ => false,
            }) {
                continue;
            }
            let definition = operations
            .iter()
            .position(|operation| {
                matches!(operation, LirOperation::RuntimeCall { result: Some(result), .. } if result == value)
            })
            .expect("every owned string is produced by a runtime call");
            let last_use = operations
                .iter()
                .enumerate()
                .filter(|(_, operation)| operation_uses_value(operation, *value))
                .map(|(index, _)| index)
                .max()
                .unwrap_or(definition);
            releases.push((last_use + 1, *value));
        }
        releases.sort_by_key(|(index, _)| std::cmp::Reverse(*index));
        for (index, value) in releases {
            operations.insert(
                index,
                LirOperation::RuntimeCall {
                    symbol: "__sev_string_release".into(),
                    arguments: vec![value],
                    result: None,
                },
            );
        }
        Ok(())
    }

    fn operation_uses_value(operation: &LirOperation, value: ValueId) -> bool {
        match operation {
            LirOperation::Aggregate { fields, .. } => fields.contains(&value),
            LirOperation::FieldGet { object, .. } => *object == value,
            LirOperation::FieldSet {
                object,
                value: field_value,
                ..
            } => *object == value || *field_value == value,
            LirOperation::Assign {
                target,
                value: assigned,
            } => *target == value || *assigned == value,
            LirOperation::Unary { operand, .. } => *operand == value,
            LirOperation::Convert { operand, .. } => *operand == value,
            LirOperation::Binary { left, right, .. } => *left == value || *right == value,
            LirOperation::Call { arguments, .. } | LirOperation::RuntimeCall { arguments, .. } => {
                arguments.contains(&value)
            }
            LirOperation::Spawn { arguments, .. } => arguments.contains(&value),
            LirOperation::SpawnFieldUpdate { value: update, .. } => *update == value,
            LirOperation::Await { task, .. } => *task == value,
            LirOperation::Return { value: returned } => *returned == Some(value),
            LirOperation::Assert {
                condition, message, ..
            } => *condition == value || *message == Some(value),
            LirOperation::If {
                condition,
                then_block,
                else_block,
            } => {
                *condition == value
                    || then_block
                        .operations
                        .iter()
                        .any(|operation| operation_uses_value(operation, value))
                    || else_block
                        .operations
                        .iter()
                        .any(|operation| operation_uses_value(operation, value))
            }
            LirOperation::While {
                condition_block,
                condition,
                body,
            } => {
                *condition == value
                    || condition_block
                        .operations
                        .iter()
                        .any(|operation| operation_uses_value(operation, value))
                    || body
                        .operations
                        .iter()
                        .any(|operation| operation_uses_value(operation, value))
            }
            LirOperation::ArtifactCall {
                inputs, outputs, ..
            } => inputs.contains(&value) || outputs.contains(&value),
            LirOperation::Coverage { .. }
            | LirOperation::Constant { .. }
            | LirOperation::Break
            | LirOperation::Continue => false,
        }
    }

    fn returns_value(operation: &LirOperation, value: ValueId) -> bool {
        match operation {
            LirOperation::Return {
                value: Some(returned),
            } => *returned == value,
            LirOperation::If {
                then_block,
                else_block,
                ..
            } => then_block
                .operations
                .iter()
                .chain(&else_block.operations)
                .any(|operation| returns_value(operation, value)),
            LirOperation::While {
                condition_block,
                body,
                ..
            } => condition_block
                .operations
                .iter()
                .chain(&body.operations)
                .any(|operation| returns_value(operation, value)),
            _ => false,
        }
    }

    fn lower_constant(value: &LiteralValue) -> Constant {
        match value {
            LiteralValue::Integer(value) => Constant::Integer(value.clone()),
            LiteralValue::Float(value) => Constant::Float(value.clone()),
            LiteralValue::Boolean(value) => Constant::Boolean(*value),
            LiteralValue::Character(value) => Constant::Integer(u32::from(*value).to_string()),
            LiteralValue::String(value) => Constant::String(value.clone()),
            LiteralValue::Bytes(value) => Constant::Bytes(value.clone()),
            LiteralValue::None => Constant::None,
            LiteralValue::Unit => Constant::Unit,
        }
    }

    fn lower_unary(operator: UnaryOperator) -> UnaryOperation {
        match operator {
            UnaryOperator::Positive => UnaryOperation::Positive,
            UnaryOperator::Negative => UnaryOperation::Negative,
            UnaryOperator::Not => UnaryOperation::Not,
        }
    }

    fn lower_binary(operator: BinaryOperator) -> BinaryOperation {
        operator
    }

    fn lower_type(
        id: TypeId,
        types: &TypeContext,
        target: &TargetSpec,
    ) -> Result<LoweredType, LoweringError> {
        let primitive = types.primitive(id).ok_or(LoweringError::NotPrimitive(id))?;
        Ok(match primitive.representation {
            PrimitiveRepresentation::Integer { bits, signed } => LoweredType::Integer {
                bits: match bits {
                    IntegerWidth::Fixed(bits) => bits,
                    IntegerWidth::Machine => target.machine_integer_bits(),
                },
                signed,
            },
            PrimitiveRepresentation::PointerInteger { signed } => LoweredType::Integer {
                bits: target.pointer_bits(),
                signed,
            },
            PrimitiveRepresentation::Float { format } => LoweredType::Float {
                format: match format {
                    FloatFormat::Float8E4M3Fn => LoweredFloatFormat::Float8E4M3Fn,
                    FloatFormat::Float8E5M2 => LoweredFloatFormat::Float8E5M2,
                    FloatFormat::Ieee(bits) => LoweredFloatFormat::Ieee(bits),
                    FloatFormat::BrainFloat16 => LoweredFloatFormat::BrainFloat16,
                    FloatFormat::Machine => LoweredFloatFormat::Ieee(target.machine_float_bits()),
                },
            },
            PrimitiveRepresentation::Boolean => LoweredType::Boolean,
            PrimitiveRepresentation::Character => LoweredType::Integer {
                bits: 32,
                signed: false,
            },
            PrimitiveRepresentation::String => LoweredType::String,
            PrimitiveRepresentation::Bytes => LoweredType::Bytes,
            PrimitiveRepresentation::None => LoweredType::None,
            PrimitiveRepresentation::Unit => LoweredType::Unit,
            PrimitiveRepresentation::Arguments => LoweredType::Arguments,
        })
    }

    #[derive(Debug, Clone, PartialEq, Eq)]
    pub enum LoweringError {
        NotPrimitive(TypeId),
        UnknownValue(severian_mir::ValueId),
        NonExhaustiveMatch(TypeId),
        UnsupportedStringOperation(BinaryOperator),
        OwnedStringEscapes(ValueId),
    }

    impl fmt::Display for LoweringError {
        fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
            write!(formatter, "{self:?}")
        }
    }

    impl std::error::Error for LoweringError {}

    #[cfg(test)]
    mod tests {
        use super::*;
        use severian_mir::{Module, Value as MirValue, ValueId as MirValueId};
        use severian_universal::{PrimitiveCategory, TypeContextBuilder, UniversalContext};

        fn pointer_context() -> (UniversalContext, TypeId) {
            let mut types = TypeContextBuilder::new();
            let id = types.register_declaration("core.usize", "usize").unwrap();
            types
                .define_primitive(
                    id,
                    PrimitiveCategory::Integer,
                    PrimitiveRepresentation::PointerInteger { signed: false },
                    false,
                )
                .unwrap();
            (UniversalContext::new(types.build()), id)
        }

        fn string_context() -> (UniversalContext, TypeId, TypeId) {
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
            let boolean = types.register_declaration("core.bool", "bool").unwrap();
            types
                .define_primitive(
                    boolean,
                    PrimitiveCategory::Boolean,
                    PrimitiveRepresentation::Boolean,
                    true,
                )
                .unwrap();
            (UniversalContext::new(types.build()), string, boolean)
        }

        #[test]
        fn usize_is_resolved_only_from_target_layout() {
            for bits in [32, 64] {
                let (context, type_id) = pointer_context();
                let mir = Module {
                    values: vec![MirValue {
                        id: MirValueId(0),
                        type_id,
                    }],
                    ..Module::default()
                };
                assert_eq!(
                    lower(
                        &mir,
                        &context.types,
                        &if bits == 32 {
                            TargetSpec::new("wasm32-unknown-wasi")
                        } else {
                            TargetSpec::new("x86_64-unknown-linux")
                        },
                    )
                    .unwrap()
                    .values[0]
                        .ty,
                    LoweredType::Integer {
                        bits,
                        signed: false,
                    }
                );
            }
        }

        #[test]
        fn string_operations_become_owned_runtime_calls_before_emission() {
            let (context, string, boolean) = string_context();
            let mir = Module {
                values: vec![
                    MirValue {
                        id: MirValueId(0),
                        type_id: string,
                    },
                    MirValue {
                        id: MirValueId(1),
                        type_id: string,
                    },
                    MirValue {
                        id: MirValueId(2),
                        type_id: string,
                    },
                    MirValue {
                        id: MirValueId(3),
                        type_id: boolean,
                    },
                ],
                initializer: MirBlock {
                    operations: vec![
                        MirOperation::Constant {
                            value: LiteralValue::String("left".into()),
                            result: MirValueId(0),
                        },
                        MirOperation::Constant {
                            value: LiteralValue::String("right".into()),
                            result: MirValueId(1),
                        },
                        MirOperation::Binary {
                            operator: BinaryOperator::Add,
                            left: MirValueId(0),
                            right: MirValueId(1),
                            result: MirValueId(2),
                        },
                        MirOperation::Binary {
                            operator: BinaryOperator::Equal,
                            left: MirValueId(2),
                            right: MirValueId(1),
                            result: MirValueId(3),
                        },
                    ],
                },
                ..Module::default()
            };
            let lir = lower(&mir, &context.types, &TargetSpec::host()).unwrap();
            let symbols = lir
                .initializer
                .operations
                .iter()
                .filter_map(|operation| match operation {
                    LirOperation::RuntimeCall { symbol, .. } => Some(symbol.as_str()),
                    _ => None,
                })
                .collect::<Vec<_>>();
            assert_eq!(
                symbols,
                [
                    "__sev_string_concat",
                    "__sev_string_compare",
                    "__sev_string_release"
                ]
            );
        }

        #[test]
        fn owned_string_escape_fails_closed_until_transfer_is_modeled() {
            let (context, string, _) = string_context();
            let mir = Module {
                values: (0..3)
                    .map(|id| MirValue {
                        id: MirValueId(id),
                        type_id: string,
                    })
                    .collect(),
                initializer: MirBlock {
                    operations: vec![
                        MirOperation::Constant {
                            value: LiteralValue::String("left".into()),
                            result: MirValueId(0),
                        },
                        MirOperation::Constant {
                            value: LiteralValue::String("right".into()),
                            result: MirValueId(1),
                        },
                        MirOperation::Binary {
                            operator: BinaryOperator::Add,
                            left: MirValueId(0),
                            right: MirValueId(1),
                            result: MirValueId(2),
                        },
                        MirOperation::Return {
                            value: Some(MirValueId(2)),
                        },
                    ],
                },
                ..Module::default()
            };
            assert!(matches!(
                lower(&mir, &context.types, &TargetSpec::host()),
                Err(LoweringError::OwnedStringEscapes(ValueId(2)))
            ));
        }

        #[test]
        fn owned_string_assignment_does_not_release_the_stored_allocation() {
            let (context, string, _) = string_context();
            let mir = Module {
                values: (0..3)
                    .map(|id| MirValue {
                        id: MirValueId(id),
                        type_id: string,
                    })
                    .collect(),
                initializer: MirBlock {
                    operations: vec![
                        MirOperation::Constant {
                            value: LiteralValue::String("left".into()),
                            result: MirValueId(0),
                        },
                        MirOperation::Constant {
                            value: LiteralValue::String("right".into()),
                            result: MirValueId(1),
                        },
                        MirOperation::Binary {
                            operator: BinaryOperator::Add,
                            left: MirValueId(0),
                            right: MirValueId(1),
                            result: MirValueId(2),
                        },
                        MirOperation::Assign {
                            target: MirValueId(0),
                            value: MirValueId(2),
                        },
                    ],
                },
                ..Module::default()
            };
            let lir = lower(&mir, &context.types, &TargetSpec::host()).unwrap();
            assert!(lir.initializer.operations.iter().any(|operation| matches!(
                operation,
                LirOperation::Assign {
                    target: ValueId(0),
                    value: ValueId(2)
                }
            )));
            assert!(!lir.initializer.operations.iter().any(|operation| matches!(
            operation,
            LirOperation::RuntimeCall { symbol, .. } if symbol == "__sev_string_release"
            )));
        }
    }
}
