use crate::*;
use severian_hir::{BindingId, BindingRef, Expression, Instruction, ValueType};
use std::collections::BTreeMap;

pub fn lower(hir: &severian_hir::Program) -> Result<Program, MirLoweringError> {
    let mut functions = hir
        .functions
        .iter()
        .map(|function| lower_function(function, function.name.clone()))
        .collect::<Result<Vec<_>, _>>()?;
    for class in &hir.classes {
        for function in class.constructors.iter().chain(&class.methods) {
            functions.push(lower_function(
                function,
                format!("{}.{}", class.name, function.name),
            )?);
        }
    }
    Ok(Program {
        hir: hir.clone(),
        functions,
    })
}

fn lower_function(
    function: &severian_hir::Function,
    name: String,
) -> Result<Function, MirLoweringError> {
    let mut builder = FunctionBuilder::default();
    let parameters = function
        .params
        .iter()
        .map(|parameter| builder.reserve_local(parameter.name.clone(), parameter.ty))
        .collect();
    let entry = builder.reserve_block();
    builder
        .lower_block(entry, &function.instructions, None)
        .map_err(|error| error.in_function(name.clone()))?;
    Ok(Function {
        id: function.id,
        name,
        native_symbol: function.native_symbol.clone(),
        decorators: function.decorators.clone(),
        parameters,
        locals: builder.locals,
        return_type: function.return_type,
        source_tensor_intrinsics: builder.source_tensor_intrinsics,
        tensor_operations: builder.tensor_operations,
        blocks: builder.blocks,
    })
}

#[derive(Default)]
struct FunctionBuilder {
    blocks: Vec<BasicBlock>,
    locals: Vec<Local>,
    bindings: BTreeMap<BindingId, LocalId>,
    source_tensor_intrinsics: usize,
    tensor_operations: Vec<TensorOp>,
}

impl FunctionBuilder {
    fn reserve_block(&mut self) -> BlockId {
        let id = BlockId(self.blocks.len() as u32);
        self.blocks.push(BasicBlock {
            id,
            operations: Vec::new(),
            terminator: Terminator::Unreachable,
        });
        id
    }

    fn reserve_local(&mut self, binding: BindingRef, ty: ValueType) -> LocalId {
        if let Some(local) = self.bindings.get(&binding.id) {
            return *local;
        }
        let id = LocalId(self.locals.len() as u32);
        self.bindings.insert(binding.id, id);
        self.locals.push(Local { id, binding, ty });
        id
    }

    fn lower_block(
        &mut self,
        block: BlockId,
        instructions: &[Instruction],
        fallthrough: Option<BlockId>,
    ) -> Result<(), MirLoweringError> {
        for (index, instruction) in instructions.iter().enumerate() {
            let rest = &instructions[index + 1..];
            match instruction {
                Instruction::Let { name, value } => {
                    let local =
                        self.reserve_local(name.clone(), value.ty().unwrap_or(ValueType::Any));
                    let value = self.value_ref(value)?;
                    self.operation(block, OperationKind::Bind(local), [value])
                }
                Instruction::TryLet { name, value, .. } => {
                    let local = self.reserve_local(name.clone(), ValueType::Any);
                    let value = self.value_ref(value)?;
                    self.operation(block, OperationKind::TryBind(local), [value])
                }
                Instruction::Assign { target, value, .. } => {
                    let target = self.value_ref(target)?;
                    let value = self.value_ref(value)?;
                    self.operation(block, OperationKind::Assign, [target, value])
                }
                Instruction::Print(value) => {
                    let value = self.value_ref(value)?;
                    self.operation(block, OperationKind::Print, [value])
                }
                Instruction::Assert(value) => {
                    let value = self.value_ref(value)?;
                    self.operation(block, OperationKind::Assert, [value])
                }
                Instruction::Evaluate(value) => {
                    let value = self.value_ref(value)?;
                    self.operation(block, OperationKind::Evaluate, [value])
                }
                Instruction::Return(value) => {
                    let value = value
                        .as_ref()
                        .map(|value| self.value_ref(value))
                        .transpose()?;
                    self.blocks[block.0 as usize].terminator = Terminator::Return(value);
                    return Ok(());
                }
                Instruction::If {
                    condition,
                    then_instructions,
                    else_instructions,
                } => {
                    let then_block = self.reserve_block();
                    let else_block = self.reserve_block();
                    let join = self.reserve_block();
                    let condition = self.value_ref(condition)?;
                    self.blocks[block.0 as usize].terminator = Terminator::Branch {
                        condition,
                        then_block,
                        else_block,
                    };
                    self.lower_block(then_block, then_instructions, Some(join))?;
                    self.lower_block(else_block, else_instructions, Some(join))?;
                    self.lower_block(join, rest, fallthrough)?;
                    return Ok(());
                }
                Instruction::While {
                    condition,
                    instructions,
                    ..
                } => {
                    let header = self.reserve_block();
                    let body = self.reserve_block();
                    let exit = self.reserve_block();
                    self.blocks[block.0 as usize].terminator = Terminator::Goto(header);
                    let condition = self.value_ref(condition)?;
                    self.blocks[header.0 as usize].terminator = Terminator::Loop {
                        condition,
                        body,
                        exit,
                    };
                    self.lower_block(body, instructions, Some(header))?;
                    self.lower_block(exit, rest, fallthrough)?;
                    return Ok(());
                }
                Instruction::For {
                    pattern,
                    iterable,
                    instructions,
                    ..
                } => {
                    let body = self.reserve_block();
                    let exit = self.reserve_block();
                    let iterable = self.value_ref(iterable)?;
                    self.blocks[block.0 as usize].terminator = Terminator::For {
                        pattern: pattern.clone(),
                        iterable,
                        body,
                        exit,
                    };
                    self.lower_block(body, instructions, Some(block))?;
                    self.lower_block(exit, rest, fallthrough)?;
                    return Ok(());
                }
                Instruction::Switch { value, arms } => {
                    let exit = self.reserve_block();
                    let mut arm_blocks = Vec::with_capacity(arms.len());
                    for arm in arms {
                        let arm_block = self.reserve_block();
                        self.lower_block(arm_block, &arm.instructions, Some(exit))?;
                        arm_blocks.push(arm_block);
                    }
                    let value = self.value_ref(value)?;
                    self.blocks[block.0 as usize].terminator = Terminator::Switch {
                        values: vec![value],
                        arms: arm_blocks,
                        exit,
                    };
                    self.lower_block(exit, rest, fallthrough)?;
                    return Ok(());
                }
                Instruction::ChannelSwitch { channels, arms, .. } => {
                    let exit = self.reserve_block();
                    let mut arm_blocks = Vec::with_capacity(arms.len());
                    for arm in arms {
                        let arm_block = self.reserve_block();
                        self.lower_block(arm_block, &arm.instructions, Some(exit))?;
                        arm_blocks.push(arm_block);
                    }
                    let values = channels
                        .iter()
                        .map(|value| self.value_ref(value))
                        .collect::<Result<Vec<_>, _>>()?;
                    self.blocks[block.0 as usize].terminator = Terminator::Switch {
                        values,
                        arms: arm_blocks,
                        exit,
                    };
                    self.lower_block(exit, rest, fallthrough)?;
                    return Ok(());
                }
                Instruction::With {
                    resources,
                    instructions,
                    ..
                } => {
                    let resources = resources
                        .iter()
                        .map(|value| self.value_ref(value))
                        .collect::<Result<Vec<_>, _>>()?;
                    self.operation(block, OperationKind::With, resources);
                    let mut combined = instructions.clone();
                    combined.extend_from_slice(rest);
                    self.lower_block(block, &combined, fallthrough)?;
                    return Ok(());
                }
                Instruction::Break => {
                    self.blocks[block.0 as usize].terminator = Terminator::Break;
                    return Ok(());
                }
                Instruction::Continue => {
                    self.blocks[block.0 as usize].terminator = Terminator::Continue;
                    return Ok(());
                }
            }
        }
        if matches!(
            self.blocks[block.0 as usize].terminator,
            Terminator::Unreachable
        ) {
            self.blocks[block.0 as usize].terminator = fallthrough
                .map(Terminator::Goto)
                .unwrap_or(Terminator::Return(None));
        }
        Ok(())
    }

    fn operation(
        &mut self,
        block: BlockId,
        kind: OperationKind,
        operands: impl IntoIterator<Item = ValueRef>,
    ) {
        self.blocks[block.0 as usize].operations.push(Operation {
            kind,
            operands: operands.into_iter().collect(),
        });
    }

    fn value_ref(&mut self, expression: &Expression) -> Result<ValueRef, MirLoweringError> {
        let mut value = ValueRef {
            id: expression.hir_id(),
            ty: expression.ty(),
            local: match expression.kind() {
                Expression::Variable(binding) => self.bindings.get(&binding.id).copied(),
                _ => None,
            },
            tensor_op: None,
        };
        let Expression::Call { target, args } = expression.kind() else {
            return Ok(value);
        };
        let Some(intrinsic) = target.tensor_intrinsic() else {
            return Ok(value);
        };
        self.source_tensor_intrinsics += 1;
        let result = match expression.ty() {
            Some(ValueType::Tensor(result)) => result,
            actual => {
                return Err(MirLoweringError::tensor(
                    intrinsic,
                    format!("recognized tensor intrinsic has non-tensor result type {actual:?}"),
                )
                .at_expression(expression.hir_id()))
            }
        };
        let inputs = tensor_operands(args, |argument| self.value_ref(argument))?;
        let operation = resolve_tensor_op(intrinsic, args, inputs, result)
            .map_err(|error| error.at_expression(expression.hir_id()))?;
        let id = TensorOpId(self.tensor_operations.len() as u32);
        self.tensor_operations.push(operation);
        value.tensor_op = Some(id);
        Ok(value)
    }
}
