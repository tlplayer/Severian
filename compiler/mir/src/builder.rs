use crate::*;
use severian_hir::{BindingId, BindingRef, Expression, Instruction, ScopedBehavior, ValueType};
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
    cleanup_stack: Vec<ScopedBehavior>,
    loop_cleanup_depths: Vec<usize>,
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
                    self.emit_cleanups(block, 0);
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
                    self.loop_cleanup_depths.push(self.cleanup_stack.len());
                    let lowered = self.lower_block(body, instructions, Some(header));
                    self.loop_cleanup_depths.pop();
                    lowered?;
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
                    self.loop_cleanup_depths.push(self.cleanup_stack.len());
                    let lowered = self.lower_block(body, instructions, Some(block));
                    self.loop_cleanup_depths.pop();
                    lowered?;
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
                    scoped_behaviors,
                    instructions,
                    ..
                } => {
                    let resources = resources
                        .iter()
                        .map(|value| self.value_ref(value))
                        .collect::<Result<Vec<_>, _>>()?;
                    self.operation(block, OperationKind::With, resources);
                    for behavior in scoped_behaviors {
                        self.operation(block, OperationKind::ScopeEnter(behavior.clone()), []);
                    }
                    let cleanup_depth = self.cleanup_stack.len();
                    self.cleanup_stack.extend(scoped_behaviors.iter().cloned());
                    let scope_exit = self.reserve_block();
                    let lowered = self.lower_block(block, instructions, Some(scope_exit));
                    self.cleanup_stack.truncate(cleanup_depth);
                    lowered?;
                    for behavior in scoped_behaviors.iter().rev() {
                        self.operation(scope_exit, OperationKind::ScopeExit(behavior.clone()), []);
                    }
                    self.lower_block(scope_exit, rest, fallthrough)?;
                    return Ok(());
                }
                Instruction::Break => {
                    let cleanup_depth = self.loop_cleanup_depths.last().copied().unwrap_or(0);
                    self.emit_cleanups(block, cleanup_depth);
                    self.blocks[block.0 as usize].terminator = Terminator::Break;
                    return Ok(());
                }
                Instruction::Continue => {
                    let cleanup_depth = self.loop_cleanup_depths.last().copied().unwrap_or(0);
                    self.emit_cleanups(block, cleanup_depth);
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

    fn emit_cleanups(&mut self, block: BlockId, depth: usize) {
        let behaviors = self.cleanup_stack[depth..]
            .iter()
            .rev()
            .cloned()
            .collect::<Vec<_>>();
        for behavior in behaviors {
            self.operation(block, OperationKind::ScopeExit(behavior), []);
        }
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
            self.lower_nested_tensor_ops(expression)?;
            return Ok(value);
        };
        let Some(intrinsic) = target.tensor_intrinsic() else {
            for argument in args {
                self.value_ref(argument)?;
            }
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
        let scalar = matches!(
            intrinsic,
            severian_hir::TensorIntrinsic::Scale | severian_hir::TensorIntrinsic::AddScalar
        )
        .then(|| {
            args.get(1)
                .map(|argument| self.value_ref(argument))
                .transpose()
        })
        .transpose()?
        .flatten();
        let inputs = tensor_operands(args, |argument| self.value_ref(argument))?;
        let operation = resolve_tensor_op(intrinsic, args, inputs, scalar, result, |argument| {
            self.value_ref(argument)
        })
        .map_err(|error| error.at_expression(expression.hir_id()))?;
        let id = TensorOpId(self.tensor_operations.len() as u32);
        self.tensor_operations.push(operation);
        value.tensor_op = Some(id);
        Ok(value)
    }

    /// Record tensor intrinsics wherever an expression evaluates its children.
    ///
    /// Tensor operations are attached to intrinsic calls themselves, but an
    /// intrinsic can be nested under any ordinary expression (most notably a
    /// method-call argument). Walking those children here keeps MIR discovery
    /// independent of how model code chooses to parenthesize an expression.
    fn lower_nested_tensor_ops(&mut self, expression: &Expression) -> Result<(), MirLoweringError> {
        let mut lower = |expression: &Expression| self.value_ref(expression).map(|_| ());
        match expression.kind() {
            Expression::Typed { expression, .. } => lower(expression),
            Expression::Ownership { value, .. }
            | Expression::Member { object: value, .. }
            | Expression::ObjectDocument { object: value, .. }
            | Expression::Task { value, .. }
            | Expression::Await(value)
            | Expression::Channel(value)
            | Expression::ChaosRule { value, .. }
            | Expression::FusedPipeline { input: value, .. }
            | Expression::Unary {
                expression: value, ..
            } => lower(value),
            Expression::List(values)
            | Expression::Tuple(values)
            | Expression::Set(values)
            | Expression::PrintArgs(values)
            | Expression::Construct { args: values, .. }
            | Expression::Variant { fields: values, .. }
            | Expression::Format { args: values, .. } => {
                for value in values {
                    lower(value)?;
                }
                Ok(())
            }
            Expression::Map(entries) => {
                for (key, value) in entries {
                    lower(key)?;
                    lower(value)?;
                }
                Ok(())
            }
            Expression::Index { object, index } => {
                lower(object)?;
                lower(index)
            }
            Expression::Slice {
                object,
                start,
                end,
                step,
            } => {
                lower(object)?;
                for bound in [start, end, step].into_iter().flatten() {
                    lower(bound)?;
                }
                Ok(())
            }
            Expression::ConstructFields { fields, .. } => {
                for (_, value) in fields {
                    lower(value)?;
                }
                Ok(())
            }
            Expression::ObjectUpdate { object, fields, .. } => {
                lower(object)?;
                for (_, value) in fields {
                    lower(value)?;
                }
                Ok(())
            }
            Expression::MethodCall { object, args, .. } => {
                lower(object)?;
                for argument in args {
                    lower(argument)?;
                }
                Ok(())
            }
            Expression::Send { value, channel } => {
                lower(value)?;
                lower(channel)
            }
            Expression::ListComprehension { element, clauses }
            | Expression::SetComprehension { element, clauses } => {
                lower(element)?;
                for clause in clauses {
                    lower(&clause.iterable)?;
                    if let Some(condition) = &clause.condition {
                        lower(condition)?;
                    }
                }
                Ok(())
            }
            Expression::MapComprehension {
                key,
                value,
                clauses,
            } => {
                lower(key)?;
                lower(value)?;
                for clause in clauses {
                    lower(&clause.iterable)?;
                    if let Some(condition) = &clause.condition {
                        lower(condition)?;
                    }
                }
                Ok(())
            }
            Expression::Conditional {
                condition,
                then_expression,
                else_expression,
            } => {
                lower(condition)?;
                lower(then_expression)?;
                lower(else_expression)
            }
            Expression::Binary { left, right, .. } => {
                lower(left)?;
                lower(right)
            }
            Expression::Call { args, .. } => {
                for argument in args {
                    lower(argument)?;
                }
                Ok(())
            }
            Expression::CallValue { callee, args, .. } => {
                lower(callee)?;
                for argument in args {
                    lower(argument)?;
                }
                Ok(())
            }
            // Lambda and closure bodies execute in their own function scope.
            Expression::Lambda { .. }
            | Expression::Closure { .. }
            | Expression::Integer(_)
            | Expression::Float(_)
            | Expression::Boolean(_)
            | Expression::String(_)
            | Expression::Variable(_)
            | Expression::Function(_) => Ok(()),
        }
    }
}
