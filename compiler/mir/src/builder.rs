use crate::*;
use severian_hir::{BindingId, BindingRef, Expression, Instruction, ValueType};
use std::collections::BTreeMap;

pub fn lower(hir: &severian_hir::Program) -> Program {
    let mut functions = hir
        .functions
        .iter()
        .map(|function| lower_function(function, function.name.clone()))
        .collect::<Vec<_>>();
    for class in &hir.classes {
        functions.extend(
            class
                .constructors
                .iter()
                .chain(&class.methods)
                .map(|function| {
                    lower_function(function, format!("{}.{}", class.name, function.name))
                }),
        );
    }
    Program {
        hir: hir.clone(),
        functions,
    }
}

fn lower_function(function: &severian_hir::Function, name: String) -> Function {
    let mut builder = FunctionBuilder::default();
    let parameters = function
        .params
        .iter()
        .map(|parameter| builder.reserve_local(parameter.name.clone(), parameter.ty))
        .collect();
    let entry = builder.reserve_block();
    builder.lower_block(entry, &function.instructions, None);
    Function {
        id: function.id,
        name,
        native_symbol: function.native_symbol.clone(),
        parameters,
        locals: builder.locals,
        return_type: function.return_type,
        blocks: builder.blocks,
    }
}

#[derive(Default)]
struct FunctionBuilder {
    blocks: Vec<BasicBlock>,
    locals: Vec<Local>,
    bindings: BTreeMap<BindingId, LocalId>,
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
    ) {
        for (index, instruction) in instructions.iter().enumerate() {
            let rest = &instructions[index + 1..];
            match instruction {
                Instruction::Let { name, value } => {
                    let local =
                        self.reserve_local(name.clone(), value.ty().unwrap_or(ValueType::Any));
                    self.operation(block, OperationKind::Bind(local), [self.value_ref(value)])
                }
                Instruction::TryLet { name, value, .. } => {
                    let local = self.reserve_local(name.clone(), ValueType::Any);
                    self.operation(
                        block,
                        OperationKind::TryBind(local),
                        [self.value_ref(value)],
                    )
                }
                Instruction::Assign { target, value, .. } => self.operation(
                    block,
                    OperationKind::Assign,
                    [self.value_ref(target), self.value_ref(value)],
                ),
                Instruction::Print(value) => {
                    self.operation(block, OperationKind::Print, [self.value_ref(value)])
                }
                Instruction::Assert(value) => {
                    self.operation(block, OperationKind::Assert, [self.value_ref(value)])
                }
                Instruction::Evaluate(value) => {
                    self.operation(block, OperationKind::Evaluate, [self.value_ref(value)])
                }
                Instruction::Return(value) => {
                    self.blocks[block.0 as usize].terminator =
                        Terminator::Return(value.as_ref().map(|value| self.value_ref(value)));
                    return;
                }
                Instruction::If {
                    condition,
                    then_instructions,
                    else_instructions,
                } => {
                    let then_block = self.reserve_block();
                    let else_block = self.reserve_block();
                    let join = self.reserve_block();
                    self.blocks[block.0 as usize].terminator = Terminator::Branch {
                        condition: self.value_ref(condition),
                        then_block,
                        else_block,
                    };
                    self.lower_block(then_block, then_instructions, Some(join));
                    self.lower_block(else_block, else_instructions, Some(join));
                    self.lower_block(join, rest, fallthrough);
                    return;
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
                    self.blocks[header.0 as usize].terminator = Terminator::Loop {
                        condition: self.value_ref(condition),
                        body,
                        exit,
                    };
                    self.lower_block(body, instructions, Some(header));
                    self.lower_block(exit, rest, fallthrough);
                    return;
                }
                Instruction::For {
                    pattern,
                    iterable,
                    instructions,
                    ..
                } => {
                    let body = self.reserve_block();
                    let exit = self.reserve_block();
                    self.blocks[block.0 as usize].terminator = Terminator::For {
                        pattern: pattern.clone(),
                        iterable: self.value_ref(iterable),
                        body,
                        exit,
                    };
                    self.lower_block(body, instructions, Some(block));
                    self.lower_block(exit, rest, fallthrough);
                    return;
                }
                Instruction::Switch { value, arms } => {
                    let exit = self.reserve_block();
                    let arm_blocks = arms
                        .iter()
                        .map(|arm| {
                            let arm_block = self.reserve_block();
                            self.lower_block(arm_block, &arm.instructions, Some(exit));
                            arm_block
                        })
                        .collect();
                    self.blocks[block.0 as usize].terminator = Terminator::Switch {
                        values: vec![self.value_ref(value)],
                        arms: arm_blocks,
                        exit,
                    };
                    self.lower_block(exit, rest, fallthrough);
                    return;
                }
                Instruction::ChannelSwitch { channels, arms, .. } => {
                    let exit = self.reserve_block();
                    let arm_blocks = arms
                        .iter()
                        .map(|arm| {
                            let arm_block = self.reserve_block();
                            self.lower_block(arm_block, &arm.instructions, Some(exit));
                            arm_block
                        })
                        .collect();
                    self.blocks[block.0 as usize].terminator = Terminator::Switch {
                        values: channels.iter().map(|value| self.value_ref(value)).collect(),
                        arms: arm_blocks,
                        exit,
                    };
                    self.lower_block(exit, rest, fallthrough);
                    return;
                }
                Instruction::With {
                    resources,
                    instructions,
                    ..
                } => {
                    let resources = resources
                        .iter()
                        .map(|value| self.value_ref(value))
                        .collect::<Vec<_>>();
                    self.operation(block, OperationKind::With, resources);
                    let mut combined = instructions.clone();
                    combined.extend_from_slice(rest);
                    self.lower_block(block, &combined, fallthrough);
                    return;
                }
                Instruction::Break => {
                    self.blocks[block.0 as usize].terminator = Terminator::Break;
                    return;
                }
                Instruction::Continue => {
                    self.blocks[block.0 as usize].terminator = Terminator::Continue;
                    return;
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

    fn value_ref(&self, expression: &Expression) -> ValueRef {
        ValueRef {
            id: expression.hir_id(),
            ty: expression.ty(),
            local: match expression.kind() {
                Expression::Variable(binding) => self.bindings.get(&binding.id).copied(),
                _ => None,
            },
        }
    }
}
