#![forbid(unsafe_code)]

use severian_hir::{Expression, FunctionId, HirId, Instruction, MatchPattern, ValueType};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct BlockId(pub u32);

#[derive(Clone, PartialEq, Eq)]
pub struct Program {
    hir: severian_hir::Program,
    pub functions: Vec<Function>,
}

impl Default for Program {
    fn default() -> Self {
        Self {
            hir: severian_hir::Program::default(),
            functions: Vec::new(),
        }
    }
}

impl std::fmt::Debug for Program {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("MirProgram")
            .field("functions", &self.functions)
            .finish()
    }
}

impl Program {
    /// The structured expression payload consumed by the current MLIR
    /// lowering. Control-flow ownership lives in MIR and consumers must enter
    /// lowering through this type.
    pub fn lowering_hir(&self) -> &severian_hir::Program {
        &self.hir
    }

    /// HIR-v2 metadata is carried through MIR as an inert sidecar. MIR and
    /// lowering do not interpret it yet, but downstream migrations can query
    /// canonical source spans and detailed types without recovering AST data.
    pub fn metadata(&self) -> &severian_hir::ProgramMetadata {
        &self.hir.metadata
    }

    pub fn source_span(&self, value: ValueRef) -> Option<severian_hir::SourceSpan> {
        value
            .id
            .and_then(|id| self.hir.metadata.sources.expression_span(id))
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Function {
    pub id: FunctionId,
    pub name: String,
    pub parameters: Vec<(String, ValueType)>,
    pub return_type: ValueType,
    pub blocks: Vec<BasicBlock>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BasicBlock {
    pub id: BlockId,
    pub operations: Vec<Operation>,
    pub terminator: Terminator,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Operation {
    pub kind: OperationKind,
    pub operands: Vec<ValueRef>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum OperationKind {
    Bind(String),
    TryBind(String),
    Assign,
    Print,
    Assert,
    Evaluate,
    With,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ValueRef {
    pub id: Option<HirId>,
    pub ty: Option<ValueType>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Terminator {
    Return(Option<ValueRef>),
    Goto(BlockId),
    Branch {
        condition: ValueRef,
        then_block: BlockId,
        else_block: BlockId,
    },
    Loop {
        condition: ValueRef,
        body: BlockId,
        exit: BlockId,
    },
    For {
        pattern: MatchPattern,
        iterable: ValueRef,
        body: BlockId,
        exit: BlockId,
    },
    Switch {
        values: Vec<ValueRef>,
        arms: Vec<BlockId>,
        exit: BlockId,
    },
    Break,
    Continue,
    Unreachable,
}

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
    let entry = builder.reserve_block();
    builder.lower_block(entry, &function.instructions, None);
    Function {
        id: function.id,
        name,
        parameters: function
            .params
            .iter()
            .map(|parameter| (parameter.name.clone(), parameter.ty))
            .collect(),
        return_type: function.return_type,
        blocks: builder.blocks,
    }
}

#[derive(Default)]
struct FunctionBuilder {
    blocks: Vec<BasicBlock>,
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
                    self.operation(block, OperationKind::Bind(name.clone()), [value_ref(value)])
                }
                Instruction::TryLet { name, value, .. } => self.operation(
                    block,
                    OperationKind::TryBind(name.clone()),
                    [value_ref(value)],
                ),
                Instruction::Assign { target, value, .. } => self.operation(
                    block,
                    OperationKind::Assign,
                    [value_ref(target), value_ref(value)],
                ),
                Instruction::Print(value) => {
                    self.operation(block, OperationKind::Print, [value_ref(value)])
                }
                Instruction::Assert(value) => {
                    self.operation(block, OperationKind::Assert, [value_ref(value)])
                }
                Instruction::Evaluate(value) => {
                    self.operation(block, OperationKind::Evaluate, [value_ref(value)])
                }
                Instruction::Return(value) => {
                    self.blocks[block.0 as usize].terminator =
                        Terminator::Return(value.as_ref().map(value_ref));
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
                        condition: value_ref(condition),
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
                        condition: value_ref(condition),
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
                        iterable: value_ref(iterable),
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
                        values: vec![value_ref(value)],
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
                        values: channels.iter().map(value_ref).collect(),
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
                    self.operation(block, OperationKind::With, resources.iter().map(value_ref));
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
}

fn value_ref(expression: &Expression) -> ValueRef {
    ValueRef {
        id: expression.hir_id(),
        ty: expression.ty(),
    }
}
