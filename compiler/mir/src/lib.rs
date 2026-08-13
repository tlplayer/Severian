#![forbid(unsafe_code)]

use severian_hir::{
    BindingId, BindingRef, Expression, FunctionId, HirId, Instruction, MatchPattern, ValueType,
};
use std::collections::{BTreeMap, BTreeSet};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct BlockId(pub u32);

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct LocalId(pub u32);

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Local {
    pub id: LocalId,
    pub binding: BindingRef,
    pub ty: ValueType,
}

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
    pub native_symbol: Option<String>,
    pub parameters: Vec<LocalId>,
    pub locals: Vec<Local>,
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
    Bind(LocalId),
    TryBind(LocalId),
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
    pub local: Option<LocalId>,
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

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VerificationError {
    pub function: String,
    pub block: Option<BlockId>,
    pub invariant: &'static str,
    pub message: String,
}

impl std::fmt::Display for VerificationError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "MIR invariant `{}` failed", self.invariant)?;
        if let Some(block) = self.block {
            write!(formatter, " in {} block {}", self.function, block.0)?;
        } else {
            write!(formatter, " in {}", self.function)?;
        }
        write!(formatter, ": {}", self.message)
    }
}

impl std::error::Error for VerificationError {}

/// Verify the structural invariants that every MIR consumer may rely on.
///
/// This deliberately lives at the MIR boundary instead of in a backend. A
/// transformation that creates an invalid CFG is therefore blamed before
/// lowering can obscure the source of the defect.
pub fn verify(program: &Program) -> Result<(), Vec<VerificationError>> {
    let mut errors = Vec::new();
    let mut function_ids = BTreeSet::new();
    for function in &program.functions {
        if !function_ids.insert(function.id) {
            errors.push(VerificationError {
                function: function.name.clone(),
                block: None,
                invariant: "unique-function-id",
                message: format!("stable function identity {:?} is reused", function.id),
            });
        }
        verify_function(function, &mut errors);
    }
    if errors.is_empty() {
        Ok(())
    } else {
        Err(errors)
    }
}

fn verify_function(function: &Function, errors: &mut Vec<VerificationError>) {
    if function.blocks.is_empty() {
        errors.push(VerificationError {
            function: function.name.clone(),
            block: None,
            invariant: "entry-block",
            message: "function has no entry block".into(),
        });
        return;
    }

    let block_count = function.blocks.len();
    let local_count = function.locals.len();
    let mut binding_ids = BTreeSet::new();
    for (index, local) in function.locals.iter().enumerate() {
        if local.id.0 as usize != index {
            errors.push(VerificationError {
                function: function.name.clone(),
                block: None,
                invariant: "dense-local-id",
                message: format!("local at index {index} has id {}", local.id.0),
            });
        }
        if !binding_ids.insert(local.binding.id) {
            errors.push(VerificationError {
                function: function.name.clone(),
                block: None,
                invariant: "unique-binding-id",
                message: format!(
                    "binding identity {:?} is assigned to multiple MIR locals",
                    local.binding.id
                ),
            });
        }
    }
    for parameter in &function.parameters {
        verify_local_target(function, None, *parameter, local_count, errors);
    }
    for (index, block) in function.blocks.iter().enumerate() {
        if block.id.0 as usize != index {
            errors.push(VerificationError {
                function: function.name.clone(),
                block: Some(block.id),
                invariant: "dense-block-id",
                message: format!("block at index {index} has id {}", block.id.0),
            });
        }
        if matches!(block.terminator, Terminator::Unreachable) {
            errors.push(VerificationError {
                function: function.name.clone(),
                block: Some(block.id),
                invariant: "terminated-block",
                message: "lowering left the block without a terminator".into(),
            });
        }
        for operation in &block.operations {
            if let OperationKind::Bind(local) | OperationKind::TryBind(local) = operation.kind {
                verify_local_target(function, Some(block.id), local, local_count, errors);
            }
            for operand in &operation.operands {
                if let Some(local) = operand.local {
                    verify_local_target(function, Some(block.id), local, local_count, errors);
                }
            }
        }
        for target in successor_blocks(&block.terminator) {
            if target.0 as usize >= block_count {
                errors.push(VerificationError {
                    function: function.name.clone(),
                    block: Some(block.id),
                    invariant: "valid-successor",
                    message: format!(
                        "terminator targets block {} but the function has {block_count} block(s)",
                        target.0
                    ),
                });
            }
        }
        verify_terminator_types(function, block, errors);
    }
}

fn verify_local_target(
    function: &Function,
    block: Option<BlockId>,
    local: LocalId,
    local_count: usize,
    errors: &mut Vec<VerificationError>,
) {
    if local.0 as usize >= local_count {
        errors.push(VerificationError {
            function: function.name.clone(),
            block,
            invariant: "valid-local",
            message: format!(
                "references local {} but the function has {local_count} local(s)",
                local.0
            ),
        });
    }
}

fn successor_blocks(terminator: &Terminator) -> Vec<BlockId> {
    match terminator {
        Terminator::Goto(target) => vec![*target],
        Terminator::Branch {
            then_block,
            else_block,
            ..
        } => vec![*then_block, *else_block],
        Terminator::Loop { body, exit, .. } | Terminator::For { body, exit, .. } => {
            vec![*body, *exit]
        }
        Terminator::Switch { arms, exit, .. } => {
            let mut targets = arms.clone();
            targets.push(*exit);
            targets
        }
        Terminator::Return(_)
        | Terminator::Break
        | Terminator::Continue
        | Terminator::Unreachable => Vec::new(),
    }
}

fn verify_terminator_types(
    function: &Function,
    block: &BasicBlock,
    errors: &mut Vec<VerificationError>,
) {
    let condition = match &block.terminator {
        Terminator::Branch { condition, .. } | Terminator::Loop { condition, .. } => {
            Some(condition)
        }
        _ => None,
    };
    if let Some(condition) = condition {
        if !matches!(condition.ty, Some(ValueType::Bool | ValueType::Any)) {
            errors.push(VerificationError {
                function: function.name.clone(),
                block: Some(block.id),
                invariant: "boolean-condition",
                message: format!("control-flow condition has type {:?}", condition.ty),
            });
        }
    }
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
