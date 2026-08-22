use crate::{CoveragePoint, FunctionId};
use severian_hir::{Callee as HirCallee, Conversion};
use severian_universal::{
    Attrs, BinaryOperator, DefId, LiteralValue, OpId, Substitution, TypeId, UnaryOperator,
};
use std::collections::BTreeMap;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct BlockId(pub u32);

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct LocalId(pub u32);

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LocalDecl {
    pub id: LocalId,
    pub ty: TypeId,
    pub mutable: bool,
    pub argument: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Place {
    pub local: LocalId,
    pub projection: Vec<Projection>,
}

impl Place {
    pub const fn local(local: LocalId) -> Self {
        Self {
            local,
            projection: Vec::new(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Projection {
    Field(u32),
    Index(LocalId),
    Dereference,
    Downcast(u32),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Operand {
    Copy(Place),
    Move(Place),
    Constant { value: LiteralValue, ty: TypeId },
    Function(DefId),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Callee {
    Direct {
        function: DefId,
        substitution: Substitution,
    },
    FunctionValue(Operand),
    Method {
        implementation: DefId,
        receiver: Operand,
        substitution: Substitution,
    },
    Constructor {
        type_def: DefId,
        variant: Option<severian_hir::VariantId>,
    },
    Intrinsic(OpId),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Rvalue {
    Use(Operand),
    Unary {
        operator: UnaryOperator,
        operand: Operand,
    },
    Binary {
        operator: BinaryOperator,
        left: Operand,
        right: Operand,
    },
    BorrowShared(Place),
    BorrowExclusive(Place),
    Convert {
        operand: Operand,
        conversion: Conversion,
    },
    Aggregate {
        type_id: TypeId,
        fields: Vec<Operand>,
    },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Statement {
    Assign(Place, Rvalue),
    Drop(Place),
    StorageLive(LocalId),
    StorageDead(LocalId),
    Assert {
        condition: Operand,
        message: Option<Operand>,
    },
    Coverage(CoveragePoint),
    Operation {
        id: OpId,
        operands: Vec<Operand>,
        results: Vec<Place>,
        attributes: Attrs,
    },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Case {
    Integer(i128),
    Boolean(bool),
    Variant(u32),
    Type(TypeId),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Terminator {
    Goto(BlockId, Vec<Operand>),
    Branch {
        condition: Operand,
        then_block: BlockId,
        else_block: BlockId,
    },
    Switch {
        discriminant: Operand,
        targets: Vec<(Case, BlockId)>,
        fallback: BlockId,
    },
    Call {
        callee: Callee,
        arguments: Vec<Operand>,
        destination: Option<Place>,
        target: BlockId,
        unwind: Option<BlockId>,
    },
    Return(Option<Operand>),
    Throw(Operand),
    Unreachable,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BasicBlock {
    pub id: BlockId,
    pub parameters: Vec<LocalId>,
    pub statements: Vec<Statement>,
    pub terminator: Terminator,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Body {
    pub entry: BlockId,
    pub blocks: Vec<BasicBlock>,
    pub locals: Vec<LocalDecl>,
    pub return_type: TypeId,
}

impl Default for Body {
    fn default() -> Self {
        Self {
            entry: BlockId(0),
            blocks: vec![BasicBlock {
                id: BlockId(0),
                parameters: Vec::new(),
                statements: Vec::new(),
                terminator: Terminator::Return(None),
            }],
            locals: Vec::new(),
            return_type: TypeId(0),
        }
    }
}

pub(crate) fn lower_program(program: &severian_hir::Program) -> (Body, BTreeMap<FunctionId, Body>) {
    let unit = program
        .modules
        .iter()
        .flat_map(|module| &module.functions)
        .next()
        .map_or(TypeId(0), |function| function.result.ty);
    let mut initializer = BodyBuilder::new(unit);
    for module in &program.modules {
        initializer.lower_statements(&module.initializer.statements, module);
    }
    let initializer = initializer.finish();

    let mut functions = BTreeMap::new();
    for module in &program.modules {
        for function in &module.functions {
            let Some(body) = &function.body else {
                continue;
            };
            let mut builder = BodyBuilder::new(function.result.ty);
            // Globals are implicit function inputs until CFG module places are
            // introduced. Giving them explicit argument locals keeps their
            // initialization and ownership state visible to verification.
            for source_module in &program.modules {
                for binding in &source_module.bindings {
                    let local = builder.local(binding.type_id, false, true);
                    builder.bindings.insert(binding.id, Place::local(local));
                    builder.entry_parameters.push(local);
                }
            }
            for parameter in &function.parameters {
                let local = builder.local(parameter.contract.ty, false, true);
                builder
                    .bindings
                    .insert(parameter.binding, Place::local(local));
                builder.entry_parameters.push(local);
            }
            builder.lower_statements(&body.statements, module);
            functions.insert(function.id, builder.finish());
        }
    }
    (initializer, functions)
}

struct BodyBuilder {
    body: Body,
    current: BlockId,
    bindings: BTreeMap<severian_hir::BindingId, Place>,
    expressions: BTreeMap<severian_hir::HirId, Place>,
    entry_parameters: Vec<LocalId>,
}

impl BodyBuilder {
    fn new(return_type: TypeId) -> Self {
        Self {
            body: Body {
                entry: BlockId(0),
                blocks: vec![BasicBlock {
                    id: BlockId(0),
                    parameters: Vec::new(),
                    statements: Vec::new(),
                    terminator: Terminator::Unreachable,
                }],
                locals: Vec::new(),
                return_type,
            },
            current: BlockId(0),
            bindings: BTreeMap::new(),
            expressions: BTreeMap::new(),
            entry_parameters: Vec::new(),
        }
    }

    fn finish(mut self) -> Body {
        self.body.blocks[self.body.entry.0 as usize].parameters =
            std::mem::take(&mut self.entry_parameters);
        if self.open(self.current) {
            self.terminate(Terminator::Return(None));
        }
        self.body
    }

    fn block(&mut self) -> BlockId {
        let id = BlockId(self.body.blocks.len() as u32);
        self.body.blocks.push(BasicBlock {
            id,
            parameters: Vec::new(),
            statements: Vec::new(),
            terminator: Terminator::Unreachable,
        });
        id
    }

    fn local(&mut self, ty: TypeId, mutable: bool, argument: bool) -> LocalId {
        let id = LocalId(self.body.locals.len() as u32);
        self.body.locals.push(LocalDecl {
            id,
            ty,
            mutable,
            argument,
        });
        id
    }

    fn open(&self, block: BlockId) -> bool {
        matches!(
            &self.body.blocks[block.0 as usize].terminator,
            Terminator::Unreachable
        )
    }

    fn terminate(&mut self, terminator: Terminator) {
        self.body.blocks[self.current.0 as usize].terminator = terminator;
    }

    fn push(&mut self, statement: Statement) {
        self.body.blocks[self.current.0 as usize]
            .statements
            .push(statement);
    }

    fn lower_statements(
        &mut self,
        statements: &[severian_hir::Statement],
        module: &severian_hir::Module,
    ) {
        for statement in statements {
            if !self.open(self.current) {
                break;
            }
            self.lower_statement(statement, module);
        }
    }

    fn lower_statement(
        &mut self,
        statement: &severian_hir::Statement,
        module: &severian_hir::Module,
    ) {
        match statement {
            severian_hir::Statement::Binding(id) => {
                let binding = module
                    .bindings
                    .iter()
                    .find(|binding| binding.id == *id)
                    .expect("typed HIR binding exists");
                let value = self.expression(&binding.value);
                let local = self.local(binding.type_id, true, false);
                let place = Place::local(local);
                self.push(Statement::StorageLive(local));
                self.push(Statement::Assign(
                    place.clone(),
                    Rvalue::Use(Operand::Copy(value)),
                ));
                self.bindings.insert(*id, place);
            }
            severian_hir::Statement::Expression(expression) => {
                self.expression(expression);
            }
            severian_hir::Statement::Return(value) => {
                let value = value
                    .as_ref()
                    .map(|expression| Operand::Copy(self.expression(expression)));
                self.terminate(Terminator::Return(value));
            }
            severian_hir::Statement::Assert {
                condition, message, ..
            } => {
                let condition = Operand::Copy(self.expression(condition));
                let message = message
                    .as_ref()
                    .map(|message| Operand::Copy(self.expression(message)));
                self.push(Statement::Assert { condition, message });
            }
            severian_hir::Statement::If {
                condition,
                then_block,
                else_block,
            } => {
                let condition = Operand::Copy(self.expression(condition));
                let then_id = self.block();
                let else_id = self.block();
                let join = self.block();
                self.terminate(Terminator::Branch {
                    condition,
                    then_block: then_id,
                    else_block: else_id,
                });
                let bindings = self.bindings.clone();
                self.current = then_id;
                self.lower_statements(&then_block.statements, module);
                if self.open(self.current) {
                    self.terminate(Terminator::Goto(join, Vec::new()));
                }
                self.bindings.clone_from(&bindings);
                self.current = else_id;
                self.lower_statements(&else_block.statements, module);
                if self.open(self.current) {
                    self.terminate(Terminator::Goto(join, Vec::new()));
                }
                self.bindings = bindings;
                self.current = join;
            }
            severian_hir::Statement::Match { subject, arms } => {
                let subject = Operand::Copy(self.expression(subject));
                let join = self.block();
                let mut targets = Vec::new();
                let mut arm_blocks = Vec::new();
                for arm in arms {
                    let block = self.block();
                    if let Some(ty) = arm.type_id {
                        targets.push((Case::Type(ty), block));
                    }
                    arm_blocks.push((block, arm));
                }
                let fallback = arm_blocks.last().map_or(join, |(block, _)| *block);
                self.terminate(Terminator::Switch {
                    discriminant: subject.clone(),
                    targets,
                    fallback,
                });
                let bindings = self.bindings.clone();
                for (block, arm) in arm_blocks {
                    self.current = block;
                    self.bindings.clone_from(&bindings);
                    if let Some(binding) = arm.binding {
                        let local =
                            self.local(arm.type_id.unwrap_or(self.body.return_type), false, false);
                        let place = Place::local(local);
                        self.push(Statement::StorageLive(local));
                        self.push(Statement::Assign(
                            place.clone(),
                            Rvalue::Use(subject.clone()),
                        ));
                        self.bindings.insert(binding, place);
                    }
                    self.lower_statements(&arm.body.statements, module);
                    if self.open(self.current) {
                        self.terminate(Terminator::Goto(join, Vec::new()));
                    }
                }
                self.bindings = bindings;
                self.current = join;
            }
        }
    }

    fn expression(&mut self, expression: &severian_hir::Expression) -> Place {
        if let Some(place) = self.expressions.get(&expression.id) {
            return place.clone();
        }
        let result = Place::local(self.local(expression.type_id, false, false));
        self.push(Statement::StorageLive(result.local));
        match &expression.kind {
            severian_hir::ExpressionKind::Literal(value) => self.push(Statement::Assign(
                result.clone(),
                Rvalue::Use(Operand::Constant {
                    value: value.clone(),
                    ty: expression.type_id,
                }),
            )),
            severian_hir::ExpressionKind::Binding(binding) => {
                let source = self.bindings[binding].clone();
                self.push(Statement::Assign(
                    result.clone(),
                    Rvalue::Use(Operand::Copy(source)),
                ));
            }
            severian_hir::ExpressionKind::Function(function) => self.push(Statement::Assign(
                result.clone(),
                Rvalue::Use(Operand::Function(*function)),
            )),
            severian_hir::ExpressionKind::Convert {
                operand,
                conversion,
            } => {
                let operand = Operand::Copy(self.expression(operand));
                self.push(Statement::Assign(
                    result.clone(),
                    Rvalue::Convert {
                        operand,
                        conversion: conversion.clone(),
                    },
                ));
            }
            severian_hir::ExpressionKind::Unary { operator, operand } => {
                let operand = Operand::Copy(self.expression(operand));
                self.push(Statement::Assign(
                    result.clone(),
                    Rvalue::Unary {
                        operator: *operator,
                        operand,
                    },
                ));
            }
            severian_hir::ExpressionKind::Binary {
                operator,
                left,
                right,
            } => {
                let left = Operand::Copy(self.expression(left));
                let right = Operand::Copy(self.expression(right));
                self.push(Statement::Assign(
                    result.clone(),
                    Rvalue::Binary {
                        operator: *operator,
                        left,
                        right,
                    },
                ));
            }
            severian_hir::ExpressionKind::Call { callee, arguments } => {
                let arguments = arguments
                    .iter()
                    .map(|argument| Operand::Copy(self.expression(argument)))
                    .collect();
                let continuation = self.block();
                let callee = self.callee(callee);
                self.terminate(Terminator::Call {
                    callee,
                    arguments,
                    destination: Some(result.clone()),
                    target: continuation,
                    unwind: None,
                });
                self.current = continuation;
            }
        }
        self.expressions.insert(expression.id, result.clone());
        result
    }

    fn callee(&self, callee: &HirCallee) -> Callee {
        match callee {
            HirCallee::Direct {
                function,
                substitution,
            } => Callee::Direct {
                function: *function,
                substitution: substitution.clone(),
            },
            HirCallee::FunctionValue(expression) => {
                Callee::FunctionValue(Operand::Copy(self.expressions[expression].clone()))
            }
            HirCallee::Method {
                implementation,
                receiver,
                substitution,
            } => Callee::Method {
                implementation: *implementation,
                receiver: Operand::Copy(self.expressions[receiver].clone()),
                substitution: substitution.clone(),
            },
            HirCallee::Constructor { type_def, variant } => Callee::Constructor {
                type_def: *type_def,
                variant: *variant,
            },
            HirCallee::Intrinsic(operation) => Callee::Intrinsic(*operation),
        }
    }
}
