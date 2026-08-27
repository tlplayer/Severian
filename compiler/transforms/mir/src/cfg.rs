use crate::{AssertionOrigin, CoverageKind, CoveragePoint, FunctionId, TaskOwner};
use severian_hir::{Callee as HirCallee, Conversion};
use severian_source::Span;
use severian_universal::{
    Attrs, BinaryOperator, DefId, ExecutionPlacement, LiteralValue, OpId, Substitution, TypeId,
    UnaryOperator,
};
use std::collections::{BTreeMap, BTreeSet};

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct BlockId(pub u32);

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct LocalId(pub u32);

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct GlobalId(pub u32);

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GlobalDecl {
    pub id: GlobalId,
    pub ty: TypeId,
    pub mutable: bool,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LocalDecl {
    pub id: LocalId,
    pub ty: TypeId,
    pub mutable: bool,
    pub argument: bool,
    pub span: Option<Span>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum PlaceBase {
    Local(LocalId),
    Global(GlobalId),
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Place {
    pub base: PlaceBase,
    pub projection: Vec<Projection>,
}

impl Place {
    pub const fn local(local: LocalId) -> Self {
        Self {
            base: PlaceBase::Local(local),
            projection: Vec::new(),
        }
    }

    pub const fn global(global: GlobalId) -> Self {
        Self {
            base: PlaceBase::Global(global),
            projection: Vec::new(),
        }
    }

    pub const fn local_id(&self) -> Option<LocalId> {
        match self.base {
            PlaceBase::Local(local) => Some(local),
            PlaceBase::Global(_) => None,
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
    AddressOf(Place),
    Convert {
        operand: Operand,
        conversion: Conversion,
    },
    Aggregate {
        type_id: TypeId,
        fields: Vec<Operand>,
    },
    Await {
        task: Operand,
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
        origin: AssertionOrigin,
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
    Spawn {
        callee: Callee,
        arguments: Vec<Operand>,
        destination: Place,
        target: BlockId,
        owner: TaskOwner,
        locked: bool,
    },
    SpawnFieldUpdate {
        place: Place,
        operator: BinaryOperator,
        value: Operand,
        destination: Place,
        target: BlockId,
        owner: TaskOwner,
        locked: bool,
    },
    Return(Option<Operand>),
    Throw(Operand),
    Unreachable,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BasicBlock {
    pub id: BlockId,
    /// Requested execution domain for this block. `None` is ordinary host
    /// execution. This is structural CFG metadata rather than an operation
    /// attribute so placement survives control-flow lowering.
    pub execution: Option<ExecutionPlacement>,
    pub parameters: Vec<LocalId>,
    pub statements: Vec<Statement>,
    pub statement_spans: Vec<Option<Span>>,
    pub terminator: Terminator,
    pub terminator_span: Option<Span>,
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
                execution: None,
                parameters: Vec::new(),
                statements: Vec::new(),
                statement_spans: Vec::new(),
                terminator: Terminator::Return(None),
                terminator_span: None,
            }],
            locals: Vec::new(),
            return_type: TypeId(0),
        }
    }
}

pub(crate) fn lower_program(
    program: &severian_hir::Program,
) -> (Vec<GlobalDecl>, Body, BTreeMap<FunctionId, Body>) {
    let unit = program
        .modules
        .iter()
        .flat_map(|module| &module.functions)
        .next()
        .map_or(TypeId(0), |function| function.result.ty);
    let mut globals = Vec::new();
    let mut global_bindings = BTreeMap::new();
    let mut global_variables = BTreeMap::new();
    for module in &program.modules {
        for statement in &module.initializer.statements {
            collect_global_bindings(
                statement,
                module,
                &mut globals,
                &mut global_bindings,
                &mut global_variables,
            );
        }
        if module.initializer.statements.is_empty() && module.functions.is_empty() {
            for binding in &module.bindings {
                let id = GlobalId(globals.len() as u32);
                let place = Place::global(id);
                globals.push(GlobalDecl {
                    id,
                    ty: binding.type_id,
                    mutable: binding.mutable,
                    span: binding.span,
                });
                global_bindings.insert(binding.id, place.clone());
                global_variables.insert(binding.variable, place);
            }
        }
    }
    let mut initializer = BodyBuilder::new(unit, global_bindings.clone(), global_variables.clone());
    for module in &program.modules {
        initializer.expressions.clear();
        initializer.lower_statements(&module.initializer.statements, module);
    }
    let initializer = initializer.finish();

    let mut functions = BTreeMap::new();
    for module in &program.modules {
        for function in &module.functions {
            let Some(body) = &function.body else {
                continue;
            };
            let mut builder = BodyBuilder::new(
                function.result.ty,
                global_bindings.clone(),
                global_variables.clone(),
            );
            for parameter in &function.parameters {
                let local = builder.local(parameter.contract.ty, true, true);
                let place = Place::local(local);
                builder.bindings.insert(parameter.binding, place.clone());
                builder
                    .variables
                    .insert(severian_hir::VariableId(parameter.binding.0), place);
                builder.entry_parameters.push(local);
            }
            builder.lower_statements(&body.statements, module);
            functions.insert(function.id, builder.finish());
        }
    }
    (globals, initializer, functions)
}

fn collect_global_bindings(
    statement: &severian_hir::Statement,
    module: &severian_hir::Module,
    globals: &mut Vec<GlobalDecl>,
    bindings: &mut BTreeMap<severian_hir::BindingId, Place>,
    variables: &mut BTreeMap<severian_hir::VariableId, Place>,
) {
    match statement {
        severian_hir::Statement::Sequence(block)
        | severian_hir::Statement::Placement { body: block, .. } => {
            for statement in &block.statements {
                collect_global_bindings(statement, module, globals, bindings, variables);
            }
        }
        severian_hir::Statement::Binding(id) => {
            let binding = module
                .bindings
                .iter()
                .find(|binding| binding.id == *id)
                .expect("typed HIR global binding exists");
            let place = if let Some(place) = variables.get(&binding.variable) {
                place.clone()
            } else {
                let id = GlobalId(globals.len() as u32);
                globals.push(GlobalDecl {
                    id,
                    ty: binding.type_id,
                    mutable: binding.mutable,
                    span: binding.span,
                });
                let place = Place::global(id);
                variables.insert(binding.variable, place.clone());
                place
            };
            bindings.insert(*id, place);
        }
        _ => {}
    }
}

struct BodyBuilder {
    body: Body,
    current: BlockId,
    bindings: BTreeMap<severian_hir::BindingId, Place>,
    variables: BTreeMap<severian_hir::VariableId, Place>,
    expressions: BTreeMap<(BlockId, severian_hir::HirId), Place>,
    entry_parameters: Vec<LocalId>,
    loops: Vec<LoopTargets>,
    catch_targets: Vec<CatchTarget>,
    terminated: BTreeSet<BlockId>,
    current_span: Option<Span>,
    execution: Option<ExecutionPlacement>,
}

#[derive(Debug, Clone, Copy)]
struct LoopTargets {
    break_target: BlockId,
    continue_target: BlockId,
}

#[derive(Debug, Clone)]
enum CatchTarget {
    Discard(BlockId),
    Bind { block: BlockId, place: Place },
}

impl BodyBuilder {
    fn new(
        return_type: TypeId,
        bindings: BTreeMap<severian_hir::BindingId, Place>,
        variables: BTreeMap<severian_hir::VariableId, Place>,
    ) -> Self {
        Self {
            body: Body {
                entry: BlockId(0),
                blocks: vec![BasicBlock {
                    id: BlockId(0),
                    execution: None,
                    parameters: Vec::new(),
                    statements: Vec::new(),
                    statement_spans: Vec::new(),
                    terminator: Terminator::Unreachable,
                    terminator_span: None,
                }],
                locals: Vec::new(),
                return_type,
            },
            current: BlockId(0),
            bindings,
            variables,
            expressions: BTreeMap::new(),
            entry_parameters: Vec::new(),
            loops: Vec::new(),
            catch_targets: Vec::new(),
            terminated: BTreeSet::new(),
            current_span: None,
            execution: None,
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
            execution: self.execution,
            parameters: Vec::new(),
            statements: Vec::new(),
            statement_spans: Vec::new(),
            terminator: Terminator::Unreachable,
            terminator_span: None,
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
            span: self.current_span,
        });
        id
    }

    fn open(&self, block: BlockId) -> bool {
        !self.terminated.contains(&block)
    }

    fn terminate(&mut self, terminator: Terminator) {
        let block = &mut self.body.blocks[self.current.0 as usize];
        block.terminator = terminator;
        block.terminator_span = self.current_span;
        self.terminated.insert(self.current);
    }

    fn push(&mut self, statement: Statement) {
        let block = &mut self.body.blocks[self.current.0 as usize];
        block.statements.push(statement);
        block.statement_spans.push(self.current_span);
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
        let span = match statement {
            severian_hir::Statement::Sequence(_) | severian_hir::Statement::Return(None) => None,
            severian_hir::Statement::Placement { span, .. } => Some(*span),
            severian_hir::Statement::FieldUpdate { value, .. }
            | severian_hir::Statement::FieldSet { value, .. }
            | severian_hir::Statement::Expression(value)
            | severian_hir::Statement::Return(Some(value)) => Some(value.span),
            severian_hir::Statement::Binding(id) => module
                .bindings
                .iter()
                .find(|binding| binding.id == *id)
                .map(|binding| binding.span),
            severian_hir::Statement::Assert { span, .. }
            | severian_hir::Statement::ExpectThrow { span, .. }
            | severian_hir::Statement::Try { span, .. }
            | severian_hir::Statement::While { span, .. }
            | severian_hir::Statement::Break { span }
            | severian_hir::Statement::Continue { span } => Some(*span),
            severian_hir::Statement::If { condition, .. }
            | severian_hir::Statement::Match {
                subject: condition, ..
            } => Some(condition.span),
        };
        self.current_span = span;
        if let Some(span) = span {
            self.push(Statement::Coverage(CoveragePoint {
                source: span.source,
                span_start: span.start,
                kind: CoverageKind::Line,
                ordinal: 0,
                key: None,
                file: None,
                line: None,
            }));
        }
        match statement {
            severian_hir::Statement::Sequence(block) => {
                self.lower_statements(&block.statements, module);
            }
            severian_hir::Statement::Placement {
                placement, body, ..
            } => {
                let previous = self.execution;
                self.execution = Some(*placement);
                let entry = self.block();
                self.execution = previous;
                let continuation = self.block();
                self.terminate(Terminator::Goto(entry, Vec::new()));

                self.execution = Some(*placement);
                self.current = entry;
                self.lower_statements(&body.statements, module);
                let reaches_continuation = self.open(self.current);
                if reaches_continuation {
                    self.terminate(Terminator::Goto(continuation, Vec::new()));
                }

                self.execution = previous;
                self.current = continuation;
                if !reaches_continuation {
                    self.terminate(Terminator::Unreachable);
                }
            }
            severian_hir::Statement::FieldSet {
                binding,
                field,
                value,
            } => {
                let mut place = self.bindings[binding].clone();
                place.projection.push(Projection::Field(*field));
                let value = Operand::Copy(self.expression(value));
                self.push(Statement::Assign(place, Rvalue::Use(value)));
            }
            severian_hir::Statement::FieldUpdate {
                binding,
                field,
                operator,
                value,
            } => {
                let mut place = self.bindings[binding].clone();
                place.projection.push(Projection::Field(*field));
                let right = Operand::Copy(self.expression(value));
                self.push(Statement::Assign(
                    place.clone(),
                    Rvalue::Binary {
                        operator: *operator,
                        left: Operand::Copy(place),
                        right,
                    },
                ));
            }
            severian_hir::Statement::Binding(id) => {
                let binding = module
                    .bindings
                    .iter()
                    .find(|binding| binding.id == *id)
                    .expect("typed HIR binding exists");
                let value = self.expression(&binding.value);
                let place = if let Some(place) = self.variables.get(&binding.variable) {
                    place.clone()
                } else {
                    let local = self.local(binding.type_id, binding.mutable, false);
                    let place = Place::local(local);
                    self.push(Statement::StorageLive(local));
                    self.variables.insert(binding.variable, place.clone());
                    place
                };
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
                condition,
                message,
                span,
                condition_span,
            } => {
                let condition = Operand::Copy(self.expression(condition));
                let message = message
                    .as_ref()
                    .map(|message| Operand::Copy(self.expression(message)));
                self.push(Statement::Assert {
                    condition,
                    message,
                    origin: AssertionOrigin {
                        statement_start: span.start,
                        condition_start: condition_span.start,
                        condition_end: condition_span.end,
                        location: None,
                    },
                });
            }
            severian_hir::Statement::ExpectThrow {
                body,
                boolean_type,
                span,
            } => {
                let caught = self.block();
                let completed = self.block();
                let join = self.block();
                self.catch_targets.push(CatchTarget::Discard(caught));
                self.lower_statements(&body.statements, module);
                self.catch_targets.pop();
                if self.open(self.current) {
                    self.terminate(Terminator::Goto(completed, Vec::new()));
                }

                self.current = caught;
                self.terminate(Terminator::Goto(join, Vec::new()));

                self.current = completed;
                let failed = self.local(*boolean_type, false, false);
                let failed_place = Place::local(failed);
                self.push(Statement::StorageLive(failed));
                self.push(Statement::Assign(
                    failed_place.clone(),
                    Rvalue::Use(Operand::Constant {
                        value: severian_universal::LiteralValue::Boolean(false),
                        ty: *boolean_type,
                    }),
                ));
                self.push(Statement::Assert {
                    condition: Operand::Copy(failed_place),
                    message: None,
                    origin: AssertionOrigin {
                        statement_start: span.start,
                        condition_start: span.start,
                        condition_end: span.end,
                        location: None,
                    },
                });
                self.terminate(Terminator::Goto(join, Vec::new()));
                self.current = join;
            }
            severian_hir::Statement::Try {
                body,
                catch_binding,
                catch_body,
                ..
            } => {
                let binding = module
                    .bindings
                    .iter()
                    .find(|binding| binding.id == *catch_binding)
                    .expect("typed HIR catch binding exists");
                let local = self.local(binding.type_id, false, false);
                let catch_place = Place::local(local);
                self.push(Statement::StorageLive(local));
                self.variables.insert(binding.variable, catch_place.clone());
                self.bindings.insert(*catch_binding, catch_place.clone());

                let caught = self.block();
                let join = self.block();
                self.catch_targets.push(CatchTarget::Bind {
                    block: caught,
                    place: catch_place,
                });
                self.lower_statements(&body.statements, module);
                self.catch_targets.pop();
                let mut reaches_join = false;
                if self.open(self.current) {
                    self.terminate(Terminator::Goto(join, Vec::new()));
                    reaches_join = true;
                }

                self.current = caught;
                self.lower_statements(&catch_body.statements, module);
                if self.open(self.current) {
                    self.terminate(Terminator::Goto(join, Vec::new()));
                    reaches_join = true;
                }
                self.current = join;
                if !reaches_join {
                    self.terminate(Terminator::Unreachable);
                }
            }
            severian_hir::Statement::If {
                condition,
                then_block,
                else_block,
            } => {
                let condition_source = condition.span.source;
                let condition_span = condition.span.start;
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
                self.push(Statement::Coverage(CoveragePoint {
                    source: condition_source,
                    span_start: condition_span,
                    kind: CoverageKind::Branch,
                    ordinal: 0,
                    key: None,
                    file: None,
                    line: None,
                }));
                self.lower_statements(&then_block.statements, module);
                let mut reaches_join = false;
                if self.open(self.current) {
                    self.terminate(Terminator::Goto(join, Vec::new()));
                    reaches_join = true;
                }
                self.bindings.clone_from(&bindings);
                self.current = else_id;
                self.push(Statement::Coverage(CoveragePoint {
                    source: condition_source,
                    span_start: condition_span,
                    kind: CoverageKind::Branch,
                    ordinal: 1,
                    key: None,
                    file: None,
                    line: None,
                }));
                self.lower_statements(&else_block.statements, module);
                if self.open(self.current) {
                    self.terminate(Terminator::Goto(join, Vec::new()));
                    reaches_join = true;
                }
                self.bindings = bindings;
                self.current = join;
                if !reaches_join {
                    self.terminate(Terminator::Unreachable);
                }
            }
            severian_hir::Statement::While {
                condition, body, ..
            } => {
                let header = self.block();
                let body_block = self.block();
                let exit = self.block();
                self.terminate(Terminator::Goto(header, Vec::new()));

                self.current = header;
                let condition = Operand::Copy(self.expression(condition));
                self.terminate(Terminator::Branch {
                    condition,
                    then_block: body_block,
                    else_block: exit,
                });

                self.loops.push(LoopTargets {
                    break_target: exit,
                    continue_target: header,
                });
                self.current = body_block;
                self.lower_statements(&body.statements, module);
                if self.open(self.current) {
                    self.terminate(Terminator::Goto(header, Vec::new()));
                }
                self.loops.pop();
                self.current = exit;
            }
            severian_hir::Statement::Break { .. } => {
                let target = self
                    .loops
                    .last()
                    .expect("semantic analysis rejects break outside loops")
                    .break_target;
                self.terminate(Terminator::Goto(target, Vec::new()));
            }
            severian_hir::Statement::Continue { .. } => {
                let target = self
                    .loops
                    .last()
                    .expect("semantic analysis rejects continue outside loops")
                    .continue_target;
                self.terminate(Terminator::Goto(target, Vec::new()));
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
                let mut reaches_join = false;
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
                        reaches_join = true;
                    }
                }
                self.bindings = bindings;
                self.current = join;
                if !reaches_join {
                    self.terminate(Terminator::Unreachable);
                }
            }
        }
    }

    fn expression(&mut self, expression: &severian_hir::Expression) -> Place {
        if let Some(place) = self.expressions.get(&(self.current, expression.id)) {
            return place.clone();
        }
        self.current_span = Some(expression.span);
        if let severian_hir::ExpressionKind::Async {
            expression: task,
            owner,
            locked,
        } = &expression.kind
        {
            let result = Place::local(self.local(expression.type_id, false, false));
            self.push(Statement::StorageLive(
                result.local_id().expect("task results are local places"),
            ));
            let severian_hir::ExpressionKind::Call { callee, arguments } = &task.kind else {
                panic!("async expressions are required to contain a call")
            };
            let arguments = arguments
                .iter()
                .map(|argument| Operand::Copy(self.expression(argument)))
                .collect();
            let continuation = self.block();
            let callee = self.callee(callee);
            self.terminate(Terminator::Spawn {
                callee,
                arguments,
                destination: result.clone(),
                target: continuation,
                owner: *owner,
                locked: *locked,
            });
            self.current = continuation;
            self.expressions
                .insert((self.current, expression.id), result.clone());
            return result;
        }
        if let severian_hir::ExpressionKind::AsyncFieldUpdate {
            binding,
            field,
            operator,
            value,
            owner,
            locked,
        } = &expression.kind
        {
            let result = Place::local(self.local(expression.type_id, false, false));
            self.push(Statement::StorageLive(
                result.local_id().expect("task results are local places"),
            ));
            let mut place = self.bindings[binding].clone();
            place.projection.push(Projection::Field(*field));
            let value = Operand::Copy(self.expression(value));
            let continuation = self.block();
            self.terminate(Terminator::SpawnFieldUpdate {
                place,
                operator: *operator,
                value,
                destination: result.clone(),
                target: continuation,
                owner: *owner,
                locked: *locked,
            });
            self.current = continuation;
            self.expressions
                .insert((self.current, expression.id), result.clone());
            return result;
        }
        if let severian_hir::ExpressionKind::Await(task) = &expression.kind {
            let result = Place::local(self.local(expression.type_id, false, false));
            self.push(Statement::StorageLive(
                result.local_id().expect("await results are local places"),
            ));
            let task = Operand::Copy(self.expression(task));
            self.push(Statement::Assign(result.clone(), Rvalue::Await { task }));
            self.expressions
                .insert((self.current, expression.id), result.clone());
            return result;
        }
        let result = Place::local(self.local(expression.type_id, false, false));
        self.push(Statement::StorageLive(
            result
                .local_id()
                .expect("expression results are local places"),
        ));
        match &expression.kind {
            severian_hir::ExpressionKind::Aggregate { class, fields } => {
                let fields = fields
                    .iter()
                    .map(|field| Operand::Copy(self.expression(field)))
                    .collect();
                self.push(Statement::Assign(
                    result.clone(),
                    Rvalue::Aggregate {
                        type_id: *class,
                        fields,
                    },
                ));
            }
            severian_hir::ExpressionKind::Field { object, index } => {
                let mut field = self.expression(object);
                field.projection.push(Projection::Field(*index));
                self.push(Statement::Assign(
                    result.clone(),
                    Rvalue::Use(Operand::Copy(field)),
                ));
            }
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
                        conversion: *conversion,
                    },
                ));
            }
            severian_hir::ExpressionKind::Fallback {
                condition,
                value,
                fallback,
            } => {
                let condition = Operand::Copy(self.expression(condition));
                let present = self.block();
                let absent = self.block();
                let join = self.block();
                self.terminate(Terminator::Branch {
                    condition,
                    then_block: present,
                    else_block: absent,
                });

                self.current = present;
                let selected = self.expression(value);
                self.push(Statement::Assign(
                    result.clone(),
                    Rvalue::Use(Operand::Copy(selected)),
                ));
                self.terminate(Terminator::Goto(join, Vec::new()));

                self.current = absent;
                let selected = self.expression(fallback);
                if self.open(self.current) {
                    self.push(Statement::Assign(
                        result.clone(),
                        Rvalue::Use(Operand::Copy(selected)),
                    ));
                    self.terminate(Terminator::Goto(join, Vec::new()));
                }
                self.current = join;
            }
            severian_hir::ExpressionKind::Throw(error) => {
                let error = Operand::Copy(self.expression(error));
                if let Some(catch) = self.catch_targets.last().cloned() {
                    let block = match catch {
                        CatchTarget::Discard(block) => block,
                        CatchTarget::Bind { block, place } => {
                            self.push(Statement::Assign(place, Rvalue::Use(error)));
                            block
                        }
                    };
                    self.terminate(Terminator::Goto(block, Vec::new()));
                } else {
                    self.terminate(Terminator::Throw(error));
                }
                self.current = self.block();
                self.terminate(Terminator::Unreachable);
            }
            severian_hir::ExpressionKind::Borrow { operand, exclusive } => {
                let source = match &operand.kind {
                    severian_hir::ExpressionKind::Binding(binding) => {
                        self.bindings[binding].clone()
                    }
                    _ => self.expression(operand),
                };
                self.push(Statement::Assign(
                    result.clone(),
                    if *exclusive {
                        Rvalue::BorrowExclusive(source)
                    } else {
                        Rvalue::BorrowShared(source)
                    },
                ));
            }
            severian_hir::ExpressionKind::AddressOf(binding) => {
                let source = self.bindings[binding].clone();
                self.push(Statement::Assign(result.clone(), Rvalue::AddressOf(source)));
            }
            severian_hir::ExpressionKind::Move(operand) => {
                let source = match &operand.kind {
                    severian_hir::ExpressionKind::Binding(binding) => {
                        self.bindings[binding].clone()
                    }
                    _ => self.expression(operand),
                };
                self.push(Statement::Assign(
                    result.clone(),
                    Rvalue::Use(Operand::Move(source)),
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
                    .collect::<Vec<_>>();
                if let HirCallee::Intrinsic {
                    operation,
                    attributes,
                } = callee
                {
                    self.push(Statement::Operation {
                        id: *operation,
                        operands: arguments,
                        results: vec![result.clone()],
                        attributes: attributes.clone(),
                    });
                    self.expressions
                        .insert((self.current, expression.id), result.clone());
                    return result;
                }
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
            severian_hir::ExpressionKind::Async { expression, .. }
            | severian_hir::ExpressionKind::Await(expression) => {
                let source = self.expression(expression);
                self.push(Statement::Assign(
                    result.clone(),
                    Rvalue::Use(Operand::Copy(source)),
                ));
            }
            severian_hir::ExpressionKind::AsyncFieldUpdate { .. } => {
                unreachable!("async field updates are lowered before ordinary expressions")
            }
        }
        self.expressions
            .insert((self.current, expression.id), result.clone());
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
            HirCallee::FunctionValue(expression) => Callee::FunctionValue(Operand::Copy(
                self.expressions[&(self.current, *expression)].clone(),
            )),
            HirCallee::Method {
                implementation,
                receiver,
                substitution,
            } => Callee::Method {
                implementation: *implementation,
                receiver: Operand::Copy(self.expressions[&(self.current, *receiver)].clone()),
                substitution: substitution.clone(),
            },
            HirCallee::Constructor { type_def, variant } => Callee::Constructor {
                type_def: *type_def,
                variant: *variant,
            },
            HirCallee::Intrinsic { operation, .. } => Callee::Intrinsic(*operation),
        }
    }
}
