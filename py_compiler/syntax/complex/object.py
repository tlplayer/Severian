from dataclasses import dataclass, field


@dataclass(frozen=True)
class ObjectType:
    name: str
    family: str
    mlir: str
    bits: int = 0
    signed: bool = False
    variants: tuple = ()
    declaration: object = None
    owns_fields: bool = False

    @property
    def parameter_ownership(self):
        return getattr(self.declaration, 'parameter_ownership', 'view')

    @property
    def binding_ownership(self):
        if self.declaration is None:
            raise ValueError(f"{self.name} has no defining ownership contract")
        return self.declaration.binding_ownership

    def require_context(self, cfg):
        pass

    @property
    def release(self):
        if not self.owns_fields and not getattr(self.declaration, 'destructor', False):
            raise AttributeError("record has no owned fields")
        return self._release

    def _release(self, value, cfg, span):
        cfg.context.provider_by_name[self.name].release(value, cfg, span)

    def element_place(self,receiver,index,cfg,env,span):
        return IndexedMethod(receiver,index,cfg,env)

    def slice_value(self, receiver, bounds, cfg, env, span, owner=None, *, closed=(True, True)):
        from py_compiler.syntax.grammar.expression import Expression
        from py_compiler.syntax.function.calls import literal_value, select, invoke
        from py_compiler.syntax.complex.tensor import ArgumentValue
        from py_compiler.frontend.lexer.lexer import lex
        from py_compiler.frontend.source.source import SourceFile
        entries = cfg.context.functions.get(self.name + '.__slice', ())
        if not entries:
            raise ValueError('type does not supply a slicing operation')
        token = lex(SourceFile(cfg.source.path + ':slice', 'slice'), cfg.syntax)[0]
        start, stop, step = bounds
        values = [cfg.expr(bound, env, cfg.context.type('int')) if bound else
                  literal_value(default, cfg.context.type('int'), cfg, span)
                  for bound, default in ((start, 0), (stop, 0), (step, 1))]
        values.extend(literal_value(bound is None, cfg.context.type('bool'), cfg, span)
                      for bound in (start, stop))
        values.extend(literal_value(flag, cfg.context.type('bool'), cfg, span) for flag in closed)
        arguments = tuple(Expression('capture', token, (ArgumentValue(value),)) for value in values)
        entry, operands = select(entries, arguments, cfg, env)
        return invoke(entry, operands, cfg, span, receiver)

    def copy_value(self, value, cfg, span):
        return cfg.context.provider_by_name[self.name].copy_value(value, cfg, span)

    def grammars(self):
        if self.declaration is None or not callable(getattr(self.declaration, "grammars", None)):
            raise ValueError(f"{self.name} has no defining object supplying grammar")
        return self.declaration.grammars()

    @property
    def generic_arguments(self):
        return getattr(self.declaration, 'arguments', ())



@dataclass(frozen=True)
class NamedDefinition:
    name: str
    binding_ownership: str = "view"

    def grammars(self):
        from py_compiler.syntax.generic.owned import WordGrammar
        return (WordGrammar(self, self.name, "IDENTIFIER"),)


@dataclass(frozen=True)
class Allocate:
    record_type: str
    plan: object = None

    def atom(self, operation):
        from py_compiler.syntax.generic.atom import Atom
        from py_compiler.syntax.primitive.catalog import primitives
        return Atom(self, (), operation.result.type, (), ('allocation',), self)

    def entry_allocations(self, operation):
        # Record storage is a fixed function-local slot. Initialization still
        # executes at the original declaration on every loop iteration.
        return self.render_operation(operation)[:2]

    def render_operation(self, operation):
        from py_compiler.mlir.src.cfg import name
        value = operation.result
        alignment = f' {{alignment = {self.plan.alignment} : i64}}' if self.plan else ''
        return [f"%count_{value.identity} = arith.constant 1 : i64",
                f"{name(value)} = llvm.alloca %count_{value.identity} x {self.record_type}{alignment} : (i64) -> !llvm.ptr",
                f"%record_empty_{value.identity} = llvm.mlir.zero : {self.record_type}",
                f"llvm.store %record_empty_{value.identity}, {name(value)} : {self.record_type}, !llvm.ptr"]


@dataclass(frozen=True)
class Invoke:
    symbol: str
    result_type: object
    ownership: tuple = ()
    result_layout: str = ""
    dependency: object = field(default=None, compare=False, repr=False)

    def atom(self, operation):
        from py_compiler.syntax.generic.atom import Atom
        return Atom(self, tuple(v.type for v in operation.operands), self.result_type if self.result_type.mlir else None,
                    self.ownership or tuple('view' for _ in operation.operands), ('call',), self)

    def entry_allocations(self, operation):
        return self.render_operation(operation)[1:3] if self.result_layout else []

    def render_operation(self, operation):
        from py_compiler.mlir.src.cfg import name
        arguments = ", ".join(name(v) for v in operation.operands)
        types = ", ".join(v.type.mlir for v in operation.operands)
        result = self.result_layout or self.result_type.mlir or "()"
        prefix = name(operation.result) + " = " if operation.result else ""
        from py_compiler.mlir.src.lib import symbol_ref
        if self.result_layout:
            tag = operation.result.identity
            return [f'%record_result_{tag} = func.call {symbol_ref(self.symbol)}({arguments}) : ({types}) -> {result}',
                    f'%record_count_{tag} = arith.constant 1 : i64',
                    f'{name(operation.result)} = llvm.alloca %record_count_{tag} x {result} : (i64) -> !llvm.ptr',
                    f'llvm.store %record_result_{tag}, {name(operation.result)} : {result}, !llvm.ptr']
        return [f"{prefix}func.call {symbol_ref(self.symbol)}({arguments}) : ({types}) -> {result}"]


@dataclass(frozen=True)
class TraitBox:
    tag: int

    def atom(self, operation):
        from py_compiler.syntax.generic.atom import Atom
        from py_compiler.syntax.primitive.catalog import primitives
        return Atom(self, (operation.operands[0].type,), operation.result.type, ('view',), (), self)

    def render_operation(self, operation):
        from py_compiler.mlir.src.cfg import name
        v = operation.result
        t = v.type.mlir
        return [f"%box_{v.identity} = llvm.mlir.undef : {t}",
                f"%tag_{v.identity} = arith.constant {self.tag} : i32",
                f"%data_{v.identity} = llvm.insertvalue {name(operation.operands[0])}, %box_{v.identity}[0] : {t}",
                f"{name(v)} = llvm.insertvalue %tag_{v.identity}, %data_{v.identity}[1] : {t}"]


@dataclass(frozen=True)
class Extract:
    index: int

    def atom(self, operation):
        from py_compiler.syntax.generic.atom import Atom
        return Atom(self, (operation.operands[0].type,), operation.result.type, ('view',), (), self)

    def render_operation(self, operation):
        from py_compiler.mlir.src.cfg import name
        return [f"{name(operation.result)} = llvm.extractvalue {name(operation.operands[0])}[{self.index}] : {operation.operands[0].type.mlir}"]


@dataclass(frozen=True)
class Assert:
    message: str

    def atom(self, operation):
        from py_compiler.syntax.generic.atom import Atom
        from py_compiler.syntax.primitive.catalog import primitives
        return Atom(self, (primitives()['bool'],), None, ('view',), ('panic',), self)

    def render_operation(self, operation):
        from py_compiler.mlir.src.cfg import name
        from py_compiler.mlir.src.lib import quoted
        return [f"cf.assert {name(operation.operands[0])}, {quoted(self.message)}"]


@dataclass(frozen=True)
class BooleanAnd:
    def atom(self, operation):
        from py_compiler.syntax.generic.atom import Atom
        from py_compiler.syntax.primitive.catalog import primitives
        boolean = primitives()['bool']
        return Atom(self, (boolean, boolean), boolean, ('copy', 'copy'), (), self)

    def render_operation(self, operation):
        from py_compiler.mlir.src.cfg import name
        return [f'{name(operation.result)} = arith.andi {name(operation.operands[0])}, {name(operation.operands[1])} : i1']


@dataclass(frozen=True)
class CountBoolean:
    def atom(self, operation):
        from py_compiler.syntax.generic.atom import Atom
        from py_compiler.syntax.primitive.catalog import primitives
        types = primitives()
        return Atom(self, (types['bool'],), types['i32'], ('copy',), (), self)

    def render_operation(self, operation):
        from py_compiler.mlir.src.cfg import name
        return [f'{name(operation.result)} = arith.extui {name(operation.operands[0])} : i1 to i32']


from py_compiler.syntax.block.union import UnionBox


@dataclass(frozen=True)
class IndexedMethod:
    receiver: object
    index: object
    cfg: object
    env: object

    @property
    def type(self):
        entries=self.cfg.context.functions.get(self.receiver.type.name+'.get',())
        types={e.result for e in entries}
        if len(types)!=1:
            raise ValueError('indexed getter requires one concrete result contract')
        return self.cfg.context.type(types.pop())

    def read(self,cfg,span):
        from py_compiler.syntax.function.calls import select,invoke
        entries=cfg.context.functions.get(self.receiver.type.name+'.get',())
        entry,values=select(entries,(self.index,),cfg,self.env)
        return invoke(entry,values,cfg,span,self.receiver)

    def preserves_shape(self, cfg):
        from py_compiler.syntax.complex.collection import indexed_write_preserves_shape
        return indexed_write_preserves_shape(self.receiver.type, cfg.context)

    def read_view(self, cfg, span, mode='view'):
        from py_compiler.syntax.function.calls import select, invoke
        entries = cfg.context.functions.get(self.receiver.type.name + ('.borrow_at' if mode == 'borrow' else '.view_at'), ())
        entry, values = select(entries, (self.index,), cfg, self.env)
        return invoke(entry, values, cfg, span, self.receiver)

    def take(self, cfg, span, value=None):
        from py_compiler.syntax.function.calls import select, invoke
        entries = cfg.context.functions.get(self.receiver.type.name + '.take', ())
        entry, values = select(entries, (self.index,), cfg, self.env)
        return invoke(entry, values, cfg, span, self.receiver)

    def write(self,cfg,value,span):
        from py_compiler.syntax.function.calls import select,invoke
        from py_compiler.syntax.grammar.expression import Expression
        from py_compiler.syntax.complex.tensor import ArgumentValue
        entries=cfg.context.functions.get(self.receiver.type.name+'.set',())
        owner = next((binding for binding in self.env.values()
                      if cfg.values.get(binding.identity) == value and binding.identity not in cfg.flow.moved),
                     cfg.call_result_owners.get(value.identity))
        entry,values=select(entries,(self.index,Expression('capture',self.index.token,(ArgumentValue(value, owner),))),cfg,self.env)
        return invoke(entry,values,cfg,span,self.receiver)
