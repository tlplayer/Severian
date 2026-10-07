from dataclasses import dataclass


@dataclass(frozen=True)
class ObjectType:
    name: str
    family: str
    mlir: str
    bits: int = 0
    signed: bool = False
    variants: tuple = ()
    declaration: object = None

    @property
    def binding_ownership(self):
        if self.declaration is None:
            raise ValueError(f"{self.name} has no defining ownership contract")
        return self.declaration.binding_ownership

    def grammars(self):
        if self.declaration is None or not callable(getattr(self.declaration, "grammars", None)):
            raise ValueError(f"{self.name} has no defining object supplying grammar")
        return self.declaration.grammars()



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

    def atom(self, operation):
        from py_compiler.syntax.generic.atom import Atom
        from py_compiler.syntax.primitive.catalog import primitives
        return Atom(self, (), operation.result.type, (), ('allocation',), self)

    def render_operation(self, operation):
        from py_compiler.mlir.src.cfg import name
        value = operation.result
        return [f"%count_{value.identity} = arith.constant 1 : i64",
                f"{name(value)} = llvm.alloca %count_{value.identity} x {self.record_type} : (i64) -> !llvm.ptr"]


@dataclass(frozen=True)
class Invoke:
    symbol: str
    result_type: object

    def atom(self, operation):
        from py_compiler.syntax.generic.atom import Atom
        return Atom(self, tuple(v.type for v in operation.operands), self.result_type if self.result_type.mlir else None,
                    tuple('view' for _ in operation.operands), ('call',), self)

    def render_operation(self, operation):
        from py_compiler.mlir.src.cfg import name
        arguments = ", ".join(name(v) for v in operation.operands)
        types = ", ".join(v.type.mlir for v in operation.operands)
        result = self.result_type.mlir or "()"
        prefix = name(operation.result) + " = " if operation.result else ""
        return [f"{prefix}func.call @{self.symbol}({arguments}) : ({types}) -> {result}"]


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
