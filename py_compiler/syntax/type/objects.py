from dataclasses import dataclass


@dataclass(frozen=True)
class ObjectType:
    name: str
    family: str
    mlir: str
    bits: int = 0
    signed: bool = False
    binding_ownership: str = "view"
    variants: tuple = ()


@dataclass(frozen=True)
class Allocate:
    record_type: str

    def render_operation(self, operation):
        from py_compiler.mlir.src.cfg import name
        value = operation.result
        return [f"%count_{value.identity} = arith.constant 1 : i64",
                f"{name(value)} = llvm.alloca %count_{value.identity} x {self.record_type} : (i64) -> !llvm.ptr"]


@dataclass(frozen=True)
class Invoke:
    symbol: str
    result_type: object

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

    def render_operation(self, operation):
        from py_compiler.mlir.src.cfg import name
        return [f"{name(operation.result)} = llvm.extractvalue {name(operation.operands[0])}[{self.index}] : {operation.operands[0].type.mlir}"]


@dataclass(frozen=True)
class Assert:
    message: str

    def render_operation(self, operation):
        from py_compiler.mlir.src.cfg import name
        from py_compiler.mlir.src.lib import quoted
        return [f"cf.assert {name(operation.operands[0])}, {quoted(self.message)}"]


@dataclass(frozen=True)
class BooleanAnd:
    def render_operation(self, operation):
        from py_compiler.mlir.src.cfg import name
        return [f'{name(operation.result)} = arith.andi {name(operation.operands[0])}, {name(operation.operands[1])} : i1']


@dataclass(frozen=True)
class CountBoolean:
    def render_operation(self, operation):
        from py_compiler.mlir.src.cfg import name
        return [f'{name(operation.result)} = arith.extui {name(operation.operands[0])} : i1 to i32']


@dataclass(frozen=True)
class UnionBox:
    variant: int

    def render_operation(self, operation):
        from py_compiler.mlir.src.cfg import name
        value = operation.result
        type_ = value.type.mlir
        return [f'%union_{value.identity} = llvm.mlir.zero : {type_}',
                f'%union_tag_{value.identity} = arith.constant {self.variant} : i32',
                f'%union_value_{value.identity} = llvm.insertvalue {name(operation.operands[0])}, %union_{value.identity}[{self.variant + 1}] : {type_}',
                f'{name(value)} = llvm.insertvalue %union_tag_{value.identity}, %union_value_{value.identity}[0] : {type_}']
