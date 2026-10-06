"""Object-supplied terminal operations; no further language expansion remains."""
from dataclasses import dataclass


@dataclass(frozen=True)
class ExternalSymbol:
    symbol: str
    abi: str
    dependencies: tuple

    def __post_init__(self):
        import re
        if not re.fullmatch(r"[A-Za-z_][A-Za-z_0-9.$]*", self.symbol):
            raise ValueError("external atom requires a valid symbol")
        if self.abi != "C":
            raise ValueError("unsupported external atom ABI: " + self.abi)
        if not self.dependencies or any(not p.endswith((".o", ".so")) for p in self.dependencies):
            raise ValueError("external atom requires .o/.so dependencies")

    def render_operation(self, operation):
        from py_compiler.mlir.src.cfg import name
        arguments = ", ".join(name(v) for v in operation.operands)
        types = ", ".join(v.type.mlir for v in operation.operands)
        result = operation.result.type.mlir if operation.result else "()"
        prefix = name(operation.result) + " = " if operation.result else ""
        return [f"{prefix}func.call @{self.symbol}({arguments}) : ({types}) -> {result}"]


@dataclass(frozen=True)
class Atom:
    owner: object
    inputs: tuple
    result: object
    ownership: tuple
    effects: tuple
    implementation: object

    def atom(self, operation):
        return self

    def verify(self, operation):
        if self.owner is None or not callable(getattr(self.implementation, "render_operation", None)):
            raise ValueError("atom requires an owning object and a lowering implementation")
        if tuple(v.type for v in operation.operands) != self.inputs:
            raise ValueError("atom input type contract mismatch")
        if (operation.result.type if operation.result else None) != self.result:
            raise ValueError("atom result type contract mismatch")
        if len(self.ownership) != len(self.inputs) or any(mode not in ("view", "copy", "borrow", "move") for mode in self.ownership):
            raise ValueError("atom requires ownership for every input")

    def render_operation(self, operation):
        self.verify(operation)
        return self.implementation.render_operation(operation)


import unittest


class AtomTests(unittest.TestCase):
    def test_input_and_result_contracts_are_checked(self):
        from py_compiler.mir.cfg.cfg import Operation, Value
        from py_compiler.syntax.primitive.catalog import primitives
        from py_compiler.syntax.generic.owned import select
        types = primitives()
        owner = types["i64"]
        grammar = select(owner, "F.operator", "+")
        operation = Operation("scalar", Value(2, owner), (Value(0, owner), Value(1, owner)), grammar, None)
        atom = grammar.atom(operation)
        atom.verify(operation)
        self.assertIn("arith.addi", atom.render_operation(operation)[0])
        from dataclasses import replace
        with self.assertRaisesRegex(ValueError, "input type"):
            atom.verify(replace(operation, operands=(Value(0, types["u8"]), Value(1, owner))))
        with self.assertRaisesRegex(ValueError, "result type"):
            atom.verify(replace(operation, result=Value(2, types["bool"])))

    def test_external_boundary_requires_supported_abi_and_dependency(self):
        with self.assertRaisesRegex(ValueError, "ABI"):
            ExternalSymbol("run", "unknown", ("run.o",))
        with self.assertRaisesRegex(ValueError, "dependencies"):
            ExternalSymbol("run", "C", ())


    def test_external_atom_survives_lir_with_link_requirements(self):
        from py_compiler.mir.cfg.cfg import Body, ExecutionBlock, Operation, Terminator, Value
        from py_compiler.hir.hir.src.program import Program
        from py_compiler.syntax.primitive.catalog import primitives
        from py_compiler.lir.src.lib import lower
        integer = primitives()['i64']
        argument, result = Value(0, integer), Value(1, integer)
        atom = Atom(integer, (integer,), integer, ('view',), ('IO',), ExternalSymbol('native_step', 'C', ('native.o',)))
        operation = Operation('external', result, (argument,), atom, None, atom)
        body = Body('run', 'run', 'atom.sev', integer, [ExecutionBlock(0, (argument,), [operation], Terminator('finish', value=result))])
        lir = lower(Program((), (), (body,)), 'x86_64-unknown-linux-gnu', 64)
        self.assertEqual(lir.dependencies, ('native.o',))
        self.assertIn('func.func private @native_step(i64) -> i64', lir.text())
        self.assertIn('func.call @native_step(%v0)', lir.text())

    def test_move_on_predecessor_rejects_later_use(self):
        from py_compiler.mir.cfg.cfg import Body, ExecutionBlock, Operation, Terminator, Edge, Value
        from py_compiler.syntax.primitive.catalog import primitives
        types = primitives()
        integer = types['i64']
        value = Value(0, integer)
        consume = Atom(integer, (integer,), None, ('move',), (), ExternalSymbol('consume', 'C', ('consume.o',)))
        operation = Operation('external', None, (value,), consume, None, consume)
        body = Body('run', 'run', 'atom.sev', integer, [
            ExecutionBlock(0, (value,), [operation], Terminator('jump', (Edge(1),))),
            ExecutionBlock(1, (), [], Terminator('finish', value=value))])
        with self.assertRaisesRegex(ValueError, 'moved'):
            body.verify()
