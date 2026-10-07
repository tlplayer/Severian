"""Dynamic values forward a concrete type descriptor and a borrowed payload address."""
from dataclasses import dataclass
from hashlib import sha256
from py_compiler.syntax.generic.atom import Atom


@dataclass(frozen=True)
class DynamicType:
    name: str = 'dynamic'
    family: str = 'dynamic'
    mlir: str = '!llvm.struct<(ptr, ptr)>'
    bits: int = 0
    binding_ownership: str = 'view'
    parameter_ownership: str = 'view'

    def grammars(self):
        from py_compiler.syntax.generic.owned import WordGrammar
        return (WordGrammar(self, self.name, 'IDENTIFIER'),)

    def from_value(self, value, cfg, span):
        if value.type == self:
            return value
        storage = value.type.mlir
        if not (storage.startswith('i') and storage[1:].isdigit() or storage in ('f16', 'bf16', 'f32', 'f64', 'f80', 'f128', '!llvm.ptr')):
            raise ValueError(f'{value.type.name} has no dynamic payload storage implementation')
        result = cfg.emit('dynamic-box', self, (value,), DynamicBox(value.type, self), span)
        cfg.stack_values.add(result.identity)
        return result


@dataclass(frozen=True)
class DynamicBox:
    concrete: object
    owner: object

    @property
    def symbol(self):
        return '__sev_type_' + sha256((self.concrete.name + ':' + self.concrete.mlir).encode()).hexdigest()

    def global_definitions(self):
        from py_compiler.mlir.src.lib import quoted
        descriptor = self.concrete.name + '\0' + self.concrete.mlir + '\0'
        return (f'llvm.mlir.global private constant @{self.symbol}({quoted(descriptor)}) : !llvm.array<{len(descriptor.encode())} x i8>',)

    def atom(self, operation):
        return Atom(self.owner, (self.concrete,), self.owner, ('view',), ('allocation',), self)

    def render_operation(self, operation):
        from py_compiler.mlir.src.cfg import name
        value, result = operation.operands[0], operation.result
        suffix = str(result.identity)
        aggregate = self.owner.mlir
        return [f'%dynamic_count_{suffix} = arith.constant 1 : i64',
                f'%dynamic_payload_{suffix} = llvm.alloca %dynamic_count_{suffix} x {self.concrete.mlir} : (i64) -> !llvm.ptr',
                f'llvm.store {name(value)}, %dynamic_payload_{suffix} : {self.concrete.mlir}, !llvm.ptr',
                f'%dynamic_descriptor_{suffix} = llvm.mlir.addressof @{self.symbol} : !llvm.ptr',
                f'%dynamic_empty_{suffix} = llvm.mlir.undef : {aggregate}',
                f'%dynamic_tagged_{suffix} = llvm.insertvalue %dynamic_descriptor_{suffix}, %dynamic_empty_{suffix}[0] : {aggregate}',
                f'{name(result)} = llvm.insertvalue %dynamic_payload_{suffix}, %dynamic_tagged_{suffix}[1] : {aggregate}']


import unittest


class DynamicTests(unittest.TestCase):
    def test_boxing_and_forwarding_retain_the_descriptor(self):
        from py_compiler.frontend.src.lib import compile_source
        from py_compiler.syntax.recognition import Syntax
        from py_compiler.mlir.src.lib import lower, render
        source = 'def identity[T](value: T) -> T:\n    return value\ndef work():\n    original: dynamic = 42\n    result = identity(original)\n'
        result = compile_source('dynamic.sev', source, Syntax())
        self.assertFalse(result.diagnostics, str(result.diagnostics))
        text = render(lower(result.program))
        self.assertIn('llvm.mlir.global private constant @__sev_type_', text)
        self.assertIn('!llvm.struct<(ptr, ptr)>', text)
        self.assertEqual(text.count('llvm.alloca'), 1)

    def test_boxed_local_cannot_escape(self):
        from py_compiler.frontend.src.lib import compile_source
        from py_compiler.syntax.recognition import Syntax
        result = compile_source('dynamic.sev', 'def bad() -> dynamic:\n    value: dynamic = 42\n    return value\n', Syntax())
        self.assertTrue(result.diagnostics)
        self.assertIn('storage belongs to this callable', str(result.diagnostics[0]))
