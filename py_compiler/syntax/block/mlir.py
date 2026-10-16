"""Library-owned, typed MLIR operations with literal attributes."""
import ast
import re
from dataclasses import dataclass
from py_compiler.syntax.generic.block import BlockProvider
from py_compiler.mlir.src.lib import quoted


@dataclass(frozen=True)
class Operation:
    name: str
    attributes: tuple = ()
    target: str = ''

    def validate(self, inputs, result):
        if not re.fullmatch(r'[a-zA-Z_][\w]*\.[\w.]+', self.name):
            raise ValueError('invalid MLIR operation name')
        if result and result.binding_ownership != 'copy':
            raise ValueError('@mlir results must have scalar copy ownership')
        if self.name == 'llvm.inline_asm':
            attrs = dict(self.attributes)
            if not self.target:
                raise ValueError('inline assembly requires an explicit target')
            if not isinstance(attrs.get('asm_string'), str) or not isinstance(attrs.get('constraints'), str):
                raise ValueError('inline assembly requires asm_string and constraints')
            if attrs.get('has_side_effects') is not True:
                raise ValueError('inline assembly requires has_side_effects=true')
            if any(t.mlir.startswith('memref') for t in inputs):
                raise ValueError('inline assembly requires scalar operands')

    def validate_target(self, target, pointer_bits):
        if self.target == 'x86_64-linux':
            if target.split('-')[0] != 'x86_64' or 'linux' not in target.split('-') or pointer_bits != 64:
                raise ValueError('MLIR operation requires x86_64 Linux with 64-bit pointers')
        elif self.target:
            raise ValueError('unsupported MLIR target contract: ' + self.target)

    def render_operation(self, operation):
        from py_compiler.mlir.src.cfg import name
        attributes = []
        for key, value in self.attributes:
            if isinstance(value, bool):
                if value:
                    attributes.append(key + ' = unit')
            elif isinstance(value, str):
                attributes.append(key + ' = ' + quoted(value))
            else:
                attributes.append(key + ' = ' + str(value) + ' : i64')
        attrs = ' {' + ', '.join(attributes) + '}' if attributes else ''
        args = ', '.join(name(v) for v in operation.operands)
        types = ', '.join(v.type.mlir for v in operation.operands)
        result = operation.result.type.mlir if operation.result else '()'
        prefix = name(operation.result) + ' = ' if operation.result else ''
        return [f'{prefix}{quoted(self.name)}({args}){attrs} : ({types}) -> {result}']


class Mlir(BlockProvider):
    spelling = '@'

    def has_body(self, node):
        return False

    def parse_header(self, items):
        text = ' '.join(t.text for t in items[1:])
        try:
            call = ast.parse(text, mode='eval').body
            if not isinstance(call, ast.Call) or not isinstance(call.func, ast.Name) or call.func.id != 'mlir' or len(call.args) != 1:
                raise ValueError('expected @mlir("dialect.operation", attributes...)')
            name = ast.literal_eval(call.args[0])
            if not isinstance(name, str):
                raise ValueError('MLIR operation name must be a string')
            attributes = {}
            for keyword in call.keywords:
                if keyword.arg is None or keyword.arg in attributes:
                    raise ValueError('MLIR attributes require unique literal names')
                value = keyword.value
                value = {'true': True, 'false': False}[value.id] if isinstance(value, ast.Name) and value.id in ('true', 'false') else ast.literal_eval(value)
                if not isinstance(value, (str, bool, int)):
                    raise ValueError('MLIR attributes require string, bool or integer literals')
                attributes[keyword.arg] = value
            target = attributes.pop('target', '')
            if not isinstance(target, str):
                raise ValueError('MLIR target must be a string')
            return Operation(name, tuple(attributes.items()), target)
        except (SyntaxError, TypeError) as failure:
            raise ValueError('invalid @mlir declaration') from failure


import unittest


class MlirTests(unittest.TestCase):
    def compile(self, source):
        from py_compiler.frontend.src.lib import compile_source
        from py_compiler.syntax.recognition import Syntax
        return compile_source('operations.sev', source, Syntax())

    def test_library_operation_is_typed_and_survives_mir(self):
        from py_compiler.mlir.src.lib import lower, render, verify_native, verifier_path
        result = self.compile('@mlir("arith.addi")\ndef add(a: int, b: int) -> int\ndef run() -> int:\n    return add(2, 3)\n')
        self.assertFalse(result.diagnostics, str(result.diagnostics))
        text = render(lower(result.program))
        self.assertIn('"arith.addi"', text)
        self.assertNotIn('__sev_io_', text)
        import shutil
        if shutil.which(verifier_path()):
            verify_native(text)

    def test_annotation_requires_a_bodyless_callable(self):
        for source in ('@mlir("arith.addi")\n', '@mlir("arith.addi")\ndef bad():\n    return\n'):
            self.assertTrue(self.compile(source).diagnostics)

    def test_inline_assembly_requires_target_and_effects(self):
        result = self.compile('@mlir("llvm.inline_asm", asm_string="syscall", constraints="={ax}")\ndef call() -> int\n')
        self.assertTrue(result.diagnostics)
        operation = Operation('llvm.inline_asm', target='x86_64-linux')
        for target, bits in (('aarch64-unknown-linux-gnu', 64), ('x86_64-apple-darwin', 64), ('x86_64-unknown-linux-gnu', 32)):
            with self.assertRaisesRegex(ValueError, 'requires x86_64 Linux'):
                operation.validate_target(target, bits)
