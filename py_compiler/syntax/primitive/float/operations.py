"""Floating operators preserve their concrete format and floor modulo."""
from py_compiler.syntax.primitive.numeric.grammar import ScalarOperation, CompoundAssignment


class FloatOperation(ScalarOperation):
    def __init__(self, owner, spelling, precedence):
        super().__init__(owner, spelling, precedence, floating=True)
        self.associativity = 'right' if spelling == '**' else 'left'

    def render_operation(self, operation):
        from py_compiler.mlir.src.cfg import name
        left, right = map(name, operation.operands)
        result, type_ = name(operation.result), self.owner.mlir
        if self.spelling == '**':
            return [f'{result} = math.powf {left}, {right} : {type_}']
        if self.spelling == '/':
            return [f'{result} = arith.divf {left}, {right} : {type_}']
        lines = [f'{result}_quotient = arith.divf {left}, {right} : {type_}']
        if self.spelling == '//':
            return lines + [f'{result} = math.floor {result}_quotient : {type_}']
        return lines + [f'{result}_floor = math.floor {result}_quotient : {type_}',
                        f'{result}_product = arith.mulf {result}_floor, {right} : {type_}',
                        f'{result} = arith.subf {left}, {result}_product : {type_}']


def grammars(owner):
    operations = (('/', 5), ('//', 5), ('%', 5), ('**', 7))
    return (*[FloatOperation(owner, op, precedence) for op, precedence in operations],
            *[CompoundAssignment(owner, op) for op, _ in operations])


import unittest


class FloatOperationTests(unittest.TestCase):
    def compile(self, source):
        from py_compiler.frontend.src.lib import compile_source
        from py_compiler.syntax.recognition import Syntax
        result = compile_source('float-operators.sev', source, Syntax())
        self.assertFalse(result.diagnostics, str(result.diagnostics))
        return result.program

    def test_every_float_format_has_owned_lowering(self):
        from py_compiler.syntax.primitive.catalog import primitives
        from py_compiler.mlir.src.lib import lower, render
        for owner in set(primitives().values()):
            if owner.family != 'float':
                continue
            for spelling, opcode in (('/', 'arith.divf'), ('//', 'math.floor'),
                                     ('%', 'arith.subf'), ('**', 'math.powf')):
                with self.subTest(type=owner.name, operator=spelling):
                    program = self.compile(f'def calculate(a: {owner.name}, b: {owner.name}) -> {owner.name}:\n    return a {spelling} b\n')
                    self.assertIn(opcode, render(lower(program)))

    def test_bitwise_operators_require_integer_owner(self):
        from py_compiler.frontend.src.lib import compile_source
        from py_compiler.syntax.recognition import Syntax
        for expression in ('a & b', 'a | b', 'a ^ b', 'a << b', 'a >> b', '~a'):
            result = compile_source('reject.sev', f'def calculate(a: float, b: float) -> float:\n    return {expression}\n', Syntax())
            self.assertTrue(result.diagnostics, expression)

    def test_native_float_results(self):
        import ctypes
        import shutil
        import subprocess
        import tempfile
        from pathlib import Path
        from py_compiler.lir.src.lib import lower, compile_object
        from py_compiler.mlir.src.lib import verifier_path
        if not shutil.which(verifier_path()) or not shutil.which('cc'):
            self.skipTest('install MLIR/LLVM tools and cc for native execution')
        source = ''.join(f'def {label}(a: float, b: float) -> float:\n    return a {op} b\n'
                         for label, op in (('divide', '/'), ('floor', '//'), ('modulo', '%'), ('power', '**')))
        program = self.compile(source)
        artifact = compile_object(lower(program, 'x86_64-unknown-linux-gnu', 64))
        with tempfile.TemporaryDirectory() as directory:
            obj, shared = Path(directory)/'operators.o', Path(directory)/'operators.so'
            obj.write_bytes(artifact.object_bytes)
            subprocess.run(['cc', '-shared', str(obj), '-o', str(shared), '-lm'], check=True, capture_output=True)
            library = ctypes.CDLL(str(shared))
            symbols = {body.declaration: body.name for body in program.bodies if body.declaration}
            for label, a, b, expected in (('divide', 7.5, 2.0, 3.75), ('floor', -7.0, 2.0, -4.0),
                                          ('modulo', -7.0, 2.0, 1.0), ('modulo', 7.0, -2.0, -1.0),
                                          ('power', 2.0, -2.0, 0.25)):
                function = getattr(library, symbols[label])
                function.argtypes = (ctypes.c_double, ctypes.c_double)
                function.restype = ctypes.c_double
                self.assertEqual(function(a, b), expected)
