"""Integer-owned division, powers, shifts and bitwise operations."""
from py_compiler.syntax.primitive.numeric.grammar import ScalarOperation, CompoundAssignment, UnaryOperation


class IntegerOperation(ScalarOperation):
    def __init__(self, owner, spelling, precedence):
        super().__init__(owner, spelling, precedence)
        self.associativity = 'right' if spelling == '**' else 'left'

    def atom(self, operation):
        from dataclasses import replace
        atom = super().atom(operation)
        return replace(atom, effects=('panic',)) if self.spelling in ('/', '//', '%', '<<', '>>', '**') else atom

    def render_operation(self, operation):
        from py_compiler.mlir.src.cfg import name
        left, right = map(name, operation.operands)
        result, type_ = name(operation.result), self.owner.mlir
        temp = lambda suffix: f'{result}_{suffix}'
        lines = []
        def constant(suffix, value):
            lines.append(f'{temp(suffix)} = arith.constant {value} : {type_}')
            return temp(suffix)
        if self.spelling in ('/', '//', '%'):
            zero = constant('zero', 0)
            lines += [f'{temp("nonzero")} = arith.cmpi ne, {right}, {zero} : {type_}',
                      f'cf.assert {temp("nonzero")}, "integer division by zero"']
            if self.owner.signed:
                minimum = constant('minimum', -(1 << (self.owner.bits - 1)))
                negative_one = constant('negative_one', -1)
                lines += [f'{temp("not_min")} = arith.cmpi ne, {left}, {minimum} : {type_}',
                          f'{temp("not_neg_one")} = arith.cmpi ne, {right}, {negative_one} : {type_}',
                          f'{temp("safe")} = arith.ori {temp("not_min")}, {temp("not_neg_one")} : i1',
                          f'cf.assert {temp("safe")}, "integer division overflow"']
        if self.spelling in ('<<', '>>'):
            width = constant('width', self.owner.bits)
            lines += [f'{temp("valid_shift")} = arith.cmpi ult, {right}, {width} : {type_}',
                      f'cf.assert {temp("valid_shift")}, "shift count outside integer width"']
        if self.spelling == '**':
            zero, one = constant('zero', 0), constant('one', 1)
            if self.owner.signed:
                lines += [f'{temp("valid_power")} = arith.cmpi sge, {right}, {zero} : {type_}',
                          f'cf.assert {temp("valid_power")}, "integer exponent must be nonnegative"']
            # Exponentiation by squaring keeps unsigned exponents unsigned and
            # supports index and every declared integer width without narrowing.
            lines += [f'{temp("power")}:3 = scf.while ({temp("base")} = {left}, {temp("exponent")} = {right}, {temp("acc")} = {one}) : ({type_}, {type_}, {type_}) -> ({type_}, {type_}, {type_}) {{',
                      f'  {temp("more")} = arith.cmpi ne, {temp("exponent")}, {zero} : {type_}',
                      f'  scf.condition({temp("more")}) {temp("base")}, {temp("exponent")}, {temp("acc")} : {type_}, {type_}, {type_}',
                      '} do {',
                      f'^bb0({temp("b")}: {type_}, {temp("e")}: {type_}, {temp("a")}: {type_}):',
                      f'  {temp("bit")} = arith.andi {temp("e")}, {one} : {type_}',
                      f'  {temp("odd")} = arith.cmpi ne, {temp("bit")}, {zero} : {type_}',
                      f'  {temp("product")} = arith.muli {temp("a")}, {temp("b")} : {type_}',
                      f'  {temp("next_acc")} = arith.select {temp("odd")}, {temp("product")}, {temp("a")} : {type_}',
                      f'  {temp("square")} = arith.muli {temp("b")}, {temp("b")} : {type_}',
                      f'  {temp("half")} = arith.shrui {temp("e")}, {one} : {type_}',
                      f'  scf.yield {temp("square")}, {temp("half")}, {temp("next_acc")} : {type_}, {type_}, {type_}',
                      '}']
            # Use a separate multi-result name so the MIR result remains scalar.
            lines.append(f'{result} = arith.addi {temp("power")}#2, {zero} : {type_}')
            return lines
        signed = self.owner.signed
        opcode = {'/': 'divsi' if signed else 'divui',
                  '//': 'floordivsi' if signed else 'divui',
                  '%': 'remsi' if signed else 'remui',
                  '<<': 'shli', '>>': 'shrsi' if signed else 'shrui',
                  '&': 'andi', '|': 'ori', '^': 'xori'}[self.spelling]
        return lines + [f'{result} = arith.{opcode} {left}, {right} : {type_}']


class Complement(UnaryOperation):
    def render_operation(self, operation):
        from py_compiler.mlir.src.cfg import name
        result, operand = name(operation.result), name(operation.operands[0])
        return [f'{result}_ones = arith.constant -1 : {self.owner.mlir}',
                f'{result} = arith.xori {operand}, {result}_ones : {self.owner.mlir}']


def grammars(owner):
    operations = (('|', 3.1), ('^', 3.2), ('&', 3.3), ('<<', 3.5), ('>>', 3.5),
                  ('/', 5), ('//', 5), ('%', 5), ('**', 7))
    return (*[IntegerOperation(owner, op, precedence) for op, precedence in operations],
            *[CompoundAssignment(owner, op) for op, _ in operations], Complement(owner, '~'))


import unittest


class IntegerOperationTests(unittest.TestCase):
    def compile(self, source):
        from py_compiler.frontend.src.lib import compile_source
        from py_compiler.syntax.recognition import Syntax
        result = compile_source('integer-operators.sev', source, Syntax())
        self.assertFalse(result.diagnostics, str(result.diagnostics))
        return result.program

    def test_every_integer_representation_has_owned_lowering(self):
        from py_compiler.syntax.primitive.catalog import primitives
        from py_compiler.mlir.src.lib import lower, render
        for owner in set(primitives().values()):
            if owner.family != 'integer':
                continue
            for spelling, opcode in (('/', 'divsi' if owner.signed else 'divui'),
                                     ('//', 'floordivsi' if owner.signed else 'divui'),
                                     ('%', 'remsi' if owner.signed else 'remui'),
                                     ('<<', 'shli'), ('>>', 'shrsi' if owner.signed else 'shrui'),
                                     ('&', 'andi'), ('|', 'ori'), ('^', 'xori')):
                with self.subTest(type=owner.name, operator=spelling):
                    program = self.compile(f'def calculate(a: {owner.name}, b: {owner.name}) -> {owner.name}:\n    return a {spelling} b\n')
                    text = render(lower(program))
                    self.assertIn('arith.' + opcode, text)
                    operation = next(o for b in program.bodies for block in b.blocks for o in block.operations if isinstance(o.payload, IntegerOperation))
                    self.assertEqual(operation.atom.owner, owner)

    def test_guards_and_unsigned_power(self):
        from py_compiler.mlir.src.lib import lower, render
        for spelling, message in (('/', 'division by zero'), ('//', 'division overflow'),
                                  ('<<', 'shift count'), ('**', 'exponent must be nonnegative')):
            program = self.compile(f'def calculate(a: int, b: int) -> int:\n    return a {spelling} b\n')
            self.assertIn(message, render(lower(program)))
        program = self.compile('def calculate(a: u64, b: u64) -> u64:\n    return a ** b\n')
        text = render(lower(program))
        self.assertIn('scf.while', text)
        self.assertIn('arith.shrui', text)
        self.assertNotIn('exponent must be nonnegative', text)

    def test_native_results(self):
        import ctypes
        import shutil
        import subprocess
        import tempfile
        from pathlib import Path
        from py_compiler.lir.src.lib import lower, compile_object
        from py_compiler.mlir.src.lib import verifier_path
        if not shutil.which(verifier_path()) or not shutil.which('cc'):
            self.skipTest('install MLIR/LLVM tools and cc for native execution')
        cases = (('divide', 'a / b', -7, 2, -3), ('floor', 'a // b', -7, 2, -4),
                 ('remainder', 'a % b', -7, 2, -1), ('power', 'a ** b', 3, 5, 243),
                 ('zero_power', 'a ** b', 0, 0, 1), ('shift', 'a >> b', -8, 2, -2),
                 ('left_shift', 'a << b', 3, 2, 12), ('bits', '(a | b) ^ (a & b)', 6, 3, 5),
                 ('complement', '~a', 0, 0, -1), ('precedence', '2 ** 3 ** 2', 0, 0, 512),
                 ('negative_power_base', '-2 ** 2', 0, 0, -4))
        source = ''.join(f'def {label}(a: int, b: int) -> int:\n    return {expression}\n' for label, expression, *_ in cases)
        program = self.compile(source)
        artifact = compile_object(lower(program, 'x86_64-unknown-linux-gnu', 64))
        with tempfile.TemporaryDirectory() as directory:
            obj, shared = Path(directory)/'operators.o', Path(directory)/'operators.so'
            obj.write_bytes(artifact.object_bytes)
            subprocess.run(['cc', '-shared', str(obj), '-o', str(shared), '-lm'], check=True, capture_output=True)
            library = ctypes.CDLL(str(shared))
            symbols = {body.declaration: body.name for body in program.bodies if body.declaration}
            for label, _, a, b, expected in cases:
                with self.subTest(operation=label):
                    function = getattr(library, symbols[label])
                    function.argtypes = (ctypes.c_int64, ctypes.c_int64)
                    function.restype = ctypes.c_int64
                    self.assertEqual(function(a, b), expected)
