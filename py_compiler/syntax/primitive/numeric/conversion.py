"""Explicit scalar conversion edges; annotations do not request conversions."""
from dataclasses import dataclass
from decimal import Decimal
from py_compiler.syntax.generic.constructor import Constructor


def bounds(owner):
    return (-(1 << (owner.bits - 1)) if owner.signed else 0,
            (1 << (owner.bits - int(owner.signed))) - 1)


def literal_conversion(target, source, value):
    if source == target:
        return target.accept_literal(source, value)
    if source.family not in target.conversion_sources:
        raise ValueError(f'{source.name} has no declared conversion to {target.name}')
    if target.family == 'bool':
        return value != 0
    if target.family == 'float':
        return Decimal(value)
    if source.family == 'float' and not value.is_finite():
        raise ValueError('non-finite value cannot convert to an integer')
    result = int(value)
    low, high = bounds(target)
    if not low <= result <= high:
        raise ValueError(f'conversion is outside {target.name} range [{low}, {high}]')
    if target.family == 'char' and (result > 0x10ffff or 0xd800 <= result <= 0xdfff):
        raise ValueError('char requires a Unicode scalar value')
    return result


class ScalarConstructor(Constructor):
    def expand(self, cfg, arguments, env):
        if len(arguments) != 1:
            raise ValueError(f'{self.owner.name} constructor requires one value')
        self.owner.require_context(cfg)
        argument = arguments[0]
        if argument.kind == 'literal' or (argument.kind == 'unary' and argument.token.text in ('+', '-') and argument.operands[0].kind == 'literal'):
            from py_compiler.frontend.parser.contract import Literal
            from py_compiler.hir.hir.src.program import resolve_literal
            from py_compiler.syntax.function.calls import literal_value
            token = argument.operands[0].token if argument.kind == 'unary' else argument.token
            spelling = (argument.token.text if argument.kind == 'unary' else '') + token.text
            owner, value = resolve_literal(Literal('', argument.token.span, spelling, token.kind, self.owner.name, True), cfg.syntax, cfg)
            return literal_value(value, owner, cfg, argument.token.span)
        value = cfg.expr(argument, env)
        if value.type == self.owner:
            return value
        if value.type.family not in self.owner.conversion_sources:
            raise ValueError(f'{value.type.name} has no declared conversion to {self.owner.name}')
        return cfg.emit('convert', self.owner, (value,), ScalarConversion(value.type, self.owner), argument.token.span)


@dataclass(frozen=True)
class ScalarConversion:
    source: object
    target: object

    def atom(self, operation):
        from py_compiler.syntax.generic.atom import Atom
        return Atom(self.target, (self.source,), self.target, ('view',), ('panic',), self)

    def render_operation(self, operation):
        from py_compiler.mlir.src.cfg import name
        source, target = self.source, self.target
        value, result = name(operation.operands[0]), name(operation.result)
        lines = []
        def emit(suffix, expression):
            temporary = result + '_' + suffix
            lines.append(f'{temporary} = {expression}')
            return temporary
        def check(predicate, message):
            lines.append(f'cf.assert {predicate}, "{message}"')
        source_type = source.mlir
        source_signed = source.signed
        if source.family == 'pointer':
            source_type = f'i{source.bits}'
            value = emit('address', f'llvm.ptrtoint {value} : !llvm.ptr to {source_type}')
        if source.mlir == 'index':
            source_type = f'i{source.bits}'
            value = emit('index', f'arith.index_cast {value} : index to {source_type}')
        if target.family == 'bool':
            zero = emit('zero', f'arith.constant {"0.0" if source.family == "float" else "0"} : {source_type}')
            opcode = 'cmpf une' if source.family == 'float' else 'cmpi ne'
            return lines + [f'{result} = arith.{opcode}, {value}, {zero} : {source_type}']
        if target.family == 'float':
            if source.family == 'float':
                if source.bits == target.bits:
                    widened = emit('wide_float', f'arith.extf {value} : {source_type} to f32')
                    return lines + [f'{result} = arith.truncf {widened} : f32 to {target.mlir}']
                opcode = 'extf' if source.bits < target.bits else 'truncf'
            else:
                opcode = 'sitofp' if source_signed else 'uitofp'
            return lines + [f'{result} = arith.{opcode} {value} : {source_type} to {target.mlir}']
        target_type = f'i{target.bits}' if target.mlir == 'index' or target.family == 'pointer' else target.mlir
        low, high = bounds(target)
        if source.family == 'float':
            # Check the truncated value, not the original fraction. Ordered
            # comparisons also reject NaN before LLVM's float-to-int operation.
            if source.bits < 64:
                value = emit('wide_float', f'arith.extf {value} : {source_type} to f64')
                source_type = 'f64'
            value = emit('truncated', f'math.trunc {value} : {source_type}')
            lower = emit('lower', f'arith.constant {low}.0 : {source_type}')
            upper = emit('upper', f'arith.constant {high + 1}.0 : {source_type}')
            ge = emit('ge', f'arith.cmpf oge, {value}, {lower} : {source_type}')
            lt = emit('lt', f'arith.cmpf olt, {value}, {upper} : {source_type}')
            valid = emit('range', f'arith.andi {ge}, {lt} : i1')
            check(valid, 'scalar conversion out of range')
            opcode = 'fptosi' if target.signed else 'fptoui'
            value = emit('converted', f'arith.{opcode} {value} : {source_type} to {target_type}')
        else:
            source_low, source_high = bounds(source)
            predicate = 's' if source_signed else 'u'
            if source_low < low:
                minimum = emit('minimum', f'arith.constant {low} : {source_type}')
                valid = emit('minimum_ok', f'arith.cmpi {predicate}ge, {value}, {minimum} : {source_type}')
                check(valid, 'scalar conversion below minimum')
            if source_high > high:
                maximum = emit('maximum', f'arith.constant {high} : {source_type}')
                valid = emit('maximum_ok', f'arith.cmpi {predicate}le, {value}, {maximum} : {source_type}')
                check(valid, 'scalar conversion above maximum')
            if source.bits != target.bits:
                opcode = ('extsi' if source_signed else 'extui') if source.bits < target.bits else 'trunci'
                value = emit('converted', f'arith.{opcode} {value} : {source_type} to {target_type}')
        if target.family == 'char':
            maximum = emit('unicode_max', f'arith.constant 1114111 : {target_type}')
            first = emit('surrogate_start', f'arith.constant 55296 : {target_type}')
            last = emit('surrogate_end', f'arith.constant 57343 : {target_type}')
            bounded = emit('unicode_range', f'arith.cmpi ule, {value}, {maximum} : {target_type}')
            before = emit('before_surrogate', f'arith.cmpi ult, {value}, {first} : {target_type}')
            after = emit('after_surrogate', f'arith.cmpi ugt, {value}, {last} : {target_type}')
            scalar = emit('not_surrogate', f'arith.ori {before}, {after} : i1')
            valid = emit('unicode_valid', f'arith.andi {bounded}, {scalar} : i1')
            check(valid, 'char requires a Unicode scalar value')
        if target.family == 'pointer':
            return lines + [f'{result} = llvm.inttoptr {value} : {target_type} to !llvm.ptr']
        if target.mlir == 'index':
            return lines + [f'{result} = arith.index_cast {value} : {target_type} to index']
        zero = emit('zero', f'arith.constant 0 : {target_type}')
        return lines + [f'{result} = arith.addi {value}, {zero} : {target_type}']


import unittest


class ConversionTests(unittest.TestCase):
    def compile(self, source):
        from py_compiler.frontend.src.lib import compile_source
        from py_compiler.syntax.recognition import Syntax
        return compile_source('conversions.sev', source, Syntax())

    def test_explicit_literal_constructors_and_annotation_boundaries(self):
        for source, expected in (('value = char(128512)\n', 128512), ('value = bool(2)\n', True),
                                 ('value = byte(16)\n', 16), ('value = int(0.8)\n', 0),
                                 ('value = u32(\'λ\')\n', 955)):
            result = self.compile(source)
            self.assertFalse(result.diagnostics, str(result.diagnostics))
            self.assertEqual(result.program.constants[0].value, expected)
        for source in ('value = char(55296)\n', 'value = char(1114112)\n', 'value = u8(256)\n',
                       'value = int(9223372036854775808)\n', 'value: byte = 1\n', 'value: char = 65\n'):
            self.assertTrue(self.compile(source).diagnostics, source)

    def test_runtime_conversions_reach_mlir(self):
        from py_compiler.mlir.src.lib import lower, render
        for source_type, target_type, opcode in (('int', 'bool', 'arith.cmpi ne'),
                                                ('float', 'bool', 'arith.cmpf une'),
                                                ('bool', 'int', 'arith.extui'),
                                                ('int', 'char', 'unicode_valid'),
                                                ('char', 'int', 'arith.extui'),
                                                ('int', 'byte', 'arith.addi'),
                                                ('byte', 'int', 'arith.addi'),
                                                ('int', 'float', 'arith.sitofp'),
                                                ('float', 'int', 'arith.fptosi'),
                                                ('i64', 'u8', 'minimum_ok'),
                                                ('u64', 'i64', 'maximum_ok'),
                                                ('bf16', 'f16', 'wide_float')):
            with self.subTest(source=source_type, target=target_type):
                result = self.compile(f'def convert(value: {source_type}) -> {target_type}:\n    return {target_type}(value)\n')
                self.assertFalse(result.diagnostics, str(result.diagnostics))
                self.assertIn(opcode, render(lower(result.program)))

    def test_pointer_address_round_trip_requires_unsafe(self):
        from py_compiler.mlir.src.lib import lower, render
        result = self.compile('def address(value: usize) -> usize:\n    unsafe:\n        raw = pointer(value)\n        return usize(raw)\n')
        self.assertFalse(result.diagnostics, str(result.diagnostics))
        text = render(lower(result.program))
        self.assertIn('llvm.inttoptr', text)
        self.assertIn('llvm.ptrtoint', text)
        self.assertTrue(self.compile('def address(value: usize) -> pointer:\n    return pointer(value)\n').diagnostics)
