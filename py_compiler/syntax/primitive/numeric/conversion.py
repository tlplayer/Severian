"""Scalar conversion edges, including implicit lossless integer conversions."""
from dataclasses import dataclass
from decimal import Decimal
from py_compiler.syntax.generic.constructor import Constructor


def bounds(owner):
    return (-(1 << (owner.bits - 1)) if owner.signed else 0,
            (1 << (owner.bits - int(owner.signed))) - 1)


def lossless_integer_conversion(source, target):
    if source.family != 'integer' or target.family != 'integer':
        return False
    source_low, source_high = bounds(source)
    target_low, target_high = bounds(target)
    return target_low <= source_low and source_high <= target_high


def known_integer(value, cfg):
    """Prove constants from SSA definitions; mutable storage and phi inputs are unknown."""
    definitions = {op.result.identity: op for block in cfg.body.blocks for op in block.operations
                   if op.result is not None}
    def evaluate(value):
        if value.type.family != 'integer':
            return None
        operation = definitions.get(value.identity)
        if operation is None:
            return None
        if operation.kind == 'constant':
            return operation.payload.value
        operands = [evaluate(v) for v in operation.operands]
        if not operands or any(v is None for v in operands):
            return None
        if operation.kind == 'convert' and isinstance(operation.payload, ScalarConversion):
            result = operands[0]
        else:
            spelling = getattr(operation.payload, 'spelling', None)
            if len(operands) == 2 and spelling in ('+', '-', '*'):
                left, right = operands
                result = {'+': lambda: left + right, '-': lambda: left - right,
                          '*': lambda: left * right}[spelling]()
            else:
                return None
        low, high = bounds(value.type)
        return result if low <= result <= high else None
    return evaluate(value)


def integer_conversion_rank(value, target, cfg):
    if lossless_integer_conversion(value.type, target):
        return 1
    if value.type.family == target.family == 'integer':
        constant = known_integer(value, cfg)
        low, high = bounds(target)
        if constant is not None and low <= constant <= high:
            return 2
    return None


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
        policy = 'checked'
        if len(arguments) == 2:
            modifier = arguments[1]
            if modifier.kind != 'name' or modifier.token.text not in ('checked', 'lossy'):
                raise ValueError('scalar conversion policy must be checked or lossy')
            policy = modifier.token.text
            arguments = arguments[:1]
        if len(arguments) != 1:
            raise ValueError(f'{self.owner.name} constructor requires one value')
        self.owner.require_context(cfg)
        argument = arguments[0]
        if policy != 'lossy' and ((argument.kind == 'literal' and argument.token.kind != 'STRING') or (argument.kind == 'unary' and argument.token.text in ('+', '-') and argument.operands[0].kind == 'literal')):
            from py_compiler.frontend.parser.contract import Literal
            from py_compiler.hir.hir.src.program import resolve_literal
            from py_compiler.syntax.function.calls import literal_value
            token = argument.operands[0].token if argument.kind == 'unary' else argument.token
            spelling = (argument.token.text if argument.kind == 'unary' else '') + token.text
            owner, value = resolve_literal(Literal('', argument.token.span, spelling, token.kind, self.owner.name, True), cfg.syntax, cfg)
            return literal_value(value, owner, cfg, argument.token.span)
        value = cfg.expr(argument, env)
        if value.type.family == 'union':
            return self.convert_union(value, argument, cfg, env, policy)
        return self.convert(value, argument, cfg, env, policy=policy)

    def convert(self, value, argument, cfg, env, binding=None, policy='checked'):
        if value.type.family == 'string' and (self.owner.family == 'float' or self.owner.family == 'integer' and self.owner.bits <= 64):
            from py_compiler.syntax.grammar.expression import Expression
            from py_compiler.syntax.complex.tensor import ArgumentValue
            from py_compiler.syntax.generic.storage import root_binding
            method = '_parse_float' if self.owner.family == 'float' else '_parse_integer' if self.owner.signed else '_parse_unsigned'
            captured = Expression('capture', argument.token, (ArgumentValue(value, binding or root_binding(argument, cfg, env)),))
            value = value.type.library_call(method, (captured,), cfg, env, argument.token.span)
        if value.type == self.owner:
            return value
        if value.type.family not in self.owner.conversion_sources:
            raise ValueError(f'{value.type.name} has no declared conversion to {self.owner.name}')
        return cfg.emit('convert', self.owner, (value,), ScalarConversion(value.type, self.owner, policy), argument.token.span)

    def convert_union(self, value, argument, cfg, env, policy='checked'):
        from py_compiler.hir.thir.src.model import Edge, Terminator
        from py_compiler.syntax.block.union import UnionTag, UnionMember
        from py_compiler.syntax.function.calls import literal_value
        from py_compiler.syntax.generic.owned import select
        from py_compiler.syntax.generic.storage import root_binding, owns_storage, release_storage
        from py_compiler.mir.ownership.flow import Binding
        span = argument.token.span
        owner = root_binding(argument, cfg, env)
        done = cfg.block((self.owner,))
        tag = cfg.emit('union-tag', cfg.context.type('i32'), (value,), UnionTag(), span)
        for index, member in enumerate(value.type.variants):
            branch, following = cfg.block(), cfg.block()
            expected = literal_value(index, tag.type, cfg, span)
            active = select(tag.type, 'F.operator', '==').expand(cfg, tag, expected, span)
            cfg.current.terminator = Terminator('conditional', (Edge(branch.identity), Edge(following.identity)), active)
            cfg.current = branch
            if not member.mlir:
                raise ValueError(f'{member.name} has no declared conversion to {self.owner.name}')
            projected = cfg.emit('union-member', member, (value,), UnionMember(index), span)
            # The projection borrows the union's payload, including a temporary union.
            binding = Binding('union-conversion:' + str(projected.identity), ':union-member', member, True, 'view', span)
            cfg.values[binding.identity] = projected
            converted = self.convert(projected, argument, cfg, env, binding, policy)
            cfg.current.terminator = Terminator('jump', (Edge(done.identity, (converted,)),))
            cfg.current = following
        cfg.current.terminator = Terminator('panic', message='invalid union tag in scalar conversion')
        cfg.current = done
        if owner is None and value.identity in cfg.owned_results and owns_storage(value.type):
            release_storage(value, cfg, span)
        return done.parameters[0]


@dataclass(frozen=True)
class ScalarConversion:
    source: object
    target: object
    policy: str = 'checked'

    def atom(self, operation):
        from py_compiler.syntax.generic.atom import Atom
        effects = () if lossless_integer_conversion(self.source, self.target) or (
            self.policy == 'lossy' and self.source.family == self.target.family == 'integer') else ('panic',)
        return Atom(self.target, (self.source,), self.target, ('view',), effects, self)

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
            if source_low < low and self.policy != 'lossy':
                minimum = emit('minimum', f'arith.constant {low} : {source_type}')
                valid = emit('minimum_ok', f'arith.cmpi {predicate}ge, {value}, {minimum} : {source_type}')
                check(valid, 'scalar conversion below minimum')
            if source_high > high and self.policy != 'lossy':
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
    def test_union_conversion_dispatches_all_numeric_variants(self):
        from py_compiler.mlir.src.lib import lower, render
        result = self.compile('def convert(value: int | f64 | u8) -> f64:\n    return f64(value)\n')
        self.assertFalse(result.diagnostics, str(result.diagnostics))
        text = render(lower(result.program))
        self.assertIn('arith.sitofp', text)
        self.assertIn('arith.uitofp', text)
        self.assertIn('invalid union tag', text)
        invalid = self.compile('def convert(value: int | None) -> f64:\n    return f64(value)\n')
        self.assertIn('None has no declared conversion', str(invalid.diagnostics))

    def test_known_integers_narrow_without_runtime_checks(self):
        from py_compiler.mlir.src.lib import lower, render
        for term in ('i64(6)', 'i64(3) + i64(3)', 'i64(-2147483648)', 'i64(2147483647)'):
            result = self.compile('def take(value: i32) -> i32:\n    return value\n'
                                  f'def work() -> i32:\n    value = {term}\n    return take(value)\n')
            self.assertFalse(result.diagnostics, str(result.diagnostics))
            self.assertNotIn('cf.assert', render(lower(result.program)))
        for term in ('i64(2147483648)', 'i64(-2147483649)'):
            result = self.compile(f'def work() -> i32:\n    return {term}\n')
            self.assertTrue(result.diagnostics)
            self.assertIn('outside i32 range', str(result.diagnostics[0]))

    def test_constant_proof_does_not_follow_unknown_loop_values(self):
        result = self.compile('def take(value: i8):\n    return\n'
                              'def work(n: i64):\n    value = i64(6)\n'
                              '    while value < n:\n        take(value)\n        value += 1\n')
        self.assertTrue(result.diagnostics)
        self.assertIn('explicit conversion', str(result.diagnostics[0]))

    def compile(self, source):
        from py_compiler.frontend.src.lib import compile_source
        from py_compiler.syntax.recognition import Syntax
        return compile_source('conversions.sev', source, Syntax())

    def test_implicit_integer_conversion_uses_the_entire_source_range(self):
        from py_compiler.mlir.src.lib import lower, render
        for source, target, opcode in (('i32', 'i64', 'arith.extsi'),
                                       ('u32', 'i64', 'arith.extui'),
                                       ('u32', 'u64', 'arith.extui'),
                                       ('i64', 'i128', 'arith.extsi')):
            with self.subTest(source=source, target=target):
                result = self.compile(f'def accept(value: {target}) -> {target}:\n    return value\n'
                                      f'def work(value: {source}) -> {target}:\n'
                                      f'    assigned: {target} = value\n    return accept(value)\n')
                self.assertFalse(result.diagnostics, str(result.diagnostics))
                text = render(lower(result.program))
                self.assertIn(opcode, text)
                self.assertNotIn('cf.assert', text)
        for source, target in (('i64', 'i32'), ('i32', 'u64'), ('u64', 'i64')):
            with self.subTest(source=source, target=target):
                result = self.compile(f'def work(value: {source}) -> {target}:\n    return value\n')
                self.assertTrue(result.diagnostics)
                self.assertIn('explicit conversion', str(result.diagnostics[0]))

    def test_integer_expression_results_widen_at_the_destination(self):
        result = self.compile('class Item:\n    value: i32\n'
                              'def small(value: i32) -> i32:\n    return value\n'
                              'def work(values: array[i32], item: Item) -> i64:\n'
                              '    indexed: i64 = values[0]\n'
                              '    field: i64 = item.value\n'
                              '    called: i64 = small(9)\n'
                              '    expression: i64 = small(2) + small(3)\n'
                              '    return indexed + field + called + expression\n')
        self.assertFalse(result.diagnostics, str(result.diagnostics))

    def test_contextual_integer_literals_check_both_i32_bounds(self):
        declaration = 'def accept(value: i32) -> i32:\n    return value\n'
        for value in (-2147483648, 2147483647):
            result = self.compile(declaration + f'def work() -> i32:\n    return accept({value})\n')
            self.assertFalse(result.diagnostics, str(result.diagnostics))
        for value in (-2147483649, 2147483648):
            result = self.compile(declaration + f'def work() -> i32:\n    return accept({value})\n')
            self.assertTrue(result.diagnostics)
            self.assertIn('outside i32 range', str(result.diagnostics[0]))

    def test_integer_overloads_prefer_exact_and_reject_equal_conversion_ranks(self):
        declarations = ('def choose(value: i32) -> i32:\n    return value\n'
                        'def choose(value: i64) -> i64:\n    return value\n')
        result = self.compile(declarations + 'def work(value: i32) -> i32:\n    return choose(value)\n')
        self.assertFalse(result.diagnostics, str(result.diagnostics))
        result = self.compile(declarations + 'def work(value: i16) -> i64:\n    return choose(value)\n')
        self.assertTrue(result.diagnostics)
        self.assertIn('ambiguous callable overload', str(result.diagnostics[0]))
        result = self.compile('def choose(value: string) -> i64:\n    return 0\n'
                              'def choose(value: i64) -> i64:\n    return value\n'
                              'def work(value: i32) -> i64:\n    return choose(value)\n')
        self.assertFalse(result.diagnostics, str(result.diagnostics))

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
