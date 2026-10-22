"""Test utility syntax. Assertions never become exported production callables."""
from py_compiler.syntax.grammar.expression import expression
from py_compiler.syntax.complex.object import Assert


class Assertion:
    def lower(self, node, cfg, env):
        scope = getattr(cfg, 'block_scope', None)
        if scope is None or not scope.contains_kind('test'):
            raise ValueError('assert requires test scope; use throw Error for runtime failures')
        if getattr(cfg, 'integration_test', False):
            term = expression(node.tokens[1:])
            if term.kind == 'binary' and term.token.text in ('==', '!='):
                left, right = term.operands
                if left.kind == 'literal':
                    left, right = right, left
                if (left.kind == 'name' and left.token.text in ('stdout', 'stderr')
                        and right.kind == 'literal' and right.token.kind == 'STRING'):
                    if node.parent != cfg.integration_scope:
                        raise ValueError('output expectations must be at the top level of an integration test')
                    from py_compiler.syntax.symbol.forms import decode_quoted
                    if not hasattr(cfg.body, 'output_expectations'):
                        cfg.body.output_expectations = []
                    cfg.body.output_expectations.append({'stream': left.token.text, 'operator': term.token.text,
                                                         'value': decode_quoted(right.token.text)})
                    return
        if (getattr(cfg, 'integration_test', False) and len(node.tokens) in (6, 7)
                and node.tokens[2].kind == 'STRING'
                and [t.text for t in node.tokens[3:]] in (['in', 'stdout', ')'], ['not', 'in', 'stdout', ')'])):
            if node.parent != cfg.integration_scope:
                raise ValueError('stdout expectations must be at the top level of an integration test')
            from py_compiler.syntax.symbol.forms import decode_quoted
            attribute = 'stdout_exclusions' if node.tokens[3].text == 'not' else 'stdout_expectations'
            if not hasattr(cfg.body, attribute):
                setattr(cfg.body, attribute, [])
            getattr(cfg.body, attribute).append(decode_quoted(node.tokens[2].text))
            return
        value = cfg.expr(expression(node.tokens[1:]), env, cfg.syntax.types['bool'])
        scenarios = getattr(cfg, 'test_scenarios', ())
        suffix = ''.join(' when ' + scenario for scenario in scenarios)
        cfg.effect('assert', (value,), Assert(f'assertion failed at {cfg.source.path}:{cfg.source.position(node.span.start)[0]}' + suffix), node.span)


class Snapshot:
    def lower(self, node, cfg, env):
        term = expression(node.tokens)
        if term.kind != 'call' or len(term.operands) != 3:
            raise ValueError('snapshot requires a name and a string value')
        from py_compiler.syntax.function.calls import literal_value
        string = cfg.context.type('string')
        values = tuple(cfg.expr(argument, env, string) for argument in term.operands[1:])
        key = literal_value('SEV_SNAPSHOT_RECORDS\0', string, cfg, node.span)
        mode = literal_value('ab\0', string, cfg, node.span)
        cfg.effect('snapshot', (*values, key, mode), SnapshotRecord(cfg.syntax.pointer_bits), node.span)
        from py_compiler.syntax.generic.storage import release_storage, root_binding
        for argument, value in zip(term.operands[1:], values):
            if value.identity in cfg.owned_results and root_binding(argument, cfg, env) is None:
                release_storage(value, cfg, node.span)


from dataclasses import dataclass


@dataclass(frozen=True)
class SnapshotRecord:
    bits: int

    @property
    def external_symbol_contracts(self):
        width = 'i' + str(self.bits)
        return (('getenv', ('!llvm.ptr',), '!llvm.ptr'),
                ('fopen', ('!llvm.ptr', '!llvm.ptr'), '!llvm.ptr'),
                ('fwrite', ('!llvm.ptr', width, width, '!llvm.ptr'), width),
                ('fclose', ('!llvm.ptr',), 'i32'))

    def atom(self, operation):
        from py_compiler.syntax.generic.atom import Atom
        return Atom(self, tuple(value.type for value in operation.operands), None, ('view',) * 4, ('IO',), self)

    def render_operation(self, operation):
        from py_compiler.mlir.src.cfg import name, temporary_suffix
        from py_compiler.syntax.generic.storage import descriptor_type
        prefix = '%snapshot_' + temporary_suffix(operation)
        width = 'i' + str(self.bits)
        descriptor = descriptor_type(self.bits)
        lines = []
        for index, value in enumerate(operation.operands):
            current = prefix + '_' + str(index)
            lines += [f'{current}_desc = builtin.unrealized_conversion_cast {name(value)} : {value.type.mlir} to {descriptor}',
                      f'{current}_base = llvm.extractvalue {current}_desc[1] : {descriptor}',
                      f'{current}_offset = llvm.extractvalue {current}_desc[2] : {descriptor}',
                      f'{current}_length = llvm.extractvalue {current}_desc[3, 0] : {descriptor}',
                      f'{current}_data = llvm.getelementptr {current}_base[{current}_offset] : (!llvm.ptr, {width}) -> !llvm.ptr, i8']
        lines += [f'{prefix}_path = func.call @getenv({prefix}_2_data) : (!llvm.ptr) -> !llvm.ptr',
                  f'{prefix}_null = llvm.mlir.zero : !llvm.ptr',
                  f'{prefix}_configured = llvm.icmp "ne" {prefix}_path, {prefix}_null : !llvm.ptr',
                  f'cf.assert {prefix}_configured, "snapshot requires the test runner"',
                  f'{prefix}_file = func.call @fopen({prefix}_path, {prefix}_3_data) : (!llvm.ptr, !llvm.ptr) -> !llvm.ptr',
                  f'{prefix}_opened = llvm.icmp "ne" {prefix}_file, {prefix}_null : !llvm.ptr',
                  f'cf.assert {prefix}_opened, "snapshot record file could not be opened"',
                  f'{prefix}_two = arith.constant 2 : i64',
                  f'{prefix}_header = llvm.alloca {prefix}_two x i64 : (i64) -> !llvm.ptr',
                  f'{prefix}_second = llvm.getelementptr {prefix}_header[1] : (!llvm.ptr) -> !llvm.ptr, i64']
        for index, destination in ((0, 'header'), (1, 'second')):
            length = f'{prefix}_{index}_length'
            if self.bits != 64:
                lines.append(f'{prefix}_size{index} = arith.extui {length} : {width} to i64')
                length = f'{prefix}_size{index}'
            lines.append(f'llvm.store {length}, {prefix}_{destination} : i64, !llvm.ptr')
        lines += [f'{prefix}_one = arith.constant 1 : {width}', f'{prefix}_bytes = arith.constant 16 : {width}']
        for index, (data, count) in enumerate(((f'{prefix}_header', f'{prefix}_bytes'),
                                              (f'{prefix}_0_data', f'{prefix}_0_length'),
                                              (f'{prefix}_1_data', f'{prefix}_1_length'))):
            lines += [f'{prefix}_written{index} = func.call @fwrite({data}, {prefix}_one, {count}, {prefix}_file) : (!llvm.ptr, {width}, {width}, !llvm.ptr) -> {width}',
                      f'{prefix}_complete{index} = arith.cmpi eq, {prefix}_written{index}, {count} : {width}',
                      f'cf.assert {prefix}_complete{index}, "snapshot record write failed"']
        lines += [f'{prefix}_closed = func.call @fclose({prefix}_file) : (!llvm.ptr) -> i32',
                  f'{prefix}_zero = arith.constant 0 : i32',
                  f'{prefix}_ok = arith.cmpi eq, {prefix}_closed, {prefix}_zero : i32',
                  f'cf.assert {prefix}_ok, "snapshot record close failed"']
        return lines


import unittest


class AssertionScopeTests(unittest.TestCase):
    def test_integration_stream_equality_is_runner_metadata(self):
        from py_compiler.frontend.src.lib import compile_source
        from py_compiler.syntax.recognition import Syntax
        result = compile_source('streams.sev', 'test with integ:\n    assert(stderr == "")\n    assert("hello" != stdout)\n', Syntax())
        self.assertFalse(result.diagnostics, str(result.diagnostics))
        body = next(body for body in result.program.bodies if body.test)
        self.assertEqual(tuple(body.test['output-expectations']), (
            {'stream': 'stderr', 'operator': '==', 'value': ''},
            {'stream': 'stdout', 'operator': '!=', 'value': 'hello'}))

    def test_assert_is_test_only_and_test_contracts_are_allowed(self):
        from py_compiler.frontend.src.lib import compile_source
        from py_compiler.syntax.recognition import Syntax
        rejected = compile_source('assert.sev', 'def work():\n    assert(true)\n', Syntax())
        self.assertIn('assert requires test scope', str(rejected.diagnostics))
        accepted = compile_source('assert.sev', 'test "contract" with {suffix true,}:\n    assert(true)\n', Syntax(include_tests=True))
        self.assertFalse(accepted.diagnostics, str(accepted.diagnostics))

    def test_suffix_checks_live_local_storage_before_cleanup(self):
        from py_compiler.frontend.src.lib import compile_source
        from py_compiler.syntax.recognition import Syntax
        result = compile_source('suffix.sev', 'test "storage" with {suffix len(value) == 2,}:\n    value = "ok"\n    assert(true)\n', Syntax(include_tests=True))
        self.assertFalse(result.diagnostics, str(result.diagnostics))
        body = next(b for b in result.program.bodies if b.test)
        operations = [op.kind for block in body.blocks for op in block.operations]
        self.assertLess(operations.index('length'), operations.index('deallocate'))


    def test_negative_stdout_expectations_are_preserved(self):
        from py_compiler.frontend.src.lib import compile_source
        from py_compiler.syntax.recognition import Syntax
        result = compile_source('stdout.sev', 'test with integ:\n    assert("bad" not in stdout)\n    assert("good" in stdout)\n', Syntax(include_tests=True))
        self.assertFalse(result.diagnostics, str(result.diagnostics))
        body = next(b for b in result.program.bodies if b.test)
        self.assertEqual(tuple(body.test['stdout-excludes']), ('bad',))
        self.assertEqual(tuple(body.test['stdout-contains']), ('good',))


from py_compiler.syntax.generic.block import BlockProvider


class AssertionBlock(BlockProvider):
    spelling = 'assert'
    required_scope = 'test'

    def recognize(self, window):
        if len(window.tokens) == 2 and window.tokens[-1].text == ':':
            return super().recognize(window)
        return None

    def lower(self, node, cfg, env, nodes, index):
        from py_compiler.syntax.grammar.expression import expression
        from py_compiler.syntax.complex.object import Assert
        if not node.children:
            raise ValueError('assert block requires conditions')
        for child in node.children:
            if child.provider is not None:
                raise ValueError('assert block contains conditions, not nested control flow')
            value = cfg.expr(expression(child.tokens), env, cfg.context.type('bool'))
            line, column = cfg.source.position(child.span.start)
            cfg.effect('assert', (value,), Assert(f'assertion failed at {cfg.source.path}:{line}:{column}'), child.span)
        return 1
