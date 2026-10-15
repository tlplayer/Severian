"""Pure, bounded SourceWindow programs implementing the common G contract."""
import ast
import operator
from dataclasses import dataclass
from py_compiler.syntax.generic.grammar import SourceWindow, Match
from py_compiler.syntax.generic.block import BlockProvider
from py_compiler.syntax.grammar.expression import expression


@dataclass(frozen=True)
class Procedure:
    parameter: str
    owner: str = ''


@dataclass(frozen=True)
class Selection:
    kind: str
    window: SourceWindow
    end: int
    captures: tuple = ()
    target: str = ''


class Returned(Exception):
    def __init__(self, value):
        self.value = value


class LoopControl(Exception):
    def __init__(self, stop):
        self.stop = stop


class Recognition:
    """No runtime values, CFG access, filesystem access, or arbitrary calls."""
    def __init__(self, node, window, budget=10000):
        self.node, self.window, self.remaining = node, window, budget
        self.env = {node.header[1].parameter: window}
        self.captures = {}

    def tick(self):
        self.remaining -= 1
        if self.remaining < 0:
            raise ValueError('procedural grammar exceeded its recognition step budget')

    def bounded(self, window):
        if (not isinstance(window, SourceWindow) or window.source is not self.window.source
                or window.start < self.window.start or window.end > self.window.end):
            raise ValueError('procedural capture escapes the original SourceWindow')
        return window

    def evaluate(self, term):
        self.tick()
        kind, spelling = term.kind, term.token.text
        if kind == 'literal':
            if spelling in ('true', 'false', 'None', 'absent'):
                return {'true': True, 'false': False, 'None': None, 'absent': None}[spelling]
            return ast.literal_eval(spelling)
        if kind == 'name':
            if spelling not in self.env:
                raise ValueError('unknown procedural recognition binding ' + spelling)
            return self.env[spelling]
        if kind == 'unary':
            value = self.evaluate(term.operands[0])
            functions = {'not': operator.not_, '-': operator.neg, '+': operator.pos}
            if spelling in functions:
                return functions[spelling](value)
        if kind == 'binary':
            left = self.evaluate(term.operands[0])
            if spelling == 'and':
                return bool(left) and bool(self.evaluate(term.operands[1]))
            if spelling == 'or':
                return bool(left) or bool(self.evaluate(term.operands[1]))
            right = self.evaluate(term.operands[1])
            functions = {'+': operator.add, '-': operator.sub, '*': operator.mul,
                         '//': operator.floordiv, '%': operator.mod, '==': operator.eq,
                         '!=': operator.ne, '<': operator.lt, '<=': operator.le,
                         '>': operator.gt, '>=': operator.ge}
            if spelling in functions:
                if spelling in ('*', '+') and isinstance(left, str):
                    raise ValueError('procedural recognition cannot synthesize detached source text')
                return functions[spelling](left, right)
        if kind == 'member':
            value = self.evaluate(term.operands[0])
            if isinstance(value, SourceWindow) and spelling in ('text', 'start', 'end'):
                return getattr(self.bounded(value), spelling)
        if kind == 'index':
            value, index = (self.evaluate(t) for t in term.operands)
            if isinstance(value, str) and isinstance(index, int) and 0 <= index < len(value):
                return value[index]
            raise ValueError('procedural index is outside its source text')
        if kind == 'call':
            callee, *arguments = term.operands
            values = [self.evaluate(arg) for arg in arguments]
            if callee.kind == 'name' and callee.token.text == 'len' and len(values) == 1:
                value = values[0]
                return len(value.text if isinstance(value, SourceWindow) else value)
            if callee.kind == 'member':
                receiver = self.evaluate(callee.operands[0])
                method = callee.token.text
                if isinstance(receiver, str) and method in ('startswith', 'endswith', 'find', 'rfind', 'isspace', 'isdigit'):
                    return getattr(receiver, method)(*values)
                if isinstance(receiver, SourceWindow):
                    receiver = self.bounded(receiver)
                    if method == 'slice' and 1 <= len(values) <= 2:
                        start, end = values[0], values[1] if len(values) == 2 else len(receiver.text)
                        if not isinstance(start, int) or not isinstance(end, int) or not 0 <= start <= end <= len(receiver.text):
                            raise ValueError('procedural slice escapes its SourceWindow')
                        start, end = receiver.start + start, receiver.start + end
                        tokens = tuple(t for t in receiver.tokens if start <= t.span.start and t.span.end <= end)
                        if any(t.span.start < start < t.span.end or t.span.start < end < t.span.end for t in receiver.tokens):
                            raise ValueError('procedural slice splits a token')
                        return SourceWindow(receiver.source, start, end, tokens)
                    if method == 'capture' and len(values) == 3:
                        name, contract, window = values
                        if not isinstance(name, str) or not isinstance(contract, str) or name in self.captures:
                            raise ValueError('procedural capture requires a unique name and type spelling')
                        self.captures[name] = (contract, self.bounded(window))
                        return window
                    if method in ('expression', 'statement', 'block'):
                        target = values.pop(0) if method == 'block' and values else ''
                        if len(values) not in (1, 2):
                            raise ValueError('procedural selection requires a source window and optional consumed count')
                        window = self.bounded(values[0])
                        end = self.window.start + values[1] if len(values) == 2 else self.window.end
                        if not self.window.start < end <= self.window.end or window.end > end:
                            raise ValueError('grammar must make progress within the original source window')
                        return Selection(method, window, end, tuple(self.captures.items()), target)
        raise ValueError('unsupported operation in pure procedural recognition: ' + spelling)

    def execute(self, nodes):
        index = 0
        while index < len(nodes):
            self.tick()
            node = nodes[index]
            spelling = node.tokens[0].text
            if spelling == 'if':
                chain = [node]
                while index + len(chain) < len(nodes) and nodes[index + len(chain)].tokens[0].text in ('elif', 'else'):
                    chain.append(nodes[index + len(chain)])
                for branch in chain:
                    if branch.tokens[0].text == 'else' or self.evaluate(expression(branch.header)):
                        self.execute(branch.children)
                        break
                index += len(chain)
                continue
            if spelling == 'while':
                condition, clauses = node.header
                if clauses:
                    raise ValueError('procedural while requires a plain condition')
                while self.evaluate(expression(condition)):
                    try:
                        self.execute(node.children)
                    except LoopControl as control:
                        if control.stop:
                            break
            elif spelling == 'return':
                raise Returned(self.evaluate(expression(node.tokens[1:])))
            elif spelling in ('break', 'continue'):
                raise LoopControl(spelling == 'break')
            elif len(node.tokens) > 2 and node.tokens[1].text in ('=', ':=', '+=', '-='):
                name, op = node.tokens[0].text, node.tokens[1].text
                value = self.evaluate(expression(node.tokens[2:]))
                if name == self.node.header[1].parameter:
                    raise ValueError('procedural input window cannot be rebound')
                if op in ('+=', '-='):
                    value = (operator.add if op == '+=' else operator.sub)(self.env[name], value)
                self.env[name] = value
            elif node.provider:
                raise ValueError('procedural recognition supports local conditionals and loops')
            else:
                self.evaluate(expression(node.tokens))
            index += 1

    def run(self):
        try:
            self.execute(self.node.children)
        except Returned as result:
            if result.value is not None and not isinstance(result.value, Selection):
                raise ValueError('procedural grammar must return a SourceWindow selection or None')
            return result.value
        except (TypeError, IndexError, KeyError, ZeroDivisionError, LoopControl) as error:
            raise ValueError('invalid procedural recognition operation') from error
        raise ValueError('procedural grammar has no recognition result')


class ProceduralBlock(BlockProvider):
    def __init__(self, node, syntax, selected=None):
        self.node, self.syntax, self.selected = node, syntax, selected
        self.spelling = node.header[0]
        if selected:
            self.attachment = selected[0].attachment
            self.continuation = selected[0].continuation
            self.indentation_unit = selected[0].indentation_unit

    def recognize(self, window):
        result = Recognition(self.node, window).run()
        if result is None:
            return None
        if result.kind != 'block' or result.end != window.end:
            raise ValueError('block grammar must select a complete header')
        target = self.syntax.block_providers.get(result.target)
        if target is None or result.target == 'grammar':
            raise ValueError('procedural block requires a registered semantic block provider')
        selected = ProceduralBlock(self.node, self.syntax, (target, result))
        return Match(selected, window, result.end, {'header': window.tokens, **{name: region for name, (_, region) in result.captures}})

    def construct(self, match):
        target, result = self.selected
        return target.parse_header((match.window.tokens[0], *result.window.tokens))

    def attach(self, previous, parent):
        return self.selected[0].attach(previous, parent)

    def validate_captures(self, cfg, env):
        from py_compiler.syntax.generic.place import expression_type
        for name, (contract, region) in self.selected[1].captures:
            actual = expression_type(expression(region.tokens), cfg, env)
            if actual != cfg.context.type(contract):
                raise ValueError('procedural capture ' + name + ' does not satisfy ' + contract)

    def lower(self, node, cfg, env, nodes, index):
        self.validate_captures(cfg, env)
        return self.selected[0].lower(node, cfg, env, nodes, index)

    def bind_scope(self, node, parent=None):
        return self.selected[0].bind_scope(node, parent)

    def expand(self, header, syntax):
        return self.selected[0].expand(header, syntax)

    def has_body(self, node):
        return self.selected[0].has_body(node)

    @property
    def has_condition(self):
        return getattr(self.selected[0], 'has_condition', False)

    def declare(self, node, context):
        return self.selected[0].declare(node, context)


import unittest


class ProceduralTests(unittest.TestCase):
    def compile(self, text):
        from py_compiler.frontend.src.lib import compile_source
        from py_compiler.syntax.recognition import Syntax
        return compile_source('procedural.sev', text, Syntax())

    def test_native_source_windows_delegate_blocks_and_typed_expressions(self):
        import json
        import tempfile
        from pathlib import Path
        from py_compiler.testing import test_package
        source = '''grammar whenever(input: SourceWindow) -> B:
    if not input.text.startswith("whenever "):
        return None
    return input.block("if", input.slice(9))

grammar unwrap(input: SourceWindow) -> int:
    if not input.text.startswith("unwrap("):
        return None
    end = len(input.text) - 1
    input.capture("value", "int", input.slice(7, end))
    return input.expression(input.slice(7, end))

test "procedural recognition preserves ordinary runtime evaluation":
    value = 0
    whenever true:
        value = unwrap(40 + 2)
    assert(value == 42)

grammar perform(input: SourceWindow) -> O:
    if input.text.startswith("perform "):
        return input.statement(input.slice(8))
    return None

test "separate selections introduce separate bindings":
    perform first = 10
    perform second = 32
    assert(first == 10)
    assert(second == 32)
'''
        with tempfile.TemporaryDirectory(prefix='sev-procedural-') as directory:
            root = Path(directory)
            (root / 'src').mkdir()
            (root / 'src/lib.sev').write_text(source)
            (root / 'package.json').write_text(json.dumps({'package': {'name': 'procedural', 'version': '0.1.0'}}))
            result = test_package(root)
            self.assertTrue(result['success'], result)

    def test_ambiguous_candidates_never_lower_runtime_bodies(self):
        source = '''grammar first(input: SourceWindow) -> B:
    if input.text.startswith("whenever "):
        return input.block("if", input.slice(9))
    return None

grammar second(input: SourceWindow) -> B:
    if input.text.startswith("whenever "):
        return input.block("if", input.slice(9))
    return None

whenever true:
    missing_function()
'''
        result = self.compile(source)
        self.assertTrue(result.diagnostics)
        self.assertIn('ambiguous B.header grammar', str(result.diagnostics))
        self.assertNotIn('no callable', str(result.diagnostics))

    def test_nested_recognition_returns_and_continuation_capture_contracts(self):
        source = '''grammar whenever(input: SourceWindow) -> B:
    if input.text.startswith("whenever "):
        return input.block("if", input.slice(9))
    else:
        return None
grammar otherwise_when(input: SourceWindow) -> B:
    if not input.text.startswith("otherwise_when "):
        return None
    input.capture("condition", "bool", input.slice(15, len(input.text) - 1))
    return input.block("elif", input.slice(15))
test:
    whenever false:
        assert(false)
    otherwise_when true:
        assert(true)
    else:
        assert(false)
'''
        self.assertFalse(self.compile(source).diagnostics)
        result = self.compile(source.replace('otherwise_when true:', 'otherwise_when 42:'))
        self.assertTrue(result.diagnostics)
        self.assertIn('does not satisfy bool', str(result.diagnostics))

    def test_statement_selection_reuses_original_semantics(self):
        source = '''grammar perform(input: SourceWindow) -> O:
    if input.text.startswith("perform "):
        return input.statement(input.slice(8))
    return None
test:
    value = 1
    perform value += 2
    assert(value == 3)
'''
        self.assertFalse(self.compile(source).diagnostics)

    def test_bounds_progress_contracts_and_step_budget(self):
        prefix = ('grammar wrapped(input: SourceWindow) -> int:\n'
                  '    if not input.text.startswith("wrapped("):\n        return None\n')
        for body, message in [
            ('return input.expression(input.slice(-1))', 'escapes'),
            ('return input.expression(input.slice(8, 9), 0)', 'progress'),
            ('while true:\n        value = 1\n    return None', 'step budget'),
            ('return "detached"', 'SourceWindow selection'),
        ]:
            with self.subTest(message=message):
                result = self.compile(prefix + '    ' + body + '\ntest:\n    value = wrapped(1)\n')
                self.assertTrue(result.diagnostics)
                self.assertIn(message, str(result.diagnostics))

    def test_selection_keeps_original_capture_spans(self):
        from py_compiler.frontend.source.source import SourceFile
        from py_compiler.frontend.lexer.lexer import lex
        from py_compiler.frontend.parser.blocks import parse_blocks
        from py_compiler.syntax.recognition import Syntax
        syntax = Syntax()
        source = SourceFile('recognizer.sev', 'grammar g(input: SourceWindow) -> int:\n    return input.expression(input.slice(2, 3))\n')
        root, errors = parse_blocks(source, lex(source, syntax), syntax)
        self.assertFalse(errors)
        original = SourceFile('original.sev', '  7  ')
        tokens = tuple(t for t in lex(original, syntax) if t.kind not in ('NEWLINE', 'EOF', 'INDENT'))
        selection = Recognition(root.children[0], SourceWindow(original, 0, 5, tokens)).run()
        self.assertIs(selection.window.source, original)
        self.assertEqual((selection.window.start, selection.window.end), (2, 3))
