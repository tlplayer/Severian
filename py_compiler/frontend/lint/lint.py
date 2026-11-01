"""One frontend lint step: general policy plus the active syntax's Y/G rules."""
from dataclasses import dataclass
from pathlib import Path
import os
import tempfile
from py_compiler.frontend.source.source import SourceFile, Diagnostic
from py_compiler.frontend.lint.rules import SourceEdit, apply_edits, declared_rules, policy


@dataclass(frozen=True)
class LintContext:
    source: object
    syntax: object
    phase: str


def provider_fixes(source, syntax, phase):
    edits = []
    for rule in declared_rules(syntax):
        if rule.phase != phase or rule.check is None:
            continue
        for finding in rule.check(LintContext(source, syntax, phase)):
            if finding.replacement is None:
                raise Diagnostic('lint', rule.name + ': ' + finding.message, source, finding.span)
            edits.append(SourceEdit(finding.span, finding.replacement, rule.name))
    return apply_edits(source, edits)


def prepare_source(source, syntax, packages=None):
    from py_compiler.frontend.lint.layout import format_source
    from py_compiler.frontend.lint.names import fix_names
    from py_compiler.frontend.lint.structure import group_operations, expand_declared_enum_defaults
    from py_compiler.frontend.lint.call_names import label_declared_arguments
    from py_compiler.frontend.lint.import_names import separate_import_statements
    settings = policy()
    unknown = set(settings['rules']) - {'source.layout', 'source.width', 'source.names'}
    if unknown:
        raise ValueError('unknown general lint rules: ' + ', '.join(sorted(unknown)))
    seen = set()
    for _ in range(settings['maximum_passes']):
        if source.text in seen:
            raise Diagnostic('lint', 'automatic fixes form a cycle', source, source.span(0, min(1, len(source.text))))
        seen.add(source.text)
        original = source.text
        source = separate_import_statements(source, syntax)
        source = SourceFile(source.path, provider_fixes(source, syntax, 'source'), source.debug_path)
        source = SourceFile(source.path, format_source(source, syntax), source.debug_path)
        source = SourceFile(source.path, fix_names(source, syntax, packages), source.debug_path)
        source = SourceFile(source.path, label_declared_arguments(source, syntax), source.debug_path)
        source = SourceFile(source.path, expand_declared_enum_defaults(source, syntax), source.debug_path)
        source = SourceFile(source.path, group_operations(source, syntax), source.debug_path)
        source = SourceFile(source.path, provider_fixes(source, syntax, 'syntax'), source.debug_path)
        if original == source.text:
            return source
    raise Diagnostic('lint', 'automatic fixes did not reach a canonical form', source,
                     source.span(0, min(1, len(source.text))))


class ResolvedLint:
    """Argument labels come from the callable selected by the actual checker."""
    def __init__(self, source, syntax):
        self.source, self.syntax, self.edits = source, syntax, []
        self.imports = {}
        self.fragments = {}
        self.spreads = {}
        self.changed_imports = False

    def register_source(self, source):
        self.imports.setdefault(source.identity, (source, []))

    def register_fragment(self, fragment, parent, offset):
        if parent.identity == self.source.identity or parent.identity in self.imports:
            if parent.text[offset:offset + len(fragment.text)] == fragment.text:
                self.fragments[fragment.identity] = (parent, offset)

    def opaque_arguments(self, arguments):
        if len(arguments) > policy()['positional_arguments'] and any(argument.kind != 'named_argument' for argument in arguments):
            token = next((argument.tokens[0] for argument in arguments if argument.tokens), None)
            if token is not None and token.span.source == self.source.identity:
                raise Diagnostic('lint', 'callable type has no parameter names for the required named-argument fix',
                                 self.source, token.span)

    def spread_arguments(self, original, expanded):
        if any(argument.kind == 'spread' for argument in original):
            self.spreads[tuple(id(argument) for argument in expanded)] = tuple(original)

    def arguments(self, arguments, parameters, *, variadic=False):
        if variadic or (len(arguments) <= policy()['positional_arguments'] and
                        not any(argument.kind == 'named_argument' for argument in arguments)):
            return
        if len(arguments) > len(parameters):
            return  # Ordinary call checking owns arity diagnostics.
        original = self.spreads.get(tuple(id(argument) for argument in arguments))
        if original:
            spread = next(argument for argument in original if argument.kind == 'spread')
            token = spread.tokens[0] if spread.tokens else spread.token
            if token.span.source == self.source.identity:
                def literal_arguments(argument):
                    if argument.kind == 'spread':
                        container = argument.operands[0]
                        if container.kind in ('tuple', 'list'):
                            values = []
                            for item in container.operands:
                                selected = literal_arguments(item)
                                if selected is None:
                                    return None
                                values.extend(selected)
                            return values
                        if container.kind == 'interval' or container.kind == 'binary' and container.token.text == '..':
                            def integer(term):
                                if term.kind == 'literal' and term.token.text.isdecimal():
                                    return int(term.token.text)
                                if term.kind == 'unary' and term.token.text in ('-', '+'):
                                    selected = integer(term.operands[0])
                                    return -selected if term.token.text == '-' else selected
                                raise ValueError('not a constant integer interval')
                            try:
                                start, stop = map(integer, container.operands)
                            except ValueError:
                                return None
                            closed = container.interval or (True, True)
                            return [str(value) for value in range(start + (not closed[0]), stop + closed[1])]
                        return None
                    if argument.kind == 'literal':
                        return [argument.token.text]
                    if argument.kind == 'unary' and argument.token.text in ('-', '+') and argument.operands[0].kind == 'literal':
                        return [argument.token.text + argument.operands[0].token.text]
                    return None
                values = []
                for argument in original:
                    selected = literal_arguments(argument)
                    if selected is None:
                        break
                    values.extend(selected)
                else:
                    if len(values) == len(arguments) and original[0].tokens and original[-1].tokens:
                        replacement = ', '.join((parameter[0] if isinstance(parameter, tuple) else parameter) + '=' + value
                                                for parameter, value in zip(parameters, values))
                        self.edits.append(SourceEdit(self.source.span(original[0].tokens[0].span.start,
                                                                       original[-1].tokens[-1].span.end),
                                                     replacement, 'G.call.arguments'))
                        return
                raise Diagnostic('lint', 'nonvariadic spread requires explicit named arguments; automatic expansion cannot preserve its ownership contract',
                                 self.source, token.span)
        # Desugared constructors are not textual calls. Inserting argument
        # labels into the endpoints of 0..3 would corrupt the user's grammar.
        for argument in arguments:
            if argument.kind == 'named_argument':
                continue
            if not argument.tokens:
                return
            token = argument.tokens[0]
            if token.span.source == self.source.identity:
                source = self.source
            elif token.span.source in self.imports:
                source = self.imports[token.span.source][0]
            elif token.span.source in self.fragments:
                continue
            else:
                return
            preceding = source.text[:token.span.start].rstrip()
            if not preceding or preceding[-1] not in ('(', ','):
                return
        for position, argument in enumerate(arguments):
            if argument.kind in ('named_argument', 'spread', 'capture'):
                continue
            tokens = argument.tokens
            if not tokens or any(token.span.source != tokens[0].span.source for token in tokens):
                continue  # A synthetic expansion is not a source-file edit.
            token = tokens[0]
            offset = 0
            if token.span.source == self.source.identity:
                source, edits = self.source, self.edits
            elif token.span.source in self.imports:
                source, edits = self.imports[token.span.source]
            elif token.span.source in self.fragments:
                source, offset = self.fragments[token.span.source]
                edits = self.edits if source.identity == self.source.identity else self.imports[source.identity][1]
            else:
                continue
            parameter = parameters[position][0] if isinstance(parameters[position], tuple) else parameters[position]
            edits.append(SourceEdit(source.span(offset + token.span.start, offset + token.span.start),
                                    parameter + '=', 'G.call.arguments'))

    def enum_default(self, case, enum, seen):
        token = case.header[0]
        if token.span.source != self.source.identity:
            return
        variants = tuple(enum.name + '.' + (variant.name.rsplit('.', 1)[-1] if hasattr(variant, 'name') else variant)
                         for variant in enum.variants)
        remaining = [name for index, name in enumerate(variants) if index not in seen]
        if remaining:
            self.edits.append(SourceEdit(token.span, ' | '.join(remaining), 'G.match.exhaustive'))

    def finish(self):
        for imported, edits in self.imports.values():
            corrected = apply_edits(imported, edits)
            if corrected != imported.text:
                rewrite_file(imported.debug_path or imported.path, imported.text, corrected)
                self.changed_imports = True
        source = SourceFile(self.source.path, apply_edits(self.source, self.edits), self.source.debug_path)
        return SourceFile(source.path, provider_fixes(source, self.syntax, 'resolved'), source.debug_path)


def rewrite_file(path, original, canonical):
    """Persist only a still-current file, preserving permissions and final bytes."""
    if canonical == original:
        return
    path = Path(path)
    if path.is_symlink():
        path = path.resolve()
    current = path.read_bytes().decode('utf-8')
    if current == canonical:
        return
    if current != original:
        raise ValueError('source changed during lint; refusing to overwrite ' + str(path))
    mode = path.stat().st_mode & 0o777
    with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', newline='', dir=path.parent,
                                     prefix='.' + path.name + '.lint-', delete=False) as stream:
        stream.write(canonical)
        temporary = stream.name
    os.chmod(temporary, mode)
    os.replace(temporary, path)


import unittest


class LintPipelineTests(unittest.TestCase):
    def test_call_labels_come_from_selected_overload_and_keep_evaluation_order(self):
        from py_compiler.frontend.src.lib import compile_source
        from py_compiler.syntax.recognition import Syntax
        text = '''def choose(first: int, second: int, third: int, fourth: int) -> int:
    return first + second + third + fourth
def work() -> int:
    return choose(1, 2, 3, fourth=4)
'''
        result = compile_source('calls.sev', text, Syntax(), lint=True)
        self.assertFalse(result.diagnostics, str(result.diagnostics))
        self.assertIn('choose(first = 1, second = 2, third = 3, fourth = 4)', result.source.text)

    def test_syntax_owned_rule_can_supply_a_fix_or_a_diagnostic(self):
        from py_compiler.syntax.recognition import Syntax
        from py_compiler.syntax.grammar.lexical import Word
        from py_compiler.frontend.lint.rules import LintRule, LintFinding
        class CustomWord(Word):
            def lint_rules(self):
                def check(context):
                    if 'old_name' in context.source.text:
                        start = context.source.text.index('old_name')
                        yield LintFinding(context.source.span(start, start + 8), 'use new_name', 'new_name')
                return (LintRule('example.word', 'source', check),)
        class CustomSyntax(Syntax):
            def recognition_order(self):
                return tuple(CustomWord(provider.keywords) if isinstance(provider, Word) else provider
                             for provider in super().recognition_order())
        result = prepare_source(SourceFile('custom.sev', 'old_name=1\n'), CustomSyntax())
        self.assertEqual(result.text, 'new_name = 1\n')

    def test_rule_without_a_fix_fails_compilation_and_cyclic_fixes_fail(self):
        from py_compiler.syntax.recognition import Syntax
        from py_compiler.syntax.grammar.lexical import Word
        from py_compiler.frontend.lint.rules import LintRule, LintFinding
        class RequiredWord(Word):
            def lint_rules(self):
                def check(context):
                    yield LintFinding(context.source.span(0, 1), 'required canonical form')
                return (LintRule('example.required', 'source', check),)
        class RequiredSyntax(Syntax):
            def recognition_order(self):
                return tuple(RequiredWord(provider.keywords) if isinstance(provider, Word) else provider
                             for provider in super().recognition_order())
        with self.assertRaisesRegex(Diagnostic, 'required canonical form'):
            prepare_source(SourceFile('required.sev', 'value = 1\n'), RequiredSyntax())
        class CyclingWord(Word):
            def lint_rules(self):
                def check(context):
                    replacement = 'right' if context.source.text.startswith('left') else 'left'
                    stop = context.source.text.index(' ')
                    yield LintFinding(context.source.span(0, stop), 'toggle', replacement)
                return (LintRule('example.cycle', 'source', check),)
        class CyclingSyntax(Syntax):
            def recognition_order(self):
                return tuple(CyclingWord(provider.keywords) if isinstance(provider, Word) else provider
                             for provider in super().recognition_order())
        with self.assertRaisesRegex(Diagnostic, 'cycle'):
            prepare_source(SourceFile('cycle.sev', 'left = 1\n'), CyclingSyntax())

    def test_persisted_fix_preserves_permissions_and_rejects_stale_source(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'source.sev'
            path.write_text('value=1\n')
            path.chmod(0o640)
            rewrite_file(path, 'value=1\n', 'value = 1\n')
            self.assertEqual(path.read_text(), 'value = 1\n')
            self.assertEqual(path.stat().st_mode & 0o777, 0o640)
            with self.assertRaisesRegex(ValueError, 'source changed'):
                rewrite_file(path, 'value=1\n', 'value = 2\n')

    def test_long_literal_is_split_without_changing_the_compiled_value(self):
        from py_compiler.frontend.src.lib import compile_source
        from py_compiler.syntax.recognition import Syntax
        value = 'text # literal, \\ with a newline\n' * 8
        from py_compiler.frontend.lint.layout import quoted
        result = compile_source('string.sev', 'def text() -> string:\n    return ' + quoted(value) + '\n', Syntax(), lint=True)
        self.assertFalse(result.diagnostics, str(result.diagnostics))
        constants = [operation.payload.value for body in result.program.bodies
                     for block in body.blocks for operation in block.operations
                     if operation.kind == 'constant' and operation.result.type.family == 'string']
        self.assertIn(value, constants)
        self.assertTrue(all(len(line) <= 80 for line in result.source.text.splitlines()))

    def test_class_constructor_and_variadic_calls_obey_their_own_contracts(self):
        from py_compiler.frontend.src.lib import compile_source
        from py_compiler.syntax.recognition import Syntax
        text = '''class Point:
    first: int
    second: int
    third: int
    fourth: int
def sum_values(values...) -> int:
    total = 0
    for value in values:
        total += value
    return total
def work() -> int:
    point = Point(1, 2, 3, 4)
    return sum_values(point.first, 2, 3, 4)
'''
        result = compile_source('construct.sev', text, Syntax(), lint=True)
        self.assertFalse(result.diagnostics, str(result.diagnostics))
        self.assertIn('Point(first = 1, second = 2, third = 3, fourth = 4)', result.source.text)
        self.assertIn('sum_values(point.first, 2, 3, 4)', result.source.text)

    def test_native_build_rewrites_sources_and_executes_canonical_cases(self):
        import json
        from py_compiler.packaging.testing import test_package
        from py_compiler.frontend.lint.layout import quoted
        long_value = 'literal # text, with spacing and escapes\n' * 7
        text = '''TRACE = 0
def nextValue(value: int) -> int:
  module.TRACE = TRACE * 10 + value
  return value
def pack(first: int, second: int, third: int, fourth: int) -> int:
  return first * 1000 + second * 100 + third * 10 + fourth
def longText() -> string:
  return LONG_LITERAL
# Two comments become one block.
# They must never execute as string expressions.
test "named fixes preserve argument evaluation order":
  module.TRACE = 0
  result = pack(nextValue(1), nextValue(2), nextValue(3), fourth=nextValue(4))
  assert(result == 1234 and TRACE == 1234)
test "wrapped condition preserves precedence":
  if true and false or true and true: # chosen path
    assert(true)
  else:
    assert(false)
test "split strings retain exact bytes":
  text = longText()
  assert(len(text) == LITERAL_LENGTH)
  assert(text[8] == u8(35))
def describeValue(value: int) -> string:
  return "value=" + string(nextValue(value)) + ", trace=" + string(TRACE)
test "formatted string chains evaluate values once in source order":
  module.TRACE = 0
  text = describeValue(7)
  assert(text == "value=7, trace=7")
test "singleton tensor shape remains a one-element tuple":
  values = tensor[u8,(3,)](tuple[u8,3](65,66,67))
  assert(len(values) == 3 and values[2] == u8(67))
class WindowFixture:
  start: int
  end: int
  cursor: int
  length: int
test "four constructor values receive their declared parameter names":
  window = WindowFixture(0, 3, 0, 3)
  assert(window.end == window.length and window.cursor == 0)
'''.replace('LONG_LITERAL', quoted(long_value)).replace('LITERAL_LENGTH', str(len(long_value)))
        with tempfile.TemporaryDirectory(prefix='sev-lint-native-') as directory:
            root = Path(directory)
            path = root / 'lib.sev'
            path.write_text(text)
            (root / 'package.json').write_text(json.dumps({
                'package': {'name': 'lint-native', 'version': '0.1.0'},
                'sources': ['lib.sev'],
            }))
            result = test_package(root, timeout=5)
            detail = json.loads(Path(result['report']).read_text()) if result.get('report') else result
            self.assertTrue(result['success'], detail)
            canonical = path.read_text()
            self.assertNotEqual(text, canonical)
            self.assertIn('first = next_value(1)', canonical)
            self.assertIn('fourth = next_value(4)', canonical)
            self.assertIn('if true\n', canonical)
            self.assertIn('WindowFixture(start = 0, end = 3, cursor = 0, length = 3)', canonical)
            self.assertNotIn('f"""', canonical)
            self.assertTrue(all(len(line) <= policy()['columns'] for line in canonical.splitlines()))

    def test_source_imports_receive_layout_naming_and_resolved_argument_fixes(self):
        from py_compiler.frontend.src.lib import compile_source
        from py_compiler.syntax.recognition import Syntax
        with tempfile.TemporaryDirectory(prefix='sev-lint-import-') as directory:
            root = Path(directory)
            helper = root / 'helper.sev'
            helper.write_text('def sumValues(first: int, second: int, third: int, fourth: int) -> int:\n'
                              '  return first + second + third + fourth\n'
                              'def importedWork() -> int:\n'
                              '  return sumValues(1, 2, 3, 4)\n')
            text = 'from "helper.sev" import *\ndef work() -> int:\n  return importedWork()\n'
            result = compile_source('main.sev', text, Syntax(), debug_path=str(root / 'main.sev'))
            self.assertFalse(result.diagnostics, str(result.diagnostics))
            self.assertIn('return imported_work()', result.source.text)
            self.assertIn('def sum_values(', helper.read_text())
            self.assertIn('sum_values(first = 1, second = 2, third = 3, fourth = 4)', helper.read_text())
            self.assertIn((str(helper), helper.read_text()), result.source_inputs)

    def test_overload_parameter_names_are_selected_by_argument_types(self):
        from py_compiler.frontend.src.lib import compile_source
        from py_compiler.syntax.recognition import Syntax
        text = '''def select(first: int, second: int, third: int, fourth: int) -> int:
    return first
def select(left: bool, middle: bool, next: bool, right: bool) -> bool:
    return left
def work() -> bool:
    return select(true, false, true, false)
'''
        result = compile_source('overload.sev', text, Syntax(), lint=True)
        self.assertFalse(result.diagnostics, str(result.diagnostics))
        self.assertIn('select(left = true, middle = false, next = true, right = false)', result.source.text)

    def test_custom_condition_provider_inherits_canonical_wrapping(self):
        from importlib import import_module
        from py_compiler.syntax.recognition import Syntax
        class When(import_module('py_compiler.syntax.block.if').Conditional):
            spelling = 'when'
        class ExtendedSyntax(Syntax):
            @property
            def block_providers(self):
                return {**super().block_providers, 'when': When()}
        result = prepare_source(SourceFile('when.sev', 'when true and false or true and true:\n  unimplemented\n'), ExtendedSyntax())
        self.assertIn('when true\n', result.text)

    def test_variadic_fixed_prefix_does_not_receive_unsupported_named_pack_arguments(self):
        from py_compiler.frontend.src.lib import compile_source
        from py_compiler.syntax.recognition import Syntax
        text = '''def pack(first: int, second: int, third: int, fourth: int, values...) -> int:
    return first + second + third + fourth
def work() -> int:
    return pack(1, 2, 3, 4)
'''
        result = compile_source('pack.sev', text, Syntax(), lint=True)
        self.assertFalse(result.diagnostics, str(result.diagnostics))
        self.assertIn('pack(1, 2, 3, 4)', result.source.text)

    def test_opaque_callable_cannot_silently_bypass_required_argument_names(self):
        from py_compiler.frontend.src.lib import compile_source
        from py_compiler.syntax.recognition import Syntax
        text = '''def work(operation: (int,int,int,int) -> int) -> int:
    return operation(1, 2, 3, 4)
'''
        result = compile_source('opaque.sev', text, Syntax(), lint=True)
        self.assertIn('callable type has no parameter names', str(result.diagnostics))

    def test_long_default_literal_and_global_literal_stay_constant(self):
        from py_compiler.frontend.src.lib import compile_source
        from py_compiler.syntax.recognition import Syntax
        from py_compiler.frontend.lint.layout import quoted
        value = 'long literal text ' * 12
        text = ('globalText = ' + quoted(value) + '\nclass Text:\n    value: string = ' + quoted(value) +
                '\ndef work() -> int:\n    text = Text()\n    return len(text.value) + len(globalText)\n')
        result = compile_source('defaults.sev', text, Syntax(), lint=True)
        self.assertFalse(result.diagnostics, str(result.diagnostics))
        self.assertIn('global_text', result.source.text)
        self.assertTrue(all(len(line) <= 80 for line in result.source.text.splitlines()))

    def test_desugared_interval_constructor_is_not_a_textual_call_fix(self):
        from py_compiler.syntax.recognition import Syntax
        from py_compiler.frontend.lexer.lexer import lex
        from py_compiler.syntax.grammar.expression import expression, Expression
        source = SourceFile('interval.sev', 'list(0..3)')
        term = expression(tuple(token for token in lex(source, Syntax()) if token.kind not in ('EOF', 'NEWLINE')))
        interval = term.operands[1]
        flags = tuple(Expression('literal', interval.token.with_text('true')) for _ in range(2))
        collector = ResolvedLint(source, Syntax())
        collector.arguments((*interval.operands, *flags), ('start', 'stop', 'start_closed', 'stop_closed'))
        self.assertEqual(collector.finish().text, source.text)

    def test_literal_spreads_receive_names_and_nontrivial_spreads_are_diagnosed(self):
        from py_compiler.frontend.src.lib import compile_source
        from py_compiler.syntax.recognition import Syntax
        text = '''def pack(first: int, second: int, third: int, fourth: int) -> int:
    return first
def work() -> int:
    return pack(...(1, 2, 3, 4))
'''
        result = compile_source('spread.sev', text, Syntax(), lint=True)
        self.assertFalse(result.diagnostics, str(result.diagnostics))
        self.assertIn('pack(first = 1, second = 2, third = 3, fourth = 4)', result.source.text)
        text = '''def pack(first: int, second: int, third: int, fourth: int) -> int:
    return first
def arguments() -> tuple[int,4]:
    return tuple[int,4](1, 2, 3, 4)
def work() -> int:
    return pack(...arguments())
'''
        result = compile_source('spread.sev', text, Syntax(), lint=True)
        self.assertIn('nonvariadic spread requires explicit named arguments', str(result.diagnostics))

    def test_native_dependency_cache_tracks_linted_source_imports(self):
        import json
        import subprocess
        from py_compiler.packaging.executable import build_source
        with tempfile.TemporaryDirectory(prefix='sev-lint-cache-') as directory:
            root = Path(directory)
            library = root / 'library'
            (library / 'src').mkdir(parents=True)
            (library / 'package.json').write_text(json.dumps({
                'package': {'name': 'lint-library', 'version': '0.1.0'},
                'sources': ['src/lib.sev'],
            }))
            helper = library / 'src/helper.sev'
            helper.write_text('def answer() -> int:\n  return 42\n')
            (library / 'src/lib.sev').write_text('from "helper.sev" import *\n')
            application = root / 'application'
            application.mkdir()
            (application / 'package.json').write_text(json.dumps({
                'package': {'name': 'lint-application', 'version': '0.1.0'},
                'dependencies': {'library': {'path': '../library'}},
            }))
            source = application / 'main.sev'
            source.write_text('import library\ndef main() -> int:\n  return library.answer()\n')
            built = build_source(source)
            self.assertEqual(subprocess.run([built['executable']], timeout=5).returncode, 42)
            self.assertTrue(build_source(source)['reused'])
            metadata = json.loads((library / 'package.pkg/metadata/realizations/dev.json').read_text())
            self.assertIn('src/helper.sev', metadata['source-inputs'])
            helper.write_text(helper.read_text().replace('return 42', 'return 43'))
            rebuilt = build_source(source)
            self.assertFalse(rebuilt['reused'])
            self.assertEqual(subprocess.run([rebuilt['executable']], timeout=5).returncode, 43)

    def test_long_test_labels_and_skip_reasons_use_their_own_grammar(self):
        from py_compiler.frontend.src.lib import compile_source
        from py_compiler.syntax.recognition import Syntax
        from py_compiler.frontend.lint.layout import quoted
        label = 'descriptive label ' * 9
        text = 'test ' + quoted(label) + ':\n    assert(true)\n'
        result = compile_source('label.sev', text, Syntax(), lint=True)
        self.assertFalse(result.diagnostics, str(result.diagnostics))
        self.assertEqual(next(body.test['label'] for body in result.program.bodies if body.test).split(), label.split())
        reason = 'explicit skip reason ' * 8
        result = compile_source('skip.sev', 'test with skip(' + quoted(reason) + '):\n    unresolved()\n', Syntax(), lint=True)
        self.assertFalse(result.diagnostics, str(result.diagnostics))

    def test_wrapped_allocation_arguments_accept_canonical_trailing_comma(self):
        from py_compiler.frontend.src.lib import compile_source
        from py_compiler.syntax.recognition import Syntax
        text = ('def work() -> int:\n    unsafe:\n'
                '        values = allocate[tensor[int,(1,2,3)],location=memory.stack,alignment=64]()\n'
                '        values[0,0,0] = 7\n        return values[0,0,0]\n')
        result = compile_source('wrapped-allocation.sev', text, Syntax(), lint=True)
        self.assertFalse(result.diagnostics, str(result.diagnostics))
        self.assertIn('alignment = 64,\n', result.source.text)

    def test_partial_named_mapping_labels_every_argument(self):
        from py_compiler.frontend.src.lib import compile_source
        from py_compiler.syntax.recognition import Syntax
        text = 'def foo(first: int, second: int, third: int) -> int:\n    return first + second + third\ndef work() -> int:\n    return foo(1, 2, third=3)\n'
        result = compile_source('mapping.sev', text, Syntax(), lint=True)
        self.assertFalse(result.diagnostics, str(result.diagnostics))
        self.assertIn('first = 1,', result.source.text)
        self.assertIn('second = 2,', result.source.text)
        self.assertIn('third = 3)', result.source.text)

    def test_enum_defaults_expand_and_new_variants_require_a_new_case(self):
        from py_compiler.frontend.src.lib import compile_source
        from py_compiler.syntax.recognition import Syntax
        for variants, pattern in (('    A\n    B\n', 'Kind.B'),
                                  ('    A(value: int)\n    B\n    C\n', 'Kind.B | Kind.C')):
            text = 'enum Kind:\n' + variants + 'def work(value: Kind) -> int:\n    return match value:\n        case A:\n            1\n        case _:\n            2\n'
            result = compile_source('enum-default.sev', text, Syntax(), lint=True)
            self.assertFalse(result.diagnostics, str(result.diagnostics))
            self.assertIn('case ' + pattern + ':', result.source.text)
            changed = result.source.text.replace('enum Kind:\n', 'enum Kind:\n    Added\n')
            invalid = compile_source('enum-default.sev', changed, Syntax(), lint=True)
            self.assertIn('non-exhaustive', str(invalid.diagnostics))

    def test_grouped_operations_preserve_sequential_execution(self):
        from py_compiler.frontend.src.lib import compile_source
        from py_compiler.syntax.recognition import Syntax
        text = ('def foo(value: int) -> int:\n    return value\n'
                'test:\n    mock(foo(0) -> 1)\n    mock(foo(0) -> 2)\n    mock(foo(0) -> 3)\n'
                '    assert(foo(0) == 3)\n    assert(true)\n    assert(true)\n'
                '    when true:\n        assert(true)\n    when true:\n        assert(true)\n    when false:\n        assert(false)\n    else:\n        assert(true)\n'
                '    throws "expected":\n        throw Error("expected")\n'
                '    throws "expected":\n        throw Error("expected")\n'
                '    throws "expected":\n        throw Error("expected")\n')
        result = compile_source('grouped.sev', text, Syntax(), lint=True)
        self.assertFalse(result.diagnostics, str(result.diagnostics))
        for operation in ('assert', 'mock', 'when', 'throws'):
            self.assertIn('    ' + operation + ':\n', result.source.text)
        import json
        from py_compiler.packaging.testing import test_package
        with tempfile.TemporaryDirectory(prefix='sev-grouped-native-') as directory:
            root = Path(directory)
            (root / 'lib.sev').write_text(text)
            (root / 'package.json').write_text(json.dumps({'package': {'name': 'grouped-native', 'version': '0.1.0'}, 'sources': ['lib.sev']}))
            built = test_package(root, timeout=5)
            detail = json.loads(Path(built['report']).read_text()) if built.get('report') else built
            self.assertTrue(built['success'], detail)

    def test_formatted_print_block_retains_exact_native_stdout(self):
        import json
        from py_compiler.packaging.testing import test_package
        from py_compiler.frontend.lint.layout import quoted
        value = 'literal {braces} with exact spacing ' * 8
        text = 'test with integ "print block":\n    print(' + quoted(value) + ')\n    assert(stdout == ' + quoted(value + '\n') + ')\n'
        with tempfile.TemporaryDirectory(prefix='sev-print-block-') as directory:
            root = Path(directory)
            path = root / 'lib.sev'
            path.write_text(text)
            (root / 'package.json').write_text(json.dumps({'package': {'name': 'print-block', 'version': '0.1.0'}, 'sources': ['lib.sev']}))
            built = test_package(root, timeout=5)
            detail = json.loads(Path(built['report']).read_text()) if built.get('report') else built
            self.assertTrue(built['success'], detail)
            self.assertIn('print(\n        """', path.read_text())

    def test_nonconstant_labels_and_malformed_interpolation_are_diagnostics(self):
        from py_compiler.frontend.src.lib import compile_source
        from py_compiler.syntax.recognition import Syntax
        for text in ('test f"label:{missing}":\n    assert(true)\n',
                     'def work() -> string:\n    return f"{value+}"\n'):
            result = compile_source('invalid-format.sev', text, Syntax(), lint=True)
            self.assertTrue(result.diagnostics)
