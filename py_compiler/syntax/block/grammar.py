"""Class sentence grammars preserve typed captures until selected expansion."""
import ast
from dataclasses import dataclass, replace
from py_compiler.syntax.generic.block import DeclarationProvider
from py_compiler.syntax.generic.grammar import Grammar as GrammarContract, Match, Registry, SourceWindow
from py_compiler.syntax.grammar.expression import Expression, expression


@dataclass(frozen=True)
class CaptureTerm:
    term: object
    env: dict
    outer: dict
    contract: object
    active_grammars: frozenset

    def apply(self, cfg, operation):
        previous = cfg.grammar_captures, cfg.active_grammars
        cfg.grammar_captures = self.outer
        # Caller expressions may use this same grammar at a deeper source
        # position. Only expansion of the grammar's own body is recursive.
        cfg.active_grammars = set(self.active_grammars)
        try:
            return operation()
        finally:
            cfg.grammar_captures, cfg.active_grammars = previous

    def type(self, cfg, expected=None):
        from py_compiler.syntax.generic.place import expression_type
        return self.apply(cfg, lambda: expression_type(self.term, cfg, self.env, expected or self.contract))

    def lower(self, cfg, expected=None):
        value = self.apply(cfg, lambda: cfg.expr(self.term, self.env, self.contract))
        if value is None or value.type != self.contract:
            raise ValueError('grammar capture does not satisfy its declared type')
        from py_compiler.syntax.function.calls import coerce
        return coerce(value, expected, cfg, self.term.token.span)

    def binding(self, cfg):
        from py_compiler.syntax.generic.storage import root_binding
        return self.apply(cfg, lambda: root_binding(cfg.capture_expression(self.term), cfg, self.env))


def capture_expression(node, captures):
    if not isinstance(node, Expression):
        return node
    if node.kind == 'capture':
        return node
    if node.kind == 'name' and node.token.text in captures:
        return Expression('capture', node.token, (captures[node.token.text],))
    if node.kind == 'reference' and node.token.text in captures:
        receiver = Expression('capture', node.token, (captures[node.token.text],))
        return Expression('member', node.operands[0], (receiver,))
    return replace(node, operands=tuple(capture_expression(part, captures) for part in node.operands))


class Grammar(DeclarationProvider):
    spelling = 'grammar'

    def parse_header(self, items):
        header = tuple(items[1:-1])
        if len(header) < 5 or header[0].kind != 'IDENTIFIER' or header[1].text != '[':
            raise ValueError('expected grammar name[arrangement]():')
        depth, end = 1, None
        for index in range(2, len(header)):
            depth += (header[index].text == '[') - (header[index].text == ']')
            if depth == 0:
                end = index
                break
        tail = tuple(t.text for t in header[end + 1:]) if end is not None else ()
        if tail[:2] != ('(', ')') or (len(tail) > 2 and (tail[2] != '->' or len(tail) < 4)):
            raise ValueError('sentence grammar requires an empty () signature')
        groups, group, depth = [], [], 0
        for token in header[2:end]:
            if token.text == ',' and depth == 0:
                groups.append(group); group = []
            else:
                group.append(token)
                if token.kind not in ('STRING', 'CHAR'):
                    depth += (token.text in ('[', '(')) - (token.text in (']', ')'))
        groups.append(group)
        arrangement, names = [], set()
        for group in groups:
            if len(group) == 1 and group[0].kind in ('STRING', 'CHAR'):
                spelling = ast.literal_eval(group[0].text)
                if not spelling:
                    raise ValueError('grammar literal must make progress')
                arrangement.append(spelling)
            elif group and (group[0].text == 'self' and len(group) == 1 or len(group) > 2 and group[1].text == ':'):
                name = group[0].text
                if name in names:
                    raise ValueError('duplicate grammar capture ' + name)
                names.add(name)
                arrangement.append((name, 'Self' if name == 'self' else ''.join(t.text for t in group[2:])))
            else:
                raise ValueError('grammar captures require a name and type')
        if 'self' not in names or not any(isinstance(part, str) for part in arrangement):
            raise ValueError('class grammar requires self and a literal boundary')
        return header[0].text, tuple(arrangement), ''.join(tail[3:]) if len(tail) > 2 else None

    def declare(self, node, context):
        raise ValueError('sentence grammar requires a class owner')

    def declare_member(self, node, context, owner):
        name, arrangement, result_name = node.header
        result = context.type(result_name) if result_name else None
        if result == context.type('absent'):
            result = None
        iterator = result is not None and result.family == 'iterator'
        if result is not None and not iterator:
            if result.binding_ownership != 'copy' and result.family != 'record':
                raise ValueError('value grammar results currently require copy value types')
            if len(node.children) != 1 or node.children[0].tokens[0].text != 'return':
                raise ValueError('value grammar requires a single return expression')
        def validate(nodes):
            for child in nodes:
                if isinstance(child.provider, DeclarationProvider):
                    raise ValueError('grammar expansion cannot introduce declarations in this implementation')
                if child.tokens[0].text in (('return',) if iterator else ('break', 'continue') if result else ('return', 'break', 'continue')):
                    raise ValueError('grammar expansion requires structured local control flow')
                validate(child.children)
        validate(node.children)
        resolved = tuple(part if isinstance(part, str) else
                         (part[0], context.type(owner.name if part[1] == 'Self' else part[1])) for part in arrangement)
        context.sentence_grammars.append((node, resolved, context.scope, result))
        context.declared_nodes.add(node.identity)


class BoundGrammar(GrammarContract):
    role = 'O.custom'

    def __init__(self, declaration, cfg, env):
        self.node, self.arrangement, self.scope, self.result = declaration
        self.cfg, self.env = cfg, env

    def recognize(self, window):
        tokens = window.tokens
        solutions = []
        def match(part, position, captures):
            if part == len(self.arrangement):
                if position == len(tokens):
                    solutions.append(captures)
                return
            item = self.arrangement[part]
            if isinstance(item, str):
                text = ''
                for end in range(position, len(tokens)):
                    if end > position and tokens[end - 1].span.end != tokens[end].span.start:
                        break
                    text += tokens[end].text
                    if text == item:
                        match(part + 1, end + 1, captures)
                    if not item.startswith(text):
                        break
                return
            name, contract = item
            for end in range(position + 1, len(tokens) + 1):
                try:
                    term = expression(tokens[position:end])
                except ValueError:
                    continue
                capture = CaptureTerm(term, self.env, self.cfg.grammar_captures, contract,
                                      frozenset(self.cfg.active_grammars))
                match(part + 1, end, {**captures, name: capture})
        match(0, 0, {})
        eligible = []
        for captures in solutions:
            try:
                valid = all(capture.type(self.cfg) == capture.contract for capture in captures.values())
            except ValueError:
                valid = False
            if valid:
                eligible.append(captures)
        if len(eligible) > 1:
            raise ValueError('ambiguous grammar capture boundaries')
        return Match(self, window, window.end, eligible[0]) if eligible else None

    def construct(self, match):
        return match.captures

    def lower(self, captures):
        cfg = self.cfg
        if self.node.identity in cfg.active_grammars:
            raise ValueError('recursive grammar expansion requires a terminating lowering')
        if not cfg.block_scope.allows(self.node.scope):
            raise ValueError('production scope cannot use a test grammar')
        previous = cfg.grammar_captures
        cfg.grammar_captures = captures
        cfg.active_grammars.add(self.node.identity)
        cfg.grammar_expansions += 1
        suffix = ':grammar:' + str(cfg.grammar_expansions)
        def clone(node):
            return replace(node, identity=node.identity + suffix, children=[clone(child) for child in node.children])
        try:
            # Captures reference caller bindings explicitly; expansion locals cannot capture caller names.
            cfg.scope([clone(child) for child in self.node.children], {}, local_names=set())
        finally:
            cfg.active_grammars.remove(self.node.identity)
            cfg.grammar_captures = previous


def expand_sentence(node, cfg, env):
    if not cfg.context.sentence_grammars:
        return False
    providers = [BoundGrammar(declaration, cfg, env) for declaration in cfg.context.sentence_grammars
                 if declaration[3] is None and tuple(cfg.declaration_scope[:len(declaration[2])]) == declaration[2]]
    window = SourceWindow(cfg.source, node.tokens[0].span.start, node.tokens[-1].span.end, node.tokens)
    match = Registry(providers).recognize('O.custom', window, optional=True)
    if match is None:
        return False
    match.provider.lower(match.provider.construct(match))
    return True


def expression_grammar(term, cfg, env):
    tokens = getattr(term, 'tokens', ())
    if not tokens or term.kind == 'capture':
        return None
    providers = [BoundGrammar(d, cfg, env) for d in cfg.context.sentence_grammars
                 if d[3] is not None and tuple(cfg.declaration_scope[:len(d[2])]) == d[2]
                 and d[0].identity not in cfg.active_grammars]
    if not providers:
        return None
    window = SourceWindow(cfg.source, tokens[0].span.start, tokens[-1].span.end, tokens)
    return Registry(providers).recognize('O.custom', window, optional=True)


def lower_result(match, cfg):
    provider = match.provider
    if provider.result.family == 'iterator':
        raise ValueError('iterator values currently require direct for consumption')
    if not cfg.block_scope.allows(provider.node.scope):
        raise ValueError('production scope cannot use a test grammar')
    previous = cfg.grammar_captures
    cfg.grammar_captures = match.captures
    cfg.active_grammars.add(provider.node.identity)
    try:
        value = cfg.expr(expression(provider.node.children[0].tokens[1:]), {}, provider.result)
        if value.type != provider.result:
            raise ValueError('grammar result does not satisfy its declared type')
        if value.type.family == 'record':
            term = expression(provider.node.children[0].tokens[1:])
            if term.kind != 'call' or cfg.value_sources.get(value.identity):
                raise ValueError('class value grammar must return an owned callable result')
            if not hasattr(cfg, 'owned_results'):
                cfg.owned_results = set()
            cfg.owned_results.add(value.identity)
        return value
    finally:
        cfg.grammar_captures = previous
        cfg.active_grammars.remove(provider.node.identity)


import unittest


class SentenceGrammarTests(unittest.TestCase):
    source = '''class Buffer:
    storage: array[int]
    grammar assign[self, "[", index: int, "]", "=", value: int]():
        self.storage[index] = value
    grammar put["put", value: int, "into", self, "at", index: int]():
        self.storage[index] = value
    grammar wrapped[self, "[", index: int, "]", "<-", value: int]():
        self.storage[index % len(self.storage)] = value

test:
    buffer = Buffer(allocate[int](4))
    buffer[1] = 42
    put 7 into buffer at 2
    assert(buffer.storage[1] == 42)
    assert(buffer.storage[2] == 7)
    buffer[5] <- 11
    assert(buffer.storage[1] == 11)
'''

    def compile(self, source):
        from py_compiler.frontend.src.lib import compile_source
        from py_compiler.syntax.recognition import Syntax
        return compile_source('grammar-storage.sev', source, Syntax(include_tests=True))

    def test_class_arrangements_reuse_checked_storage(self):
        result = self.compile(self.source)
        self.assertFalse(result.diagnostics, str(result.diagnostics))
        from py_compiler.mlir.src.lib import render, lower
        text = render(lower(result.program))
        self.assertIn('memref.store', text)
        self.assertIn('memory index outside allocation', text)

    def test_custom_store_cannot_mutate_a_view(self):
        source = self.source.replace('    buffer[1] = 42', '    alias = view buffer\n    alias[1] = 42')
        result = self.compile(source)
        self.assertTrue(result.diagnostics)
        self.assertIn('view', str(result.diagnostics))

    def test_ambiguous_grammars_do_not_execute_either_body(self):
        duplicate = ('    grammar duplicate[self, "[", index: int, "]", "=", value: int]():\n'
                     '        missing_function()\n')
        result = self.compile(self.source.replace('\ntest:', '\n' + duplicate + '\ntest:'))
        self.assertTrue(result.diagnostics)
        self.assertIn('ambiguous O.custom grammar', str(result.diagnostics))
        self.assertNotIn('no callable', str(result.diagnostics))

    def test_selected_failure_does_not_fall_back(self):
        result = self.compile(self.source.replace('        self.storage[index] = value', '        missing_function()', 1))
        self.assertTrue(result.diagnostics)
        self.assertIn('no callable', str(result.diagnostics))

    def test_capture_type_mismatch_never_expands_body(self):
        result = self.compile(self.source.replace('buffer[1] = 42', 'buffer[1] = true'))
        self.assertTrue(result.diagnostics)

    def test_default_storage_evaluates_rhs_before_index(self):
        source = '''def index() -> int:
    return 0
def rhs() -> int:
    return 7
def work():
    values = allocate[int](1)
    values[index()] = rhs()
'''
        result = self.compile(source)
        self.assertFalse(result.diagnostics, str(result.diagnostics))
        body = next(body for body in result.program.bodies if body.declaration == 'work')
        calls = [operation.payload.symbol for block in body.blocks for operation in block.operations if operation.kind == 'call']
        self.assertEqual(len(calls), 2)
        self.assertIn('rhs', calls[0])
        self.assertIn('index', calls[1])

    def test_nested_index_assignment_uses_common_target_resolution(self):
        source = '''test:
    a = allocate[int](4)
    b = allocate[int](4)
    c = allocate[int](4)
    a[0] = 1
    b[1] = 2
    c[2] = 42
    c[0] = 3
    b[3] = 2
    a[b[c[0]]] = c[b[a[0]]]
    assert(a[2] == 42)
'''
        result = self.compile(source)
        self.assertFalse(result.diagnostics, str(result.diagnostics))

    def test_result_grammar_declaration_is_typed(self):
        result = self.compile('class Value:\n    value: int\n    grammar read[self, "[", index: int, "]"]() -> int:\n        return self.value\ntest:\n    item = Value(42)\n    assert(item[0] == 42)\n')
        self.assertFalse(result.diagnostics, str(result.diagnostics))

    def test_procedural_grammar_requires_a_recognition_contract(self):
        result = self.compile('grammar block(input: SourceWindow) -> B:\n    unimplemented\n')
        self.assertTrue(result.diagnostics)
        self.assertIn('expected grammar name', str(result.diagnostics))

    def test_imported_grammar_requires_an_interface_definition(self):
        from py_compiler.frontend.src.lib import compile_source
        from py_compiler.syntax.recognition import Syntax
        package = {'exports': [], 'objects': ['/unused.o'], 'types': [
            {'name': 'Buffer', 'fields': [('value', 'int')], 'defaults': [0], 'memory-defaults': []}]}
        result = compile_source('imported-grammar.sev',
            'import library\ntest:\n    value = library.Buffer()\n    value[0] = 1\n',
            Syntax(include_tests=True), native_packages={'library': package})
        self.assertTrue(result.diagnostics)
        self.assertIn('no indexed storage contract', str(result.diagnostics))
