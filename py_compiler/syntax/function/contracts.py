"""Callable declarations retain parameter, result and acceptance contracts."""
from dataclasses import dataclass, field
from py_compiler.syntax.grammar.expression import expression


@dataclass
class Callable:
    node: object
    name: str
    parameters: tuple
    result: str | None
    guards: tuple
    receiver: object = None
    body: object = None
    compiling: bool = False
    scope: tuple = ()
    complexity: dict = field(default_factory=dict)
    obligations: tuple = ()
    suffix: tuple = ()
    failure_actions: tuple = ()
    imports: tuple = ()
    parameter_ownership: dict = field(default_factory=dict)
    templates: tuple = ()
    type_arguments: dict = field(default_factory=dict)
    callable_arguments: dict = field(default_factory=dict)
    realization_key: tuple = ()

    defaults: dict = field(default_factory=dict)
    receiver_ownership: str | None = None
    declaration_signature: tuple | None = None
    symbol_namespace: str = ''
    result_ownership: str = 'own'
    infer_result_ownership: bool = False
    variadic: str | None = None

    def __post_init__(self):
        # Inference rewrites parameters after compilation; names must stay stable
        # between recursive calls, body creation and later references.
        if self.declaration_signature is None:
            self.declaration_signature = tuple(annotation or '?' for _, annotation in self.parameters)

    @property
    def symbol(self):
        source = self.node.tokens[0].lexeme.source
        line, column = source.position(self.node.span.start)
        # Nested lexical scopes currently contain source snapshot identities.
        # Expose their offsets without putting the snapshot hash in the name.
        def readable(value):
            return str(value).replace(source.identity + ':', 'scope_')
        if self.realization_key:
            arguments = [readable(value) if kind == 'type' else
                         parameter + '=' + readable(value).removeprefix('__sev_fn_')
                         for parameter, kind, value in self.realization_key]
        elif self.templates:
            arguments = [parameter.name for parameter in self.templates]
        else:
            arguments = [readable(annotation) for annotation in self.declaration_signature]
        # The source qualifier distinguishes overload declarations and files.
        # It deliberately avoids '@', which ELF linkers use for symbol versions.
        qualifier = '::package=' + self.symbol_namespace if self.symbol_namespace else ''
        return f"__sev_fn_{readable(self.name)}[{','.join(arguments)}]::{source.path}:{line}:{column}{qualifier}"


def signature(node, owner=""):
    from py_compiler.syntax.complex.generic import template_header
    templates, items = template_header(list(node.header))
    if len(items) < 3 or not items[0].text.isidentifier() or items[1].text != "(":
        raise ValueError("expected def name(parameters) -> Type")
    depth, end = 0, -1
    for index in range(1, len(items)):
        depth += (items[index].text == '(') - (items[index].text == ')')
        if depth == 0:
            end = index
            break
    if end < 2:
        raise ValueError("missing parameter-list delimiter")
    groups, group, depth = [], [], 0
    for token in items[2:end]:
        if token.text == ',' and depth == 0:
            groups.append(group); group = []
        else:
            group.append(token)
            depth += (token.text in ('[', '(', '{')) - (token.text in (']', ')', '}'))
    if group:
        groups.append(group)
    parameters, parameter_ownership, defaults = [], {}, {}
    receiver_ownership = None
    variadic = None
    for group_index, group in enumerate(groups):
        if not group:
            raise ValueError('empty parameter')
        name = group[0].text
        if group[-1].text == '...':
            if group_index != len(groups) - 1 or len(group) < 2 or defaults:
                raise ValueError('variadic parameter must be last and cannot follow defaults')
            variadic = name
            group = group[:-1]
        if name == 'self':
            if not owner or parameters or receiver_ownership is not None:
                raise ValueError('self must be the first method parameter')
            words = [t.text for t in group[1:]]
            if not words:
                receiver_ownership = 'view'
            elif words[:1] == [':'] and words[-1:] == ['Self']:
                receiver_ownership = words[1] if len(words) == 3 else 'view'
                if receiver_ownership not in ('view', 'borrow'):
                    raise ValueError('receiver currently supports view or borrow')
            else:
                raise ValueError('expected self: view Self or self: borrow Self')
            continue
        if group[0].kind != 'IDENTIFIER' or any(p[0] == name for p in parameters):
            raise ValueError('parameter name is invalid or duplicated')
        cursor, annotation = 1, None
        if cursor < len(group) and group[cursor].text == ':':
            cursor += 1
            if cursor < len(group) and group[cursor].text in ('view', 'borrow', 'move', 'copy', 'mirror'):
                parameter_ownership[name] = group[cursor].text; cursor += 1
            start = cursor
            while cursor < len(group) and group[cursor].text != '=':
                cursor += 1
            annotation = ''.join(t.text for t in group[start:cursor])
            if not annotation:
                raise ValueError('parameter annotation is missing')
        if cursor < len(group):
            if group[cursor].text != '=':
                raise ValueError('expected a parameter default')
            default = expression(group[cursor + 1:])
            def literal_default(value):
                if value.kind == 'literal':
                    return True
                if value.kind in ('list', 'tuple'):
                    return all(literal_default(element) for element in value.operands)
                return (value.kind == 'unary' and value.token.text in ('+', '-')
                        and value.operands[0].kind == 'literal')
            if not literal_default(default):
                raise ValueError('parameter default requires a literal or collection of literals')
            defaults[name] = default
        elif defaults:
            raise ValueError('required parameter cannot follow a default parameter')
        parameters.append((name, None if annotation == 'infer' else annotation))
    cursor, result, guards = end + 1, None, []
    result_ownership = 'own'
    complexity, obligations, suffix, failure_actions = {}, [], [], []
    if cursor < len(items) and items[cursor].text == "->":
        if cursor + 1 == len(items):
            raise ValueError("missing result annotation")
        cursor += 1
        if items[cursor].text in ('borrow', 'view', 'move'):
            result_ownership = 'own' if items[cursor].text == 'move' else items[cursor].text
            cursor += 1
        start = cursor
        while cursor < len(items) and items[cursor].text != 'with':
            cursor += 1
        result = ''.join(t.text for t in items[start:cursor])
        if result == 'infer':
            result = None
    if cursor < len(items):
        if items[cursor].text != 'with':
            raise ValueError('expected with before acceptance clauses')
        from importlib import import_module
        for clause in import_module('py_compiler.syntax.block.with').clause_groups(items[cursor + 1:]):
            add_clause(clause, guards, obligations, complexity, suffix, failure_actions)
    for index, token in enumerate(items):
        if token.text == "complexity" and index >= 2 and items[index - 1].text == "." and items[index - 2].text not in (items[0].text, "F"):
            raise ValueError("complexity contract must name its declaring function")
    # Unspecified parameter types are implicit templates, checked against each
    # caller's concrete types. Variadic parameters already specialize as packs.
    if not variadic:
        from py_compiler.syntax.complex.generic import TemplateParameter
        reserved = {token.text for token in node.tokens}
        inferred = []
        for index, (name, annotation) in enumerate(parameters):
            if annotation is None:
                annotation = '__inferred_' + name
                while annotation in reserved:
                    annotation += '_'
                reserved.add(annotation)
                inferred.append(TemplateParameter(annotation))
            parameters[index] = (name, annotation)
        templates = (*templates, *inferred)
    entry = Callable(node, (owner + "." if owner else "") + items[0].text,
                    tuple(parameters), result, tuple(guards), complexity=complexity, obligations=tuple(obligations), suffix=tuple(suffix), failure_actions=tuple(failure_actions), parameter_ownership=parameter_ownership, templates=templates, defaults=defaults, receiver_ownership=receiver_ownership)
    entry.result_ownership = result_ownership
    entry.variadic = variadic
    return entry


def pure_predicate(node, parameters, context=None):
    if node.kind == 'call' and len(node.operands) == 2:
        callee, argument = node.operands
        if context is not None and callee.kind == 'name' and not context.lookup_functions(callee.token.text, context.scope):
            from py_compiler.syntax.prelude import builtin_call_providers
            provider = builtin_call_providers().get(callee.token.text)
            proof = getattr(provider, 'predicate_argument', None)
            if proof and proof(argument, parameters, context):
                pure_predicate(argument, parameters, context)
                return
        if callee.kind == 'member' and callee.token.text == 'suffix':
            pure_predicate(callee.operands[0], parameters, context)
            pure_predicate(argument, parameters, context)
            return
    if node.kind == "chain":
        for operand in node.operands[::2]:
            pure_predicate(operand, parameters, context)
        return
    if node.kind in ("member", "reference") and node.operands and node.token.text == "self" and "self" in parameters:
        return
    if node.kind == "member":
        pure_predicate(node.operands[0], parameters, context)
        return
    if node.kind == "literal":
        return
    if node.kind == "name" and node.token.text in parameters:
        return
    if node.kind == "unary" and node.token.text in ("not", "+", "-"):
        pure_predicate(node.operands[0], parameters, context)
        return
    if node.kind == "binary" and node.token.text in ("and", "or", "==", "!=", "<", "<=", ">", ">=", "in"):
        for operand in node.operands:
            pure_predicate(operand, parameters, context)
        return
    raise ValueError("dispatch predicate lacks a pure, terminating, stable-read provider")


def integer_dispatch_domain(entry, context):
    """Prove only conjunctions of integer bounds; unsupported predicates stay unknown."""
    bounds = {}
    names = {name: (index, annotation) for index, (name, annotation) in enumerate(entry.parameters)}

    def path(term):
        fields = []
        while term.kind == 'member':
            fields.insert(0, term.token.text)
            term = term.operands[0]
        if term.kind != 'name' or term.token.text not in names:
            return None
        index, annotation = names[term.token.text]
        type_ = context.type(annotation)
        for field in fields:
            annotation = dict(getattr(type_.declaration, 'fields', ())).get(field) if hasattr(type_, 'declaration') else None
            if annotation is None:
                return None
            type_ = context.type(annotation)
        if type_.family != 'integer':
            return None
        key = (index, *fields)
        minimum = -(1 << (type_.bits - 1)) if type_.signed else 0
        maximum = (1 << (type_.bits - (1 if type_.signed else 0))) - 1
        bounds.setdefault(key, [minimum, maximum])
        return key

    def number(term):
        sign = 1
        if term.kind == 'unary' and term.token.text in ('-', '+'):
            sign = -1 if term.token.text == '-' else 1
            term = term.operands[0]
        if term.kind != 'literal' or term.token.kind != 'NUMBER':
            return None
        try:
            return sign * int(term.token.text.replace('_', ''), 0)
        except ValueError:
            return None

    def comparison(left, operator, right):
        key, value = path(left), number(right)
        if key is None or value is None:
            key, value = path(right), number(left)
            operator = {'<': '>', '<=': '>=', '>': '<', '>=': '<=', '==': '=='}.get(operator)
        if key is None or value is None or operator not in ('<', '<=', '>', '>=', '=='):
            return False
        interval = bounds[key]
        if operator in ('>', '>=', '=='):
            interval[0] = max(interval[0], value + (operator == '>'))
        if operator in ('<', '<=', '=='):
            interval[1] = min(interval[1], value - (operator == '<'))
        return True

    def visit(term):
        if term.kind == 'binary' and term.token.text == 'and':
            return all(visit(child) for child in term.operands)
        if term.kind == 'binary':
            return comparison(term.operands[0], term.token.text, term.operands[1])
        if term.kind == 'chain':
            return all(comparison(term.operands[i - 1], term.operands[i].text, term.operands[i + 1])
                       for i in range(1, len(term.operands), 2))
        return False

    return bounds if all(visit(guard) for guard in entry.guards) else None


from py_compiler.syntax.complex.big_o import BigO


def add_clause(tokens, guards, obligations, complexity, suffix, failure_actions):
    depth, action = 0, None
    for index, token in enumerate(tokens):
        if token.text == '->' and depth == 0:
            action = expression(tokens[index + 1:])
            if action.kind != 'call':
                raise ValueError('contract failure action requires a call')
            tokens = tokens[:index]
            break
        depth += (token.text in ('(', '[', '{')) - (token.text in (')', ']', '}'))
    if not tokens:
        raise ValueError('contract requires a condition before ->')
    words = [t.text for t in tokens]
    from py_compiler.syntax.prelude import clause_providers
    provider = clause_providers().get(words[0])
    if provider is not None:
        lengths = tuple(len(items) for items in (guards, obligations, suffix))
        provider.attach(tokens[1:], guards, obligations, suffix)
        if action is not None:
            for items, length in zip((guards, obligations, suffix), lengths):
                if len(items) > length:
                    failure_actions.append((items[-1], action))
        return
    if ':=' in words:
        if action is not None:
            raise ValueError('complexity declarations do not take failure actions')
        if len(words) != 9 or words[1:4] != ['.', 'complexity', '.'] or words[4] not in ('time', 'space') or words[5:8] != [':=', 'BigO', '.']:
            raise ValueError('expected function.complexity.time/space := BigO.variant')
        try:
            value = BigO[words[8]]
        except KeyError:
            raise ValueError('unknown BigO variant') from None
        if words[4] in complexity:
            raise ValueError('duplicate complexity contract')
        complexity[words[4]] = value
        return
    condition = expression(tokens)
    guards.append(condition)
    if action is not None:
        failure_actions.append((condition, action))


import unittest


class CallableSymbolTests(unittest.TestCase):
    def test_length_predicate_requires_provider_purity_support(self):
        from py_compiler.frontend.src.lib import compile_source
        from py_compiler.syntax.recognition import Syntax
        result = compile_source('length-contract.sev',
            'def bounded(values: tensor[int]) -> int with len(values) < 10000:\n'
            '    return len(values)\n', Syntax())
        self.assertFalse(result.diagnostics, result.diagnostics)

    def test_packages_with_identical_source_have_distinct_symbols(self):
        from py_compiler.frontend.src.lib import compile_source
        from py_compiler.syntax.recognition import Syntax
        source = 'def value() -> int:\n    return 42\n'
        packages = [compile_source('src/lib.sev', source, Syntax(), interface=name)
                    for name in ('first', 'second')]
        for result in packages:
            self.assertFalse(result.diagnostics, str(result.diagnostics))
        symbols = [{body.name for body in result.program.bodies} for result in packages]
        self.assertTrue(symbols[0].isdisjoint(symbols[1]))

    def declaration(self, path, text):
        from py_compiler.frontend.source.source import SourceFile
        from py_compiler.frontend.lexer.lexer import lex
        from py_compiler.frontend.parser.blocks import parse_blocks
        from py_compiler.syntax.recognition import Syntax
        source, syntax = SourceFile(path, text), Syntax()
        root, errors = parse_blocks(source, lex(source, syntax), syntax)
        self.assertFalse(errors, str(errors))
        return signature(root.children[0])

    def test_readable_declaration_name_is_stable_after_inference(self):
        entry = self.declaration('src/example.sev', 'def identity(value):\n    return value\n')
        original = entry.symbol
        self.assertEqual(original, '__sev_fn_identity[__inferred_value]::src/example.sev:1:1')
        entry.parameters = (('value', 'i32'),)
        self.assertEqual(entry.symbol, original)

    def test_types_callable_arguments_and_source_files_are_distinct(self):
        from dataclasses import replace
        source = 'def identity[T](value: T) -> T:\n    return value\n'
        template = self.declaration('src/one.sev', source)
        integer = replace(template, templates=(), realization_key=(('T', 'type', 'i32'),))
        string = replace(template, templates=(), realization_key=(('T', 'type', 'string'),))
        self.assertTrue(integer.symbol.startswith('__sev_fn_identity[i32]::'))
        self.assertTrue(string.symbol.startswith('__sev_fn_identity[string]::'))
        other = replace(self.declaration('src/two.sev', source), templates=(), realization_key=integer.realization_key)
        self.assertEqual(len({integer.symbol, string.symbol, other.symbol}), 3)
        left = replace(integer, realization_key=(*integer.realization_key, ('F', 'callable', '__sev_fn_left[int]::ops.sev:1:1')))
        right = replace(integer, realization_key=(*integer.realization_key, ('F', 'callable', '__sev_fn_right[int]::ops.sev:3:1')))
        self.assertNotEqual(left.symbol, right.symbol)
        self.assertIn('F=left[int]', left.symbol)


class ParameterDefaultTests(unittest.TestCase):
    def test_collection_defaults_preserve_literal_syntax(self):
        from py_compiler.frontend.src.lib import compile_source
        from py_compiler.syntax.recognition import Syntax
        source = 'def count(values: tuple[int, 2] = (3, 4)) -> int:\n    return values[0] + values[1]\ntest:\n    assert(count() == 7)\n'
        result = compile_source('collection-default.sev', source, Syntax())
        self.assertFalse(result.diagnostics, str(result.diagnostics))

    def test_typed_parameter_defaults_and_nested_delimiters(self):
        from py_compiler.frontend.src.lib import compile_source
        from py_compiler.syntax.recognition import Syntax
        source = 'def scale(value: int, factor: int = 2) -> int:\n    return value * factor\ndef work() -> int:\n    return scale(3)\n'
        result = compile_source('defaults.sev', source, Syntax())
        self.assertFalse(result.diagnostics, str(result.diagnostics))

    def test_required_parameter_after_default_is_rejected(self):
        from py_compiler.frontend.src.lib import compile_source
        from py_compiler.syntax.recognition import Syntax
        result = compile_source('defaults.sev', 'def bad(a: int = 1, b: int) -> int:\n    return b\n', Syntax())
        self.assertTrue(result.diagnostics)

class InferredParameterTests(unittest.TestCase):
    def compile(self, source, **options):
        from py_compiler.frontend.src.lib import compile_source
        from py_compiler.syntax.recognition import Syntax
        return compile_source('inferred.sev',source,Syntax(include_tests=True),**options)

    def test_omitted_and_explicit_infer_specialize_per_consumer(self):
        for annotation in ('',': infer'):
            source='def identity(value'+annotation+') -> infer:\n    local_value: infer = value\n    return local_value\ntest:\n    assert(identity(7) == 7)\n    assert(identity(true))\n'
            result=self.compile(source)
            self.assertFalse(result.diagnostics,str(result.diagnostics))
            bodies=[body for body in result.program.bodies if body.declaration=='identity']
            self.assertEqual({body.result_type.name for body in bodies},{'i64','bool'})
            self.assertEqual(len(bodies),2)

    def test_inferred_defaults_supply_types(self):
        result=self.compile('def identity(value = 7):\n    return value\ntest:\n    assert(identity() == 7)\n    assert(identity(false) == false)\n')
        self.assertFalse(result.diagnostics,str(result.diagnostics))

    def test_length_requirement_is_checked_at_consumer(self):
        for annotation in ('',': infer'):
            definition='def count(value'+annotation+'):\n    return len(value)\n'
            result=self.compile(definition)
            self.assertFalse(result.diagnostics,str(result.diagnostics))
            self.assertEqual(len(result.templates),1)
            valid=self.compile(definition+'class Sized:\n    def len(self: view Self) -> int:\n        return 3\ntest:\n    assert(count(Sized()) == 3)\n')
            self.assertFalse(valid.diagnostics,str(valid.diagnostics))
            invalid=self.compile(definition+'test:\n    count(1.5)\n')
            self.assertTrue(invalid.diagnostics)
            self.assertIn('f64 does not provide len',str(invalid.diagnostics))

    def test_inferred_container_constraints_are_deferred(self):
        source='''def bounded(values: infer) -> int with len(values) < 10:
    return len(values)
class Sized:
    def len(self: view Self) -> int:
        return 3
test:
    assert(bounded(Sized()) == 3)
'''
        result=self.compile(source)
        self.assertFalse(result.diagnostics,str(result.diagnostics))

    def test_imported_inferred_callable_enforces_consumer_operations(self):
        definition='def count(value: infer):\n    return len(value)\n'
        producer=self.compile(definition)
        self.assertFalse(producer.diagnostics,str(producer.diagnostics))
        package={'package':{'name':'inferred'},'objects':['inferred.o'],
                 'exports':[],'types':[],'templates':producer.templates}
        consumer=self.compile('import inferred\ntest:\n    inferred.count(1.5)\n',c_packages={'inferred':package})
        self.assertTrue(consumer.diagnostics)
        self.assertIn('f64 does not provide len',str(consumer.diagnostics))

    def test_inference_cannot_override_an_explicit_borrow(self):
        source='''class Box:
    value: int
def consume(value: move Box) -> move Box:
    return value
def forward(value: borrow infer):
    return consume(value)
test:
    value = Box(3)
    forward(value)
'''
        result=self.compile(source)
        self.assertTrue(result.diagnostics)
        self.assertIn('cannot move owned storage from a view or borrow',str(result.diagnostics))

    def test_c_insertion_sort_infers_container_element_and_ownership(self):
        import json
        import tempfile
        from pathlib import Path
        from py_compiler.packaging.testing import test_package
        source='''from collections import list
from util import range
def insertion_sort(a: infer) -> infer:
    for i in range(1, len(a)):
        key = a[i]
        j = i - 1
        while j >= 0 and a[j] > key:
            a[j + 1] = a[j]
            j -= 1
        a[j + 1] = key
    return a

def implicit(a):
    return insertion_sort(a)

test "inferred integer and float lists have independent specializations":
    empty = implicit(list[int]())
    assert(len(empty) == 0)
    integers = implicit([3, -1, 3, 0, -1])
    assert(integers == [-1, -1, 0, 3, 3])
    floats = implicit([3.5, -1.25, 0.0])
    assert(floats == [-1.25, 0.0, 3.5])
    ordered = implicit([1, 2, 3, 4])
    assert(ordered == [1, 2, 3, 4])
    reversed = implicit([4, 3, 2, 1])
    assert(reversed == [1, 2, 3, 4])
    singleton = implicit([7])
    assert(singleton == [7])
'''
        repository=Path(__file__).resolve().parents[3]
        with tempfile.TemporaryDirectory(prefix='sev-inferred-sort-') as directory:
            root=Path(directory)
            (root/'main.sev').write_text(source)
            (root/'package.json').write_text(json.dumps({'package':{'name':'inferred.sort','version':'0.1.0'},
                'sources':['main.sev'],'dependencies':{
                    'collections':{'path':str(repository/'library/core/collections/collection')},
                    'util':{'path':str(repository/'library/core/utils/c')}}}))
            result=test_package(root,memory=True,coverage=True,timeout=10)
            report=json.loads(Path(result['report']).read_text()) if 'report' in result else result
            self.assertTrue(result['success'],report)
