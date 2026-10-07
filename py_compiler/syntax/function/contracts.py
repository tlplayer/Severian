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
    imports: tuple = ()
    parameter_ownership: dict = field(default_factory=dict)
    templates: tuple = ()
    type_arguments: dict = field(default_factory=dict)
    callable_arguments: dict = field(default_factory=dict)
    realization_key: tuple = ()

    declaration_signature: tuple | None = None

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
        return f"__sev_fn_{readable(self.name)}[{','.join(arguments)}]::{source.path}:{line}:{column}"


def signature(node, owner=""):
    from py_compiler.syntax.complex.generic import template_header
    templates, items = template_header(list(node.header))
    if len(items) < 3 or items[0].kind != "IDENTIFIER" or items[1].text != "(":
        raise ValueError("expected def name(parameters) -> Type")
    end = next((i for i, item in enumerate(items) if item.text == ")"), -1)
    if end < 2:
        raise ValueError("missing parameter-list delimiter")
    parameters, cursor = [], 2
    parameter_ownership = {}
    while cursor < end:
        name = items[cursor].text
        if items[cursor].kind != "IDENTIFIER" or any(p[0] == name for p in parameters):
            raise ValueError("parameter name is invalid or duplicated")
        cursor += 1
        type_name = None
        if cursor < end and items[cursor].text == ":":
            cursor += 1
            if cursor == end:
                raise ValueError("parameter annotation is missing")
            if items[cursor].text in ('view', 'move', 'borrow', 'copy', 'mirror'):
                parameter_ownership[name] = items[cursor].text
                cursor += 1
            start, depth = cursor, 0
            while cursor < end:
                word = items[cursor].text
                if word == ',' and depth == 0:
                    break
                depth += (word == '[') - (word == ']')
                cursor += 1
            type_name = ''.join(t.text for t in items[start:cursor])
            if not type_name or depth:
                raise ValueError('invalid parameter type')
        parameters.append((name, type_name))
        if cursor < end:
            if items[cursor].text != ",":
                raise ValueError("expected parameter separator")
            cursor += 1
    cursor, result, guards = end + 1, None, []
    complexity, obligations, suffix = {}, [], []
    if cursor < len(items) and items[cursor].text == "->":
        if cursor + 1 == len(items):
            raise ValueError("missing result annotation")
        cursor += 1
        start = cursor
        while cursor < len(items) and items[cursor].text != 'with':
            cursor += 1
        result = ''.join(t.text for t in items[start:cursor])
    if cursor < len(items):
        if items[cursor].text != "with" or cursor + 2 >= len(items) or items[cursor + 1].text != "{" or items[-1].text != "}":
            raise ValueError("with requires a braced acceptance contract")
        current, depth = [], 0
        for token in items[cursor + 2:-1]:
            if token.text == "," and depth == 0:
                if current:
                    add_clause(current, guards, obligations, complexity, suffix)
                    current = []
            else:
                current.append(token)
                depth += int(token.text == "(") - int(token.text == ")")
        if current:
            add_clause(current, guards, obligations, complexity, suffix)
    for index, token in enumerate(items):
        if token.text == "complexity" and index >= 2 and items[index - 1].text == "." and items[index - 2].text not in (items[0].text, "F"):
            raise ValueError("complexity contract must name its declaring function")
    return Callable(node, (owner + "." if owner else "") + items[0].text,
                    tuple(parameters), result, tuple(guards), complexity=complexity, obligations=tuple(obligations), suffix=tuple(suffix), parameter_ownership=parameter_ownership, templates=templates)


def pure_predicate(node, parameters):
    if node.kind == "chain":
        for operand in node.operands[::2]:
            pure_predicate(operand, parameters)
        return
    if node.kind == "literal":
        return
    if node.kind == "name" and node.token.text in parameters:
        return
    if node.kind == "unary" and node.token.text in ("not", "+", "-"):
        pure_predicate(node.operands[0], parameters)
        return
    if node.kind == "binary" and node.token.text in ("and", "or", "==", "!=", "<", "<=", ">", ">="):
        for operand in node.operands:
            pure_predicate(operand, parameters)
        return
    raise ValueError("dispatch predicate lacks a pure, terminating, stable-read provider")


from py_compiler.syntax.complex.big_o import BigO


def add_clause(tokens, guards, obligations, complexity, suffix):
    words = [t.text for t in tokens]
    from py_compiler.syntax.prelude import clause_providers
    provider = clause_providers().get(words[0])
    if provider is not None:
        provider.attach(tokens[1:], guards, obligations, suffix)
        return
    if ':=' in words:
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
    guards.append(expression(tokens))


import unittest


class CallableSymbolTests(unittest.TestCase):
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
        self.assertEqual(original, '__sev_fn_identity[?]::src/example.sev:1:1')
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
