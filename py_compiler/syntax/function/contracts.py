"""Callable declarations retain parameter, result and acceptance contracts."""
from dataclasses import dataclass, field
from hashlib import sha256
from py_compiler.syntax.sentence.syntax import expression


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

    @property
    def symbol(self):
        return "__sev_fn_" + sha256((self.node.identity + self.name).encode()).hexdigest()


def signature(node, owner=""):
    items = list(node.header)
    if len(items) < 3 or items[0].kind != "IDENTIFIER" or items[1].text != "(":
        raise ValueError("expected def name(parameters) -> Type")
    end = next((i for i, item in enumerate(items) if item.text == ")"), -1)
    if end < 2:
        raise ValueError("missing parameter-list delimiter")
    parameters, cursor = [], 2
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
            type_name = items[cursor].text
            cursor += 1
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
        result, cursor = items[cursor + 1].text, cursor + 2
        while cursor < len(items) and items[cursor].text == "|":
            if cursor + 1 >= len(items):
                raise ValueError("missing union variant")
            result += " | " + items[cursor + 1].text
            cursor += 2
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
                    tuple(parameters), result, tuple(guards), complexity=complexity, obligations=tuple(obligations), suffix=tuple(suffix))


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


from enum import IntEnum


class BigO(IntEnum):
    constant = 0
    linear = 1
    quadratic = 2
    undefined = 3
    exponential = 4
    combinatorial = 5


def add_clause(tokens, guards, obligations, complexity, suffix):
    words = [t.text for t in tokens]
    if words[0] in ('suffix', 'defer'):
        suffix.append(expression(tokens[1:]))
        return
    if words[0] == 'prefix':
        guards.append(expression(tokens[1:]))
        return
    if words[0] == 'fix':
        obligations.append(expression(tokens[1:]))
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
