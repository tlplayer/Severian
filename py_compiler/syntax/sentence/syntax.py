"""Expression and sentence grammar; recognition never executes source code."""
from dataclasses import dataclass


@dataclass(frozen=True)
class Expression:
    kind: str
    token: object
    operands: tuple = ()


OWNERSHIP = frozenset(("copy", "view", "borrow", "move"))


def expression(tokens):
    if not tokens:
        raise ValueError('expected an expression')
    from py_compiler.syntax.generic.grammar import SourceWindow
    from py_compiler.syntax.recognition import Syntax
    source = tokens[0].lexeme.source
    if source is None:
        raise ValueError('expression grammar requires original source ownership')
    window = SourceWindow(source, tokens[0].span.start, tokens[-1].span.end, tuple(tokens))
    syntax = tokens[0].syntax or Syntax()
    return syntax.registry.construct('F', window)


def parse_expression(tokens, syntax):
    from py_compiler.syntax.function.grammar import operators
    precedence = {grammar.spelling: grammar.precedence for grammar in operators(syntax)}
    cursor = 0
    def parse(minimum=0):
        nonlocal cursor
        if cursor == len(tokens):
            raise ValueError("expected an expression")
        token = tokens[cursor]
        cursor += 1
        if token.text in OWNERSHIP or token.text in ("not", "+", "-"):
            left = Expression("unary", token, (parse(6),))
        elif token.text == "(":
            left = parse()
            if cursor == len(tokens) or tokens[cursor].text != ")":
                raise ValueError("expected closing parenthesis")
            cursor += 1
        elif token.kind in ("NUMBER", "CHAR", "STRING") or token.text in ("true", "false", "None", "absent"):
            left = Expression("literal", token)
        elif token.text in ("module", "local", "self") and cursor + 1 < len(tokens) and tokens[cursor].text == "." and tokens[cursor + 1].kind == "IDENTIFIER":
            left = Expression("reference", token, (tokens[cursor + 1],))
            cursor += 2
        elif token.kind == "IDENTIFIER":
            left = Expression("name", token)
        else:
            raise ValueError(f"unsupported expression keyword {token.text!r}")
        while cursor < len(tokens) and tokens[cursor].text in ("(", "."):
            if tokens[cursor].text == ".":
                cursor += 1
                if cursor == len(tokens) or tokens[cursor].kind != "IDENTIFIER":
                    raise ValueError("member access requires a name")
                left = Expression("member", tokens[cursor], (left,))
                cursor += 1
                continue
            cursor += 1
            arguments = []
            while cursor < len(tokens) and tokens[cursor].text != ")":
                arguments.append(parse())
                if cursor < len(tokens) and tokens[cursor].text == ",":
                    cursor += 1
                else:
                    break
            if cursor == len(tokens) or tokens[cursor].text != ")":
                raise ValueError("expected closing call parenthesis")
            cursor += 1
            left = Expression("call", left.token, (left, *arguments))
        while cursor < len(tokens) and precedence.get(tokens[cursor].text, -1) >= minimum:
            operator = tokens[cursor]
            cursor += 1
            right = parse(precedence[operator.text] + 1)
            comparisons = ('==', '!=', '<', '<=', '>', '>=')
            if operator.text in comparisons and left.kind == 'binary' and left.token.text in comparisons:
                left = Expression('chain', operator, (left.operands[0], left.token, left.operands[1], operator, right))
            elif operator.text in comparisons and left.kind == 'chain':
                left = Expression('chain', operator, (*left.operands, operator, right))
            else:
                left = Expression("binary", operator, (left, right))
        return left
    result = parse()
    if cursor != len(tokens):
        raise ValueError(f"unsupported expression suffix {tokens[cursor].text!r}")
    return result
