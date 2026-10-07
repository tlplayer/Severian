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
    cursor = [0]
    def parse(minimum=0):
        if cursor[0] == len(tokens):
            raise ValueError("expected an expression")
        token = tokens[cursor[0]]
        cursor[0] += 1
        owner = syntax.expression_providers.get(token.text)
        if owner is not None:
            left = owner.parse_prefix(token, parse, tokens, cursor)
        elif token.text in OWNERSHIP or token.text in ("not", "+", "-"):
            left = Expression("unary", token, (parse(6),))
        elif token.text == "(":
            left = parse()
            if cursor[0] == len(tokens) or tokens[cursor[0]].text != ")":
                raise ValueError("expected closing parenthesis")
            cursor[0] += 1
        elif token.kind in ("NUMBER", "CHAR", "STRING") or token.text in ("true", "false", "None", "absent"):
            left = Expression("literal", token)
        elif token.text in ("module", "local", "self") and cursor[0] + 1 < len(tokens) and tokens[cursor[0]].text == "." and tokens[cursor[0] + 1].kind == "IDENTIFIER":
            left = Expression("reference", token, (tokens[cursor[0] + 1],))
            cursor[0] += 2
        elif token.kind == "IDENTIFIER":
            left = Expression("name", token)
        else:
            raise ValueError(f"unsupported expression keyword {token.text!r}")
        while cursor[0] < len(tokens) and tokens[cursor[0]].text in ("(", "."):
            if tokens[cursor[0]].text == ".":
                cursor[0] += 1
                if cursor[0] == len(tokens) or tokens[cursor[0]].kind != "IDENTIFIER":
                    raise ValueError("member access requires a name")
                left = Expression("member", tokens[cursor[0]], (left,))
                cursor[0] += 1
                continue
            cursor[0] += 1
            arguments = []
            while cursor[0] < len(tokens) and tokens[cursor[0]].text != ")":
                arguments.append(parse())
                if cursor[0] < len(tokens) and tokens[cursor[0]].text == ",":
                    cursor[0] += 1
                else:
                    break
            if cursor[0] == len(tokens) or tokens[cursor[0]].text != ")":
                raise ValueError("expected closing call parenthesis")
            cursor[0] += 1
            left = Expression("call", left.token, (left, *arguments))
        while cursor[0] < len(tokens) and precedence.get(tokens[cursor[0]].text, -1) >= minimum:
            operator = tokens[cursor[0]]
            cursor[0] += 1
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
    if cursor[0] != len(tokens):
        raise ValueError(f"unsupported expression suffix {tokens[cursor[0]].text!r}")
    return result
