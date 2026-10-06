"""Expression and sentence grammar; recognition never executes source code."""
from dataclasses import dataclass


@dataclass(frozen=True)
class Expression:
    kind: str
    token: object
    operands: tuple = ()


PRECEDENCE = {"or": 1, "and": 2, "==": 3, "!=": 3, "<": 3, ">": 3, "<=": 3, ">=": 3,
              "+": 4, "-": 4, "*": 5}
OWNERSHIP = frozenset(("copy", "view", "borrow", "move"))


def expression(tokens):
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
        elif token.kind in ("NUMBER", "CHAR", "STRING") or token.text in ("true", "false", "None", "absent", "unit"):
            left = Expression("literal", token)
        elif token.text in ("global", "local", "self") and cursor + 1 < len(tokens) and tokens[cursor].text == "." and tokens[cursor + 1].kind == "IDENTIFIER":
            left = Expression("reference", token, (tokens[cursor + 1],))
            cursor += 2
        elif token.kind == "IDENTIFIER":
            left = Expression("name", token)
        else:
            raise ValueError(f"unsupported expression keyword {token.text!r}")
        if cursor < len(tokens) and tokens[cursor].text == "(" and left.kind == "name":
            cursor += 1
            argument = parse()
            if cursor == len(tokens) or tokens[cursor].text != ")":
                raise ValueError("expected closing primitive constructor parenthesis")
            cursor += 1
            left = Expression("construct", token, (argument,))
        while cursor < len(tokens) and PRECEDENCE.get(tokens[cursor].text, -1) >= minimum:
            operator = tokens[cursor]
            cursor += 1
            left = Expression("binary", operator, (left, parse(PRECEDENCE[operator.text] + 1)))
        return left
    result = parse()
    if cursor != len(tokens):
        raise ValueError(f"unsupported expression suffix {tokens[cursor].text!r}")
    return result
