"""Expression and sentence grammar; recognition never executes source code."""
from dataclasses import dataclass


@dataclass(frozen=True)
class Expression:
    kind: str
    token: object
    operands: tuple = ()


from py_compiler.syntax.operator.ownership import OWNERSHIP


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


def type_spelling(node):
    if node.kind == 'name':
        return node.token.text
    if node.kind == 'index':
        return type_spelling(node.operands[0]) + '[' + type_spelling(node.operands[1]) + ']'
    raise ValueError('expected a concrete type argument')


def parse_expression(tokens, syntax):
    from py_compiler.syntax.function.grammar import operators
    bindings = {grammar.spelling: grammar for grammar in operators(syntax)}
    precedence = {spelling: grammar.precedence for spelling, grammar in bindings.items()}
    cursor = [0]
    def parse(minimum=0, strict=False):
        if cursor[0] == len(tokens):
            raise ValueError("expected an expression")
        token = tokens[cursor[0]]
        cursor[0] += 1
        owner = syntax.expression_providers.get(token.text)
        if owner is not None:
            left = owner.parse_prefix(token, parse, tokens, cursor)
        elif token.text in ("not", "+", "-", "~"):
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
        while cursor[0] < len(tokens) and tokens[cursor[0]].text in ("(", ".", "["):
            if tokens[cursor[0]].text == "[":
                bracket = tokens[cursor[0]]
                cursor[0] += 1
                arguments = [parse()]
                while cursor[0] < len(tokens) and tokens[cursor[0]].text == ',':
                    cursor[0] += 1
                    arguments.append(parse())
                index = arguments[0] if len(arguments) == 1 else Expression('template_arguments', bracket, tuple(arguments))
                if cursor[0] == len(tokens) or tokens[cursor[0]].text != "]":
                    raise ValueError("expected closing index bracket")
                cursor[0] += 1
                left = Expression("index", bracket, (left, index))
                continue
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
        while cursor[0] < len(tokens):
            level = precedence.get(tokens[cursor[0]].text, -1)
            if level < minimum or (strict and level == minimum):
                break
            operator = tokens[cursor[0]]
            cursor[0] += 1
            binding = bindings[operator.text]
            right = parse(binding.precedence, strict=binding.associativity != 'right')
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


import unittest


class OperatorBindingTests(unittest.TestCase):
    def parse(self, text):
        from py_compiler.frontend.lexer.lexer import lex
        from py_compiler.frontend.source.source import SourceFile
        from py_compiler.syntax.recognition import Syntax
        syntax = Syntax()
        tokens = [t for t in lex(SourceFile('operators.sev', text), syntax)
                  if t.kind not in ('NEWLINE', 'EOF', 'INDENT', 'DEDENT')]
        return parse_expression(tokens, syntax)

    def test_power_is_right_associative_and_binds_before_unary(self):
        tree = self.parse('2 ** 3 ** 2')
        self.assertEqual(tree.operands[1].token.text, '**')
        tree = self.parse('-2 ** 2')
        self.assertEqual((tree.kind, tree.operands[0].token.text), ('unary', '**'))

    def test_bitwise_precedence_and_left_associative_division(self):
        tree = self.parse('1 | 2 ^ 3 & 4 << 1 + 2 * 3')
        for operator in ('|', '^', '&', '<<', '+', '*'):
            self.assertEqual(tree.token.text, operator)
            tree = tree.operands[1]
        tree = self.parse('8 / 2 / 2')
        self.assertEqual(tree.operands[0].token.text, '/')
