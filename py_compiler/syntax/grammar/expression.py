"""Expression and sentence grammar; recognition never executes source code."""
from dataclasses import dataclass


@dataclass(frozen=True)
class Expression:
    kind: str
    token: object
    operands: tuple = ()
    tokens: tuple = ()
    interval: tuple | None = None


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
    if node.kind in ('name', 'literal'):
        return node.token.text
    if node.kind == 'tuple':
        return '(' + ','.join(type_spelling(item) for item in node.operands) + (',' if len(node.operands) == 1 else '') + ')'
    if node.kind == 'call' and node.operands[0].token.text == 'tuple':
        return 'tuple(' + ','.join(type_spelling(item) for item in node.operands[1:]) + ')'
    if node.kind == 'unary' and node.token.text in ('-', '+'):
        return node.token.text + type_spelling(node.operands[0])
    if node.kind == 'template_arguments':
        return ','.join(type_spelling(item) for item in node.operands)
    if node.kind == 'binary' and node.token.text == '|':
        return type_spelling(node.operands[0]) + '|' + type_spelling(node.operands[1])
    if node.kind == 'slice' and node.operands[0] is not None and node.operands[1] is not None and node.operands[2] is None:
        return type_spelling(node.operands[0]) + ':' + type_spelling(node.operands[1])
    if node.kind == 'template_binding':
        return node.token.text.join(type_spelling(item) for item in node.operands)
    if node.kind == 'member':
        return type_spelling(node.operands[0]) + '.' + node.token.text
    if node.kind == 'index':
        return type_spelling(node.operands[0]) + '[' + type_spelling(node.operands[1]) + ']'
    raise ValueError('expected a concrete type argument')


def parse_expression(tokens, syntax):
    from py_compiler.syntax.function.grammar import operators
    bindings = {grammar.spelling: grammar for grammar in operators(syntax)}
    precedence = {spelling: grammar.precedence for spelling, grammar in bindings.items()}
    cursor = [0]
    def parse(minimum=0, strict=False):
        start = cursor[0]
        if cursor[0] == len(tokens):
            raise ValueError("expected an expression")
        token = tokens[cursor[0]]
        cursor[0] += 1
        owner = syntax.expression_providers.get(token.text)
        if owner is not None:
            left = owner.parse_prefix(token, parse, tokens, cursor)
        elif token.text == '...':
            left = Expression('spread', token, (parse(6),))
        elif token.text == "&":
            left = Expression("address", token, (parse(6),))
        elif token.text in ("not", "!", "+", "-", "~"):
            # Logical negation includes comparisons, but stops before and/or.
            left = Expression("unary", token.with_text('not') if token.text == '!' else token,
                              (parse(2.5 if token.text in ('not', '!') else 6),))
        elif token.text == "(":
            elements, comma = [], False
            while cursor[0] < len(tokens) and tokens[cursor[0]].text != ')':
                elements.append(parse())
                if cursor[0] < len(tokens) and tokens[cursor[0]].text == ',':
                    cursor[0] += 1
                    comma = True
                else:
                    break
            if cursor[0] == len(tokens) or tokens[cursor[0]].text != ")":
                raise ValueError("expected closing parenthesis")
            cursor[0] += 1
            left = elements[0] if len(elements) == 1 and not comma else Expression('tuple', token, tuple(elements))
        elif token.text == '[':
            elements = []
            comprehension = None
            while cursor[0] < len(tokens) and tokens[cursor[0]].text != ']':
                elements.append(parse())
                if cursor[0] < len(tokens) and tokens[cursor[0]].text == 'for':
                    if len(elements) != 1:
                        raise ValueError('comprehension requires one result expression')
                    cursor[0] += 1
                    if cursor[0] + 1 >= len(tokens) or tokens[cursor[0]].kind != 'IDENTIFIER' or tokens[cursor[0] + 1].text != 'in':
                        raise ValueError('comprehension requires for name in iterable')
                    target = tokens[cursor[0]]
                    cursor[0] += 2
                    source = parse(1)
                    predicate = None
                    if cursor[0] < len(tokens) and tokens[cursor[0]].text == 'if':
                        cursor[0] += 1
                        predicate = parse()
                    comprehension = (elements[0], Expression('name', target), source, predicate)
                    break
                if cursor[0] < len(tokens) and tokens[cursor[0]].text == ',':
                    cursor[0] += 1
                else:
                    break
            if cursor[0] == len(tokens) or tokens[cursor[0]].text != ']':
                raise ValueError('expected closing list bracket')
            cursor[0] += 1
            left = Expression('comprehension', token, comprehension) if comprehension else Expression('list', token, tuple(elements))
        elif token.text == '{':
            entries = []
            collection_kind = None
            while cursor[0] < len(tokens) and tokens[cursor[0]].text != '}':
                key = parse()
                pair = cursor[0] < len(tokens) and tokens[cursor[0]].text == ':'
                kind = 'map' if pair else 'set'
                if collection_kind is not None and collection_kind != kind:
                    raise ValueError('cannot mix mapping entries and set elements')
                collection_kind = kind
                if pair:
                    cursor[0] += 1
                    entries.extend((key,parse()))
                else:
                    entries.append(key)
                if cursor[0] < len(tokens) and tokens[cursor[0]].text == ',':
                    cursor[0] += 1
                else:
                    break
            if cursor[0] == len(tokens) or tokens[cursor[0]].text != '}':
                raise ValueError('expected closing map brace')
            cursor[0] += 1
            left = Expression(collection_kind or 'map', token, tuple(entries))
        elif token.kind == "FSTRING":
            left = Expression("interpolation", token)
        elif token.kind in ("NUMBER", "CHAR", "STRING") or token.text in ("true", "false", "None", "absent"):
            left = Expression("literal", token)
        elif token.text in ("module", "local", "self") and cursor[0] + 1 < len(tokens) and tokens[cursor[0]].text == "." and tokens[cursor[0] + 1].kind == "IDENTIFIER":
            left = Expression("reference", token, (tokens[cursor[0] + 1],))
            cursor[0] += 2
        elif (token.kind == "IDENTIFIER" or token.text == 'self'
              or token.kind == 'KEYWORD' and cursor[0] < len(tokens) and tokens[cursor[0]].text == '('):
            left = Expression("name", token)
        else:
            raise ValueError(f"unsupported expression keyword {token.text!r}")
        while cursor[0] < len(tokens) and tokens[cursor[0]].text in ("(", ".", "["):
            interval_call = False
            if tokens[cursor[0]].text == '(':
                depth = 0
                for following in tokens[cursor[0]:]:
                    if following.text in ('(', '[', '{'):
                        depth += 1
                    elif following.text in (')', ']', '}'):
                        depth -= 1
                        if depth == 0:
                            break
                    elif depth == 1 and following.text == ':':
                        interval_call = True
                        break
            if tokens[cursor[0]].text == "[" or interval_call:
                bracket = tokens[cursor[0]]
                cursor[0] += 1
                def argument():
                    value = None if cursor[0] < len(tokens) and tokens[cursor[0]].text == ':' else parse()
                    if cursor[0] < len(tokens) and tokens[cursor[0]].text == ':':
                        separator = tokens[cursor[0]]
                        cursor[0] += 1
                        stop = None if cursor[0] < len(tokens) and tokens[cursor[0]].text in (']', ')', ':', ',') else parse()
                        step = None
                        if cursor[0] < len(tokens) and tokens[cursor[0]].text == ':':
                            cursor[0] += 1
                            step = None if cursor[0] < len(tokens) and tokens[cursor[0]].text in (']', ')', ',') else parse()
                        return Expression('slice', separator, (value, stop, step))
                    if cursor[0] < len(tokens) and tokens[cursor[0]].text == '=':
                        separator = tokens[cursor[0]]
                        cursor[0] += 1
                        value = Expression('template_binding', separator, (value, parse()))
                    return value
                arguments = [argument()]
                while cursor[0] < len(tokens) and tokens[cursor[0]].text == ',':
                    cursor[0] += 1
                    arguments.append(argument())
                index = arguments[0] if len(arguments) == 1 else Expression('template_arguments', bracket, tuple(arguments))
                if cursor[0] == len(tokens) or tokens[cursor[0]].text not in ("]", ")"):
                    raise ValueError("expected closing index bracket")
                if index.kind == 'slice':
                    from dataclasses import replace
                    index = replace(index, interval=(bracket.text == '[', tokens[cursor[0]].text == ']'))
                elif tokens[cursor[0]].text != ']':
                    raise ValueError('mixed range boundaries require a colon')
                cursor[0] += 1
                left = Expression("index", bracket, (left, index))
                continue
            if tokens[cursor[0]].text == ".":
                cursor[0] += 1
                if cursor[0] == len(tokens) or not tokens[cursor[0]].text.isidentifier():
                    raise ValueError("member access requires a name")
                left = Expression("member", tokens[cursor[0]], (left,))
                cursor[0] += 1
                continue
            cursor[0] += 1
            arguments = []
            while cursor[0] < len(tokens) and tokens[cursor[0]].text != ")":
                item = parse()
                if cursor[0] < len(tokens) and tokens[cursor[0]].text == '=':
                    if item.kind != 'name':
                        raise ValueError('named call argument requires a parameter name')
                    cursor[0] += 1
                    item = Expression('named_argument', item.token, (parse(),))
                elif any(argument.kind == 'named_argument' for argument in arguments):
                    raise ValueError('positional arguments must precede named arguments')
                arguments.append(item)
                if cursor[0] < len(tokens) and tokens[cursor[0]].text == ",":
                    cursor[0] += 1
                else:
                    break
            if cursor[0] == len(tokens) or tokens[cursor[0]].text != ")":
                raise ValueError("expected closing call parenthesis")
            cursor[0] += 1
            left = Expression("call", left.token, (left, *arguments))
        from dataclasses import replace
        left = replace(left, tokens=tuple(tokens[start:cursor[0]]))
        while cursor[0] < len(tokens):
            if tokens[cursor[0]].text == 'is' and minimum <= 3:
                operator = tokens[cursor[0]]
                cursor[0] += 1
                negate = cursor[0] < len(tokens) and tokens[cursor[0]].text == 'not'
                cursor[0] += int(negate)
                left = Expression('identity', operator, (left, parse(3, strict=True)))
                if negate:
                    left = Expression('unary', operator.with_text('not'), (left,))
                continue
            if (tokens[cursor[0]].text == 'not' and cursor[0] + 1 < len(tokens)
                    and tokens[cursor[0] + 1].text == 'in'):
                level = precedence.get('in', -1)
                if level < minimum or (strict and level == minimum):
                    break
                operator = tokens[cursor[0]]
                cursor[0] += 2
                membership = Expression('binary', operator.with_text('in'), (left, parse(level, strict=True)))
                left = Expression('unary', operator, (membership,))
                continue
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
            left = replace(left, tokens=tuple(tokens[start:cursor[0]]))
        if minimum == 0 and cursor[0] < len(tokens) and tokens[cursor[0]].text == 'if':
            operator = tokens[cursor[0]]
            cursor[0] += 1
            condition = parse(1)
            if cursor[0] == len(tokens) or tokens[cursor[0]].text != 'else':
                raise ValueError('conditional expression requires else')
            cursor[0] += 1
            left = Expression('conditional', operator, (condition, left, parse()))
        if minimum == 0 and cursor[0] < len(tokens) and tokens[cursor[0]].text == 'else':
            operator = tokens[cursor[0]]
            cursor[0] += 1
            if cursor[0] < len(tokens) and tokens[cursor[0]].text == 'throw':
                throwing = tokens[cursor[0]]
                cursor[0] += 1
                right = Expression('fallback_throw', throwing, (parse(),))
            else:
                right = parse()
            left = Expression('fallback', operator, (left, right))
        from dataclasses import replace
        return replace(left, tokens=tuple(tokens[start:cursor[0]]))
    result = parse()
    if cursor[0] != len(tokens):
        raise ValueError(f"unsupported expression suffix {tokens[cursor[0]].text!r}")
    return result


import unittest


class IntervalTests(unittest.TestCase):
    def test_boundary_kinds_and_nested_calls(self):
        from py_compiler.frontend.lexer.lexer import lex
        from py_compiler.frontend.source.source import SourceFile
        from py_compiler.frontend.parser.blocks import parse_blocks
        from py_compiler.syntax.recognition import Syntax
        syntax = Syntax()
        for spelling, closed in (('a[0:1]', (True, True)), ('a(0:1]', (False, True)),
                                 ('a[0:1)', (True, False)), ('a(0:1)', (False, False))):
            source = SourceFile('interval.sev', spelling)
            tokens = [t for t in lex(source, syntax) if t.kind not in ('NEWLINE', 'EOF')]
            term = expression(tokens)
            self.assertEqual(term.operands[1].interval, closed)
            wrapped = SourceFile('nested.sev', 'for i, value in enumerate(' + spelling + '):\n    break\n')
            _, errors = parse_blocks(wrapped, lex(wrapped, syntax), syntax)
            self.assertFalse(errors, errors)
        bad = SourceFile('mismatch.sev', 'call(1]\n')
        self.assertTrue(parse_blocks(bad, lex(bad, syntax), syntax)[1])

    def test_c_list_tuple_and_string_boundaries(self):
        import tempfile
        import json
        from pathlib import Path
        from py_compiler.packaging.testing import test_package
        library = Path(__file__).resolve().parents[3] / 'library/core/collections/collection'
        source = '''from collections import list
test "mathematical endpoints on lists and tuples":
    values = [1, 2, 3]
    assert(values(0:] == (2, 3))
    assert(values(0:][:1] == (2, 3))
    assert(len(values(0:][:1](0:1)) == 0)
    assert(values[0:1] == (1, 2))
    assert(values[0:1) == (1,))
    assert(values[-2:-1] == (2, 3))
    assert(values[2:0:-1] == (3, 2, 1))
    assert(values(2:0:-1) == (2,))
    assert(values[-100:100] == values)
    assert(len(values(100:]) == 0)
    assert(len(values[:-100]) == 0)
    pair = (1, 2, 3)
    assert(pair(0:] == (2, 3))
    assert(pair[:1] == (1, 2))
test "mathematical endpoints on text":
    assert("abc"(0:] == "bc")
    assert("abc"[:1] == "ab")
    assert("abc"(0:1) == "")
'''
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'src').mkdir()
            (root / 'src/lib.sev').write_text(source)
            (root / 'package.json').write_text(json.dumps({'package': {'name': 'intervals', 'version': '1'},
                'dependencies': {'collections': {'path': str(library)}}}))
            result = test_package(root)
            detail = Path(result['report']).read_text() if 'report' in result else result
            self.assertTrue(result['success'], detail)


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

    def test_negation_contains_comparison_but_stops_before_and_or(self):
        tree = self.parse('not left() == right() and enabled or fallback()')
        self.assertEqual(tree.token.text, 'or')
        conjunction = tree.operands[0]
        self.assertEqual(conjunction.token.text, 'and')
        negation = conjunction.operands[0]
        self.assertEqual((negation.kind, negation.token.text), ('unary', 'not'))
        comparison = negation.operands[0]
        self.assertEqual(comparison.token.text, '==')
        self.assertEqual([node.operands[0].token.text for node in comparison.operands], ['left', 'right'])

    def test_comparison_chain_keeps_one_middle_call(self):
        tree = self.parse('first() < middle() <= last()')
        self.assertEqual(tree.kind, 'chain')
        self.assertEqual([node.operands[0].token.text for node in tree.operands[::2]],
                         ['first', 'middle', 'last'])
        self.assertEqual([token.text for token in tree.operands[1::2]], ['<', '<='])
