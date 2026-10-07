"""Lambda owns parameter capture; closure representation remains an explicit requirement."""
from py_compiler.syntax.keywords.keyword import Keyword


class Lambda(Keyword):
    def __init__(self):
        super().__init__('lambda')

    def parse_prefix(self, token, parse, tokens, cursor):
        from py_compiler.syntax.grammar.expression import Expression
        names = []
        while cursor[0] < len(tokens) and tokens[cursor[0]].text != ':':
            name = tokens[cursor[0]]
            if name.kind != 'IDENTIFIER' or name.text in names:
                raise ValueError('lambda requires distinct parameter names')
            names.append(name.text)
            cursor[0] += 1
            if cursor[0] < len(tokens) and tokens[cursor[0]].text == ',':
                cursor[0] += 1
            elif cursor[0] < len(tokens) and tokens[cursor[0]].text != ':':
                raise ValueError('lambda parameters require commas')
        if cursor[0] == len(tokens):
            raise ValueError("lambda requires ':' before its expression")
        cursor[0] += 1
        return Expression('owned', token, (self, tuple(names), parse()))

    def lower_expression(self, node, cfg, env, expected):
        raise ValueError('lambda has no declared closure/ownership/lowering implementation')
