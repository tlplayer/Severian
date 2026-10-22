from py_compiler.syntax.symbol.symbol import Symbol, TokenForm


class Range(Symbol):
    token_forms = (TokenForm('RANGE', '..'),)

    def grammars(self):
        return (*super().grammars(), RangeOperation(self))


from py_compiler.syntax.generic.owned import OwnedGrammar


class RangeOperation(OwnedGrammar):
    spelling = '..'
    precedence = 2.75
    associativity = 'left'

    def __init__(self, owner):
        super().__init__(owner, 'F.operator', ('..',))

    def recognize(self, window):
        from py_compiler.syntax.generic.grammar import Match
        if len(window.tokens) == 1 and window.tokens[0].text == '..':
            return Match(self, window, window.end, {})
        return None

    def lower_expression(self, cfg, node, env):
        from py_compiler.syntax.grammar.expression import Expression
        callee = Expression('name', node.token.with_text('range', kind='IDENTIFIER'))
        return cfg.expr(Expression('call', node.token, (callee, *node.operands)), env)
