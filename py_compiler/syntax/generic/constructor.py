from py_compiler.syntax.generic.owned import OwnedGrammar, Capture
from py_compiler.syntax.generic.grammar import Match


class Constructor(OwnedGrammar):
    spelling = 'construct'

    def __init__(self, owner):
        super().__init__(owner, 'F.constructor', (owner.name, '(', Capture('value', owner), ')'))

    def recognize(self, window):
        tokens = window.tokens
        if len(tokens) >= 3 and tokens[0].text == self.owner.name and tokens[1].text == '(' and tokens[-1].text == ')':
            return Match(self, window, window.end, {'value': tokens[2:-1]})
        return None

    def expand(self, cfg, arguments, env):
        if len(arguments) != 1:
            raise ValueError(f'{self.owner.name} constructor requires one value')
        return cfg.expr(arguments[0], env, self.owner)
