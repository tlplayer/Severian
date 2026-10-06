from dataclasses import dataclass


@dataclass(frozen=True)
class Primitive:
    name: str
    family: str
    mlir: str
    bits: int = 0
    signed: bool = False

    @property
    def no_result(self):
        return False

    @property
    def binding_ownership(self):
        return "copy"

    def grammars(self):
        return (Constructor(self),)

    def render_constant(self, value, symbol):
        raise ValueError(f'{self.name} does not supply constant lowering')

    def module_storage(self, symbol):
        raise ValueError(f'{self.name} does not supply module storage')

    def accept_literal(self, owner, value):
        if owner != self:
            raise ValueError(f'{owner.name} literal does not satisfy {self.name}')
        return value


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
