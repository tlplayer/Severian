"""With contract phases, applied by the owning block's semantic provider."""
from py_compiler.syntax.complex.object import Assert


class ContractPhases:
    def __init__(self, entry, cfg, env):
        self.entry, self.cfg, self.env = entry, cfg, env
        self.checking = False

    def check(self, expressions, phase):
        if self.checking:
            return
        self.checking = True
        try:
            for term in expressions:
                value = self.cfg.expr(term, self.env, self.cfg.syntax.types['bool'])
                line, column = self.cfg.source.position(term.token.span.start)
                message = f'{phase} contract failed in {self.entry.name} at {self.cfg.source.path}:{line}:{column}'
                self.cfg.effect('assert', (value,), Assert(message), term.token.span)
        finally:
            self.checking = False

    def enter(self):
        self.check(self.entry.guards, 'prefix')
        self.check(self.entry.obligations, 'fix')

    def during(self):
        self.check(self.entry.obligations, 'fix')

    def leave(self):
        self.during()
        self.check(self.entry.suffix, 'suffix')
