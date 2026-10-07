from py_compiler.syntax.symbol.symbol import Symbol, TokenForm


class Percent(Symbol):
    token_forms = (TokenForm('PERCENT', '%'),)
