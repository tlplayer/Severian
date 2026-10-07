from py_compiler.syntax.symbol.symbol import Symbol, TokenForm


class Plus(Symbol):
    token_forms = (TokenForm('PLUS', '+'),)
