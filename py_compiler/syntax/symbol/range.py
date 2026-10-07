from py_compiler.syntax.symbol.symbol import Symbol, TokenForm


class Range(Symbol):
    token_forms = (TokenForm('RANGE', '..'),)
