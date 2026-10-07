from py_compiler.syntax.symbol.symbol import Symbol, TokenForm


class Dot(Symbol):
    token_forms = (TokenForm('DOT', '.'),)
