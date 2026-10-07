from py_compiler.syntax.symbol.symbol import Symbol, TokenForm


class Greater(Symbol):
    token_forms = (TokenForm('GREATER', '>'),)
