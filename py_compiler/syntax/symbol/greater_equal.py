from py_compiler.syntax.symbol.symbol import Symbol, TokenForm


class GreaterEqual(Symbol):
    token_forms = (TokenForm('GREATER_EQUAL', '>='),)
