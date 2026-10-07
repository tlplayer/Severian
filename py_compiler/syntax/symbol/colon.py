from py_compiler.syntax.symbol.symbol import Symbol, TokenForm


class Colon(Symbol):
    token_forms = (TokenForm('COLON', ':'),)
