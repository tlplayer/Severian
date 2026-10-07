from py_compiler.syntax.symbol.symbol import Symbol, TokenForm


class Semicolon(Symbol):
    token_forms = (TokenForm('SEMICOLON', ';'),)
