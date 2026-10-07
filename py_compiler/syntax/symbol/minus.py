from py_compiler.syntax.symbol.symbol import Symbol, TokenForm


class Minus(Symbol):
    token_forms = (TokenForm('MINUS', '-'),)
