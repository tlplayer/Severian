from py_compiler.syntax.symbol.symbol import Symbol, TokenForm


class Comma(Symbol):
    token_forms = (TokenForm('COMMA', ','),)
