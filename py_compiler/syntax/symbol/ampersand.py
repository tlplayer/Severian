from py_compiler.syntax.symbol.symbol import Symbol, TokenForm


class Ampersand(Symbol):
    token_forms = (TokenForm('AMPERSAND', '&'),)
