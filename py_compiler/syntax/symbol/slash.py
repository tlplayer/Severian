from py_compiler.syntax.symbol.symbol import Symbol, TokenForm


class Slash(Symbol):
    token_forms = (TokenForm('SLASH', '/'),)
