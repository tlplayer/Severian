from py_compiler.syntax.symbol.symbol import Symbol, TokenForm


class Tilde(Symbol):
    token_forms = (TokenForm('TILDE', '~'),)
