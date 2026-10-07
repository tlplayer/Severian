from py_compiler.syntax.symbol.symbol import Symbol, TokenForm


class Ellipsis(Symbol):
    token_forms = (TokenForm('ELLIPSIS', '...'),)
