from py_compiler.syntax.symbol.symbol import Symbol, TokenForm


class Star(Symbol):
    token_forms = (TokenForm('STAR', '*'),)
