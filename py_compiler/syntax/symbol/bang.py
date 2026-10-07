from py_compiler.syntax.symbol.symbol import Symbol, TokenForm


class Bang(Symbol):
    token_forms = (TokenForm('BANG', '!'),)
