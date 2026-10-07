from py_compiler.syntax.symbol.symbol import Symbol, TokenForm


class At(Symbol):
    token_forms = (TokenForm('AT', '@'),)
