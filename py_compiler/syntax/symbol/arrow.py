from py_compiler.syntax.symbol.symbol import Symbol, TokenForm


class Arrow(Symbol):
    token_forms = (TokenForm('ARROW', '->'),)
