from py_compiler.syntax.symbol.symbol import Symbol, TokenForm


class Caret(Symbol):
    token_forms = (TokenForm('CARET', '^'),)
