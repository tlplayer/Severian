from py_compiler.syntax.symbol.symbol import Symbol, TokenForm


class PlusEqual(Symbol):
    token_forms = (TokenForm('PLUS_EQUAL', '+='),)
