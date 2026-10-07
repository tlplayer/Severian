from py_compiler.syntax.symbol.symbol import Symbol, TokenForm


class NotEqual(Symbol):
    token_forms = (TokenForm('NOT_EQUAL', '!='),)
