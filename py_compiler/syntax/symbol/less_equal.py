from py_compiler.syntax.symbol.symbol import Symbol, TokenForm


class LessEqual(Symbol):
    token_forms = (TokenForm('LESS_EQUAL', '<='),)
