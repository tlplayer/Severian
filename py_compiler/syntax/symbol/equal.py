from py_compiler.syntax.symbol.symbol import Symbol, TokenForm


class Equal(Symbol):
    token_forms = (TokenForm('EQUAL', '='),)
