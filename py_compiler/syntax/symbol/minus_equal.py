from py_compiler.syntax.symbol.symbol import Symbol, TokenForm


class MinusEqual(Symbol):
    token_forms = (TokenForm('MINUS_EQUAL', '-='),)
