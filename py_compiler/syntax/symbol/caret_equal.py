from py_compiler.syntax.symbol.symbol import Symbol, TokenForm


class CaretEqual(Symbol):
    token_forms = (TokenForm('CARET_EQUAL', '^='),)
