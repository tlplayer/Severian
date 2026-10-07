from py_compiler.syntax.symbol.symbol import Symbol, TokenForm


class PercentEqual(Symbol):
    token_forms = (TokenForm('PERCENT_EQUAL', '%='),)
