from py_compiler.syntax.symbol.symbol import Symbol, TokenForm


class StarEqual(Symbol):
    token_forms = (TokenForm('STAR_EQUAL', '*='),)
