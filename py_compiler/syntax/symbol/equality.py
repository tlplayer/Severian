from py_compiler.syntax.symbol.symbol import Symbol, TokenForm


class Equality(Symbol):
    token_forms = (TokenForm('EQUALITY', '=='),)
