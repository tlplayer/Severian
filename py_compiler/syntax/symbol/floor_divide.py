from py_compiler.syntax.symbol.symbol import Symbol, TokenForm


class FloorDivide(Symbol):
    token_forms = (TokenForm('FLOOR_DIVIDE', '//'),)
