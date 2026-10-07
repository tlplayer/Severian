from py_compiler.syntax.symbol.symbol import Symbol, TokenForm


class FloorDivideEqual(Symbol):
    token_forms = (TokenForm('FLOOR_DIVIDE_EQUAL', '//='),)
