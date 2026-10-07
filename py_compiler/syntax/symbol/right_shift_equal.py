from py_compiler.syntax.symbol.symbol import Symbol, TokenForm


class RightShiftEqual(Symbol):
    token_forms = (TokenForm('RIGHT_SHIFT_EQUAL', '>>='),)
