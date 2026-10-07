from py_compiler.syntax.symbol.symbol import Symbol, TokenForm


class LeftShiftEqual(Symbol):
    token_forms = (TokenForm('LEFT_SHIFT_EQUAL', '<<='),)
