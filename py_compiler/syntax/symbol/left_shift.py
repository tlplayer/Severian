from py_compiler.syntax.symbol.symbol import Symbol, TokenForm


class LeftShift(Symbol):
    token_forms = (TokenForm('LEFT_SHIFT', '<<'),)
