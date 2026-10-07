from py_compiler.syntax.symbol.symbol import Symbol, TokenForm


class RightShift(Symbol):
    token_forms = (TokenForm('RIGHT_SHIFT', '>>'),)
