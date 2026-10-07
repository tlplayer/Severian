from py_compiler.syntax.symbol.symbol import Symbol, TokenForm


class RightParen(Symbol):
    token_forms = (TokenForm('RIGHT_PAREN', ')'),)
