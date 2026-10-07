from py_compiler.syntax.symbol.symbol import Symbol, TokenForm


class RightBracket(Symbol):
    token_forms = (TokenForm('RIGHT_BRACKET', ']'),)
