from py_compiler.syntax.symbol.symbol import Symbol, TokenForm


class LeftParen(Symbol):
    token_forms = (TokenForm('LEFT_PAREN', '('),)
