from py_compiler.syntax.symbol.symbol import Symbol, TokenForm


class RightBrace(Symbol):
    token_forms = (TokenForm('RIGHT_BRACE', '}'),)
