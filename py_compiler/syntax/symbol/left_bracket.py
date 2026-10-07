from py_compiler.syntax.symbol.symbol import Symbol, TokenForm


class LeftBracket(Symbol):
    token_forms = (TokenForm('LEFT_BRACKET', '['),)
