from py_compiler.syntax.symbol.symbol import Symbol, TokenForm


class LeftBrace(Symbol):
    token_forms = (TokenForm('LEFT_BRACE', '{'),)
