from py_compiler.syntax.symbol.symbol import Symbol, TokenForm


class SlashEqual(Symbol):
    token_forms = (TokenForm('SLASH_EQUAL', '/='),)
