from py_compiler.syntax.symbol.symbol import Symbol, TokenForm


class ColonEqual(Symbol):
    token_forms = (TokenForm('COLON_EQUAL', ':='),)
