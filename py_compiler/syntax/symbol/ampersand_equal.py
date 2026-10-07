from py_compiler.syntax.symbol.symbol import Symbol, TokenForm


class AmpersandEqual(Symbol):
    token_forms = (TokenForm('AMPERSAND_EQUAL', '&='),)
