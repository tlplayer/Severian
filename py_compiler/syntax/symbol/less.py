from py_compiler.syntax.symbol.symbol import Symbol, TokenForm


class Less(Symbol):
    token_forms = (TokenForm('LESS', '<'),)
