from py_compiler.syntax.symbol.symbol import Symbol, TokenForm


class Conversion(Symbol):
    token_forms = (TokenForm('CONVERSION', '<=>'),)
