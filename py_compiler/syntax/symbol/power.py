from py_compiler.syntax.symbol.symbol import Symbol, TokenForm


class Power(Symbol):
    token_forms = (TokenForm('POWER', '**'),)
