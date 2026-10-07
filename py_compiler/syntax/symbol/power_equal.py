from py_compiler.syntax.symbol.symbol import Symbol, TokenForm


class PowerEqual(Symbol):
    token_forms = (TokenForm('POWER_EQUAL', '**='),)
