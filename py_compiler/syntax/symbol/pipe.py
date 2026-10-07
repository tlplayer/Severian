from py_compiler.syntax.symbol.symbol import Symbol, TokenForm


class Pipe(Symbol):
    token_forms = (TokenForm('PIPE', '|'),)
