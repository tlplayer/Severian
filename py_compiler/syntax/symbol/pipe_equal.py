from py_compiler.syntax.symbol.symbol import Symbol, TokenForm


class PipeEqual(Symbol):
    token_forms = (TokenForm('PIPE_EQUAL', '|='),)
