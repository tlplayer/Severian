from py_compiler.syntax.symbol.symbol import Symbol, TokenForm


class QuestionEqual(Symbol):
    token_forms = (TokenForm('QUESTION_EQUAL', '?='),)
