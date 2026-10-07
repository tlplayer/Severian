"""Symbol declarations own lexer token forms, independently of operations."""
from dataclasses import dataclass
from py_compiler.syntax.generic.owned import OwnedGrammar
from py_compiler.syntax.generic.grammar import Match


@dataclass(frozen=True)
class TokenForm:
    kind: str
    spelling: str

    def __post_init__(self):
        if not self.kind or not self.spelling:
            raise ValueError('token forms require a kind and a nonempty spelling')


class Symbol:
    token_forms = ()

    @property
    def name(self):
        return type(self).__name__

    def grammars(self):
        return tuple(SymbolGrammar(self, form) for form in self.token_forms)


class SymbolGrammar(OwnedGrammar):
    def __init__(self, owner, form):
        super().__init__(owner, 'Y', (form.spelling,))
        self.form = form

    def recognize(self, window):
        end = window.start + len(self.form.spelling)
        if end > window.end or window.source.text[window.start:end] != self.form.spelling:
            return None
        return Match(self, window, end, {'symbol': self.owner, 'form': self.form})

    def construct(self, match):
        return self.form.kind, match.end


import unittest


class SymbolTests(unittest.TestCase):
    def test_every_symbol_reaches_a_typed_token_and_original_lexeme(self):
        from py_compiler.frontend.source.source import SourceFile
        from py_compiler.frontend.lexer.lexer import lex
        from py_compiler.syntax.recognition import Syntax
        syntax = Syntax()
        for owner in syntax.symbols:
            for form in owner.token_forms:
                with self.subTest(symbol=owner.name):
                    source = SourceFile('symbols.sev', form.spelling)
                    token = lex(source, syntax)[0]
                    self.assertEqual(token.kind, form.kind)
                    self.assertEqual(token.form, form)
                    self.assertIs(token.symbol, owner)
                    self.assertEqual(token.text, form.spelling)
                    self.assertEqual((token.span.start, token.span.end), (0, len(form.spelling)))
                    self.assertIs(token.lexeme.source, source)

    def test_longest_form_is_independent_of_declaration_order(self):
        from py_compiler.frontend.source.source import SourceFile
        from py_compiler.frontend.lexer.lexer import lex
        from py_compiler.syntax.recognition import Syntax
        syntax = Syntax()
        source = SourceFile('symbols.sev', '**= ** * //= // / <<= << <= <=> < ... .. .')
        expected = ['POWER_EQUAL', 'POWER', 'STAR', 'FLOOR_DIVIDE_EQUAL', 'FLOOR_DIVIDE',
                    'SLASH', 'LEFT_SHIFT_EQUAL', 'LEFT_SHIFT', 'LESS_EQUAL', 'CONVERSION',
                    'LESS', 'ELLIPSIS', 'RANGE', 'DOT']
        for owners in (syntax.symbols, tuple(reversed(syntax.symbols))):
            self.assertEqual([t.kind for t in lex(source, Syntax(symbols=owners))[:-1]], expected)

    def test_new_symbol_owner_needs_no_lexer_changes(self):
        from py_compiler.frontend.source.source import SourceFile
        from py_compiler.frontend.lexer.lexer import lex
        from py_compiler.syntax.recognition import Syntax
        class Pipeline(Symbol):
            token_forms = (TokenForm('PIPELINE', '|>'),)
        owner = Pipeline()
        syntax = Syntax(symbols=(*Syntax().symbols, owner))
        token = lex(SourceFile('extension.sev', '|>'), syntax)[0]
        self.assertEqual(token.kind, 'PIPELINE')
        self.assertIs(token.symbol, owner)

    def test_equal_length_conflicts_remain_diagnostics(self):
        from py_compiler.frontend.source.source import SourceFile, Diagnostic
        from py_compiler.frontend.lexer.lexer import lex
        from py_compiler.syntax.recognition import Syntax
        class Duplicate(Symbol):
            token_forms = (TokenForm('OTHER_PLUS', '+'),)
        with self.assertRaisesRegex(Diagnostic, 'ambiguous'):
            lex(SourceFile('ambiguous.sev', '+'), Syntax(symbols=(*Syntax().symbols, Duplicate())))

    def test_longest_match_respects_window_end(self):
        from py_compiler.frontend.source.source import SourceFile
        from py_compiler.syntax.generic.grammar import SourceWindow
        from py_compiler.syntax.recognition import Syntax
        syntax = Syntax()
        source = SourceFile('bounded.sev', '**=')
        for end, kind in ((1, 'STAR'), (2, 'POWER'), (3, 'POWER_EQUAL')):
            self.assertEqual(syntax.registry.construct('Y', SourceWindow(source, 0, end)), (kind, end))
