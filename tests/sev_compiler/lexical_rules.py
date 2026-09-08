#!/usr/bin/env python3
"""Imported lexical bodies execute without rebuilding either compiler."""
import hashlib
import unittest

from migration import MigrationCase, ROOT


NUMBER_RULE = '''
class SigilInteger: LexicalRule:
    def digit(character: string) -> bool:
        return character >= "0" and character <= "9"

    def scan(input: LexemeInput, start: int) -> LexemeMatch:
        characters = input.characters
        if characters[start] != "~":
            return LexemeMatch(start, None, false)
        cursor := start + 1
        while cursor < len(characters) and digit(characters[cursor]):
            cursor += 1
        if cursor == start + 1:
            return LexemeMatch(start, None, false)
        return LexemeMatch(cursor, TokenKind.Integer, value=lexeme_text(input, start + 1, cursor))
'''


class LexicalRules(MigrationCase):
    def test_rule_body_changes_without_frontend_or_binary_changes(self):
        paths = [p for part in ('lexer', 'parser')
                 for p in (ROOT / 'sev_compiler/frontend' / part).rglob('*.sev')]
        before = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
        self.write(NUMBER_RULE, 'lexical.sev')
        self.native('''
            import "lexical.sev"
            test:
                assert(~42 == 42)
                assert(f"{~42}" == "42")
        ''')
        self.write(NUMBER_RULE.replace(
            'value=lexeme_text(input, start + 1, cursor)', 'value="73"'), 'lexical.sev')
        self.native('''
            import "lexical.sev"
            test:
                assert(~42 == 73)
        ''')
        self.assertEqual(before, {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in paths})
        # MigrationCase checks the source compiler binary digest in tearDown.

    def test_rule_is_inactive_without_import(self):
        self.write(NUMBER_RULE, 'lexical.sev')
        self.rejects('test:\n    assert(~42 == 42)\n', r'(unexpected character|expected an expression)')

    def test_new_string_delimiter_and_unicode_source(self):
        self.write('''
            class BacktickString: LexicalRule:
                def scan(input: LexemeInput, start: int) -> LexemeMatch:
                    characters = input.characters
                    if characters[start] != "`":
                        return LexemeMatch(start, None, false)
                    cursor := start + 1
                    while cursor < len(characters) and characters[cursor] != "`":
                        cursor += 1
                    if cursor == len(characters):
                        throw "unterminated backtick string"
                    return LexemeMatch(cursor + 1, TokenKind.String, value=lexeme_text(input, start + 1, cursor))
        ''', 'lexical.sev')
        self.native(r'''
            import "lexical.sev"
            test:
                assert("λ" == `λ`)
                assert(`` == "")
                assert(`hello` == "hello")
                assert(`"\q"` == "\"\\q\"")
        ''')
        self.rejects('import "lexical.sev"\ntest:\n    value = `unterminated\n',
                     r'unterminated backtick string')

    def test_symbol_rule_resolves_normalized_spelling_through_y_and_g(self):
        self.write('''
            class EscapedPlus: LexicalRule:
                def scan(input: LexemeInput, start: int) -> LexemeMatch:
                    characters = input.characters
                    if start + 2 < len(characters) and characters[start] == "`" and characters[start + 1] == "+" and characters[start + 2] == "`":
                        return LexemeMatch(start + 3, TokenKind.Symbol, value="+")
                    return LexemeMatch(start, None, false)
        ''', 'lexical.sev')
        self.native('import "lexical.sev"\ntest:\n    assert(4 `+` 2 == 6)\n')

    def test_progress_and_bounds_are_checked(self):
        for end in ('start', 'len(input.characters) + 1'):
            self.write(NUMBER_RULE.replace(
                'LexemeMatch(cursor, TokenKind.Integer,',
                f'LexemeMatch({end}, TokenKind.Integer,'), 'lexical.sev')
            self.rejects('import "lexical.sev"\ntest:\n    assert(~42 == 42)\n',
                         r'nonempty source range')

    def test_ambiguous_rules_and_explicit_priority(self):
        self.write(NUMBER_RULE, 'first.sev')
        second = NUMBER_RULE.replace('SigilInteger', 'OtherInteger').replace(
            'value=lexeme_text(input, start + 1, cursor)', 'value="73"')
        self.write(second, 'second.sev')
        program = 'import "first.sev"\nimport "second.sev"\ntest:\n    assert(~42 == 73)\n'
        self.rejects(program, r'ambiguous lexical rules')
        self.write(second.replace('class OtherInteger: LexicalRule:',
                                  'class OtherInteger: LexicalRule:\n    priority: int = 1'), 'second.sev')
        self.native(program)

    def test_longest_imported_rule_wins(self):
        self.write(NUMBER_RULE, 'first.sev')
        longer = NUMBER_RULE.replace('SigilInteger', 'LongerInteger').replace(
            '        return LexemeMatch(cursor, TokenKind.Integer,',
            '        if cursor < len(characters) and characters[cursor] == "!":\n'
            '            cursor += 1\n'
            '        return LexemeMatch(cursor, TokenKind.Integer,').replace(
                'value=lexeme_text(input, start + 1, cursor)', 'value="73"')
        self.write(longer, 'second.sev')
        self.native('import "first.sev"\nimport "second.sev"\ntest:\n    assert(~42! == 73)\n')

    def test_execution_budget_stops_nonterminating_rule(self):
        self.write(NUMBER_RULE.replace('cursor += 1', 'cursor = cursor'), 'lexical.sev')
        self.rejects('import "lexical.sev"\ntest:\n    assert(~42 == 42)\n',
                     r'execution budget exceeded')

    def test_missing_scan_is_rejected_at_registration(self):
        self.write('class Missing: LexicalRule:\n    pass\n', 'lexical.sev')
        self.rejects('import "lexical.sev"\ntest:\n    assert(true)\n',
                     r'missing scan implementation')


if __name__ == '__main__':
    unittest.main(verbosity=2)
