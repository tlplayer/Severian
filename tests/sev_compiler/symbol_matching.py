"""Native lexer regressions for cached operator spellings."""
import unittest

from migration import MigrationCase, contract


class SymbolMatching(MigrationCase):
    def test_overlapping_operators_and_punctuation(self):
        self.native('''
def adjust(value: int) -> int:
    if value >= 10:
        return value - 1
    return value + 1

test:
    value := 5
    value += 2
    assert(adjust(10) == 9)
    assert(adjust(5) == 6)
    assert(value != 3 and value <= 7)
    values = [10, 20]
    assert(values[1] == 20)
''')

    def test_registered_operator_uses_longest_match(self):
        self.native(contract(symbol='<~>') + '''
test:
    assert((2 <~> 7) == 27)
    assert(2 < 7)
    assert(7 > 2)
''')

    def test_registered_unicode_symbol_preserves_character_offsets(self):
        self.native(contract(symbol='⊕') + '''
test:
    assert((2 ⊕ 7) == 27)
    assert(((2 ⊕ 7) ⊕ 3) == 273)
''')


if __name__ == '__main__':
    unittest.main(verbosity=2)
