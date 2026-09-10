#!/usr/bin/env python3
"""Additional source-contract regressions using the unchanged source compiler."""
import hashlib
import unittest

from migration import MigrationCase, ROOT, contract


class SourceContracts(MigrationCase):
    def test_imported_punctuation_keeps_frontend_and_binary_unchanged(self):
        def frontend_snapshot():
            return {
                str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                for component in ("lexer", "parser")
                for path in (ROOT / "sev_compiler/frontend" / component).rglob("*.sev")
            }

        before = frontend_snapshot()
        try:
            for spelling in ("|>", "|~>"):
                self.rejects(f"test:\n    assert(4 {spelling} 2 == 42)\n",
                             r"(unexpected character|expected an expression)",
                             name="unregistered.sev")
                self.write(contract("Pipeline", spelling), "language.sev")
                self.native(f'''
                    import * from "language.sev"
                    test:
                        assert(4 {spelling} 2 == 42)
                        assert(f"{{4 {spelling} 2}}" == "42")
                ''')
        finally:
            # MigrationCase.tearDown also checks the compiler binary digest.
            self.assertEqual(frontend_snapshot(), before)

    def test_inherited_semantic_body_can_bind_a_new_symbol(self):
        self.native(contract() + '''
trait Derived: Fuse:
    symbol: Y = <!~>
extend int:
    operator <!~>[G:Derived](right: Self) -> Self
test:
    assert(4 <!~> 2 == 42)
''')

    def test_requirement_resolves_even_without_instantiation(self):
        self.rejects('''
            trait Addable:
                operator +[G:Missing](other: Self) -> Self
        ''', r"unknown operator constraint Missing")

    def test_grammar_parameter_can_be_renamed(self):
        self.native(contract().replace("[G:Fuse]", "[Syntax:Fuse]") + '''
test:
    assert(4 <~> 2 == 42)
''')

    def test_new_symbol_and_user_type_share_a_generic_requirement(self):
        self.native('''
            trait Fuse: G:
                symbol: Y = <~>
                arity: int = 2
                precedence: int = 7
                associativity: Associativity = Left
                left: Operand = Value[Self]
                right: Operand = Value[Self]
                result: T = Self
                evaluation: Evaluation = LeftToRight
                effects: Effects = Pure
                def semantic(left: Self, right: Self) -> Self
            trait Fusible:
                operator <~>[G:Fuse](right: Self) -> Self
            extend int:
                operator <~>[G:Fuse](right: Self) -> Self:
                    return self * 10 + right
            class Box: Fusible:
                amount: int
                operator <~>[G:Fuse](right: Self) -> Self:
                    return Box(amount * 10 + right.amount)
            def combine[T: Fusible](left: T, right: T) -> T:
                return left <~> right
            test:
                assert(combine(4, 2) == 42)
                assert(combine(Box(4), Box(2)).amount == 42)
        ''')

    def test_binding_symbol_must_match_its_contract(self):
        program = contract().replace("operator <~>[G:Fuse]", "operator +[G:Fuse]")
        self.rejects(program, r"symbol does not match its grammar contract")

    def test_semantic_operand_types_must_match_the_implementation(self):
        program = contract().replace("semantic(left: int", "semantic(left: float")
        self.rejects(program, r"semantic operand type does not match")

    def test_semantic_result_type_must_match_the_implementation(self):
        program = contract().replace(
            "semantic(left: int, right: int) -> int",
            "semantic(left: int, right: int) -> float")
        self.rejects(program, r"semantic result type does not match")


if __name__ == "__main__":
    unittest.main(verbosity=2)
