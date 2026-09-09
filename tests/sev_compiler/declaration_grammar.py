#!/usr/bin/env python3
"""Source-defined declaration grammar, exercised without rebuilding the compiler."""
import hashlib
import os
import re
import unittest

from migration import MigrationCase, ROOT


class DeclarationGrammar(MigrationCase):
    def test_enum_syntax_requires_its_prelude_provider(self):
        sysroot = self.directory / "sysroot"
        target = sysroot / "sev_compiler/universal/prelude.sev"
        target.parent.mkdir(parents=True)
        origin = ROOT / "sev_compiler/universal/prelude.sev"
        providers = []
        enum_provider = ""
        for line in origin.read_text().splitlines():
            match = re.fullmatch(r'import "([^"]+)"(.*)', line)
            if not match:
                continue
            path = (origin.parent / match[1]).resolve()
            entry = f'import "{os.path.relpath(path, target.parent)}"{match[2]}\n'
            if path.name == "enum.sev":
                enum_provider = entry
            else:
                providers.append(entry)
        self.assertTrue(enum_provider)
        target.write_text("".join(providers))
        self.native('test:\n    assert(20 + 22 == 42)\n', sysroot=sysroot)
        self.rejects('enum Color:\n    Red\n    Blue\n',
                     "expected a type name", sysroot=sysroot)
        target.write_text("".join(providers) + enum_provider)
        example = ROOT / "docs/examples/01-types/05-enums/01-enum-basics.sev"
        self.native(example.read_text(), sysroot=sysroot)

    def test_declaration_symbol_can_be_punctuation(self):
        self.write('''
            trait Choice: G:
                symbol: Y = <|
                form: GrammarForm = GrammarForm.Declaration
                members: DeclarationMembers = DeclarationMembers.Callables
                representation: PrimitiveRepresentation = PrimitiveRepresentation.TaggedAggregate
        ''', "language.sev")
        self.native('''
            import "language.sev"
            <| Value:
                Empty
                Number(value: int)
            def read(item: Value) -> int:
                match item:
                    case Number:
                        return value
                    case Empty:
                        return 0
            test:
                assert(read(Number(42)) == 42)
        ''')

    def test_primitive_controls_keyword_and_representation(self):
        before = {str(path): hashlib.sha256(path.read_bytes()).hexdigest()
                  for path in (ROOT / "sev_compiler/frontend").rglob("*.sev")}
        primitive = (ROOT / "sev_compiler/universal/primitive/enum.sev").read_text()
        primitive = primitive.replace("trait enum:", "trait Choice:")
        for word in ("choice", "alternative"):
            self.write(primitive.replace("symbol: Y = enum", f"symbol: Y = {word}"),
                       "language.sev")
            self.native(f'''
                import "language.sev"
                {word} Value:
                    Empty
                    Number(value: int)
                def read(item: Value) -> int:
                    match item:
                        case Empty:
                            return 0
                        case Number:
                            return value
                test:
                    assert(read(Number(42)) == 42)
                    assert(read(Value.Empty) == 0)
            ''')
        # The same declaration path also creates ordinary product types.
        product = primitive.replace("symbol: Y = enum", "symbol: Y = product")
        product = product.replace("DeclarationMembers.Callables", "DeclarationMembers.Fields")
        product = product.replace("PrimitiveRepresentation.TaggedAggregate",
                                  "PrimitiveRepresentation.RecordAggregate")
        self.write(product, "language.sev")
        self.native('''
            import "language.sev"
            product Pair:
                left: int
                right: int
            test:
                pair = Pair(20, 22)
                assert(pair.left + pair.right == 42)
        ''')
        after = {str(path): hashlib.sha256(path.read_bytes()).hexdigest()
                 for path in (ROOT / "sev_compiler/frontend").rglob("*.sev")}
        self.assertEqual(before, after)

    def test_invalid_contract_and_conflicting_grammar_are_rejected(self):
        self.rejects('''
            trait Broken: G:
                symbol: Y = broken
                form: GrammarForm = GrammarForm.Declaration
                members: DeclarationMembers = DeclarationMembers.Fields
                representation: PrimitiveRepresentation = PrimitiveRepresentation.TaggedAggregate
        ''', "declaration members do not match the representation")
        self.rejects('''
            trait Conflict: G:
                symbol: Y = enum
                form: GrammarForm = GrammarForm.Declaration
                members: DeclarationMembers = DeclarationMembers.Fields
                representation: PrimitiveRepresentation = PrimitiveRepresentation.RecordAggregate
        ''', "conflicting declaration grammar")


if __name__ == "__main__":
    unittest.main(verbosity=2)
