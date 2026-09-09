#!/usr/bin/env python3
"""Native checks for scalar operations selected from source grammar maps."""
import shutil
import unittest

from migration import MigrationCase, ROOT


class GrammarScalar(MigrationCase):
    def overlay(self):
        root = self.directory / "sysroot"
        (root / "sev_compiler").mkdir(parents=True)
        shutil.copytree(ROOT / "sev_compiler/universal", root / "sev_compiler/universal")
        (root / "library").symlink_to(ROOT / "library", target_is_directory=True)
        return root, root / "sev_compiler/universal/grammar/contracts.sev"

    def test_map_edit_changes_integer_operation_without_rebuilding(self):
        root, contracts = self.overlay()
        self.native('test:\n    assert(3 + 4 == 7)\n', sysroot=root)
        text = contracts.read_text()
        start = text.index("trait Add: G:")
        end = text.index("\ntrait ", start + 1)
        declaration = text[start:end]
        self.assertIn('ScalarOperation("addi")', declaration)
        declaration = declaration.replace('ScalarOperation("addi")', 'ScalarOperation("muli")')
        contracts.write_text(text[:start] + declaration + text[end:])
        self.native('''
            test:
                assert(3 + 4 == 12)
                assert(3.0 + 4.0 == 7.0)
        ''', sysroot=root)

    def test_new_operator_uses_its_type_map(self):
        self.native('''
            trait Combine: G:
                symbol: Y = <~>
                precedence: int = 7
                associativity: Associativity = Left
                operations: {T: ScalarOperation} = {
                    int: ScalarOperation("addi"),
                    float: ScalarOperation("mulf"),
                }
            test:
                assert((3 <~> 4) == 7)
                assert((3.0 <~> 4.0) == 12.0)
        ''')

    def test_empty_map_does_not_fall_back_to_a_builtin(self):
        root, contracts = self.overlay()
        text = contracts.read_text()
        start = text.index("trait Add: G:")
        end = text.index("\ntrait ", start + 1)
        declaration = text[start:end]
        map_start = declaration.index("{\n", declaration.index("operations:"))
        map_end = declaration.index("}", map_start)
        declaration = declaration[:map_start] + "{}" + declaration[map_end + 1:]
        contracts.write_text(text[:start] + declaration + text[end:])
        self.rejects('test:\n    value = 3 + 4\n', r"operator", sysroot=root)


if __name__ == "__main__":
    unittest.main(verbosity=2)
