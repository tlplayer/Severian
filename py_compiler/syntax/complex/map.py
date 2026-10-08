"""Brace literals and map[K,V,C] default to ordered B-tree storage."""
from dataclasses import dataclass
from py_compiler.syntax.complex.dictionary import Dictionary


@dataclass(frozen=True)
class Map(Dictionary):
    name: str = 'map'
    family: str = 'map'


import unittest


class MapTests(unittest.TestCase):
    def test_string_literal_keys_and_duplicate_updates(self):
        from py_compiler.frontend.src.lib import compile_source
        from py_compiler.syntax.recognition import Syntax
        from py_compiler.mlir.src.lib import lower, render
        result = compile_source('map.sev', 'def work() -> int:\n    d = {"first":34,"second":8,"first":42}\n    return d["first"]\n', Syntax())
        self.assertFalse(result.diagnostics, str(result.diagnostics))
        binding = next(b for body in result.program.bodies for b in body.bindings if b.name == 'd')
        self.assertEqual(binding.type.name, 'map[string,i64,btree]')
        self.assertIn('sev_dict_set', render(lower(result.program)))

    def test_empty_map_uses_annotation(self):
        from py_compiler.frontend.src.lib import compile_source
        from py_compiler.syntax.recognition import Syntax
        result = compile_source('map.sev', 'def work() -> int:\n    d: map[int,int] = {}\n    d[1] = 3\n    return d[1]\n', Syntax())
        self.assertFalse(result.diagnostics, str(result.diagnostics))
