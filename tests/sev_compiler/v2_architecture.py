#!/usr/bin/env python3
"""Behavioral V2 graph gates against the unchanged source compiler executable."""
import unittest

from migration import MigrationCase


class SemanticGraph(MigrationCase):
    def checked_graph(self, text, **kwargs):
        ir = self.agent_ir(text, **kwargs)
        definitions = {node["id"]: node for node in ir["definitions"]}
        self.assertEqual(ir["source_coordinates"], "utf8-bytes")
        for edge in ir["dependencies"]:
            self.assertIn(edge["from"], definitions)
            self.assertIn(edge["to"], definitions)
            if edge["kind"] == "call":
                self.assertEqual(definitions[edge["from"]]["kind"], "function")
                self.assertEqual(definitions[edge["to"]]["kind"], "function")
        flattened = [node for group in ir["dependency_components"] for node in group]
        self.assertCountEqual(flattened, definitions)
        component = {node: i for i, group in enumerate(ir["dependency_components"])
                     for node in group}
        for edge in ir["dependencies"]:
            left = definitions[edge["from"]]["submodule"]
            right = definitions[edge["to"]]["submodule"]
            if left == right:
                self.assertLessEqual(component[edge["to"]], component[edge["from"]])
            else:
                self.assertIn({"from": left, "to": right},
                              ir["package_graph"]["dependencies"])
        for group in ir["dependency_components"]:
            self.assertEqual(len({definitions[node]["submodule"] for node in group}), 1)
        for function in ir["functions"]:
            for block in function["blocks"]:
                for operation in block["operations"]:
                    if operation["kind"] == "Call":
                        target = operation["function"]
                        self.assertIn(target, definitions)
                        self.assertEqual(definitions[target]["kind"], "function")
                        self.assertIn({"from": function["definition"], "to": target,
                                       "kind": "call"}, ir["dependencies"])
        return ir

    def test_call_references_and_dependency_order(self):
        ir = self.checked_graph('''
            def twice(value: int) -> int:
                return value + value
            def answer() -> int:
                return twice(21)
            test:
                assert(answer() == 42)
        ''')
        twice = self.definition(ir, "twice", "function")
        answer = self.definition(ir, "answer", "function")
        self.assertIn({"from": answer["id"], "to": twice["id"], "kind": "call"},
                      ir["dependencies"])

    def test_mutual_recursion_is_one_component(self):
        text = '''
            def even(value: int) -> bool:
                if value == 0:
                    return true
                return odd(value - 1)
            def odd(value: int) -> bool:
                if value == 0:
                    return false
                return even(value - 1)
            test:
                assert(even(8))
                assert(odd(7))
        '''
        ir = self.checked_graph(text)
        identities = {self.definition(ir, name, "function")["id"]
                      for name in ("even", "odd")}
        self.assertIn(identities, [set(group) for group in ir["dependency_components"]])
        self.native(text)

    def test_unicode_provenance_uses_utf8_bytes(self):
        subject = self.write('''
            # π 😀 preceding a declaration must not shift its byte origin
            def greeting() -> string:
                return "héllo 😀"
            test:
                assert(greeting() == "héllo 😀")
        ''')
        ir = self.checked_graph(subject)
        origin = self.definition(ir, "greeting", "function")["source"]
        data = subject.read_bytes()
        self.assertEqual(origin["start"], data.index(b"def greeting"))
        self.assertIn('"héllo 😀"', data[origin["start"]:origin["end"]].decode())
        self.assertIsNone(self.definition(ir, "$initializer", "function")["source"])

    def test_import_aliases_reference_one_grammar(self):
        self.write('''
            trait LocalGrammar: G:
                pass
        ''', "grammar.sev")
        ir = self.checked_graph('''
            import * from "grammar.sev" as first
            import * from "grammar.sev" as second
            test:
                assert(true)
        ''')
        nodes = [node for node in ir["definitions"]
                 if any(name.endswith(".LocalGrammar")
                        for name in node.get("aliases", []))]
        self.assertEqual(len(nodes), 1)
        self.assertEqual(set(nodes[0]["aliases"]),
                         {"first.LocalGrammar", "second.LocalGrammar"})

    def test_package_owns_explicit_multifile_membership(self):
        self.write('''
            {"package": {"name": "graph-fixture", "version": "0.1.0",
              "metadata": {"semantic": {"module": "frontend", "submodules": {
                "analysis": ["subject.sev", "helper.sev"],
                "storage": ["storage.sev"]}}}},
              "lib": {"path": "subject.sev"}}
        ''', "package.json")
        self.write('''
            def helper(value: int) -> int:
                return value + 1
        ''', "helper.sev")
        self.write('''
            def stored() -> int:
                return 41
        ''', "storage.sev")
        ir = self.checked_graph('''
            import * from "helper.sev"
            import * from "storage.sev"
            def answer() -> int:
                return helper(stored())
            test:
                assert(answer() == 42)
        ''')
        answer = self.definition(ir, "answer", "function")
        helper = self.definition(ir, "helper", "function")
        stored = self.definition(ir, "stored", "function")
        self.assertEqual(answer["submodule"], helper["submodule"])
        self.assertNotEqual(answer["submodule"], stored["submodule"])
        units = ir["package_graph"]["units"]
        self.assertEqual(units[answer["submodule"]]["module"], "frontend")
        self.assertEqual(units[answer["submodule"]]["submodule"], "analysis")
        self.assertEqual(units[stored["submodule"]]["submodule"], "storage")


if __name__ == "__main__":
    unittest.main(verbosity=2)
