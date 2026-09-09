#!/usr/bin/env python3
"""Source tuple and shared field-pack acceptance through the source compiler."""
import os
import unittest

from migration import MigrationCase, ROOT


class Tuples(MigrationCase):
    def test_tuple_example(self):
        example = ROOT / "docs/examples/01-types/01-basic/05-tuples.sev"
        self.native(example.read_text())

    def test_empty_singleton_heterogeneous_and_nested(self):
        self.native('''
            test:
                empty = ()
                singleton = (42,)
                mixed = (42, true, 2.5)
                nested = (mixed, singleton)
                assert(string(empty) == "()")
                assert(string(singleton) == "(42,)")
                assert(string(mixed) == "(42,true,2.5)")
                assert(nested[0][2] == 2.5)
                assert(nested[1][0] == 42)
                assert((42) == 42)
                assert(string((1,2,3,4)[::-1]) == "(4,3,2,1)")
                assert(string(mixed[1:]) == "(true,2.5)")
                assert(string(mixed[2:2]) == "()")
        ''')

    def test_other_class_uses_same_pack_construction_and_projection(self):
        self.native('''
            class Bundle[...Elements]:
                elements: ...Elements
            test:
                bundle = Bundle(7, false, 1.5)
                assert(bundle.elements[0] == 7)
                assert(not bundle.elements[1])
                assert(bundle.elements[2] == 1.5)
                empty = Bundle()
                assert(len(empty.elements) == 0)
        ''')

    def test_syntax_constructor_binding_is_hygienic(self):
        self.native('''
            test:
                tuple = 99
                value = (1, true)
                assert(tuple == 99)
                assert(value[0] == 1)
                assert(value[1])
        ''')

    def test_source_grammar_and_semantic_body_change_without_rebuilding(self):
        service = os.path.relpath(ROOT / "sev_compiler/universal/component/services.sev",
                                  self.directory)
        for body, assertion in [
            ("return builders.construct(Bundle, values)", "assert(value[1] == 2.5)"),
            ("return values[0]", "assert(value == 7)"),
        ]:
            self.native(f'''
                import "{service}" as builders
                class Bundle[...Elements]:
                    elements: ...Elements
                trait BundleSyntax: G:
                    symbol: Y = <|
                    form: GrammarForm = GrammarForm.Delimited
                    closing: string = "|>"
                    separator: string = ","
                    grouping: bool = false
                    def semantic(values: list[Ex]) -> Ex:
                        {body}
                test:
                    value = <|7, 2.5|>
                    {assertion}
            ''')

    def test_expression_helper_uses_its_source_body(self):
        self.native('''
            def apply(constructor: F, arguments: list[Ex]) -> Ex:
                return constructor(...arguments)
            class Packet[...Types]:
                fields: ...Types
            trait PacketSyntax: G:
                symbol: Y = <|
                form: GrammarForm = GrammarForm.Delimited
                closing: string = "|>"
                separator: string = ","
                grouping: bool = false
                def semantic(values: list[Ex]) -> Ex:
                    return apply(Packet, values)
            test:
                packet = <|5, true|>
                assert(packet[0] == 5)
                assert(packet[1])
        ''')

    def test_same_grammar_helper_constructs_an_ordinary_record(self):
        service = os.path.relpath(ROOT / "sev_compiler/universal/component/services.sev",
                                  self.directory)
        self.native(f'''
            import "{service}" as builders
            class Point:
                x: int
                y: int
            trait PointSyntax: G:
                symbol: Y = <|
                form: GrammarForm = GrammarForm.Delimited
                closing: string = "|>"
                separator: string = ","
                grouping: bool = false
                def semantic(values: list[Ex]) -> Ex:
                    return builders.construct(Point, values)
            test:
                point = <|3, 4|>
                assert(point.x == 3)
                assert(point.y == 4)
        ''')

    def test_tuple_part_of_collections_example(self):
        self.native('''
            test:
                origin = (0.0, 0.0)
                x = origin[0]
                assert(x == 0.0)
                assert(string(origin) == "(0,0)")
        ''')

    def test_index_diagnostics(self):
        self.rejects("test:\n    flag: bool = string((1,))\n",
                     "(expected type|expected scalar type)")
        self.rejects("test:\n    value = (1, true)\n    print(value[2])\n",
                     "index is out of bounds")
        self.rejects("test:\n    value = (1, true)\n    print(value[::0])\n",
                     "slice step cannot be zero")

    def test_explicit_type_context_and_nominal_identity(self):
        self.native('''
            def pair() -> tuple[i8, bool]:
                return (12, true)
            test:
                value = pair()
                first: i8 = value[0]
                assert(first == 12)
                assert(value[1])
        ''')
        self.rejects('''
            class Bundle[...T]:
                values: ...T
            test:
                value: tuple[int] = Bundle(1)
        ''', "does not match expected type")

    def test_pack_iteration_and_length_evaluate_receiver_once(self):
        self.native('''
            class Bundle[...T]:
                items: ...T
            class Counter:
                value: int
                def make() -> Bundle[int, int]:
                    value += 1
                    return Bundle(10, 20)
                def empty() -> Bundle[]:
                    value += 1
                    return Bundle()
            test:
                counter := Counter(0)
                total := 0
                for item in counter.make().items:
                    total += item
                assert(total == 30)
                assert(counter.value == 1)
                assert(len(counter.empty().items) == 0)
                assert(counter.value == 2)
                for item in counter.empty().items:
                    assert(false)
                assert(counter.value == 3)
        ''')

    def test_construction_order(self):
        self.native('''
            class Counter:
                value: int
                def next() -> int:
                    value += 1
                    return value
            test:
                counter := Counter(0)
                values = (counter.next(), counter.next(), counter.next())
                assert(values[0] == 1)
                assert(values[1] == 2)
                assert(values[2] == 3)
                assert(counter.value == 3)
        ''')


if __name__ == "__main__":
    unittest.main()
