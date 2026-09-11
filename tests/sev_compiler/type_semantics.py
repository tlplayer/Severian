#!/usr/bin/env python3
"""Regressions for the shared semantics used by docs/examples/01-types."""
import unittest
import owned_records

from migration import COMPILER, MigrationCase, ROOT


class TypeSemantics(MigrationCase):
    def test_basic_examples_unchanged(self):
        directory = ROOT / 'docs/examples/01-types/01-basic'
        for filename, expected in [
            ('01-primitives.sev', '10\n1000000\n0.5\ntrue\na\nhello\n'),
            ('02-inference.sev', '10:int\n0.5:float\nl:char\ntrue:bool\nseverian:string\n'),
            ('03-conversion.sev', '10\n0.5\n10.5\ntrue\nseverian\n10.5!\n'),
            ('04-strings.sev', 'Severian\n8\nSEVERIAN\ntrue\nHello, Severian!\nSever\n'),
        ]:
            with self.subTest(example=filename):
                executable = self.directory / filename.removesuffix('.sev')
                self.succeeds([COMPILER, 'build', directory / filename,
                               '--sysroot', ROOT, '-o', executable])
                self.assertEqual(self.succeeds([executable]), expected)

    def test_interpolation_views_input_without_consuming_it(self):
        self.native('''
            def formatted(value: string) -> string:
                return f"Hello, {value}!"
            test:
                name = "Sev" + "erian"
                print(f"{name}:{type(name)}")
                greeting = formatted(name)
                assert(name[:5] == "Sever")
                drop(name)
                assert(greeting == "Hello, Severian!")
                print(greeting)
                empty = ""
                assert(f"{empty}" == "")
                assert(empty == "")
        ''', expected="Severian:string\nHello, Severian!\n")
        owned_records.OwnedRecords.check_allocations(self)

    def test_interpolation_evaluates_expressions_once_and_keeps_results_alive(self):
        self.native('''
            def text() -> string:
                print("text")
                return "λ" + "😀"
            def number() -> int:
                print("number")
                return 42
            def identity(value: string) -> string:
                return f"{value}"
            test:
                original = "Sev" + "erian"
                result = identity(original)
                print(original)
                drop(original)
                assert(result == "Severian")
                assert(f"{text()}:{number()}" == "λ😀:42")
                print(result)
        ''', expected="Severian\ntext\nnumber\nSeverian\n")
        owned_records.OwnedRecords.check_allocations(self)

    def test_character_conversion_for_concatenation(self):
        self.native('''
            test:
                count = 10
                ratio = 0.5
                character = '!'
                combined = string(count + ratio) + character
                assert(combined == "10.5!")
                assert("hello" + 'λ' == "helloλ")
                result: string = "hello" + '😀'
                assert(result == "hello😀")
                appended := "hello"
                appended += '!'
                assert(appended == "hello!")
        ''')

    def test_interpolation_preserves_nested_move_effects(self):
        self.rejects('''
            def consume(value: move string) -> int:
                return value.length()
            def formatted(value: string) -> string:
                return f"{consume(move value)}"
            def main():
                name = "Severian"
                print(formatted(name))
                print(name)
        ''', "use after drop or move")

    def test_for_initializer_runs_once_and_has_loop_scope(self):
        self.native('''
            def initial() -> int:
                print("initial")
                return 0
            def values() -> list[int]:
                print("values")
                return [10, 20, 30]
            test:
                total := 0
                for value in values() with index := initial():
                    index += 1
                    if index == 2:
                        continue
                    total += value
                    assert(index == 1 or index == 3)
                assert(total == 40)
                for value in range(0) with index := initial():
                    assert(false)
                for value in range(5) with index := 0:
                    index += 1
                    if index == 2:
                        break
                    total += value
                assert(total == 40)
        ''', expected="initial\nvalues\ninitial\n")
        self.rejects('''
            def main():
                for value in range(3) with index := 0:
                    index += 1
                print(index)
        ''', "unknown name index")

    def test_symbol_generic_example(self):
        path = ROOT / "docs/examples/01-types/05-generics/23-symbol-generic.sev"
        self.native(path.read_text())

    def test_wrapped_parameters_arguments_and_source_delimiters(self):
        self.native('''
            enum Packet:
                Empty
                Data(
                    value: int,
                )
            class Box[
                T,
            ]:
                value: T
            def add(
                left: int,
                right: int = 2,
            ) -> int:
                return left + right
            def boxed(
                value: Box[
                    int,
                ],
            ) -> int:
                return value.value
            test:
                values = [
                    add(
                        20,
                        right=22,
                    ),
                    7,
                ]
                pair = (
                    values[0],
                    boxed(
                        Box[int](9),
                    ),
                )
                assert(pair[0] == 42)
                assert(pair[1] == 9)
                if true:
                    assert(add(
                        40
                    ) == 42)
                assert(values[1] == 7)
                packet = Data(
                    pair[0],
                )
                match packet:
                    case Data(value):
                        assert(value == 42)
                    case Empty:
                        assert(false)
        ''')
        self.rejects('''
            def value(
                first: int
                second: int
            ) -> int:
                return first
        ''', "expected closing parenthesis")

    def test_enum_examples(self):
        for name in ("01-enum-basics", "02-enum-payloads"):
            with self.subTest(example=name):
                path = ROOT / f"docs/examples/01-types/06-enums/{name}.sev"
                self.native(path.read_text())

    def test_enum_payloads_and_match_scopes(self):
        self.native('''
            enum Choice:
                Empty
                Number(value: int)
                Pair(left: int, right: int)
            def total(choice: Choice) -> int:
                match choice:
                    case Empty:
                        return 0
                    case Number(value):
                        return value
                    case Pair(left, right):
                        return left + right
            test:
                assert(total(Choice.Empty) == 0)
                assert(total(Number(42)) == 42)
                assert(total(Choice.Pair(right=22, left=20)) == 42)
                Empty = 7
                assert(Empty == 7)
            test with compiler:
                reject:
                    value = Number(true)
                reject:
                    def incomplete(choice: Choice) -> int:
                        match choice:
                            case Empty:
                                return 0
                reject:
                    def duplicate(choice: Choice) -> int:
                        match choice:
                            case Empty:
                                return 0
                            case Empty:
                                return 1
                            case _:
                                return 2
                reject:
                    choice = Number(42)
                    match choice:
                        case Number(value):
                            assert(value == 42)
                        case _:
                            pass
                    assert(value == 42)
        ''')

    def test_enum_evaluation_order_and_nominal_identity(self):
        self.native('''
            enum Item:
                Pair(left: int, right: int)
                Empty
            def mark(value: int) -> int:
                print(value)
                return value
            def make() -> Item:
                print(1)
                return Item.Pair(right=mark(2), left=mark(3))
            test:
                match make():
                    case Pair(left, right):
                        assert(left == 3)
                        assert(right == 2)
                    case Empty:
                        assert(false)
        ''', expected="1\n2\n3\n")
        self.rejects('''
            enum First:
                Value
            enum Second:
                Value
            def wrong() -> First:
                return Second.Value
        ''', "value is not a variant of the expected enum")
        self.rejects('''
            enum First:
                Value
            enum Second:
                Value
            def ambiguous():
                value = Value
        ''', "ambiguous tagged variant")

    def test_additive_extension_example(self):
        path = ROOT / "docs/examples/01-types/08-extend/01-basic-extend.sev"
        self.native(path.read_text())
        self.rejects('''
            class Counter:
                value: int
                def get() -> int:
                    return value
            extend Counter:
                def get() -> int:
                    return 0
        ''', "extension cannot replace existing member get")

    def test_compiler_cases_compile_declarations_in_isolation(self):
        self.native('''
            test with compiler:
                accept:
                    class Box:
                        value: int
                    box = Box(42)
                    assert(box.value == 42)
                reject:
                    class Box:
                        value: int
                    box = Box(true)
                accept:
                    class Box:
                        value: bool
                    box = Box(true)
                    assert(box.value)
        ''')

    def test_list_size_and_named_source_operator(self):
        path = ROOT / "docs/examples/01-types/05-generics/17-block-generic.sev"
        self.native(path.read_text())
        self.native('''
            operator doubled(value: int) -> int:
                return value + value
            test:
                assert(doubled(21) == 42)
                assert(len([1, 2, 3]) == 3)
                assert(size([1, 2, 3]) == 3)
        ''')

    def test_none_return_and_drawable_example(self):
        path = ROOT / "docs/examples/01-types/07-traits/01-point-drawable.sev"
        self.native(path.read_text())
        self.native('''
            def implicit() -> None:
                pass
            def explicit() -> None:
                return
            test:
                assert(implicit() is None)
                assert(explicit() is None)
        ''')

    def test_unsigned_widths_and_full_u64_range(self):
        self.native('''
            test:
                small: u16 = 65535
                word: u32 = 4294967295
                large: u64 = 18446744073709551615
                assert(small > 32767)
                assert(word > 2147483647)
                assert(large > 9223372036854775807)
                assert(string(large) == "18446744073709551615")
                assert(large / 10 == 1844674407370955161)
                assert(large % 10 == 5)
                assert(u32(small) == 65535)
                assert(i64(word) == 4294967295)
                assert(u64(i64(42)) == 42)
        ''')
        self.rejects('test:\n    value: u64 = 18446744073709551616\n', "outside u64")
        self.rejects('test:\n    value: u64 = -1\n', "outside u64")

    def test_unsigned_lists_keep_element_types(self):
        self.native('''
            test:
                empty: list[u64] = []
                assert(size(empty) == 0)
                values: list[u64] := [18446744073709551615, 42]
                assert(values[0] > 9223372036854775807)
                values = [1, 2, 3]
                assert(size(values) == 3)
                assert(values[2] == 3)
        ''')

    def test_unsigned_generic_examples(self):
        for name in ("06-type-generic", "22-kind-generic", "25-compiler-term-generic"):
            with self.subTest(example=name):
                path = ROOT / f"docs/examples/01-types/05-generics/{name}.sev"
                self.native(path.read_text())

    def test_power(self):
        self.native('''
            test:
                assert(2 ** 10 == 1024)
                assert(2.0 ** 0.5 > 1.414)
                assert(9.0 ** 0.5 == 3.0)
                assert(2 ** 0 == 1)
        ''')

    def test_constructor_initializes_fields(self):
        path = ROOT / "docs/examples/01-types/02-classes/01-inferred-field-mutation.sev"
        self.native(path.read_text())

    def test_constructor_overloads(self):
        path = ROOT / "docs/examples/01-types/02-classes/05-overloaded-constructors.sev"
        self.native(path.read_text() + '''
test:
    assert(X(20, 22).value == 42)
    assert(X(17).value == 17)
''')

    def test_field_visibility_and_compiler_test_setup(self):
        path = ROOT / "docs/examples/01-types/02-classes/02-read-write-fields.sev"
        self.native(path.read_text())

    def test_constructor_checks_field_types_and_initialization(self):
        self.rejects('''
            class Wrong:
                value: int
                def Wrong():
                    value := true
            test:
                value = Wrong()
        ''', "boolean cannot initialize an integer")
        self.rejects('''
            class Missing:
                value: int
                def Missing():
                    pass
            test:
                value = Missing()
        ''', "unknown name value")

    def test_expression_generic_name_is_not_a_compiler_capability(self):
        path = ROOT / "docs/examples/01-types/05-generics/09-expression-generic.sev"
        self.native(path.read_text())


if __name__ == "__main__":
    unittest.main(verbosity=2)
