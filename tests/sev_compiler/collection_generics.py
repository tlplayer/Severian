#!/usr/bin/env python3
"""Container applications use the source compiler's ordinary generic machinery."""
import unittest

from migration import MigrationCase


class CollectionGenerics(MigrationCase):
    def test_named_arguments_preserve_type_identity_and_nested_layouts(self):
        self.native('''
            class Pair[Left, Right]:
                left: Left
                right: Right
            class Box[T]:
                value: T
            type Mixed = Pair[Right:string, Left:int]
            def first(value: Mixed) -> int:
                return value.left
            test:
                value = Pair[Left:int, Right:string](42, "text")
                assert(first(value) == 42)
                nested = Box[T:Pair[Right:string, Left:int]](value)
                assert(nested.value.right == "text")
                assert(Pair[int, float](7, 2.5).right == 2.5)
                data = [1, 2, 3]
                start = 1
                stop = 3
                assert(len(data[start:stop]) == 2)
        ''')

    def test_named_argument_diagnostics(self):
        self.rejects('class Duplicate[T, T]:\n    value: T\n', 'duplicate generic parameter')
        declaration = 'class Pair[Left, Right]:\n    left: Left\n    right: Right\n'
        for arguments, diagnostic in (
            ('Left:int, Unknown:int', 'unknown type parameter'),
            ('Left:int, Left:int', 'duplicate type argument'),
            ('int, Left:int', 'duplicate type argument'),
            ('Right:int, int', 'positional type argument follows named'),
            ('Left:int', 'missing type argument'),
        ):
            with self.subTest(arguments=arguments):
                self.rejects(declaration + f'type Invalid = Pair[{arguments}]\n', diagnostic)

    def test_container_constructor_is_a_compile_time_argument(self):
        self.native('''
            trait Readable[Element]:
                def get() -> Element
            class Box[Value]:
                value: Value
                def get() -> Value:
                    return value
            class Alternative[T]:
                value: T
                def get() -> T:
                    return value
            class Collection[T, C:Readable[Element:T]]:
                storage: C[T]
                def Collection(value: T):
                    storage = C[T](value)
                def get() -> T:
                    return storage.get()
            type Adapted[Element, Backend] = Collection[T:Element, C:Backend]
            test:
                first = Collection[T:int, C:Box](42)
                second = Collection[C:Alternative, T:string]("owned")
                assert(first.get() == 42)
                assert(second.get() == "owned")
                alias = Adapted[Backend:Box, Element:int](9)
                assert(alias.get() == 9)
        ''')
        self.rejects('''
            trait Readable[T]:
                def get() -> T
            class Wrong[T]:
                value: T
                def get() -> bool:
                    return true
            class Collection[T, C:Readable[T]]:
                storage: C[T]
            value = Collection[T:int, C:Wrong](Wrong[int](42))
        ''', 'container does not satisfy trait')

    def test_applied_traits_in_generic_function_bodies(self):
        self.native('''
            trait Readable[T]:
                def get() -> T
            trait Input[T]: Readable[T]:
                def ready() -> bool
            class Box[T]: Input[T]:
                value: T
                def get() -> T:
                    return value
                def ready() -> bool:
                    return true
            def read[T, C:Readable[T]](source: C, witness: T) -> T:
                return source.get()
            def forward[T, C:Input[T]](source: C, witness: T) -> T:
                return read(source, witness)
            test:
                assert(read(Box[int](42), 0) == 42)
                assert(forward(Box[int](7), 0) == 7)
                assert(read(Box[string]("owned"), "") == "owned")
                assert(read[C:Box[int], T:int](Box[int](9), 0) == 9)
        ''')

    def test_explicit_function_types_and_result_conversion(self):
        self.native('''
            def identity[T](value: T) -> T:
                return value
            def right[Left, Right](left: Left, right: Right) -> Right:
                return right
            test:
                assert(identity[T:int](42) == 42)
                assert(right[Right:string, Left:int](7, "result") == "result")
        ''')

    def test_explicit_function_arguments_respect_declaration_scope(self):
        self.write('''
            def identity[T](value: T) -> T:
                return value
            def read() -> int:
                return identity[T:int](42)
        ''', name='provider.sev')
        self.native('''
            import * from "provider.sev" as provider
            def identity[T](value: T) -> T:
                print("wrong declaration")
                return value
            test:
                assert(provider.read() == 42)
        ''')

    def test_union_element_type_survives_construction_and_lookup(self):
        self.native('''
            class Slot[T]:
                value: T
                def get() -> T:
                    return value
            def render(value: int | string) -> string:
                match value:
                    case number: int:
                        return string(number)
                    case text: string:
                        return text
            test:
                number = Slot[T:int | string](42)
                text = Slot[T:int | string]("owned")
                assert(render(number.get()) == "42")
                assert(render(text.get()) == "owned")
        ''')

    def test_applied_trait_class_contract(self):
        self.native('''
            trait Readable[T]:
                def get() -> T
            class Box[T]: Readable[T]:
                value: T
                def get() -> T:
                    return value
            test:
                assert(Box[int](42).get() == 42)
                assert(Box[string]("owned").get() == "owned")
        ''')
        self.rejects('''
            trait Readable[T]:
                def get() -> T
            class Wrong[T]: Readable[T]:
                value: T
                def get() -> bool:
                    return true
            wrong = Wrong[int](42)
        ''', 'does not satisfy trait')
        self.rejects('''
            trait Readable[T]:
                def get[T](value: T) -> T
            class Wrong[T]: Readable[T]:
                value: T
                def get(value: T) -> T:
                    return value
            wrong = Wrong[int](42)
        ''', 'generic trait methods require method argument inference')


if __name__ == '__main__':
    unittest.main()
