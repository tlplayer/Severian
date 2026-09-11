#!/usr/bin/env python3
import unittest
from migration import MigrationCase

class ContainerConstruction(MigrationCase):
    def test_default_provider_and_variadic_factory(self):
        self.native('''
            class Storage[T]:
                values: list[T]
                def Storage():
                    values = []
                def add(value: T):
                    values.append(value)
                def first() -> T:
                    return values[0]
                def count() -> int:
                    return len(values)
            class Bag[T, C=Storage]:
                storage: C[T]
                def Bag(*values: T):
                    storage = C[T]()
                    for value in values:
                        storage.add(value)
                def first() -> T:
                    return storage.first()
                def count() -> int:
                    return storage.count()
            test:
                values = Bag[T:int](42, 7, 9)
                assert(values.first() == 42)
                assert(values.count() == 3)
                assert(Bag[string]("owned", "text").first() == "owned")
                assert(Bag[int]().count() == 0)
        ''')

    def test_dependent_defaults_and_diagnostics(self):
        self.native('''
            class Pair[T, R=T]:
                first: T
                second: R
            type Both[T, R=T] = Pair[T, R]
            test:
                assert(Pair[int](7, 42).second == 42)
                assert(Both[string]("first", "second").second == "second")
        ''')
        self.rejects('class Bad[T=int, R]:\n    value: T\n', 'required type parameter follows a default')
        self.rejects('class Pair[T, R=T]:\n    first: T\n    second: R\nPair[int](1, "wrong")\n', 'expected scalar type')

if __name__ == '__main__':
    unittest.main()
