"""Unsafe grants pointer access within a lexical execution scope."""
from py_compiler.syntax.generic.block import BlockProvider


def require_unsafe(context, operation):
    if not getattr(context, 'unsafe_depth', 0):
        raise ValueError(f'{operation} require an unsafe block; help: use array[T, N] instead')


class Unsafe(BlockProvider):
    spelling = 'unsafe'

    def parse_header(self, items):
        header = super().parse_header(items)
        if header:
            raise ValueError('unsafe takes no header operands')
        return header

    def declare(self, node, context):
        context.unsafe_depth += 1
        try:
            super().declare(node, context)
        finally:
            context.unsafe_depth -= 1

    def lower(self, node, cfg, env, nodes, index):
        # Block locals stay local even in a module initializer. Existing outer
        # places retain their storage, so writes still update their bindings.
        previous_factory = cfg.storage_factory
        cfg.storage_factory = None
        cfg.unsafe_depth += 1
        try:
            cfg.scope(node.children, dict(env))
            cfg.flow.end_scope(cfg.live_binding_ids(env))
        finally:
            cfg.unsafe_depth -= 1
            cfg.storage_factory = previous_factory
        return 1


import unittest


class UnsafeTests(unittest.TestCase):
    def compile(self, text):
        from py_compiler.frontend.src.lib import compile_source
        from py_compiler.syntax.recognition import Syntax
        return compile_source('unsafe.sev', text, Syntax())

    def test_pointer_construction_and_nested_scope_lower_to_mlir(self):
        from py_compiler.mlir.src.lib import lower, render
        result = self.compile('unsafe:\n    address: pointer = pointer(0)\n    unsafe:\n        nested: pointer = pointer(4096)\n    copied = address\n')
        self.assertFalse(result.diagnostics, str(result.diagnostics))
        text = render(lower(result.program))
        self.assertIn('llvm.mlir.zero : !llvm.ptr', text)
        self.assertIn('llvm.inttoptr', text)

    def test_pointer_requires_unsafe_in_both_literal_and_executable_paths(self):
        for text in ('address = pointer(0)\n', 'address: pointer = 0\n',
                     'def work():\n    address = pointer(0)\n',
                     'def identity(value: pointer) -> pointer:\n    return value\n',
                     'unsafe:\n    address = pointer(0)\nother = pointer(1)\n',
                     'unsafe:\n    def work():\n        address = pointer(0)\n'):
            with self.subTest(source=text):
                result = self.compile(text)
                self.assertTrue(result.diagnostics)
                self.assertIn('require an unsafe block', str(result.diagnostics[0]))

    def test_pointer_local_does_not_escape_unsafe_scope(self):
        result = self.compile('def work():\n    unsafe:\n        address = pointer(0)\n    leaked = address\n')
        self.assertTrue(result.diagnostics)
        self.assertIn('unknown binding', str(result.diagnostics[0]))

    def test_outer_mutation_and_return_flow_are_preserved(self):
        result = self.compile('def work() -> int:\n    value = 1\n    unsafe:\n        value = 2\n    return value\n')
        self.assertFalse(result.diagnostics, str(result.diagnostics))
        body = next(b for b in result.program.bodies if b.declaration == 'work')
        constants = {o.result.identity: o.payload.value for block in body.blocks for o in block.operations if o.kind == 'constant'}
        self.assertEqual(constants[body.blocks[-1].terminator.value.identity], 2)
        result = self.compile('def work() -> int:\n    unsafe:\n        return 3\n')
        self.assertFalse(result.diagnostics, str(result.diagnostics))

    def test_unsafe_header_rejects_operands(self):
        self.assertTrue(self.compile('unsafe true:\n    value = 1\n').diagnostics)

    def test_allocation_requires_lexical_unsafe_with_array_help(self):
        for source in (
            'value = allocate[int](3)\n',
            'def work():\n    value = allocate[int](3)\n',
            'def work():\n    unsafe:\n        value = allocate[int](3)\n    other = allocate[int](1)\n',
            'unsafe:\n    def work():\n        value = allocate[int](3)\n',
            'def work():\n    value = allocate[int](3)\nunsafe:\n    work()\n',
        ):
            with self.subTest(source=source):
                result = self.compile(source)
                self.assertTrue(result.diagnostics)
                self.assertIn('allocate[T] calls require an unsafe block', str(result.diagnostics))
                self.assertIn('help: use array[T, N] instead', str(result.diagnostics))

    def test_typed_pointer_operations_require_unsafe_with_array_help(self):
        for source in (
            'def read(p: pointer[int]) -> int:\n    return p[0]\n',
            'def write(p: borrow pointer[int]):\n    p[0] = 1\n',
            'def identity(p: pointer[int]) -> pointer[int]:\n    return p\n',
            'def convert(a: array[int]):\n    p = pointer[int](a)\n',
            'def convert(a: array[int]):\n    p: pointer[int] = a\n',
            'def address():\n    value = 1\n    p = &value\n',
            'def read(p: pointer[int]) -> int:\n    unsafe:\n        value = p[0]\n    return p[0]\n',
        ):
            with self.subTest(source=source):
                result = self.compile(source)
                self.assertTrue(result.diagnostics)
                self.assertIn('pointer[T] values require an unsafe block', str(result.diagnostics))
                self.assertIn('help: use array[T, N] instead', str(result.diagnostics))

    def test_explicit_unsafe_allows_allocation_and_typed_pointers(self):
        result = self.compile('def work() -> int:\n    unsafe:\n        a = allocate[int](3)\n        p = pointer[int](a)\n        unsafe:\n            p[0] = 7\n        return p[0]\n')
        self.assertFalse(result.diagnostics, str(result.diagnostics))

    def test_array_access_does_not_require_unsafe(self):
        result = self.compile('def read(a: array[int,3]) -> int:\n    return a[0]\ndef write(a: borrow array[int,3]):\n    a[0] = 7\n')
        self.assertFalse(result.diagnostics, str(result.diagnostics))
