from py_compiler.syntax.block.record import RecordProvider, TypeDeclaration, register_template, record_header
from py_compiler.syntax.function.contracts import signature
from py_compiler.syntax.complex.object import ObjectType


class Trait(RecordProvider):
    spelling = "trait"

    def parse_header(self, items):
        header = tuple(items[1:-1] if items[-1].text == ':' else items[1:])
        record_header(header)
        return header

    def declare(self, node, context):
        if register_template(self, node, context):
            return
        if record_header(node.header)[2]:
            raise ValueError('trait inheritance is not implemented')
        fields, defaults, members = self.fields(node, context)
        methods = tuple(signature(member, ".".join((*context.scope, node.header[0].text))) for member in members)
        for method in methods:
            method.scope = context.scope
            if not method.templates:
                method.parameters = tuple((name, context.type(annotation).name if annotation and annotation != 'Self' else annotation)
                                          for name, annotation in method.parameters)
                if method.result and method.result != 'Self':
                    method.result = context.type(method.result).name
            if any(annotation is None for _, annotation in method.parameters):
                raise ValueError("trait parameter contracts require type annotations")
        context.register(TypeDeclaration(node.identity, ".".join((*context.scope, node.header[0].text)), self.spelling,
                                         fields, (), (), node.span, context.source.path, defaults, methods,
                                         getattr(node, 'record_template', ''), getattr(node, 'record_arguments', ())))
        name = ".".join((*context.scope, node.header[0].text))
        context.types[name] = ObjectType(name, "trait", "!llvm.struct<(!llvm.ptr, i32)>", declaration=context.by_name[name])
        if not getattr(context, 'importing_trait', False):
            from py_compiler.syntax.block.function import Function
            previous = context.scope
            context.scope = (*previous, node.header[0].text)
            try:
                for method in methods:
                    if method.node.children and method.receiver_ownership is None:
                        Function().register(method.node, context)
            finally:
                context.scope = previous


    def satisfy(self, target, implementation, context):
        def resolved(annotation):
            from py_compiler.syntax.complex.generic import substitute
            return context.type(substitute(annotation, {'Self': context.types[implementation.name]}))
        for field in target.fields:
            if field not in implementation.fields:
                raise ValueError(f"{implementation.name} does not satisfy {target.name}.{field[0]}: {field[1]}")
        for method in target.methods:
            name = method.name.rsplit(".", 1)[-1]
            candidates = context.functions.get(implementation.name + "." + name, [])
            expected = tuple(resolved(t) for _, t in method.parameters)
            if not candidates:
                if method.node.children and method.receiver_ownership is None:
                    continue
                if method.node.children:
                    from py_compiler.syntax.block.function import Function
                    from importlib import import_module
                    Function().register(method.node, context, import_module("py_compiler.syntax.block.class").Receiver(implementation))
                    candidates = context.functions[implementation.name + "." + name]
                    for entry in candidates:
                        entry.trait_fallback = True
                else:
                    raise ValueError(f"{implementation.name} is missing {target.name}.{name}")
            matches = []
            for entry in candidates:
                if entry.result_ownership != method.result_ownership:
                    continue
                actual = tuple(resolved(t) for _, t in entry.parameters)
                if actual == expected and (method.node.children and method.receiver_ownership is None or resolved(entry.result or "absent") == resolved(method.result or "absent")):
                    if any(entry.parameter_ownership.get(n, 'view') != method.parameter_ownership.get(m, 'view') for (n, _), (m, _) in zip(entry.parameters, method.parameters)):
                        continue
                    if method.receiver_ownership == 'view' and entry.receiver_ownership == 'borrow':
                        continue
                    matches.append(entry)
            if len(matches) != 1:
                raise ValueError(f"{implementation.name}.{name} requires exactly one implementation of the trait callable contract")


import unittest


class GenericTraitTests(unittest.TestCase):
    def test_shared_syntax_keeps_parallel_trait_checks_in_their_module(self):
        from concurrent.futures import ThreadPoolExecutor
        from threading import Barrier
        from unittest.mock import patch
        from py_compiler.frontend.src.lib import compile_source
        from py_compiler.syntax.recognition import Syntax
        syntax = Syntax()
        barrier = Barrier(2)
        satisfy = Trait.satisfy

        def synchronized(provider, target, implementation, context):
            barrier.wait(timeout=5)
            return satisfy(provider, target, implementation, context)

        def compile_module(name):
            return compile_source(name + '.sev',
                'trait Readable:\n    def read(self: view Self) -> int\n'
                f'class {name}: Readable\n'
                '    def read(self: view Self) -> int:\n        return 1\n', syntax)

        with patch.object(Trait, 'satisfy', synchronized), ThreadPoolExecutor(max_workers=2) as pool:
            results = list(pool.map(compile_module, ('First', 'Second')))
        for result in results:
            self.assertFalse(result.diagnostics, str(result.diagnostics))

    source = ('trait Readable[T]:\n    def read(self: view Self) -> T\n'
              'class Cell[T = int]: Readable[T]\n    value: T\n'
              '    def read(self: view Self) -> T:\n        return self.value\n')

    def compile(self, source):
        from py_compiler.frontend.src.lib import compile_source
        from py_compiler.syntax.recognition import Syntax
        return compile_source('trait-contracts.sev', source, Syntax())

    def test_constraint_realizes_to_direct_method_call(self):
        result = self.compile(self.source + 'def read[T: Readable[int]](value: T) -> int:\n    return value.read()\ndef work() -> int:\n    value = Cell[int](3)\n    return read(value)\n')
        self.assertFalse(result.diagnostics, str(result.diagnostics))
        body = next(body for body in result.program.bodies if body.declaration == 'read')
        calls = [operation.payload.symbol for block in body.blocks for operation in block.operations if operation.kind == 'call']
        self.assertTrue(any('Cell[i64].read' in symbol for symbol in calls))
        self.assertEqual(len(body.blocks), 1)

    def test_wrong_trait_argument_is_rejected(self):
        result = self.compile(self.source + 'def read[T: Readable[i32]](value: T) -> int:\n    return 1\ndef work() -> int:\n    value = Cell[int](3)\n    return read(value)\n')
        self.assertTrue(result.diagnostics)
        self.assertIn('does not satisfy', str(result.diagnostics[0]))

    def test_generic_trait_default_method(self):
        source = ('trait Default[T]:\n    def value(self: view Self) -> T:\n        return T(1)\n'
                  'class Item: Default[int]\n    unused: int\n'
                  'def work() -> int:\n    item = Item(0)\n    return item.value()\n')
        result = self.compile(source)
        self.assertFalse(result.diagnostics, str(result.diagnostics))
