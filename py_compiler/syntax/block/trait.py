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
        self.context = context
        methods = tuple(signature(member, ".".join((*context.scope, node.header[0].text))) for member in members)
        for method in methods:
            if any(annotation is None for _, annotation in method.parameters):
                raise ValueError("trait parameter contracts require type annotations")
        context.register(TypeDeclaration(node.identity, ".".join((*context.scope, node.header[0].text)), self.spelling,
                                         fields, (), (), node.span, context.source.path, defaults, methods,
                                         getattr(node, 'record_template', ''), getattr(node, 'record_arguments', ())))
        name = ".".join((*context.scope, node.header[0].text))
        context.types[name] = ObjectType(name, "trait", "!llvm.struct<(!llvm.ptr, i32)>", declaration=context.by_name[name])

    def satisfy(self, target, implementation):
        def resolved(annotation):
            from py_compiler.syntax.complex.generic import substitute
            return self.context.type(substitute(annotation, {'Self': self.context.types[implementation.name]}))
        for field in target.fields:
            if field not in implementation.fields:
                raise ValueError(f"{implementation.name} does not satisfy {target.name}.{field[0]}: {field[1]}")
        for method in target.methods:
            name = method.name.rsplit(".", 1)[-1]
            candidates = self.context.functions.get(implementation.name + "." + name, [])
            expected = tuple(resolved(t) for _, t in method.parameters)
            if not candidates:
                if method.node.children:
                    from py_compiler.syntax.block.function import Function
                    from importlib import import_module
                    Function().register(method.node, self.context, import_module("py_compiler.syntax.block.class").Receiver(implementation))
                    candidates = self.context.functions[implementation.name + "." + name]
                else:
                    raise ValueError(f"{implementation.name} is missing {target.name}.{name}")
            matches = []
            for entry in candidates:
                actual = tuple(resolved(t) for _, t in entry.parameters)
                if actual == expected and resolved(entry.result or "absent") == resolved(method.result or "absent"):
                    if any(entry.parameter_ownership.get(n, 'view') != method.parameter_ownership.get(m, 'view') for (n, _), (m, _) in zip(entry.parameters, method.parameters)):
                        continue
                    if method.receiver_ownership == 'view' and entry.receiver_ownership == 'borrow':
                        continue
                    matches.append(entry)
            if len(matches) != 1:
                raise ValueError(f"{implementation.name}.{name} requires exactly one implementation of the trait callable contract")


import unittest


class GenericTraitTests(unittest.TestCase):
    source = ('trait Readable[T]:\n    def read(self: view Self) -> T\n'
              'class Cell[T = int]: Readable[T]:\n    value: T\n'
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
                  'class Item: Default[int]:\n    unused: int\n'
                  'def work() -> int:\n    item = Item(0)\n    return item.value()\n')
        result = self.compile(source)
        self.assertFalse(result.diagnostics, str(result.diagnostics))
