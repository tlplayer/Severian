"""Bounded memory length comes from its descriptor, including pointer views."""
from dataclasses import dataclass
from py_compiler.syntax.generic.atom import Atom


@dataclass(frozen=True)
class Length:
    stride: int = 1

    def predicate_argument(self, argument, parameters, context):
        if not isinstance(parameters, dict):
            return False
        from py_compiler.syntax.function.contracts import predicate_value_type
        type_ = predicate_value_type(argument, parameters, context)
        if type_ is None:
            return False
        entries = context.functions.get(type_.name + '.len', ())
        if entries:
            from py_compiler.syntax.grammar.expression import expression
            from py_compiler.syntax.function.contracts import pure_predicate
            for entry in entries:
                children = entry.node.children
                if len(children) != 1 or children[0].tokens[0].text != 'return':
                    return False
                try:
                    pure_predicate(expression(children[0].tokens[1:]), {'self'} | {name for name, _ in type_.declaration.fields})
                except ValueError:
                    return False
            return True
        return type_.family in ('array', 'pointer', 'tensor', 'tuple', 'string')

    def call(self, type_, arguments, cfg, env, span):
        if type_ is not None or len(arguments) != 1:
            raise ValueError('len requires one bounded array or pointer')
        value = cfg.expr(arguments[0], env)
        entries = cfg.context.functions.get(value.type.name + '.len', ())
        if entries:
            from py_compiler.syntax.function.calls import select, invoke
            entry, operands = select(entries, (), cfg, env)
            result = invoke(entry, operands, cfg, span, value)
        elif callable(getattr(value.type, 'length', None)):
            result = value.type.length(value, cfg, span)
        else:
            if value.type.family not in ('array', 'pointer') or not value.type.mlir.startswith('memref<'):
                raise ValueError(f'{value.type.name} does not provide len; a sized container is required')
            from py_compiler.syntax.complex.array import record_storage
            stride = record_storage(value.type, cfg.context)[0] if getattr(value.type, 'aggregate', False) else 1
            result = cfg.emit('length', cfg.context.type('int'), (value,), Length(stride), span)
        from py_compiler.syntax.generic.storage import release_temporary
        release_temporary(value, cfg, span)
        return result

    def atom(self, operation):
        return Atom(self, (operation.operands[0].type,), operation.result.type, ('view',), ('read',), self)

    def render_operation(self, operation):
        from py_compiler.mlir.src.cfg import name
        source, result = operation.operands[0], operation.result
        lines = [f'%dimension_{result.identity} = arith.constant 0 : index',
                 f'%length_{result.identity} = memref.dim {name(source)}, %dimension_{result.identity} : {source.type.mlir}']
        length = f'%length_{result.identity}'
        if self.stride != 1:
            lines += [f'%stride_{result.identity} = arith.constant {self.stride} : index',
                      f'%elements_{result.identity} = arith.divui {length}, %stride_{result.identity} : index']
            length = f'%elements_{result.identity}'
        if result.type.mlir == 'index':
            lines.append(f'{name(result)} = arith.addi {length}, %dimension_{result.identity} : index')
        else:
            lines.append(f'{name(result)} = arith.index_cast {length} : index to {result.type.mlir}')
        return lines


import unittest


class TemporaryLengthTests(unittest.TestCase):
    def test_iterator_length_keeps_frozen_owned_capture_alive(self):
        from py_compiler.frontend.src.lib import compile_source
        from py_compiler.syntax.recognition import Syntax
        result = compile_source('captured-length.sev', '''class Word:
    text: string
    def len() -> int:
        return len(text)
    grammar iterate["iterator", "(", self, ")"]() -> iterator[int]:
        index = 0
        while index < len(self):
            yield index
            index += 1
def word() -> Word:
    return Word("hello")
def count() -> int:
    total = 0
    for item in [word()]:
        for index in item:
            total += index
    return total
''', Syntax(include_tests=False))
        self.assertFalse(result.diagnostics, str(result.diagnostics))
        for body in result.program.bodies:
            body.verify()

    def test_length_releases_an_unbound_string_result_once(self):
        from py_compiler.frontend.src.lib import compile_source
        from py_compiler.syntax.recognition import Syntax
        result = compile_source('temporary-length.sev', '''def text() -> string:
    return "hello"
def count() -> int:
    return len(text())
''', Syntax(include_tests=False))
        self.assertFalse(result.diagnostics, str(result.diagnostics))
        body = next(body for body in result.program.bodies if body.declaration == 'count')
        operations = [operation for block in body.blocks for operation in block.operations]
        self.assertEqual(sum(op.kind == 'call' for op in operations), 1)
        self.assertEqual(sum(op.kind == 'deallocate' for op in operations), 1)
        self.assertLess(next(i for i, op in enumerate(operations) if op.kind == 'length'),
                        next(i for i, op in enumerate(operations) if op.kind == 'deallocate'))

    def test_length_preserves_a_live_binding(self):
        from py_compiler.frontend.src.lib import compile_source
        from py_compiler.syntax.recognition import Syntax
        result = compile_source('bound-length.sev', '''def count() -> int:
    text = "hello"
    first = len(text)
    return first + len(text)
''', Syntax(include_tests=False))
        self.assertFalse(result.diagnostics, str(result.diagnostics))
        body = next(body for body in result.program.bodies if body.declaration == 'count')
        operations = [operation for block in body.blocks for operation in block.operations]
        self.assertEqual(sum(op.kind == 'deallocate' for op in operations), 1)
        self.assertGreater(next(i for i, op in enumerate(operations) if op.kind == 'deallocate'),
                           max(i for i, op in enumerate(operations) if op.kind == 'length'))
