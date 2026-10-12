"""Native layout queries share the representation used by MLIR record lowering."""
from dataclasses import dataclass


@dataclass(frozen=True)
class Layout:
    size: int
    alignment: int
    fields: tuple = ()


def layout_of(type_, context, active=()):
    if type_.name in active:
        raise ValueError('recursive inline layout requires pointer indirection')
    if type_.family == 'record':
        offset, alignment, fields = 0, 1, []
        for field, name in type_.declaration.fields:
            field_type = context.type(name)
            if field_type.mlir.startswith('memref'):
                word = context.type('usize').bits // 8
                layout = Layout(5 * word, word)
            else:
                layout = layout_of(field_type, context, (*active, type_.name))
            offset = (offset + layout.alignment - 1) // layout.alignment * layout.alignment
            fields.append((field, offset))
            offset += layout.size
            alignment = max(alignment, layout.alignment)
        return Layout((offset + alignment - 1) // alignment * alignment, alignment, tuple(fields))
    if type_.family == 'array':
        if type_.count is None:
            raise ValueError('dynamic array has no fixed inline element layout')
        element = layout_of(type_.element, context, active)
        return Layout(element.size * type_.count, element.alignment)
    if type_.mlir.startswith('i') and type_.mlir[1:].isdigit():
        size = max(1, (int(type_.mlir[1:]) + 7) // 8)
        if size & (size - 1):
            raise ValueError('native layout requires a target contract for this integer width')
        return Layout(size, size)
    if type_.mlir in ('f16', 'bf16', 'f32', 'f64'):
        size = 2 if type_.mlir in ('f16', 'bf16') else int(type_.mlir[1:]) // 8
        return Layout(size, size)
    raise ValueError(f'{type_.name} has no supported native layout contract')


@dataclass(frozen=True)
class Offsets:
    target: object
    fields: tuple
    byte_type: object
    family: str = 'offsets'
    binding_ownership: str = 'copy'
    parameter_ownership: str = 'copy'

    @property
    def name(self):
        return f'offset[{self.target.name}]'

    @property
    def mlir(self):
        return '!llvm.struct<(' + ', '.join(self.byte_type.mlir for _ in self.fields) + ')>'

    def field_place(self, receiver, field, cfg, span, write=False):
        if write:
            raise ValueError('layout offsets are immutable')
        for index, (name, _) in enumerate(self.fields):
            if field == name:
                return OffsetField(receiver, index, self.byte_type)
        raise ValueError(f'{self.target.name} has no field {field!r}')

    def atom(self, operation):
        from py_compiler.syntax.generic.atom import Atom
        return Atom(self, (), self, (), (), self)

    def render_operation(self, operation):
        from py_compiler.mlir.src.cfg import name
        tag = str(operation.result.identity)
        lines = [f'%offsets_{tag}_0 = llvm.mlir.undef : {self.mlir}']
        for index, (_, offset) in enumerate(self.fields):
            lines.append(f'%offset_{tag}_{index} = arith.constant {offset} : {self.byte_type.mlir}')
            result = name(operation.result) if index == len(self.fields) - 1 else f'%offsets_{tag}_{index + 1}'
            lines.append(f'{result} = llvm.insertvalue %offset_{tag}_{index}, %offsets_{tag}_{index}[{index}] : {self.mlir}')
        if not self.fields:
            lines = [f'{name(operation.result)} = llvm.mlir.undef : {self.mlir}']
        return lines


@dataclass(frozen=True)
class OffsetField:
    receiver: object
    index: int
    type: object

    def read(self, cfg, span):
        from py_compiler.syntax.complex.object import Extract
        return cfg.emit('field-offset', self.type, (self.receiver,), Extract(self.index), span)


@dataclass(frozen=True)
class LayoutQuery:
    name: str

    def call(self, type_, arguments, cfg, env, span):
        if type_ is None or arguments:
            raise ValueError(f'{self.name}[T]() requires a concrete type and no values')
        layout = layout_of(type_, cfg.context)
        if self.name == 'offset':
            if type_.family != 'record':
                raise ValueError('offset[T]() requires a class')
            offsets = Offsets(type_, layout.fields, cfg.context.type('byte'))
            return cfg.emit('layout-offsets', offsets, (), offsets, span)
        from py_compiler.syntax.function.calls import literal_value
        return literal_value(layout.size if self.name == 'bytes' else layout.alignment,
                             cfg.context.type('byte'), cfg, span)


import unittest


class LayoutTests(unittest.TestCase):
    def test_typed_offsets_and_nested_layout(self):
        from py_compiler.frontend.src.lib import compile_source
        from py_compiler.syntax.recognition import Syntax
        source = ('class Header:\n    kind: u8\n    length: u32\n'
                  'class Pair:\n    left: Header\n    right: Header\n'
                  'test:\n    offsets := offset[Header]()\n'
                  '    assert(offsets.kind == 0B)\n    assert(offsets.length == 4B)\n'
                  '    assert(bytes[Header]() == 8B)\n    assert(alignment[Header]() == 4B)\n'
                  '    assert(bytes[Pair]() == 16B)\n')
        result = compile_source('layout.sev', source, Syntax())
        self.assertFalse(result.diagnostics, str(result.diagnostics))
        bad = compile_source('layout.sev', source + 'test:\n    value = offset[Header]().missing\n', Syntax())
        self.assertTrue(bad.diagnostics)
        self.assertIn('has no field', str(bad.diagnostics))
