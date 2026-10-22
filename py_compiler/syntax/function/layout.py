"""C layout queries share the representation used by MLIR record lowering."""
from dataclasses import dataclass


@dataclass(frozen=True)
class Layout:
    size: int
    alignment: int
    fields: tuple = ()


def layout_of(type_, context, active=()):
    if type_.name in active:
        raise ValueError('recursive inline layout requires pointer indirection')
    if not type_.mlir:
        return Layout(1, 1)
    if callable(getattr(type_, 'storage_layout', None)):
        return type_.storage_layout(context)
    if type_.family == 'record':
        offset, alignment, fields = 0, 1, []
        for field, name in type_.declaration.fields:
            field_type = context.type(name)
            if field_type.mlir.startswith('memref'):
                word = context.type('usize').bits // 8
                layout = Layout((3 + 2 * getattr(field_type, 'rank', 1)) * word, word)
            else:
                layout = layout_of(field_type, context, (*active, type_.name))
            offset = (offset + layout.alignment - 1) // layout.alignment * layout.alignment
            fields.append((field, offset))
            offset += layout.size
            alignment = max(alignment, layout.alignment)
        if type_.owns_fields:
            offset += len(type_.declaration.fields)
        return Layout((offset + alignment - 1) // alignment * alignment, alignment, tuple(fields))
    if type_.family in ('array', 'tensor', 'tuple'):
        if type_.count is None:
            raise ValueError('dynamic array has no fixed inline element layout')
        element = layout_of(type_.element, context, active)
        return Layout(element.size * type_.count, element.alignment)
    if type_.family == 'union':
        offset, alignment = 4, 4
        for member in type_.variants:
            layout = storage_layout(member, context, active)
            offset = (offset + layout.alignment - 1) // layout.alignment * layout.alignment + layout.size
            alignment = max(alignment, layout.alignment)
        return Layout((offset + alignment - 1) // alignment * alignment, alignment)
    if type_.family == 'iterator':
        word = context.type('usize').bits // 8
        return Layout(3 * word, word)
    if type_.family == 'dynamic':
        word = context.type('usize').bits // 8
        return Layout(2 * word, word)
    if type_.family == 'shared':
        word = context.type('usize').bits // 8
        return Layout(10 * word, word)
    if type_.mlir in ('!llvm.ptr', 'index'):
        word = context.type('usize').bits // 8
        return Layout(word, word)
    if type_.mlir.startswith('i') and type_.mlir[1:].isdigit():
        size = max(1, (int(type_.mlir[1:]) + 7) // 8)
        if size & (size - 1):
            raise ValueError('c layout requires a target contract for this integer width')
        return Layout(size, size)
    if type_.mlir in ('f16', 'bf16', 'f32', 'f64'):
        size = 2 if type_.mlir in ('f16', 'bf16') else int(type_.mlir[1:]) // 8
        return Layout(size, size)
    raise ValueError(f'{type_.name} has no supported c layout contract')


def storage_layout(type_, context, active=()):
    if type_.mlir.startswith('memref'):
        word = context.type('usize').bits // 8
        return Layout((3 + 2 * getattr(type_, 'rank', 1)) * word, word)
    return layout_of(type_, context, active)


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
        if type_ is not None and len(arguments) == 1 and self.name in ('offset', 'alignment'):
            return self.field(type_, arguments[0], cfg, env, span)
        if type_ is None and len(arguments) == 1:
            argument = arguments[0]
            if argument.kind == 'name' and cfg.lookup(argument, env) is None:
                type_ = cfg.context.type(argument.token.text)
            else:
                value = cfg.expr(argument, env)
                type_ = value.type.declaration if value.type.family == 'type' else value.type
                if self.name == 'bytes' and value.type.family == 'record':
                    entries = cfg.context.functions.get(value.type.name + '.bytes', ())
                    if entries:
                        from py_compiler.syntax.function.calls import select, invoke
                        entry, values = select(entries, (), cfg, env)
                        if entry.receiver_ownership != 'view' or cfg.context.type(entry.result) != cfg.context.type('byte'):
                            raise ValueError('bytes method must read its receiver and return byte')
                        return invoke(entry, values, cfg, span, value)
                if self.name == 'bytes' and value.type.family in ('tensor', 'tuple', 'array'):
                    from py_compiler.syntax.function.length import Length
                    from py_compiler.syntax.function.calls import literal_value
                    from py_compiler.syntax.primitive.numeric.conversion import ScalarConversion
                    from py_compiler.syntax.generic.owned import select
                    count = type_.length(value,cfg,span) if hasattr(type_, 'length') else cfg.emit('length',cfg.context.type('int'),(value,),Length(),span)
                    size = literal_value(layout_of(type_.element,cfg.context).size,count.type,cfg,span)
                    count = select(count.type,'F.operator','*').expand(cfg,count,size,span)
                    return cfg.emit('byte-count',cfg.context.type('byte'),(count,),ScalarConversion(count.type,cfg.context.type('byte')),span)
            arguments = ()
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

    def field(self, type_, argument, cfg, env, span):
        import ast
        from py_compiler.syntax.function.calls import literal_value
        if type_.family != 'record':
            raise ValueError(self.name + '[T](field) requires a class')
        offsets = dict(layout_of(type_, cfg.context).fields)
        values = {name: offsets[name] if self.name == 'offset' else
                  storage_layout(cfg.context.type(annotation), cfg.context).alignment
                  for name, annotation in type_.declaration.fields}
        byte = cfg.context.type('byte')
        if argument.kind == 'literal' and argument.token.kind == 'STRING':
            field = ast.literal_eval(argument.token.text)
            if field not in values:
                raise ValueError(f'{type_.name} has no field {field!r}')
            return literal_value(values[field], byte, cfg, span)
        from py_compiler.hir.thir.src.model import Edge, Terminator
        from py_compiler.syntax.grammar.expression import Expression
        from py_compiler.syntax.complex.tensor import ArgumentValue
        from py_compiler.syntax.generic.storage import root_binding, release_storage
        value = cfg.expr(argument, env, cfg.context.type('string'))
        owner = root_binding(argument, cfg, env)
        capture_owner = owner
        if capture_owner is None:
            from py_compiler.mir.ownership.flow import Binding
            capture_owner = Binding('layout-name:' + str(value.identity), ':layout-name', value.type, True, 'view', span)
            cfg.values[capture_owner.identity] = value
        done = cfg.block((byte,))
        for field, amount in values.items():
            import json
            name = Expression('literal', argument.token.with_text(json.dumps(field), kind='STRING'))
            captured = Expression('capture', argument.token, (ArgumentValue(value, capture_owner),))
            matches = value.type.library_call('_equal', (captured, name), cfg, env, span)
            branch, following = cfg.block(), cfg.block()
            cfg.current.terminator = Terminator('conditional', (Edge(branch.identity), Edge(following.identity)), matches)
            cfg.current = branch
            result = literal_value(amount, byte, cfg, span)
            cfg.current.terminator = Terminator('jump', (Edge(done.identity, (result,)),))
            cfg.current = following
        cfg.current.terminator = Terminator('panic', message='unknown field in layout query')
        cfg.current = done
        if owner is None and value.identity in cfg.owned_results:
            release_storage(value, cfg, span)
        return done.parameters[0]


class TargetQuery:
    def call(self, type_, arguments, cfg, env, span):
        import ast
        from py_compiler.syntax.function.calls import literal_value
        if type_ is not None or len(arguments) != 1 or arguments[0].kind != 'literal' or arguments[0].token.kind != 'STRING':
            raise ValueError('target requires one literal fact name')
        fact = ast.literal_eval(arguments[0].token.text)
        triple = cfg.syntax.target
        parts = triple.split('-')
        architecture = parts[0]
        operating_system = next((part for part in parts[1:] if part in
            ('linux', 'windows', 'win32', 'darwin', 'macos', 'ios', 'freebsd', 'netbsd', 'openbsd', 'wasi', 'emscripten')), 'none')
        little = ('x86_64', 'i386', 'i486', 'i586', 'i686', 'aarch64', 'arm', 'thumb', 'riscv32', 'riscv64', 'wasm32', 'wasm64', 'mipsel', 'mips64el', 'powerpc64le')
        big = ('aarch64_be', 'armeb', 'thumbeb', 's390x', 'sparc', 'sparcv9', 'mips', 'mips64', 'powerpc', 'powerpc64')
        if fact == 'byte_order':
            if architecture in big or architecture.startswith(('armeb', 'thumbeb')):
                value = 'big'
            elif architecture in little or architecture.startswith(('armv', 'thumbv')):
                value = 'little'
            else:
                raise ValueError('target byte order is unknown for ' + architecture)
        elif fact in ('architecture', 'operating_system', 'triple'):
            value = {'architecture': architecture, 'operating_system': operating_system, 'triple': triple}[fact]
        elif fact == 'pointer_bits':
            return literal_value(cfg.syntax.pointer_bits, cfg.context.type('int'), cfg, span)
        else:
            raise ValueError('unknown target fact: ' + fact)
        return literal_value(value, cfg.context.type('string'), cfg, span)


import unittest


class LayoutTests(unittest.TestCase):
    def test_target_facts_use_the_compilation_target(self):
        from py_compiler.frontend.src.lib import compile_source
        from py_compiler.syntax.recognition import Syntax
        result = compile_source('target.sev', 'def arch() -> string:\n    return target("architecture")\n'
            'def endian() -> string:\n    return target("byte_order")\n'
            'def bits() -> int:\n    return target("pointer_bits")\n', Syntax(pointer_bits=32, target='armebv7-unknown-linux-gnueabi'))
        self.assertFalse(result.diagnostics, str(result.diagnostics))
        constants = {constant.value for constant in result.program.constants}
        self.assertTrue({'armebv7', 'big', 32} <= constants)

    def test_named_field_layout_is_checked(self):
        from py_compiler.frontend.src.lib import compile_source
        from py_compiler.syntax.recognition import Syntax
        source = 'class Item:\n    tag: u8\n    value: u64\n'
        result = compile_source('field-layout.sev', source + 'test:\n    assert(offset[Item]("value") == 8B)\n    assert(alignment[Item]("value") == 8B)\n', Syntax())
        self.assertFalse(result.diagnostics, str(result.diagnostics))
        bad = compile_source('field-layout.sev', source + 'test:\n    value = offset[Item]("missing")\n', Syntax())
        self.assertIn('has no field', str(bad.diagnostics))
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
