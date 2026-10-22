"""Storage providers used through generic read/write tools."""
from dataclasses import dataclass
from hashlib import sha256


def descriptor_type(bits, rank=1):
    return f'!llvm.struct<(!llvm.ptr, !llvm.ptr, i{bits}, !llvm.array<{rank} x i{bits}>, !llvm.array<{rank} x i{bits}>)>'


def stored_type(type_, context):
    if not type_.mlir:
        return 'i1'
    if type_.mlir == 'index':
        return 'i' + str(context.type('usize').bits)
    if type_.family == 'record':
        return context.provider_by_name[type_.name].layout(type_.declaration, context)
    if type_.mlir.startswith('memref'):
        return descriptor_type(context.type('usize').bits, getattr(type_, 'rank', 1))
    return type_.mlir


def owns_storage(type_):
    return type_.family not in ('pointer', 'trait') and (type_.family == 'string' or callable(getattr(type_, 'release', None)))


def copy_storage(value, cfg, span):
    if value.type.family in ('pointer', 'trait'):
        return value
    if value.type.family == 'string':
        from py_compiler.syntax.complex.array import MemoryOperation
        return cfg.emit('memory-copy', value.type, (value,), MemoryOperation('copy'), span)
    copy = getattr(value.type, 'copy_value', None)
    return copy(value, cfg, span) if copy else value


def release_storage(value, cfg, span):
    if value.type.family in ('pointer', 'trait'):
        return
    if value.type.family == 'string':
        from py_compiler.syntax.complex.array import MemoryOperation
        cfg.effect('deallocate', (value,), MemoryOperation('release'), span)
    else:
        release = getattr(value.type, 'release', None)
        if release:
            cfg.stack_values.discard(value.identity)
            release(value, cfg, span)


def pack_value(lines, value, type_, storage, label):
    if type_.mlir == 'index':
        lines.append(f'{label} = arith.index_cast {value} : index to {storage}')
    elif type_.family == 'record':
        lines.append(f'{label} = llvm.load {value} : !llvm.ptr -> {storage}')
    elif type_.mlir.startswith('memref'):
        lines.append(f'{label} = builtin.unrealized_conversion_cast {value} : {type_.mlir} to {storage}')
    else:
        return value
    return label


def unpack_value(lines, value, type_, storage, label):
    if type_.mlir == 'index':
        lines.append(f'{label} = arith.index_cast {value} : {storage} to index')
    elif type_.family == 'record':
        lines.extend([f'{label}_one = arith.constant 1 : i64',
                      f'{label} = llvm.alloca {label}_one x {storage} : (i64) -> !llvm.ptr',
                      f'llvm.store {value}, {label} : {storage}, !llvm.ptr'])
    elif type_.mlir.startswith('memref'):
        lines.append(f'{label} = builtin.unrealized_conversion_cast {value} : {storage} to {type_.mlir}')
    else:
        lines.append(f'{label} = builtin.unrealized_conversion_cast {value} : {storage} to {type_.mlir}')


def contains_borrowed_fields(type_, context, active=()):
    if type_.name in active:
        return False
    if type_.family in ('pointer', 'trait'):
        return True
    active = (*active, type_.name)
    if type_.family == 'record':
        return any(contains_borrowed_fields(context.type(t), context, active) for _, t in type_.declaration.fields)
    if type_.family == 'shared':
        return contains_borrowed_fields(type_.payload, context, active)
    if type_.family in ('array', 'tensor', 'tuple'):
        return contains_borrowed_fields(type_.element, context, active)
    if type_.family == 'union':
        return any(contains_borrowed_fields(t, context, active) for t in type_.variants)
    return type_.family in ('iterator', 'dynamic')


def slot_key(term):
    if term.kind == 'literal' and term.token.kind == 'NUMBER':
        try:
            return int(term.token.text)
        except ValueError:
            pass
    return '*'


def element_slot(term):
    """Keep nested constant projections distinct; unknown projections may alias."""
    path = []
    while term.kind in ('index', 'member'):
        path.append(slot_key(term.operands[1]) if term.kind == 'index' else ('field', term.token.text))
        term = term.operands[0]
    if '*' in path:
        return '*'
    return path[0] if len(path) == 1 else tuple(reversed(path))


def root_binding(node, cfg, env):
    while node.kind in ('member', 'index'):
        if node.kind == 'member' and node.operands[0].kind == 'name':
            key = node.operands[0].token.text + '.' + node.token.text
            place = cfg.context.imported_globals.get(key)
            if place is not None:
                from py_compiler.mir.ownership.flow import Binding
                return Binding(place.identity, key, place.type, place.constant, 'view' if place.constant else 'borrow', node.token.span)
        node = node.operands[0]
    if node.kind == 'capture':
        return node.operands[0].binding(cfg)
    return cfg.lookup(node, env) if node.kind in ('name', 'reference') else None


@dataclass(frozen=True)
class GlobalPlace:
    identity: str
    type: object
    exported_name: str = ''
    constant: bool = False
    external_symbol: str = ''

    @property
    def symbol(self):
        if self.external_symbol:
            return self.external_symbol
        return "__sev_global_" + sha256(self.identity.encode()).hexdigest()

    def declaration(self):
        if self.external_symbol and hasattr(self.type, 'global_storage_type'):
            return f'llvm.mlir.global external @{self.symbol}() : {self.type.global_storage_type}'
        if self.external_symbol:
            return f'memref.global "public" @{self.symbol} : memref<{self.type.mlir}>'
        storage = getattr(self.type, 'module_storage', None)
        if storage is None:
            raise ValueError(self.type.name + ' has no module storage contract; bind it in a local scope')
        declaration = storage(self.symbol)
        return declaration.replace('memref.global "private"', 'memref.global "public"', 1) if self.exported_name else declaration

    def read(self, cfg, span):
        if self.external_symbol and self not in cfg.body.storage:
            cfg.body.storage.append(self)
        return cfg.emit("read", self.type, (), self, span)

    def write(self, cfg, value, span):
        if self.external_symbol:
            if self.constant:
                raise ValueError('cannot assign to an imported constant')
            if self not in cfg.body.storage:
                cfg.body.storage.append(self)
        cfg.effect("write", (value,), self, span)

    def atom(self, operation):
        from py_compiler.syntax.generic.atom import Atom
        if operation.result is not None:
            return Atom(self, (), self.type, (), ('read',), self)
        return Atom(self, (self.type,), None, ('copy',), ('mutation',), self)

    def render_operation(self, operation):
        from py_compiler.mlir.src.cfg import name, temporary_suffix
        suffix = temporary_suffix(operation)
        address = f"%global_{suffix}"
        if hasattr(self.type, 'global_storage_type'):
            storage = self.type.global_storage_type
            lines = [f'{address} = llvm.mlir.addressof @{self.symbol} : !llvm.ptr']
            descriptor = f'%global_descriptor_{suffix}'
            if operation.result:
                lines += [f'{descriptor} = llvm.load {address} : !llvm.ptr -> {storage}',
                          f'{name(operation.result)} = builtin.unrealized_conversion_cast {descriptor} : {storage} to {self.type.mlir}']
            else:
                lines += [f'{descriptor} = builtin.unrealized_conversion_cast {name(operation.operands[0])} : {self.type.mlir} to {storage}',
                          f'llvm.store {descriptor}, {address} : {storage}, !llvm.ptr']
            return lines
        lines = [f"{address} = memref.get_global @{self.symbol} : memref<{self.type.mlir}>"]
        if operation.result:
            lines.append(f"{name(operation.result)} = memref.load {address}[] : memref<{self.type.mlir}>")
        else:
            lines.append(f"memref.store {name(operation.operands[0])}, {address}[] : memref<{self.type.mlir}>")
        return lines


@dataclass(frozen=True)
class FieldPlace:
    receiver: object
    index: int
    record_type: str
    type: object
    storage_type: str = ''

    @property
    def tracked(self):
        return getattr(self.receiver.type, 'owns_fields', False)

    def clear(self, cfg, span):
        if self.tracked:
            cfg.effect('field-clear', (self.receiver,), FieldState(self, False), span)

    def when_initialized(self, cfg, span, action):
        if not self.tracked:
            return action()
        from py_compiler.hir.thir.src.model import Edge, Terminator
        condition = cfg.emit('field-initialized', cfg.context.type('bool'), (self.receiver,), FieldState(self), span)
        body, done = cfg.block(), cfg.block()
        cfg.current.terminator = Terminator('conditional', (Edge(body.identity), Edge(done.identity)), condition)
        cfg.current = body
        action()
        cfg.current.terminator = Terminator('jump', (Edge(done.identity),))
        cfg.current = done

    def destroy(self, cfg, span):
        def destroy():
            value = self.read(cfg, span)
            self.clear(cfg, span)
            release_storage(value, cfg, span)
        self.when_initialized(cfg, span, destroy)

    def take(self, cfg, span, value=None):
        value = self.read(cfg, span) if value is None else value
        if self.type.family == 'record':
            value = cfg.emit('memory-copy', self.type, (value,), StorageTransfer(stored_type(self.type, cfg.context)), span)
        self.clear(cfg, span)
        cfg.owned_results.add(value.identity)
        return value

    def read(self, cfg, span):
        return cfg.emit("read", self.type, (self.receiver,), self, span)

    def write(self, cfg, value, span):
        if contains_borrowed_fields(self.type, cfg.context):
            dependencies = cfg.value_dependencies.get(value.identity, set())
            cfg.value_dependencies.setdefault(self.receiver.identity, set()).update(dependencies)
            for owner in cfg.value_origins.get(self.receiver.identity, ()):
                # Receiver projections can alias distinct elements. Retain the
                # union until container destruction rather than overwrite another
                # element's dependency with the same record field index.
                slot = ('field', self.index)
                previous = cfg.flow.retained.get(owner, {}).get(slot, frozenset())
                cfg.flow.retain(owner, slot, previous | dependencies)
        cfg.effect("write", (self.receiver, value), self, span)

    def atom(self, operation):
        from py_compiler.syntax.generic.atom import Atom
        if operation.result is not None:
            return Atom(self, (self.receiver.type,), self.type, ('view',), ('read',), self)
        return Atom(self, (self.receiver.type, self.type), None, ('borrow', 'move' if owns_storage(self.type) else 'copy'), ('mutation',), self)

    def render_operation(self, operation):
        from py_compiler.mlir.src.cfg import name, temporary_suffix
        suffix = temporary_suffix(operation)
        address = f"%field_{suffix}_{self.index}"
        lines = [f"{address} = llvm.getelementptr {name(self.receiver)}[0, {self.index}] : (!llvm.ptr) -> !llvm.ptr, {self.record_type}"]
        if operation.result:
            guard = []
            if self.tracked:
                guard = FieldState(self).render_load(operation, f'%field_ready_{suffix}')
                guard.append(f'cf.assert %field_ready_{suffix}, "read of moved or uninitialized record field"')
            if self.type.family == 'record':
                lines[0] = lines[0].replace(address + ' =', name(operation.result) + ' =', 1)
            elif self.type.mlir.startswith('memref'):
                lines.append(f"%descriptor_{suffix} = llvm.load {address} : !llvm.ptr -> {self.storage_type}")
                lines.append(f"{name(operation.result)} = builtin.unrealized_conversion_cast %descriptor_{suffix} : {self.storage_type} to {self.type.mlir}")
            else:
                lines.append(f"{name(operation.result)} = llvm.load {address} : !llvm.ptr -> {self.type.mlir}")
            lines = guard + lines
        else:
            value, storage = name(operation.operands[-1]), self.storage_type or self.type.mlir
            if self.type.family == 'record':
                lines.append(f'%aggregate_{suffix}_{self.index} = llvm.load {value} : !llvm.ptr -> {storage}')
                value = f'%aggregate_{suffix}_{self.index}'
            elif self.type.mlir.startswith('memref'):
                lines.append(f'%descriptor_{suffix}_{self.index} = builtin.unrealized_conversion_cast {value} : {self.type.mlir} to {storage}')
                value = f'%descriptor_{suffix}_{self.index}'
            lines.append(f"llvm.store {value}, {address} : {storage}, !llvm.ptr")
            if self.tracked:
                lines.extend(FieldState(self, True).render_operation(operation))
        return lines


@dataclass(frozen=True)
class FieldState:
    place: FieldPlace
    initialized: object = None

    def atom(self, operation):
        from py_compiler.syntax.generic.atom import Atom
        return Atom(self, tuple(v.type for v in operation.operands), operation.result.type if operation.result else None,
                    tuple('view' for _ in operation.operands), ('read', 'mutation'), self)

    def address(self, operation):
        from py_compiler.mlir.src.cfg import name, temporary_suffix
        index = len(self.place.receiver.type.declaration.fields) + self.place.index
        address = f'%field_flag_{temporary_suffix(operation)}_{index}'
        return address, f'{address} = llvm.getelementptr {name(operation.operands[0])}[0, {index}] : (!llvm.ptr) -> !llvm.ptr, {self.place.record_type}'

    def render_load(self, operation, result):
        address, line = self.address(operation)
        return [line, f'{result} = llvm.load {address} : !llvm.ptr -> i1']

    def render_operation(self, operation):
        from py_compiler.mlir.src.cfg import name, temporary_suffix
        if self.initialized is None:
            return self.render_load(operation, name(operation.result))
        address, line = self.address(operation)
        flag = f'%field_flag_value_{temporary_suffix(operation)}'
        return [line, f'{flag} = arith.constant {int(self.initialized)} : i1',
                f'llvm.store {flag}, {address} : i1, !llvm.ptr']


@dataclass(frozen=True)
class StorageTransfer:
    storage: str

    def atom(self, operation):
        from py_compiler.syntax.generic.atom import Atom
        return Atom(self, (operation.operands[0].type,), operation.result.type, ('move',), (), self)

    def entry_allocations(self, operation):
        return self.render_operation(operation)[1:3] if operation.result.type.family == 'record' else []

    def render_operation(self, operation):
        from py_compiler.mlir.src.cfg import name, temporary_suffix
        tag = temporary_suffix(operation)
        lines = []
        packed = pack_value(lines, name(operation.operands[0]), operation.result.type, self.storage, f'%transfer_{tag}')
        unpack_value(lines, packed, operation.result.type, self.storage, name(operation.result))
        return lines


class BindingPlace:
    """Grammar receiver bound to an existing SSA or addressable place."""
    def __init__(self, binding, rebind=True):
        self.binding, self.type, self.rebind = binding, binding.type, rebind

    def capture(self, cfg, ownership, *, preserve_shape=False):
        from contextlib import contextmanager
        @contextmanager
        def lease():
            cfg.flow.read(self.binding)
            if ownership != 'borrow':
                raise ValueError('this storage operation requires borrow self')
            if self.binding.ownership == 'view':
                raise ValueError('cannot acquire writable self from a view')
            if self.rebind and self.binding.constant and self.binding.ownership != "borrow":
                raise ValueError('cannot update a constant binding')
            identity = cfg.flow.resolve(self.binding.identity)
            cfg.record_lifetime('element-check' if preserve_shape else 'check', identity, self.binding.name, (self.binding.identity,))
            conflicts = [loan for loan, owner in cfg.flow.loans.items() if identity in (owner if isinstance(owner, frozenset) else (owner,)) and loan != self.binding.identity
                         and not (preserve_shape and loan in cfg.flow.shape_loans)]
            if conflicts or cfg.flow.borrowers(identity):
                raise ValueError('cannot acquire writable self while a borrow is live')
            key = self.binding.identity + ':grammar-borrow'
            cfg.flow.begin_loan(key, identity)
            try:
                yield
            finally:
                cfg.flow.end_loan(key)
        return lease()

    def read(self, cfg, span):
        return cfg.read_binding(self.binding, span)

    def write(self, cfg, value, span):
        if value.type != self.type:
            raise ValueError('grammar expansion result does not satisfy destination storage')
        identity = cfg.flow.resolve(self.binding.identity)
        cfg.values[identity] = value
        if identity in cfg.storage:
            cfg.storage[identity].write(cfg, value, span)


class AddressPlace:
    def __init__(self, place, binding=None):
        self.place, self.binding, self.type = place, binding, place.type

    def capture(self, cfg, ownership):
        from contextlib import nullcontext
        if self.binding:
            from py_compiler.syntax.complex.array import ElementPlace, RecordElementPlace
            stable = isinstance(self.place, (ElementPlace, RecordElementPlace))
            if hasattr(self.place, 'preserves_shape'):
                stable = self.place.preserves_shape(cfg)
            return BindingPlace(self.binding, rebind=False).capture(cfg, ownership, preserve_shape=stable)
        if ownership != 'borrow':
            raise ValueError('field store requires borrow self')
        return nullcontext()

    def read(self, cfg, span):
        return self.place.read(cfg, span)

    def write(self, cfg, value, span):
        return self.place.write(cfg, value, span)


import unittest


class FieldStateTests(unittest.TestCase):
    def test_initialized_and_cleared_flags_pass_c_verification(self):
        from py_compiler.frontend.src.lib import compile_source
        from py_compiler.syntax.recognition import Syntax
        from py_compiler.mlir.src.lib import lower, render, verify_c
        source = '''class Buffer:
    data: array[int]
def work():
    buffer = Buffer(array[int,1]())
    taken = move buffer.data
    drop buffer
    drop taken
'''
        result = compile_source('field-state.sev', source, Syntax())
        self.assertFalse(result.diagnostics, str(result.diagnostics))
        text = render(lower(result.program))
        verify_c(text)
