"""Storage providers used through generic read/write tools."""
from dataclasses import dataclass
from hashlib import sha256


def descriptor_type(bits):
    return f'!llvm.struct<(!llvm.ptr, !llvm.ptr, i{bits}, !llvm.array<1 x i{bits}>, !llvm.array<1 x i{bits}>)>'


def stored_type(type_, context):
    if type_.family == 'record':
        return context.provider_by_name[type_.name].layout(type_.declaration, context)
    if type_.mlir.startswith('memref'):
        return descriptor_type(context.type('usize').bits)
    return type_.mlir


def owns_storage(type_):
    return type_.family == 'string' or callable(getattr(type_, 'release', None))


def copy_storage(value, cfg, span):
    if value.type.family == 'string':
        from py_compiler.syntax.complex.array import MemoryOperation
        return cfg.emit('memory-copy', value.type, (value,), MemoryOperation('copy'), span)
    copy = getattr(value.type, 'copy_value', None)
    return copy(value, cfg, span) if copy else value


def release_storage(value, cfg, span):
    if value.type.family == 'string':
        from py_compiler.syntax.complex.array import MemoryOperation
        cfg.effect('deallocate', (value,), MemoryOperation('release'), span)
    else:
        release = getattr(value.type, 'release', None)
        if release:
            cfg.stack_values.discard(value.identity)
            release(value, cfg, span)


def pack_value(lines, value, type_, storage, label):
    if type_.family == 'record':
        lines.append(f'{label} = llvm.load {value} : !llvm.ptr -> {storage}')
    elif type_.mlir.startswith('memref'):
        lines.append(f'{label} = builtin.unrealized_conversion_cast {value} : {type_.mlir} to {storage}')
    else:
        return value
    return label


def unpack_value(lines, value, type_, storage, label):
    if type_.family == 'record':
        lines.extend([f'{label}_one = arith.constant 1 : i64',
                      f'{label} = llvm.alloca {label}_one x {storage} : (i64) -> !llvm.ptr',
                      f'llvm.store {value}, {label} : {storage}, !llvm.ptr'])
    elif type_.mlir.startswith('memref'):
        lines.append(f'{label} = builtin.unrealized_conversion_cast {value} : {storage} to {type_.mlir}')
    else:
        lines.append(f'{label} = builtin.unrealized_conversion_cast {value} : {storage} to {type_.mlir}')


def root_binding(node, cfg, env):
    while node.kind in ('member', 'index'):
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
        if self.external_symbol:
            return f'memref.global "private" @{self.symbol} : memref<{self.type.mlir}>'
        declaration = self.type.module_storage(self.symbol)
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

    def read(self, cfg, span):
        return cfg.emit("read", self.type, (self.receiver,), self, span)

    def write(self, cfg, value, span):
        cfg.effect("write", (self.receiver, value), self, span)

    def atom(self, operation):
        from py_compiler.syntax.generic.atom import Atom
        if operation.result is not None:
            return Atom(self, (self.receiver.type,), self.type, ('view',), ('read',), self)
        return Atom(self, (self.receiver.type, self.type), None, ('borrow', 'move' if callable(getattr(self.type, 'release', None)) else 'copy'), ('mutation',), self)

    def render_operation(self, operation):
        from py_compiler.mlir.src.cfg import name, temporary_suffix
        suffix = temporary_suffix(operation)
        address = f"%field_{suffix}_{self.index}"
        lines = [f"{address} = llvm.getelementptr {name(self.receiver)}[0, {self.index}] : (!llvm.ptr) -> !llvm.ptr, {self.record_type}"]
        if operation.result:
            if self.type.family == 'record':
                lines[0] = lines[0].replace(address + ' =', name(operation.result) + ' =', 1)
            elif self.type.mlir.startswith('memref'):
                lines.append(f"%descriptor_{suffix} = llvm.load {address} : !llvm.ptr -> {self.storage_type}")
                lines.append(f"{name(operation.result)} = builtin.unrealized_conversion_cast %descriptor_{suffix} : {self.storage_type} to {self.type.mlir}")
            else:
                lines.append(f"{name(operation.result)} = llvm.load {address} : !llvm.ptr -> {self.type.mlir}")
        else:
            value, storage = name(operation.operands[-1]), self.storage_type or self.type.mlir
            if self.type.family == 'record':
                lines.append(f'%aggregate_{suffix}_{self.index} = llvm.load {value} : !llvm.ptr -> {storage}')
                value = f'%aggregate_{suffix}_{self.index}'
            elif self.type.mlir.startswith('memref'):
                lines.append(f'%descriptor_{suffix}_{self.index} = builtin.unrealized_conversion_cast {value} : {self.type.mlir} to {storage}')
                value = f'%descriptor_{suffix}_{self.index}'
            lines.append(f"llvm.store {value}, {address} : {storage}, !llvm.ptr")
        return lines


class BindingPlace:
    """Grammar receiver bound to an existing SSA or addressable place."""
    def __init__(self, binding, rebind=True):
        self.binding, self.type, self.rebind = binding, binding.type, rebind

    def capture(self, cfg, ownership):
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
            conflicts = [loan for loan, owner in cfg.flow.loans.items() if owner == identity and loan != self.binding.identity]
            if conflicts:
                raise ValueError('cannot acquire writable self while a borrow is live')
            key = self.binding.identity + ':grammar-borrow'
            cfg.flow.loans[key] = identity
            try:
                yield
            finally:
                cfg.flow.loans.pop(key, None)
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
            return BindingPlace(self.binding, rebind=False).capture(cfg, ownership)
        if ownership != 'borrow':
            raise ValueError('field store requires borrow self')
        return nullcontext()

    def read(self, cfg, span):
        return self.place.read(cfg, span)

    def write(self, cfg, value, span):
        return self.place.write(cfg, value, span)
