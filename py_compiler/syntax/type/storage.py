"""Storage providers used through generic read/write tools."""
from dataclasses import dataclass
from hashlib import sha256


@dataclass(frozen=True)
class GlobalPlace:
    identity: str
    type: object

    @property
    def symbol(self):
        return "__sev_global_" + sha256(self.identity.encode()).hexdigest()

    def declaration(self):
        return self.type.module_storage(self.symbol)

    def read(self, cfg, span):
        return cfg.emit("read", self.type, (), self, span)

    def write(self, cfg, value, span):
        cfg.effect("write", (value,), self, span)

    def atom(self, operation):
        from py_compiler.syntax.generic.atom import Atom
        if operation.result is not None:
            return Atom(self, (), self.type, (), ('read',), self)
        return Atom(self, (self.type,), None, ('copy',), ('mutation',), self)

    def render_operation(self, operation):
        from py_compiler.mlir.src.cfg import name
        suffix = str(operation.result.identity) if operation.result else f"store_{operation.span.start}_{operation.operands[-1].identity}"
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

    def read(self, cfg, span):
        return cfg.emit("read", self.type, (self.receiver,), self, span)

    def write(self, cfg, value, span):
        cfg.effect("write", (self.receiver, value), self, span)

    def atom(self, operation):
        from py_compiler.syntax.generic.atom import Atom
        if operation.result is not None:
            return Atom(self, (self.receiver.type,), self.type, ('view',), ('read',), self)
        return Atom(self, (self.receiver.type, self.type), None, ('borrow', 'copy'), ('mutation',), self)

    def render_operation(self, operation):
        from py_compiler.mlir.src.cfg import name
        suffix = str(operation.result.identity) if operation.result else f"store_{operation.span.start}_{operation.operands[-1].identity}"
        address = f"%field_{suffix}_{self.index}"
        lines = [f"{address} = llvm.getelementptr {name(self.receiver)}[0, {self.index}] : (!llvm.ptr) -> !llvm.ptr, {self.record_type}"]
        if operation.result:
            lines.append(f"{name(operation.result)} = llvm.load {address} : !llvm.ptr -> {self.type.mlir}")
        else:
            lines.append(f"llvm.store {name(operation.operands[-1])}, {address} : {self.type.mlir}, !llvm.ptr")
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
                raise ValueError('cannot acquire writable self while another borrow is live')
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
