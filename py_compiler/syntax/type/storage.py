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
        if self.type.family not in ("integer", "float", "bool", "char", "byte"):
            raise ValueError("global storage provider currently requires a scalar element")
        zero = "0.0" if self.type.family == "float" else "0"
        return f'memref.global "private" @{self.symbol} : memref<{self.type.mlir}> = dense<{zero}>'

    def read(self, cfg, span):
        return cfg.emit("read", self.type, (), self, span)

    def write(self, cfg, value, span):
        cfg.effect("write", (value,), self, span)

    def render_operation(self, operation):
        from py_compiler.mlir.src.cfg import name
        suffix = str(operation.result.identity) if operation.result else f"store_{operation.span.start}"
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

    def render_operation(self, operation):
        from py_compiler.mlir.src.cfg import name
        suffix = str(operation.result.identity) if operation.result else f"store_{operation.span.start}"
        address = f"%field_{suffix}"
        lines = [f"{address} = llvm.getelementptr {name(self.receiver)}[0, {self.index}] : (!llvm.ptr) -> !llvm.ptr, {self.record_type}"]
        if operation.result:
            lines.append(f"{name(operation.result)} = llvm.load {address} : !llvm.ptr -> {self.type.mlir}")
        else:
            lines.append(f"llvm.store {name(operation.operands[-1])}, {address} : {self.type.mlir}, !llvm.ptr")
        return lines
