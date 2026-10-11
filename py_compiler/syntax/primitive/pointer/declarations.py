from py_compiler.syntax.primitive.contract import Primitive


class Pointer(Primitive):
    def specialize(self, element):
        return TypedPointer(element, self.bits)

    @property
    def parameter_ownership(self):
        return 'view'

    conversion_sources = frozenset(('integer', 'pointer'))

    def grammars(self):
        from py_compiler.syntax.primitive.pointer.operations import grammars
        return (*super().grammars(), *grammars(self))

    def require_context(self, context):
        if not getattr(context, 'unsafe_depth', 0):
            raise ValueError('pointer values require an unsafe block')

    def render_constant(self, value, symbol):
        if value == 0:
            return [], [f'%value = llvm.mlir.zero : {self.mlir}']
        return [], [f'%address = arith.constant {value} : i{self.bits}',
                    f'%value = llvm.inttoptr %address : i{self.bits} to {self.mlir}']

    def accept_literal(self, owner, value):
        if owner.family != 'integer':
            return super().accept_literal(owner, value)
        if not 0 <= value < (1 << self.bits):
            raise ValueError('pointer address outside target width')
        return value


def declaration(pointer_bits):
    return Pointer('pointer', 'pointer', '!llvm.ptr', pointer_bits)


from dataclasses import dataclass


@dataclass(frozen=True)
class TypedPointer:
    """A bounded typed handle; binding effects decide owner, view, or borrow."""
    element: object
    bits: int
    family: str = 'pointer'
    signed: bool = False
    binding_ownership: str = 'view'
    parameter_ownership: str = 'view'
    location: str = 'heap'

    def __post_init__(self):
        from py_compiler.syntax.complex.array import ArrayType
        ArrayType(self.element)

    @property
    def name(self):
        return f'pointer[{self.element.name}]'

    @property
    def mlir(self):
        return f'memref<?x{self.element.mlir}>'

    def require_context(self, cfg):
        pass

    def grammars(self):
        return ()

    def from_value(self, value, cfg, span):
        if value.type == self:
            return value
        from py_compiler.syntax.complex.array import ArrayType
        if not isinstance(value.type, (ArrayType, TypedPointer)) or value.type.element != self.element:
            raise ValueError('pointer[T] requires storage with the same element type')
        from py_compiler.syntax.primitive.pointer.operations import BoundedAddress
        result = cfg.emit('bounded-address', self, (value,), BoundedAddress(), span)
        if value.identity in cfg.stack_values:
            cfg.stack_values.add(result.identity)
        return result

    def element_place(self, receiver, index, cfg, env, span):
        from py_compiler.syntax.complex.array import ArrayType
        return ArrayType.element_place(self, receiver, index, cfg, env, span)

    def release(self, value, cfg, span):
        from py_compiler.syntax.complex.array import ArrayType
        return ArrayType.release(self, value, cfg, span)

    def copy_value(self, value, cfg, span):
        from py_compiler.syntax.complex.array import ArrayType
        return ArrayType.copy_value(self, value, cfg, span)
