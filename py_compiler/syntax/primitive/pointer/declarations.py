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
    """A non-owning address; T supplies the dereferenced value representation."""
    element: object
    bits: int
    family: str = 'pointer'
    mlir: str = '!llvm.ptr'
    signed: bool = False
    binding_ownership: str = 'view'
    parameter_ownership: str = 'view'

    @property
    def name(self):
        return f'pointer[{self.element.name}]'

    def require_context(self, cfg):
        # Carrying an address does not dereference it.
        pass

    def grammars(self):
        from py_compiler.syntax.primitive.pointer.operations import grammars
        return grammars(self)

    def from_value(self, value, cfg, span):
        if value.type == self:
            return value
        from py_compiler.syntax.complex.array import ArrayType
        if not isinstance(value.type, ArrayType) or value.type.element != self.element:
            raise ValueError('pointer[T] requires an array with the same element type')
        from py_compiler.syntax.primitive.pointer.operations import ArrayAddress
        return cfg.emit('array-address', self, (value,), ArrayAddress(self), span)

    def element_place(self, receiver, index, cfg, env, span):
        if not cfg.unsafe_depth:
            raise ValueError('pointer indexing requires an unsafe block')
        from py_compiler.syntax.generic.owned import select
        from py_compiler.syntax.primitive.pointer.operations import PointerPlace
        storage = self.element.mlir
        if not (storage.startswith('i') and storage[1:].isdigit() or
                storage in ('f16', 'bf16', 'f32', 'f64', 'f80', 'f128', '!llvm.ptr') or
                storage.startswith(('!llvm.struct<', '!llvm.array<'))):
            raise ValueError(f'{self.element.name} has no LLVM-compatible pointer element representation')
        index = select(cfg.context.type('isize'), 'F.constructor', 'construct').expand(cfg, (index,), env)
        return PointerPlace(receiver, index, self.element)
