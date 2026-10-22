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
        from py_compiler.syntax.block.unsafe import require_unsafe
        require_unsafe(context, 'pointer values')

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
    """An unsafe typed handle; binding effects decide owner, view, or borrow."""
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
        return 'memref<?xi8>' if self.element.family == 'record' else f'memref<?x{self.element.mlir}>'

    @property
    def aggregate(self):
        from py_compiler.syntax.complex.array import ArrayType
        return ArrayType(self.element).aggregate

    def require_context(self, cfg):
        from py_compiler.syntax.block.unsafe import require_unsafe
        require_unsafe(cfg, 'pointer[T] values')

    def grammars(self):
        return ()

    def from_value(self, value, cfg, span):
        self.require_context(cfg)
        if value.type == self:
            return value
        from py_compiler.syntax.complex.array import ArrayType
        from py_compiler.syntax.complex.tensor import TensorType
        if (not isinstance(value.type, (ArrayType, TypedPointer, TensorType)) or value.type.element != self.element
                or getattr(value.type, 'rank', 1) != 1 or getattr(value.type, 'strided', False)):
            raise ValueError('pointer[T] requires storage with the same element type')
        from py_compiler.syntax.primitive.pointer.operations import BoundedAddress
        result = cfg.emit('bounded-address', self, (value,), BoundedAddress(), span)
        if value.identity in cfg.stack_values:
            cfg.stack_values.add(result.identity)
        return result

    def reinterpret(self, value, cfg, span):
        self.require_context(cfg)
        if not isinstance(value.type, TypedPointer) or value.type.element == self.element:
            return self.from_value(value, cfg, span)
        if self.aggregate or value.type.aggregate:
            raise ValueError('pointer reinterpretation requires scalar element storage')
        from py_compiler.syntax.function.layout import layout_of
        from py_compiler.syntax.primitive.pointer.operations import PointerReinterpret
        source = layout_of(value.type.element, cfg.context)
        target = layout_of(self.element, cfg.context)
        return cfg.emit('pointer-reinterpret', self, (value,),
                        PointerReinterpret(source.size, target.size, target.alignment, self.bits), span)

    def element_place(self, receiver, index, cfg, env, span):
        from py_compiler.syntax.complex.array import ArrayType, ElementPlace
        from py_compiler.syntax.grammar.expression import Expression
        from py_compiler.syntax.complex.tensor import ArgumentValue
        offset = cfg.expr(index, env)
        if offset.type.name == 'byte':
            self.require_context(cfg)
            if self.aggregate:
                raise ValueError('byte offsets currently require scalar pointer elements')
            from py_compiler.syntax.function.layout import layout_of
            position = cfg.emit('pointer-byte-offset', cfg.context.type('index'), (offset,),
                                ByteOffset(layout_of(self.element, cfg.context).size), span)
            return ElementPlace(receiver, position, self.element)
        captured = Expression('capture', index.token, (ArgumentValue(offset),))
        return ArrayType.element_place(self, receiver, captured, cfg, env, span)

    def release(self, value, cfg, span):
        from py_compiler.syntax.complex.array import ArrayType
        return ArrayType.release(self, value, cfg, span)

    def copy_value(self, value, cfg, span):
        from py_compiler.syntax.complex.array import ArrayType
        return ArrayType.copy_value(self, value, cfg, span)


@dataclass(frozen=True)
class ByteOffset:
    stride: int

    def atom(self, operation):
        from py_compiler.syntax.generic.atom import Atom
        return Atom(self, (operation.operands[0].type,), operation.result.type, ('view',), ('panic',), self)

    def render_operation(self, operation):
        from py_compiler.mlir.src.cfg import name
        source, result = operation.operands[0], name(operation.result)
        prefix = result + '_byte'
        return [f'{prefix} = arith.index_cast {name(source)} : {source.type.mlir} to index',
                f'{prefix}_stride = arith.constant {self.stride} : index',
                f'{prefix}_zero = arith.constant 0 : index',
                f'{prefix}_remainder = arith.remui {prefix}, {prefix}_stride : index',
                f'{prefix}_aligned = arith.cmpi eq, {prefix}_remainder, {prefix}_zero : index',
                f'cf.assert {prefix}_aligned, "pointer byte offset is not element-aligned"',
                f'{result} = arith.divui {prefix}, {prefix}_stride : index']
