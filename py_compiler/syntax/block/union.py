"""Union owns its semantic variants, tagged representation and boxing atom."""
from dataclasses import dataclass
from py_compiler.syntax.generic.owned import WordGrammar
from py_compiler.syntax.generic.definition import TypeDefinition


@dataclass(frozen=True)
class UnionType:
    variants: tuple
    family: str = 'union'
    binding_ownership: str = 'view'

    def __post_init__(self):
        if not self.variants or any(not t.mlir or t.mlir.startswith('memref') for t in self.variants):
            raise ValueError('union variant has no LLVM value representation')

    @property
    def name(self):
        return ' | '.join(t.name for t in self.variants)

    @property
    def mlir(self):
        return '!llvm.struct<(i32, ' + ', '.join(t.mlir for t in self.variants) + ')>'

    def grammars(self):
        return (WordGrammar(self, self.name, 'IDENTIFIER'),)


class Union(TypeDefinition):
    def __init__(self):
        super().__init__('union')

    def resolve(self, variants):
        return UnionType(tuple(variants))


@dataclass(frozen=True)
class UnionBox:
    variant: int

    def atom(self, operation):
        from py_compiler.syntax.generic.atom import Atom
        result = operation.result.type
        return Atom(self, (result.variants[self.variant],), result, ('view',), (), self)

    def render_operation(self, operation):
        from py_compiler.mlir.src.cfg import name
        value = operation.result
        type_ = value.type.mlir
        return [f'%union_{value.identity} = llvm.mlir.zero : {type_}',
                f'%union_tag_{value.identity} = arith.constant {self.variant} : i32',
                f'%union_value_{value.identity} = llvm.insertvalue {name(operation.operands[0])}, %union_{value.identity}[{self.variant + 1}] : {type_}',
                f'{name(value)} = llvm.insertvalue %union_tag_{value.identity}, %union_value_{value.identity}[0] : {type_}']
