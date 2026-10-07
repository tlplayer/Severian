from decimal import Decimal
from py_compiler.syntax.primitive.contract import Primitive
from py_compiler.syntax.primitive.numeric.grammar import NumericLiteral, scalar_grammars


class Integer(Primitive):
    conversion_sources = frozenset(('integer', 'float', 'bool', 'char', 'byte', 'pointer'))
    def render_constant(self, value, symbol):
        return [], [f'%value = arith.constant {int(value)} : {self.mlir}']

    def module_storage(self, symbol):
        return f'memref.global "private" @{symbol} : memref<{self.mlir}> = dense<0>'

    def grammars(self):
        return (*super().grammars(), *((NumericLiteral(self, "integer"),) if self.name == 'i64' else ()), *scalar_grammars(self), *self.integer_grammars())

    def integer_grammars(self):
        from py_compiler.syntax.primitive.int.operations import grammars
        return grammars(self)

    def decode(self, spelling):
        text = spelling.replace('_', '')
        return int(text, 0 if text.lstrip('+-').lower().startswith(('0x', '0o', '0b')) else 10)

    def accept_literal(self, owner, value):
        if owner.family != 'integer':
            return super().accept_literal(owner, value)
        low = -(1 << (self.bits - 1)) if self.signed else 0
        high = (1 << (self.bits - int(self.signed))) - 1
        if not low <= value <= high:
            raise ValueError(f'literal {value} is outside {self.name} range [{low}, {high}]')
        return value


class Float(Primitive):
    conversion_sources = frozenset(('integer', 'float', 'bool'))
    def render_constant(self, value, symbol):
        spelling = str(value)
        if '.' not in spelling:
            parts = spelling.upper().split('E')
            spelling = parts[0] + '.0' + ('E' + parts[1] if len(parts) == 2 else '')
        return [], [f'%value = arith.constant {spelling} : {self.mlir}']

    def module_storage(self, symbol):
        return f'memref.global "private" @{symbol} : memref<{self.mlir}> = dense<0.0>'

    def grammars(self):
        return (*super().grammars(), *((NumericLiteral(self, "float"),) if self.name == 'f64' else ()), *scalar_grammars(self, floating=True), *self.float_grammars())

    def float_grammars(self):
        from py_compiler.syntax.primitive.float.operations import grammars
        return grammars(self)

    def decode(self, spelling):
        return Decimal(spelling.replace('_', ''))

    def accept_literal(self, owner, value):
        if owner.family not in ('integer', 'float'):
            return super().accept_literal(owner, value)
        result = Decimal(value)
        if not result.is_finite():
            raise ValueError('non-finite numeric literals require an explicit provider')
        return result
