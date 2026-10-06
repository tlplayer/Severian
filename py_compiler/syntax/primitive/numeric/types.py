from decimal import Decimal
from py_compiler.syntax.primitive.contract import Primitive
from py_compiler.syntax.primitive.numeric.grammar import NumericLiteral, scalar_grammars


class Integer(Primitive):
    def render_constant(self, value, symbol):
        return [], [f'%value = arith.constant {int(value)} : {self.mlir}']

    def module_storage(self, symbol):
        return f'memref.global "private" @{symbol} : memref<{self.mlir}> = dense<0>'

    def grammars(self):
        return (*super().grammars(), *((NumericLiteral(self),) if self.name == 'i64' else ()), *scalar_grammars(self))

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
    def render_constant(self, value, symbol):
        spelling = str(value)
        if '.' not in spelling:
            parts = spelling.upper().split('E')
            spelling = parts[0] + '.0' + ('E' + parts[1] if len(parts) == 2 else '')
        return [], [f'%value = arith.constant {spelling} : {self.mlir}']

    def module_storage(self, symbol):
        return f'memref.global "private" @{symbol} : memref<{self.mlir}> = dense<0.0>'

    def grammars(self):
        return (*super().grammars(), *((NumericLiteral(self),) if self.name == 'f64' else ()), *scalar_grammars(self))

    def decode(self, spelling):
        return Decimal(spelling.replace('_', ''))

    def accept_literal(self, owner, value):
        if owner.family not in ('integer', 'float'):
            return super().accept_literal(owner, value)
        result = Decimal(value)
        if not result.is_finite():
            raise ValueError('non-finite numeric literals require an explicit provider')
        return result


class Byte(Integer):
    def grammars(self):
        return (*Primitive.grammars(self), NumericLiteral(self), *scalar_grammars(self))

    def decode(self, spelling):
        text = spelling[:-1].replace('_', '')
        if any(c in text.lower() for c in '.e'):
            raise ValueError('byte quantities require an integral amount')
        value = int(text, 10)
        if not -(1 << 63) <= value < (1 << 63):
            raise ValueError('byte quantity exceeds its signed 64-bit representation')
        return value

    def accept_literal(self, owner, value):
        return Primitive.accept_literal(self, owner, value)
