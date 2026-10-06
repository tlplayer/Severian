"""Numeric type definitions own literal recognition and operation expansion."""
from py_compiler.syntax.generic.owned import OwnedGrammar, Capture, select
from py_compiler.syntax.generic.grammar import Match
from py_compiler.syntax.symbol.forms import NUMBER


class NumericLiteral(OwnedGrammar):
    def __init__(self, owner):
        super().__init__(owner, 'Y', ('numeric-literal',))

    def recognize(self, window):
        found = NUMBER.match(window.source.text, window.start, window.end)
        if found is None:
            return None
        end, spelling = found.end(), found.group().lower()
        if end < window.end and (window.source.text[end].isalnum() or window.source.text[end] == '_'):
            raise ValueError('invalid numeric literal or unsupported suffix')
        based = spelling.startswith(('0x', '0o', '0b'))
        family = 'byte' if spelling.endswith('b') and not based else ('float' if not based and any(c in spelling for c in '.e') else 'integer')
        if family != self.owner.family:
            return None
        return Match(self, window, end, {'kind': 'NUMBER', 'type': self.owner})

    def construct(self, match):
        return 'NUMBER', match.end


class ScalarOperation(OwnedGrammar):
    associativity = 'left'

    def __init__(self, owner, spelling, precedence, comparison=False):
        super().__init__(owner, 'F.operator', (Capture('self', owner), spelling, Capture('value', owner)),
                         (Capture('self', owner), Capture('value', owner, 'copy')))
        self.spelling, self.precedence, self.comparison = spelling, precedence, comparison

    def recognize(self, window):
        if len(window.tokens) == 1 and window.tokens[0].text == self.spelling:
            return Match(self, window, window.end, {'operator': window.tokens[0]})
        return None

    def expand(self, cfg, left, right, span):
        if left.type != self.owner or right.type != self.owner:
            raise ValueError(f'{self.owner.name}.{self.spelling} needs a declared conversion from {right.type.name}')
        result = cfg.syntax.types['bool'] if self.comparison else self.owner
        return cfg.emit('scalar', result, (left, right), self, span)

    def render_operation(self, operation):
        from py_compiler.mlir.src.cfg import name
        left, right = operation.operands
        floating = self.owner.family == 'float'
        if not self.comparison:
            opcode = {'+': 'add', '-': 'sub', '*': 'mul'}[self.spelling] + ('f' if floating else 'i')
        elif floating:
            predicate = {'==': 'oeq', '!=': 'une', '<': 'olt', '<=': 'ole', '>': 'ogt', '>=': 'oge'}[self.spelling]
            opcode = 'cmpf ' + predicate + ','
        else:
            predicate = {'==': 'eq', '!=': 'ne', '<': 'lt', '<=': 'le', '>': 'gt', '>=': 'ge'}[self.spelling]
            if self.spelling not in ('==', '!='):
                predicate = ('s' if self.owner.signed else 'u') + predicate
            opcode = 'cmpi ' + predicate + ','
        return [f'{name(operation.result)} = arith.{opcode} {name(left)}, {name(right)} : {self.owner.mlir}']


class CompoundAssignment(OwnedGrammar):
    def __init__(self, owner, operation):
        super().__init__(owner, 'O.assignment', (Capture('self', owner, 'borrow'), operation + '=', Capture('value', owner, 'copy')),
                         (Capture('self', owner, 'borrow'), Capture('value', owner, 'copy')))
        self.spelling, self.operation = operation + '=', operation

    def recognize(self, window):
        if any(t.text == self.spelling for t in window.tokens):
            return Match(self, window, window.end, {'tokens': window.tokens})
        return None

    def expand(self, cfg, place, value, span):
        # borrow self -> self = self + value. The place owns pointer/SSA storage.
        with place.capture(cfg, self.captures[0].ownership):
            previous = place.read(cfg, span)
            result = select(self.owner, 'F.operator', self.operation).expand(cfg, previous, value, span)
            place.write(cfg, result, span)
        return result


def scalar_grammars(owner, arithmetic=True):
    rules = [ScalarOperation(owner, op, 3, True) for op in ('==', '!=', '<', '<=', '>', '>=')]
    if arithmetic:
        rules.extend(ScalarOperation(owner, op, precedence) for op, precedence in (('+', 4), ('-', 4), ('*', 5)))
        rules.extend(CompoundAssignment(owner, op) for op in ('+', '-', '*'))
        rules.append(UnaryOperation(owner, "+"))
        if owner.signed or owner.family == "float":
            rules.append(UnaryOperation(owner, "-"))
    return tuple(rules)


class UnaryOperation(OwnedGrammar):
    def __init__(self, owner, spelling):
        super().__init__(owner, 'F.unary', (spelling, Capture('value', owner)))
        self.spelling = spelling

    def recognize(self, window):
        if window.tokens and window.tokens[0].text == self.spelling:
            return Match(self, window, window.end, {'value': window.tokens[1:]})
        return None

    def expand(self, cfg, operand, span):
        if operand.type != self.owner:
            raise ValueError('unary grammar operand violates its capture contract')
        if self.spelling == '+':
            return operand
        if self.spelling == '-':
            from py_compiler.syntax.function.calls import literal_value
            zero = literal_value(0, self.owner, cfg, span)
            return select(self.owner, 'F.operator', '-').expand(cfg, zero, operand, span)
        return cfg.emit('unary', self.owner, (operand,), self, span)

    def render_operation(self, operation):
        from py_compiler.mlir.src.cfg import name
        if self.spelling != 'not':
            raise ValueError('unary grammar lacks a target operation')
        value = operation.result
        return [f'%not_true_{value.identity} = arith.constant true',
                f'{name(value)} = arith.xori {name(operation.operands[0])}, %not_true_{value.identity} : i1']
