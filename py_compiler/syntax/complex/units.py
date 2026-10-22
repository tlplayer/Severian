"""Dimensioned numeric literals normalize once; arithmetic preserves dimensions."""
from dataclasses import dataclass
from decimal import Decimal
import re
from py_compiler.syntax.primitive.numeric.types import Float
from py_compiler.syntax.generic.owned import OwnedGrammar, WordGrammar
from py_compiler.syntax.generic.grammar import Match

# Dimensions are data, time, temperature. A byte remains the existing integral
# memory-size type; arithmetic with dimensioned values normalizes it to bits.
UNITS = {
    'b': ((1,0,0),'1','0'),
    'KB': ((1,0,0),'8000','0'), 'MB': ((1,0,0),'8000000','0'),
    'GB': ((1,0,0),'8000000000','0'), 'TB': ((1,0,0),'8000000000000','0'),
    'KiB': ((1,0,0),'8192','0'), 'MiB': ((1,0,0),'8388608','0'),
    'GiB': ((1,0,0),'8589934592','0'), 'TiB': ((1,0,0),'8796093022208','0'),
    'ns': ((0,1,0),'0.000000001','0'), 'us': ((0,1,0),'0.000001','0'),
    'ms': ((0,1,0),'0.001','0'), 's': ((0,1,0),'1','0'),
    'min': ((0,1,0),'60','0'), 'hr': ((0,1,0),'3600','0'), 'day': ((0,1,0),'86400','0'),
    'Hz': ((0,-1,0),'1','0'), 'kHz': ((0,-1,0),'1000','0'),
    'MHz': ((0,-1,0),'1000000','0'), 'GHz': ((0,-1,0),'1000000000','0'),
    'C': ((0,0,1),'1','0'), 'K': ((0,0,1),'1','-273.15'),
    'F': ((0,0,1),'F','0'), 'pct': ((0,0,0),'0.01','0'),
}
PATTERN = re.compile(r'(?:\d[\d_]*(?:\.\d[\d_]*)?|\.\d[\d_]*)(?:[eE][+-]?\d+)?(' + '|'.join(sorted(UNITS,key=len,reverse=True)) + r')(?![\w])')


def match_literal(text,start=0,end=None):
    return PATTERN.match(text,start,len(text) if end is None else end)


def quantity(dimensions):
    names={(1,0,0):'data_quantity',(0,1,0):'duration',(0,-1,0):'frequency',(0,0,1):'temperature',(0,0,0):'ratio'}
    return Quantity(names.get(dimensions,'quantity['+','.join(map(str,dimensions))+']'),'unit','f64',64,True,dimensions)


@dataclass(frozen=True)
class Quantity(Float):
    dimensions: tuple = (0,0,0)

    def grammars(self):
        from py_compiler.syntax.primitive.numeric.grammar import UnaryOperation, CompoundAssignment
        return (WordGrammar(self,self.name,'IDENTIFIER'), UnitLiteral(self),
                *(QuantityOperation(self, op) for op in ('+', '-', '*', '/', '==', '!=', '<', '<=', '>', '>=')),
                *(CompoundAssignment(self, op) for op in ('+', '-', '*', '/')),
                UnaryOperation(self, '+'), UnaryOperation(self, '-'))

    def decode(self, spelling):
        sign=-1 if spelling.startswith('-') else 1
        text=spelling.lstrip('+-')
        matched=match_literal(text)
        if matched is None or matched.end()!=len(text):
            raise ValueError('invalid dimensioned literal')
        suffix=matched.group(1)
        amount=Decimal(text[:-len(suffix)].replace('_',''))*sign
        _,scale,offset=UNITS[suffix]
        return (amount-32)*5/9 if scale=='F' else amount*Decimal(scale)+Decimal(offset)

    def accept_literal(self,owner,value):
        if not isinstance(owner,Quantity) or owner.dimensions!=self.dimensions:
            raise ValueError('incompatible unit dimensions')
        return value

    def binary_operation(self,node,left,cfg,env):
        return binary(node,left,cfg,env)


class UnitLiteral(OwnedGrammar):
    def __init__(self,owner):
        super().__init__(owner,'Y',('unit-literal',))

    def recognize(self,window):
        matched=match_literal(window.source.text,window.start,window.end)
        if matched is None or UNITS[matched.group(1)][0]!=self.owner.dimensions:
            return None
        return Match(self,window,matched.end(),{'kind':'NUMBER','type':self.owner})

    def construct(self,match):
        return 'NUMBER',match.end


def normalized(value,cfg,span):
    from py_compiler.syntax.primitive.numeric.conversion import ScalarConversion
    from py_compiler.syntax.function.calls import literal_value
    from py_compiler.syntax.generic.owned import select
    scalar=cfg.context.type('f64')
    if value.type.family=='unit':
        # Representation-preserving reinterpretation; dimensions stay semantic.
        converted=cfg.emit('unit-scalar',scalar,(value,),UnitCast(),span)
        return converted,value.type.dimensions
    if value.type.family not in ('integer','float','byte'):
        raise ValueError('unit arithmetic requires numeric quantities')
    converted=value if value.type==scalar else cfg.emit('convert',scalar,(value,),ScalarConversion(value.type,scalar),span)
    dimensions=(0,0,0)
    if value.type.family=='byte':
        dimensions=(1,0,0)
        converted=select(scalar,'F.operator','*').expand(cfg,converted,literal_value(8,scalar,cfg,span),span)
    return converted,dimensions


def binary(node,left,cfg,env):
    right=cfg.expr(node.operands[1],env)
    return calculate(node.token.text, left, right, cfg, node.token.span)


def dimensions_of(type_):
    if type_.family == 'unit':
        return type_.dimensions
    if type_.family == 'byte':
        return (1, 0, 0)
    if type_.family in ('integer', 'float'):
        return (0, 0, 0)
    raise ValueError('unit arithmetic requires numeric quantities')


def result_type(op, left, right, context):
    ad, bd = dimensions_of(left), dimensions_of(right)
    if op in ('+','-','==','!=','<','<=','>','>='):
        if ad!=bd: raise ValueError('incompatible unit dimensions')
        dimensions=ad
    elif op in ('*','/'):
        dimensions=tuple(x+y if op=='*' else x-y for x,y in zip(ad,bd))
    else:
        raise ValueError('unsupported dimensioned operation '+op)
    if op in ('==','!=','<','<=','>','>='):
        return context.type('bool')
    if dimensions == (0, 0, 0) and not (op in ('+', '-') and (left.family == 'unit' or right.family == 'unit')):
        return context.type('f64')
    target = quantity(dimensions)
    context.types[target.name] = target
    return target


def calculate(op, left, right, cfg, span):
    from py_compiler.syntax.generic.owned import select
    target = result_type(op, left.type, right.type, cfg.context)
    a, _ = normalized(left, cfg, span)
    b, _ = normalized(right, cfg, span)
    result=select(a.type,'F.operator',op).expand(cfg,a,b,span)
    if target.family != 'unit':
        return result
    return cfg.emit('dimensioned-value',target,(result,),UnitCast(),span)


from py_compiler.syntax.primitive.numeric.grammar import ScalarOperation


class QuantityOperation(ScalarOperation):
    def __init__(self, owner, spelling):
        comparison = spelling in ('==', '!=', '<', '<=', '>', '>=')
        super().__init__(owner, spelling, 3 if comparison else 5 if spelling in ('*', '/') else 4,
                         comparison=comparison, floating=True)

    def expand(self, cfg, left, right, span):
        return calculate(self.spelling, left, right, cfg, span)


@dataclass(frozen=True)
class UnitCast:
    def atom(self,operation):
        from py_compiler.syntax.generic.atom import Atom
        return Atom(self,(operation.operands[0].type,),operation.result.type,('copy',),(),self)

    def render_operation(self,operation):
        from py_compiler.mlir.src.cfg import name
        return [f'{name(operation.result)} = builtin.unrealized_conversion_cast {name(operation.operands[0])} : f64 to f64']
