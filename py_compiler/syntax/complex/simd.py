"""Host SIMD elementwise arithmetic with checked shapes and a scalar tail."""
from dataclasses import dataclass
from py_compiler.syntax.generic.atom import Atom


def elementwise(node,left,cfg,env):
    from py_compiler.syntax.complex.tensor import TensorType
    right=cfg.expr(node.operands[1],env)
    if (left.type.rank != 1 or right.type.family != 'tensor' or right.type.rank != 1
            or left.type.element != right.type.element or left.type.element.mlir not in ('f32','f64')
            or node.token.text not in ('+','*') or left.type.strided or right.type.strided):
        raise ValueError('SIMD elementwise arithmetic requires contiguous rank-one matching floating tensors')
    result=TensorType(left.type.element)
    value=cfg.emit('simd-elementwise',result,(left,right),Elementwise(node.token.text),node.token.span)
    cfg.owned_results.add(value.identity)
    cfg.body.placements=[*getattr(cfg.body,'placements',()),{'requested':'simd','selected':'host-vector','lanes':4}]
    return value


@dataclass(frozen=True)
class Elementwise:
    operator: str

    def atom(self,operation):
        return Atom(self,tuple(v.type for v in operation.operands),operation.result.type,('view','view'),('read','allocation'),self)

    def render_operation(self,operation):
        from py_compiler.mlir.src.cfg import name,temporary_suffix
        left,right=operation.operands; result=operation.result
        p='%simd_'+temporary_suffix(operation)
        scalar=left.type.element.mlir; vector=f'vector<4x{scalar}>'
        op='arith.mulf' if self.operator=='*' else 'arith.addf'
        a,b=name(left),name(right)
        return [f'{p}z = arith.constant 0 : index',f'{p}one = arith.constant 1 : index',f'{p}four = arith.constant 4 : index',
                f'{p}n = memref.dim {a}, {p}z : {left.type.mlir}',f'{p}m = memref.dim {b}, {p}z : {right.type.mlir}',
                f'{p}ok = arith.cmpi eq, {p}n, {p}m : index',f'cf.assert {p}ok, "SIMD tensor shapes do not match"',
                f'{name(result)} = memref.alloc({p}n) : {result.type.mlir}',
                f'{p}chunks = arith.divui {p}n, {p}four : index',f'{p}end = arith.muli {p}chunks, {p}four : index',
                f'scf.for {p}i = {p}z to {p}end step {p}four {{',
                f'  {p}a = vector.load {a}[{p}i] : {left.type.mlir}, {vector}',
                f'  {p}b = vector.load {b}[{p}i] : {right.type.mlir}, {vector}',
                f'  {p}v = {op} {p}a, {p}b : {vector}',
                f'  vector.store {p}v, {name(result)}[{p}i] : {result.type.mlir}, {vector}', '}',
                f'scf.for {p}j = {p}end to {p}n step {p}one {{',
                f'  {p}x = memref.load {a}[{p}j] : {left.type.mlir}',f'  {p}y = memref.load {b}[{p}j] : {right.type.mlir}',
                f'  {p}r = {op} {p}x, {p}y : {scalar}',f'  memref.store {p}r, {name(result)}[{p}j] : {result.type.mlir}', '}']
