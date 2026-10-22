"""Persistent read sharing; only the writer copies a shared backing allocation."""
from dataclasses import dataclass, replace
from py_compiler.syntax.complex.shared import Shared, SharedOperation
from py_compiler.syntax.generic.atom import Atom


@dataclass(frozen=True)
class Mirror(Shared):
    family: str = 'mirror'
    parameter_ownership: str = 'view'

    @property
    def name(self):
        return 'mirror[' + self.payload.name + ']'

    def length(self,value,cfg,span):
        from py_compiler.syntax.function.length import Length
        data = self.data(value,cfg,span)
        from py_compiler.syntax.complex.array import record_storage
        stride = record_storage(self.payload,cfg.context)[0] if getattr(self.payload,'aggregate',False) else 1
        return cfg.emit('length',cfg.context.type('int'),(data,),Length(stride),span)

    def copy_value(self,value,cfg,span):
        from py_compiler.syntax.generic.storage import copy_storage
        return self.from_value(copy_storage(self.data(value,cfg,span),cfg,span),cfg,span)

    def binary_operation(self,node,left,cfg,env):
        from py_compiler.syntax.complex.collection import collection_namespace
        from py_compiler.syntax.function.calls import select,invoke
        from py_compiler.syntax.grammar.expression import Expression
        from py_compiler.syntax.complex.tensor import ArgumentValue
        from py_compiler.syntax.generic.storage import root_binding
        if node.token.text not in ('==','!='):
            raise ValueError('mirror comparison requires == or !=')
        right=cfg.expr(node.operands[1],env)
        terms=tuple(Expression('capture',node.token,(ArgumentValue(value,root_binding(term,cfg,env)),)) for value,term in zip((left,right),node.operands))
        entry,values=select(cfg.context.lookup_functions(collection_namespace(cfg.context)+'equal',cfg.declaration_scope),terms,cfg,env)
        result=invoke(entry,values,cfg,node.token.span)
        if node.token.text=='!=':
            from py_compiler.syntax.generic.owned import select as grammar
            result=grammar(result.type,'F.unary','not').expand(cfg,result,node.token.span)
        return result

    def element_place(self,receiver,index,cfg,env,span):
        owner=next((b for b in env.values() if cfg.values.get(b.identity)==receiver and b.identity not in cfg.flow.moved),None)
        if owner is None:
            owner=cfg.call_result_owners.get(receiver.identity)
        original=self.payload.element_place(self.data(receiver,cfg,span),index,cfg,env,span)
        return MirrorPlace(self,receiver,original,owner)

    def writable(self,receiver,owner,cfg,span):
        from py_compiler.hir.thir.src.model import Edge,Terminator
        from py_compiler.syntax.generic.storage import copy_storage
        if owner is None or owner.ownership in ('view','borrow'):
            raise ValueError('mirror mutation requires its owning binding')
        unique=cfg.emit('mirror-unique',cfg.context.type('bool'),(receiver,),Unique(self),span)
        keep,detach,done=cfg.block(),cfg.block(),cfg.block((self,))
        cfg.current.terminator=Terminator('conditional',(Edge(keep.identity),Edge(detach.identity)),unique)
        keep.terminator=Terminator('jump',(Edge(done.identity,(receiver,)),))
        cfg.current=detach
        copied=copy_storage(self.data(receiver,cfg,span),cfg,span)
        independent=self.from_value(copied,cfg,span)
        self.release(receiver,cfg,span)
        cfg.current.terminator=Terminator('jump',(Edge(done.identity,(independent,)),))
        cfg.current=done
        result=done.parameters[0]
        cfg.values[owner.identity]=result
        cfg.flow.mirror_readers.discard(owner.identity)
        return self.data(result,cfg,span)


@dataclass(frozen=True)
class Unique:
    type: Mirror

    def atom(self,operation):
        return Atom(self,(self.type,),operation.result.type,('view',),('read',),self)

    def render_operation(self,operation):
        from py_compiler.mlir.src.cfg import name,temporary_suffix
        p='%mirror_'+temporary_suffix(operation);t=self.type
        return [f'{p}counter = llvm.extractvalue {name(operation.operands[0])}[1] : {t.mlir}',
                f'{p}ref = builtin.unrealized_conversion_cast {p}counter : {t.descriptor} to memref<1xi64>',
                f'{p}zero = arith.constant 0 : index',f'{p}count = memref.load {p}ref[{p}zero] : memref<1xi64>',
                f'{p}one = arith.constant 1 : i64',f'{name(operation.result)} = arith.cmpi eq, {p}count, {p}one : i64']


@dataclass(frozen=True)
class MirrorPlace:
    owner_type: Mirror
    receiver: object
    place: object
    owner: object

    @property
    def type(self):return self.place.type

    def read(self,cfg,span):return self.place.read(cfg,span)

    def write(self,cfg,value,span):
        payload=self.owner_type.writable(self.receiver,self.owner,cfg,span)
        replace(self.place,receiver=payload).write(cfg,value,span)


def capture(value,owner,cfg,env,span):
    if owner is None or owner.ownership in ('view','borrow'):
        raise ValueError('mirror requires an owning binding')
    if owner.identity in cfg.flow.mirror_readers:
        raise ValueError('cannot mirror a mirror before it detaches')
    cfg.flow.transfer(owner)
    if not isinstance(value.type,Mirror):
        if value.type.family not in ('array','tensor','tuple') or getattr(value.type,'rank',1)!=1:
            raise ValueError('mirror currently requires rank-one owned storage')
        type_=Mirror(value.type,cfg.context.type('usize').bits)
        value=type_.from_value(value,cfg,span)
        updated=replace(owner,type=type_)
        env[owner.name]=updated
        cfg.values[owner.identity]=value
        cfg.body.bindings=[updated if b.identity==owner.identity else b for b in cfg.body.bindings]
        cfg.context.types[type_.name]=type_
    result=cfg.emit('mirror-retain',value.type,(value,),SharedOperation('retain',value.type),span)
    cfg.owned_results.add(result.identity)
    return result
