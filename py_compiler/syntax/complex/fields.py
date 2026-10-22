"""Reflective field access validates every candidate before committing stores."""
import ast
from dataclasses import replace
from py_compiler.syntax.grammar.expression import expression, Expression
from py_compiler.syntax.function.calls import literal_value, coerce
from py_compiler.syntax.generic.owned import select
from py_compiler.hir.thir.src.model import Edge, Terminator


def split_constraints(node):
    children, constraints = [], []
    for child in node.children:
        tokens=list(child.tokens)
        if child.provider or not any(t.text=='{' for t in tokens):
            children.append(child); continue
        start=next(i for i,t in enumerate(tokens) if t.text=='{')
        end=next(i for i in range(start+1,len(tokens)) if tokens[i].text=='}')
        groups,group,depth=[],[],0
        for token in tokens[start+1:end]:
            if token.text==',' and depth==0:
                if group: groups.append(group)
                group=[]
            else:
                group.append(token)
                depth+=(token.text in ('(', '['))-(token.text in (')',']'))
        if group: groups.append(group)
        for group in groups:
            arrow=next((i for i,t in enumerate(group) if t.text=='->'),None)
            condition=expression(group if arrow is None else group[:arrow])
            action=expression(group[arrow+1:]) if arrow is not None else None
            constraints.append((tokens[0].text,condition,action))
        children.append(replace(child,tokens=tuple(tokens[:start]+tokens[end+1:])))
    return replace(node,children=children),tuple(constraints)


def guard(condition, message, cfg, span):
    from py_compiler.syntax.complex.error import raise_error
    passed,failed=cfg.block(),cfg.block()
    cfg.current.terminator=Terminator('conditional',(Edge(passed.identity),Edge(failed.identity)),condition)
    flow,values=cfg.flow.fork(),dict(cfg.values)
    cfg.current=failed
    raise_error(cfg,message,span)
    cfg.current, cfg.flow, cfg.values = passed,flow,values


def validate(receiver, field, value, cfg, env, span, proposed=None):
    from py_compiler.mir.ownership.flow import Binding
    scope=dict(env)
    provider=cfg.context.provider_by_name[receiver.type.name]
    for name,_ in receiver.type.declaration.fields:
        candidate = (proposed[name] if proposed and name in proposed else value if name == field else provider.field_place(receiver,name,cfg,span).read(cfg,span))
        binding=Binding('candidate:'+str(candidate.identity)+':'+name,name,candidate.type,False,'view',span)
        scope[name]=binding
        cfg.values[binding.identity]=candidate
    for name,condition,action in receiver.type.declaration.field_constraints:
        if name!=field: continue
        result=cfg.expr(condition,scope,cfg.context.type('bool'))
        message='ConstraintError: '+name
        if action is not None:
            # Field arrows describe failure predicates, unlike function guards.
            result=select(result.type,'F.unary','not').expand(cfg,result,span)
            if action.kind=='call' and len(action.operands)==2 and action.operands[1].kind=='literal' and action.operands[1].token.kind=='STRING':
                message += ': ' + ast.literal_eval(action.operands[1].token.text)
        guard(result,message,cfg,span)


def validate_object(receiver,cfg,env,span):
    provider=cfg.context.provider_by_name[receiver.type.name]
    for field in dict.fromkeys(name for name,_,_ in receiver.type.declaration.field_constraints):
        validate(receiver,field,provider.field_place(receiver,field,cfg,span).read(cfg,span),cfg,env,span)


def access(receiver,method,arguments,cfg,env,span,owner):
    if method == 'set':
        return update(receiver,arguments,cfg,env,span,owner)
    from py_compiler.syntax.block.union import UnionType
    if len(arguments) != 1:
        raise ValueError('get requires one field name')
    key = arguments[0]
    literal = ast.literal_eval(key.token.text) if key.kind == 'literal' and key.token.kind == 'STRING' else None
    provider = cfg.context.provider_by_name[receiver.type.name]
    fields = [(name,cfg.context.type(type_)) for name,type_ in receiver.type.declaration.fields if not name.startswith('_')]
    if literal is not None:
        if literal not in dict(fields):
            raise ValueError('unknown field '+literal)
        value = provider.field_place(receiver,literal,cfg,span).read(cfg,span)
        if value.type.binding_ownership != 'copy':
            cfg.call_result_owners[value.identity] = owner
            cfg.call_result_modes[value.identity] = 'view'
        return value
    key_value = field_name(key,cfg,env,span)
    temporary = retain_argument(key,key_value,cfg,env,span)
    if not fields:
        raise ValueError('object exposes no readable fields')
    result_type = UnionType(tuple(t for _,t in fields)).with_layout(cfg.context)
    if len(result_type.variants) == 1:
        result_type = result_type.variants[0]
    done = cfg.block((result_type,))
    initial_flow,initial_values = cfg.flow.fork(),dict(cfg.values)
    for field,type_ in fields:
        yes,no = cfg.block(),cfg.block()
        equal = string_equal(key_value,literal_value(field,key_value.type,cfg,span),cfg,env,span)
        cfg.current.terminator = Terminator('conditional',(Edge(yes.identity),Edge(no.identity)),equal)
        cfg.current = yes
        output = coerce(provider.field_place(receiver,field,cfg,span).read(cfg,span),result_type,cfg,span)
        cfg.current.terminator = Terminator('jump',(Edge(done.identity,(output,)),))
        cfg.current = no
        cfg.flow,cfg.values = initial_flow.fork(),dict(initial_values)
    from py_compiler.syntax.complex.error import raise_error
    raise_error(cfg,'FieldError: unknown field',span)
    cfg.current,cfg.flow,cfg.values = done,initial_flow,initial_values
    finish_arguments((temporary,),cfg,env,span)
    if result_type.binding_ownership != 'copy':
        cfg.call_result_owners[done.parameters[0].identity] = owner
        cfg.call_result_modes[done.parameters[0].identity] = 'view'
    return done.parameters[0]


def string_equal(left,right,cfg,env,span):
    from py_compiler.syntax.complex.tensor import ArgumentValue
    from py_compiler.frontend.lexer.lexer import lex
    from py_compiler.frontend.source.source import SourceFile
    token=lex(SourceFile('field-equality','=='),cfg.syntax)[0]
    terms=tuple(Expression('capture',token,(ArgumentValue(v, next((b for b in env.values() if cfg.values.get(b.identity) == v and b.identity not in cfg.flow.moved), None) or cfg.call_result_owners.get(v.identity)),)) for v in (left,right))
    return left.type.binary_operation(Expression('binary',token,terms),left,cfg,env)


def update(receiver, arguments, cfg, env, span, owner):
    """Stage names and values, validate the prospective object, then commit."""
    from py_compiler.syntax.generic.storage import BindingPlace, copy_storage, owns_storage
    if owner is None or owner.ownership == 'view':
        raise ValueError('set requires an owner or exclusive borrow')
    with BindingPlace(owner,rebind=False).capture(cfg,'borrow'):
        pass
    if len(arguments) == 1 and arguments[0].kind == 'map':
        terms = arguments[0].operands
        updates = list(zip(terms[::2],terms[1::2]))
    elif len(arguments) == 2:
        updates = [tuple(arguments)]
    else:
        raise ValueError('set requires a name and value, or grouped field updates')
    provider = cfg.context.provider_by_name[receiver.type.name]
    fields = {name:cfg.context.type(type_) for name,type_ in receiver.type.declaration.fields if not name.startswith('_')}
    staged, static_names = [],set()
    temporaries = []
    for key,term in updates:
        name = ast.literal_eval(key.token.text) if key.kind == 'literal' and key.token.kind == 'STRING' else None
        if name is not None:
            if name not in fields: raise ValueError('unknown field '+name)
            if name in static_names: raise ValueError('duplicate grouped field update '+name)
            static_names.add(name)
        key_value = field_name(key,cfg,env,span) if name is None else None
        if key_value is not None:
            temporaries.append(retain_argument(key,key_value,cfg,env,span))
        value = cfg.expr(term,env,fields[name] if name is not None else None)
        temporaries.append(retain_argument(term,value,cfg,env,span))
        matching = [field for field,type_ in fields.items() if type_ == value.type and (name is None or name == field)]
        if not matching: raise ValueError('field update has incompatible value type')
        if name is None:
            from py_compiler.syntax.generic.owned import select
            found = literal_value(False,cfg.context.type('bool'),cfg,span)
            for field in matching:
                equal = string_equal(key_value,literal_value(field,key_value.type,cfg,span),cfg,env,span)
                from py_compiler.syntax.operator.logical import BooleanOr
                found = cfg.emit('field-match',found.type,(found,equal),BooleanOr(),span)
            guard(found,'FieldError: unknown field or incompatible field value',cfg,span)
        staged.append((name,key_value,value))
    proposed = {}
    for field,type_ in fields.items():
        candidate = provider.field_place(receiver,field,cfg,span).read(cfg,span)
        for name,key,value in staged:
            if name is not None:
                if name == field: candidate = value
                continue
            if value.type != type_: continue
            equal = string_equal(key,literal_value(field,key.type,cfg,span),cfg,env,span)
            yes,no,done = cfg.block(),cfg.block(),cfg.block((type_,))
            cfg.current.terminator = Terminator('conditional',(Edge(yes.identity),Edge(no.identity)),equal)
            yes.terminator = Terminator('jump',(Edge(done.identity,(value,)),))
            no.terminator = Terminator('jump',(Edge(done.identity,(candidate,)),))
            cfg.current = done
            candidate = done.parameters[0]
        proposed[field] = candidate
    for field in dict.fromkeys(name for name,_,_ in receiver.type.declaration.field_constraints):
        validate(receiver,field,proposed[field],cfg,env,span,proposed)
    # Copy every candidate before touching any field, including swaps of owners.
    copied = {field:copy_storage(value,cfg,span) for field,value in proposed.items()}
    for field,value in copied.items():
        place = provider.field_place(receiver,field,cfg,span,write=True)
        if owns_storage(place.type): place.destroy(cfg,span)
        place.write(cfg,value,span)
    finish_arguments(temporaries,cfg,env,span)
    cfg.call_result_owners[receiver.identity]=owner
    cfg.call_result_modes[receiver.identity]='borrow'
    return receiver


def retain_argument(term,value,cfg,env,span):
    from py_compiler.syntax.generic.storage import root_binding,owns_storage
    from py_compiler.mir.ownership.flow import Binding
    if not owns_storage(value.type) or value.identity not in cfg.owned_results or root_binding(term,cfg,env) is not None:
        return None
    name='<field argument '+str(value.identity)+'>'
    binding=Binding(cfg.body.identity+':'+name,name,value.type,False,'own',span)
    env[name]=binding
    cfg.local_names.add(name)
    cfg.values[binding.identity]=value
    cfg.body.bindings.append(binding)
    cfg.record_lifetime('declare',binding.identity,name)
    return binding


def finish_arguments(bindings,cfg,env,span):
    for binding in bindings:
        if binding is not None:
            cfg.cleanup_memory((binding,),span)
            env.pop(binding.name,None)
            cfg.local_names.discard(binding.name)


def field_name(term,cfg,env,span):
    from py_compiler.syntax.generic.storage import root_binding,copy_storage
    value=cfg.expr(term,env,cfg.context.type('string'))
    # A computed name must remain stable through every dispatch comparison.
    if root_binding(term,cfg,env) is not None:
        return value
    copied=copy_storage(value,cfg,span)
    cfg.owned_results.add(copied.identity)
    return copied
