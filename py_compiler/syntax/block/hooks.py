"""Structured hook phases enter in declaration order and exit in reverse."""
from dataclasses import dataclass, replace
from py_compiler.syntax.generic.block import BlockProvider
from py_compiler.syntax.function.calls import literal_value, coerce
from py_compiler.syntax.grammar.expression import Expression
from py_compiler.syntax.complex.tensor import ArgumentValue
from py_compiler.mir.ownership.flow import Binding


def parse_phases(tokens):
    from py_compiler.syntax.grammar.expression import expression
    if len(tokens)<2 or tokens[0].text!='{' or tokens[-1].text!='}':
        raise ValueError('hook phases require with { prefix callback(...), suffix callback(...), }')
    groups=[]; current=[]; depth=0
    for token in tokens[1:-1]:
        if token.text==',' and depth==0:
            if current: groups.append(current); current=[]
            continue
        current.append(token)
        depth += (token.text in ('(', '[', '{')) - (token.text in (')', ']', '}'))
    if current: groups.append(current)
    phases=[]
    for group in groups:
        if group[0].text not in ('prefix','suffix'):
            raise ValueError('hook phase must be prefix or suffix')
        # F.args... is a compiler pack, not an arbitrary runtime splat.
        body=[]; i=1
        while i<len(group):
            if i+3<len(group) and [t.text for t in group[i:i+4]]==['F','.','args','...']:
                body.extend((group[i+3],*group[i:i+3])); i+=4
            else: body.append(group[i]); i+=1
        term=expression(tuple(body))
        if term.kind!='call': raise ValueError('hook phase requires a callback call')
        def nested_call(value):
            return isinstance(value,Expression) and (value.kind=='call' or any(nested_call(x) for x in value.operands))
        if any(nested_call(value) for value in term.operands[1:]):
            raise ValueError('hook arguments cannot contain nested calls; use a nonthrowing callback')
        phases.append((group[0].text,term))
    if not phases: raise ValueError('hook annotation requires at least one phase')
    return tuple(phases)


class Without(BlockProvider):
    spelling='without'


def register_hook(node,context):
    if len(node.header)<2 or tuple(t.text for t in node.header[-2:])!=('with','context'):
        return False
    name=node.header[0].text
    if not hasattr(context,'hook_definitions'):
        context.hook_definitions={};context.hook_groups={}
    if name in context.hook_definitions:
        raise ValueError('duplicate hook '+name)
    phases={getattr(n.provider,'spelling',''):n for n in node.children}
    if set(phases)-{'with','without'} or len(phases)!=len(node.children):
        raise ValueError('hook requires one entry and one exit phase')
    context.hook_definitions[name]=node
    for annotation in node.decorators:
        context.hook_groups.setdefault(annotation.name,[]).append(name)
    return True


class Hooks:
    def __init__(self,entry,cfg,env):
        self.cfg,self.env,self.entry=cfg,env,entry
        definitions=getattr(cfg.context,'hook_definitions',{})
        names=[]
        self.groups=[]
        self.entered=[]
        self.running=False
        cfg.hooks=self
        for annotation in entry.node.decorators:
            if annotation.phases:
                self.groups.append(annotation)
                continue
            before=len(names)
            if annotation.name in definitions:
                names.append(annotation.name)
            elif annotation.name in getattr(cfg.context,'hook_groups',{}):
                group=cfg.context.hook_groups[annotation.name]
                chosen=annotation.arguments or tuple(group)
                if any(name not in group for name in chosen):
                    raise ValueError('hook group contains an unknown selection')
                names.extend(chosen)
            self.groups.extend(definitions[name] for name in names[before:])
        self.definitions=[definitions[name] for name in names]
        self.context=None
        if not self.groups:return
        self.started=cfg.emit('hook-clock',cfg.context.type('f64'),(),MonotonicClock(),entry.node.span)
        if not names:
            self.enter()
            return
        type_=cfg.context.type('HookContext')
        provider=cfg.context.provider_by_name[type_.name]
        self.context=provider.instantiate(type_.declaration,(),cfg,env,entry.node.span)
        self.owner=Binding(entry.node.identity+':hook-context',':hook-context',type_,False,'own',entry.node.span)
        env[self.owner.name]=self.owner
        cfg.values[self.owner.identity]=self.context
        cfg.body.bindings.append(self.owner)
        self.provider=provider
        self.set('function',literal_value(entry.name.rsplit('.',1)[-1],cfg.context.type('string'),cfg,entry.node.span))
        self.enter()

    def set(self,name,value):
        from py_compiler.syntax.generic.storage import copy_storage,owns_storage
        place=self.provider.field_place(self.context,name,self.cfg,self.entry.node.span,write=True)
        if value is None:
            from py_compiler.syntax.block.union import UnionNone
            absent=next((i for i,t in enumerate(place.type.variants) if t.name in ('None','absent')),None)
            if absent is None: raise ValueError('hook result/error context must be optional')
            value=self.cfg.emit('union-none',place.type,(),UnionNone(absent),self.entry.node.span)
        else:
            value=coerce(value,place.type,self.cfg,self.entry.node.span)
        if owns_storage(place.type):
            value=copy_storage(value,self.cfg,self.entry.node.span)
            place.destroy(self.cfg,self.entry.node.span)
        place.write(self.cfg,value,self.entry.node.span)

    def phase(self,definition,kind):
        nodes=[n for n in definition.children if getattr(n.provider,'spelling','')==kind]
        if not nodes:return
        cfg=self.cfg
        binding=Binding(definition.identity+':context:'+kind,'context',self.context.type,False,'borrow',definition.span)
        cfg.values[binding.identity]=self.context
        scope={**self.env,'context':binding}
        cfg.body.bindings.append(binding)
        cfg.grammar_expansions+=1
        suffix=':hook:'+str(cfg.grammar_expansions)
        def clone(n):
            return replace(n,identity=n.identity+suffix,children=[clone(c) for c in n.children])
        cfg.scope([clone(n) for n in nodes[0].children],scope)

    def enter(self):
        for group in self.groups:
            if getattr(group,'phases',()): self.callbacks(group,'prefix')
            else: self.phase(group,'with')
            self.entered.append(group)

    def callbacks(self,group,phase,result=None,error=None,duration=None):
        from py_compiler.syntax.function.calls import select,invoke,literal_value,coerce
        from py_compiler.syntax.complex.error import error_members,is_error
        from py_compiler.syntax.generic.storage import owns_storage,release_storage
        cfg=self.cfg
        scope=dict(self.env)
        temporaries=[]
        def capture(value):
            binding=Binding(self.entry.node.identity+':hook-view:'+str(cfg.next_value)+':'+str(len(scope)),
                            '<hook view>',value.type,False,'view',self.entry.node.span)
            cfg.body.bindings.append(binding); cfg.values[binding.identity]=value
            scope[binding.identity]=binding
            return Expression('capture',self.entry.node.tokens[0],(ArgumentValue(value,binding),))
        def optional(value,types):
            names=[t.name for t in types if t and t.name not in ('None','absent')]
            target=cfg.context.type('|'.join(dict.fromkeys([*names,'None'])))
            if value is not None and value.type.family=='union' and error_members(value.type):
                from py_compiler.syntax.block.union import UnionMember
                successes=[t for t in value.type.variants if not is_error(t)]
                if len(successes)==1:
                    value=cfg.emit('union-member',successes[0],(value,),UnionMember(value.type.variants.index(successes[0])),self.entry.node.span)
                else:
                    from py_compiler.syntax.block.union import Union, UnionWiden
                    success_type=Union().resolve(successes,cfg.context)
                    value=cfg.emit('union-narrow',success_type,(value,),UnionWiden(narrow=True),self.entry.node.span)
            if value is None:
                value=literal_value(None,cfg.context.type('None'),cfg,self.entry.node.span)
            value=coerce(value,target,cfg,self.entry.node.span)
            # Union wrapping owns any copied payload; release after observers.
            if value.identity in cfg.owned_results: temporaries.append(value)
            return capture(value)
        def transform(term):
            if term.kind=='member' and term.operands[0].kind=='index':
                indexed=term.operands[0]; base,position=indexed.operands
                if base.kind=='member' and base.token.text=='params' and base.operands[0].kind=='name' and base.operands[0].token.text=='F':
                    if position.kind!='literal' or not position.token.text.isdigit():
                        raise ValueError('F.params requires a constant parameter index')
                    index=int(position.token.text)
                    if index>=len(self.entry.parameters): raise ValueError('F.params index outside parameter count')
                    name,annotation=self.entry.parameters[index]
                    default=self.entry.defaults.get(name)
                    fields={'name':name,'type':annotation or self.env[name].type.name,
                            'ownership':self.entry.parameter_ownership.get(name,self.env[name].ownership),
                            'default':' '.join(t.text for t in default.tokens) if default else ''}
                    if term.token.text not in fields: raise ValueError('unknown parameter metadata field')
                    return capture(literal_value(fields[term.token.text],cfg.context.type('string'),cfg,term.token.span))
            if term.kind=='name' and term.token.text=='R':
                if phase=='prefix': raise ValueError('R is unavailable during prefix')
                target=cfg.body.result_type or cfg.context.type('absent')
                outcome=error if error is not None else result
                if outcome is None:
                    outcome=literal_value(None,cfg.context.type('absent'),cfg,term.token.span)
                outcome=coerce(outcome,target,cfg,term.token.span)
                if outcome.identity in cfg.owned_results and outcome is not result and outcome is not error:
                    temporaries.append(outcome)
                return capture(outcome)
            if term.kind=='member' and term.operands[0].kind=='name':
                root=term.operands[0].token.text; member=term.token.text
                if root=='F':
                    if member=='name': return capture(literal_value(self.entry.name.rsplit('.',1)[-1],cfg.context.type('string'),cfg,term.token.span))
                    if member=='args':
                        return Expression('tuple',term.token,tuple(Expression('name',term.token.with_text(name)) for name,_ in self.entry.parameters))
                    if member=='params': raise ValueError('F.params is compile-time metadata; select F.params[index].name, .type, .ownership, or .default')
                    raise ValueError('unknown function hook binding F.'+member)
                if root=='T': raise ValueError('T bindings require a type target')
                if root=='context':
                    if member in ('result','error','duration') and phase=='prefix':
                        raise ValueError('context.'+member+' is unavailable during prefix')
                    if member=='result':
                        target=cfg.body.result_type
                        successes=[t for t in (target.variants if target and target.family=='union' else (target,)) if t and not is_error(t)]
                        return optional(result,successes)
                    if member=='error': return optional(error,error_members(cfg.body.result_type) or (cfg.context.type('Error'),))
                    if member=='duration': return capture(duration)
                    raise ValueError('unknown hook context binding '+member)
            return replace(term,operands=tuple(transform(x) if isinstance(x,Expression) else x for x in term.operands))
        for kind,term in group.phases:
            if kind!=phase: continue
            call=transform(term)
            callee=call.operands[0]
            if callee.kind!='name': raise ValueError('hook callbacks must name a function')
            arguments=call.operands[1:]
            from py_compiler.syntax.function.calls import expand_spreads
            if any(a.kind=='spread' for a in arguments): arguments=expand_spreads(arguments,cfg,scope)
            entries=cfg.context.lookup_functions(callee.token.text,cfg.declaration_scope)
            callback,values=select(entries,arguments,cfg,scope)
            # Hooks observe outcomes; they cannot consume or mutate borrowed state.
            for (name,_),value in zip(callback.parameters,values):
                if callback.parameter_ownership.get(name,getattr(value.type,'parameter_ownership','view')) in ('borrow','move','own'):
                    raise ValueError('hook callbacks require read-only arguments')
            def throwing(entry,seen):
                if id(entry) in seen: return False
                seen.add(id(entry))
                if error_members(cfg.context.type(entry.result or 'absent')): return True
                def explicit(nodes):
                    return any((n.tokens and n.tokens[0].text=='throw') or explicit(n.children) for n in nodes)
                if explicit(entry.node.children): return True
                body=getattr(entry,'body',None)
                if body is not None:
                    for block in body.blocks:
                        for operation in block.operations:
                            dependency=getattr(operation.payload,'dependency',None)
                            if dependency is not None and hasattr(dependency,'node') and throwing(dependency,seen): return True
                return False
            if throwing(callback,set()):
                raise ValueError('hook callbacks must be nonthrowing')
            value=invoke(callback,values,cfg,term.token.span)
            if value is not None and value.identity in cfg.owned_results and owns_storage(value.type):
                release_storage(value,cfg,term.token.span)
        for value in temporaries:
            if owns_storage(value.type): release_storage(value,cfg,self.entry.node.span)

    def finish(self,result=None,error=None):
        if not self.groups or self.running:return
        self.running=True
        try:
            cfg=self.cfg
            now=cfg.emit('hook-clock',cfg.context.type('f64'),(),MonotonicClock(),self.entry.node.span)
            from py_compiler.syntax.generic.owned import select
            duration=select(now.type,'F.operator','-').expand(cfg,now,self.started,self.entry.node.span)
            if self.context is not None:
                self.set('duration',duration); self.set('result',result)
                message=None
                if error is not None:
                    message=cfg.context.provider_by_name[error.type.name].field_place(error,'message',cfg,self.entry.node.span).read(cfg,self.entry.node.span)
                self.set('error',message)
            for group in reversed(self.entered):
                if getattr(group,'phases',()): self.callbacks(group,'suffix',result,error,duration)
                else: self.phase(group,'without')
        finally:
            self.running=False


@dataclass(frozen=True)
class MonotonicClock:
    external_symbol_contracts=(('clock_gettime',('i32','!llvm.ptr'),'i32'),)

    def atom(self,operation):
        from py_compiler.syntax.generic.atom import Atom
        return Atom(self,(),operation.result.type,(),('IO',),self)

    def render_operation(self,operation):
        from py_compiler.mlir.src.cfg import name,temporary_suffix
        p='%clock_'+temporary_suffix(operation)
        return [f'{p}_one = arith.constant 1 : i64',
                f'{p}_storage = llvm.alloca {p}_one x !llvm.struct<(i64,i64)> : (i64) -> !llvm.ptr',
                f'{p}_monotonic = arith.constant 1 : i32',
                f'{p}_status = func.call @clock_gettime({p}_monotonic, {p}_storage) : (i32,!llvm.ptr) -> i32',
                f'{p}_zero = arith.constant 0 : i32',
                f'{p}_ok = arith.cmpi eq, {p}_status, {p}_zero : i32',
                f'cf.assert {p}_ok, "monotonic clock failed"',
                f'{p}_time = llvm.load {p}_storage : !llvm.ptr -> !llvm.struct<(i64,i64)>',
                f'{p}_seconds = llvm.extractvalue {p}_time[0] : !llvm.struct<(i64,i64)>',
                f'{p}_nanos = llvm.extractvalue {p}_time[1] : !llvm.struct<(i64,i64)>',
                f'{p}_s = arith.sitofp {p}_seconds : i64 to f64',
                f'{p}_ns = arith.sitofp {p}_nanos : i64 to f64',
                f'{p}_scale = arith.constant 1.0e-9 : f64',
                f'{p}_fraction = arith.mulf {p}_ns, {p}_scale : f64',
                f'{name(operation.result)} = arith.addf {p}_s, {p}_fraction : f64']


import unittest


class AnnotationPhaseTests(unittest.TestCase):
    def compile(self,source):
        from py_compiler.frontend.src.lib import compile_source
        from py_compiler.syntax.recognition import Syntax
        return compile_source('annotation-phases.sev',source,Syntax(include_tests=True))

    def test_scalar_result_and_argument_pack(self):
        source='''def enter(x: int, y: int):
    if x != 4 or y != 6:
        panic("wrong arguments")
def leave(result: int):
    if result != 10:
        panic("wrong result")
@monitor with {
    prefix enter(F.args...),
    suffix leave(R),
}
def add(x: int, y: int) -> int:
    return x + y
'''
        result=self.compile(source)
        self.assertFalse(result.diagnostics,str(result.diagnostics))
        from py_compiler.mlir.src.lib import lower,render,verify_c
        verify_c(render(lower(result.program)))

    def test_phase_and_target_diagnostics(self):
        cases=(
            ('@monitor with {prefix observe(R),}\ndef work() -> int:\n    return 1\n','unavailable during prefix'),
            ('@monitor with {suffix observe(T.fields),}\ndef work() -> int:\n    return 1\n','type target'),
            ('@monitor with {suffix observe(R),}\nclass Value:\n    value: int\n','function target'),
            ('@monitor with {suffix 1 + 2,}\ndef work():\n    return\n','callback call'),
            ('@monitor with {suffix observe(fail()),}\ndef work():\n    return\n','nested calls'),
            ('def bad() -> int | Error:\n    throw Error("bad")\n@monitor with {suffix bad(),}\ndef work():\n    return\n','nonthrowing'),
        )
        for source,message in cases:
            with self.subTest(message=message):
                result=self.compile(source)
                self.assertIn(message,str(result.diagnostics))

    def test_nonthrowing_rule_checks_indirect_calls(self):
        result=self.compile('''def fail():
    throw Error("bad")
def callback():
    fail()
@monitor with {suffix callback(),}
def work():
    return
''')
        self.assertIn('nonthrowing',str(result.diagnostics))

    def test_suffix_cannot_read_a_consumed_argument(self):
        result=self.compile('''def observe(value: view tensor[int]):
    return
@monitor with {suffix observe(F.args...),}
def consume(value: move tensor[int]) -> tensor[int]:
    return value
''')
        self.assertTrue(result.diagnostics)
        self.assertIn('moved',str(result.diagnostics))

    def test_parameter_metadata_fields_are_typed_strings(self):
        result=self.compile('''def metadata(name: string, default: string):
    return
@monitor with {prefix metadata(F.params[0].name, F.params[1].default),}
def add(x: int, y: int = 2) -> int:
    return x + y
''')
        self.assertFalse(result.diagnostics,str(result.diagnostics))


class CompleteResultHookTests(unittest.TestCase):
    def test_generic_observer_receives_all_success_and_error_variants(self):
        from py_compiler.frontend.src.lib import compile_source
        from py_compiler.syntax.recognition import Syntax
        source = '''def observe[R](result: view R):
    if result is E:
        if result.message != "failed":
            panic("lost error payload")
@monitor with {suffix observe(R),}
def choose(kind: int) -> int | string | E:
    if kind == 0:
        return 42
    if kind == 1:
        return "text"
    throw E("failed")
def forward(kind: int) -> int | string | E:
    return choose(kind)
test:
    assert(forward(0) == 42)
    assert(forward(1) == "text")
    failed ?= forward(2)
    assert(failed is E)
    assert(failed.message == "failed")
'''
        from py_compiler.syntax.prelude import PreludeTests
        result = compile_source('complete-result.sev', source, Syntax(include_tests=True),
                                c_packages=PreludeTests().library_packages())
        self.assertFalse(result.diagnostics, str(result.diagnostics))
        for body in result.program.bodies:
            body.verify()
