"""Enum matching checks coverage and lowers typed payloads to ordinary CFG."""
from py_compiler.syntax.generic.block import BlockProvider


class Match(BlockProvider):
    spelling = "match"

    def parse_header(self, items):
        from py_compiler.syntax.grammar.expression import expression
        header = super().parse_header(items)
        expression(header)
        return header

    def declare(self, node, context):
        if not node.children or any(getattr(child.provider, 'spelling', None) != 'case' for child in node.children):
            raise ValueError('match requires case blocks')
        super().declare(node, context)

    def lower(self, node, cfg, env, nodes, index):
        from dataclasses import replace
        from importlib import import_module
        from py_compiler.syntax.grammar.expression import expression
        term = expression(node.header)
        subject = cfg.expr(term, env)
        retain_subject(node,term,subject,cfg,env)
        if subject is not None and getattr(subject.type, 'enum_payloads', False):
            from py_compiler.syntax.generic.storage import root_binding
            owner=root_binding(term,cfg,env) or cfg.call_result_owners.get(subject.identity)
            return lower_payload_match(node, subject, cfg, env, owner)
        if subject is None or subject.type.family != 'enum':
            raise ValueError('match currently requires a payload-free enum subject')
        seen, chain, wildcard = set(), [], False
        for position, case in enumerate(node.children):
            pattern = expression(case.header)
            if wildcard:
                raise ValueError('unreachable case after wildcard')
            if pattern.kind == 'name' and pattern.token.text == '_':
                wildcard = True
                ordinal = None
            elif pattern.kind == 'name':
                ordinal = subject.type.variant(pattern.token.text)
            elif pattern.kind == 'member' and pattern.operands[0].kind == 'name':
                if cfg.context.type(pattern.operands[0].token.text) != subject.type:
                    raise ValueError('case enum type does not match its subject')
                ordinal = subject.type.variant(pattern.token.text)
            else:
                raise ValueError('enum case requires a named variant or wildcard')
            if ordinal is not None:
                if ordinal in seen:
                    raise ValueError('duplicate enum case')
                seen.add(ordinal)
            branch = EnumBranch(subject, ordinal)
            chain.append(replace(case, provider=branch))
        if not wildcard and len(seen) != len(subject.type.variants):
            raise ValueError('non-exhaustive enum match')
        chain[-1].provider.has_condition = False
        lower_chain(node,cfg,chain,env)
        return 1


class Case(BlockProvider):
    spelling = 'case'

    def attach(self, previous, parent):
        if getattr(parent.provider, 'spelling', None) not in ('match', 'select'):
            raise ValueError('case requires an enclosing match')


class EnumBranch:
    def __init__(self, subject, ordinal):
        self.subject, self.ordinal = subject, ordinal
        self.has_condition = ordinal is not None

    def condition_value(self, cfg, node, env):
        from py_compiler.syntax.function.calls import literal_value
        from py_compiler.syntax.generic.owned import select
        variant = literal_value(self.ordinal, self.subject.type, cfg, node.span)
        return select(self.subject.type, 'F.operator', '==').expand(cfg, self.subject, variant, node.span)


import unittest


class EnumMatchTests(unittest.TestCase):
    def compile(self, source):
        from py_compiler.frontend.src.lib import compile_source
        from py_compiler.syntax.recognition import Syntax
        return compile_source('enum-match.sev', source, Syntax())

    def test_exhaustive_match_returns_on_every_branch(self):
        from py_compiler.mlir.src.lib import lower, render
        result = self.compile('enum Kind:\n    A\n    B\ndef work(kind: Kind) -> int:\n'
                              '    match kind:\n        case A:\n            return 1\n'
                              '        case Kind.B:\n            return 2\n')
        self.assertFalse(result.diagnostics, str(result.diagnostics))
        self.assertIn('cf.cond_br', render(lower(result.program)))

    def test_invalid_matches_are_rejected(self):
        for cases, diagnostic in (
            ('        case A:\n            return 1\n', 'non-exhaustive'),
            ('        case A:\n            return 1\n        case A:\n            return 2\n', 'duplicate enum case'),
            ('        case _:\n            return 1\n        case B:\n            return 2\n', 'unreachable case'),
            ('        case C:\n            return 1\n', 'unknown Kind variant'),
        ):
            with self.subTest(cases=cases):
                result = self.compile('enum Kind:\n    A\n    B\ndef work(kind: Kind) -> int:\n    match kind:\n' + cases)
                self.assertTrue(result.diagnostics)
                self.assertIn(diagnostic, str(result.diagnostics[0]))


def lower_payload_match(node, subject, cfg, env, owner=None):
    from dataclasses import replace
    from importlib import import_module
    from py_compiler.syntax.grammar.expression import expression
    seen, chain, wildcard = set(),[],False
    for case in node.children:
        if wildcard or len(seen) == len(subject.type.variants):
            raise ValueError('unreachable case after complete enum coverage')
        term=expression(case.header)
        if term.kind == 'name' and term.token.text == '_':
            wildcard=True
            chain.append(replace(case,provider=PayloadBranch(subject,None)))
            continue
        callee=term.operands[0] if term.kind=='call' else term
        if callee.kind not in ('name','member'):
            raise ValueError('payload case requires a named enum variant')
        if callee.kind == 'member':
            if callee.operands[0].kind != 'name' or cfg.context.type(callee.operands[0].token.text) != subject.type:
                raise ValueError('case enum type does not match its subject')
        leaf=callee.token.text
        ordinal=next((i for i,t in enumerate(subject.type.variants) if t.name.rsplit('.',1)[-1]==leaf),None)
        if ordinal is None:
            raise ValueError('unknown payload enum variant ' + leaf)
        if ordinal in seen:
            raise ValueError('duplicate enum case')
        seen.add(ordinal)
        fields=subject.type.variants[ordinal].declaration.fields
        names=term.operands[1:] if term.kind=='call' else ()
        if term.kind == 'call' and (len(names)!=len(fields) or any(n.kind!='name' for n in names)):
            raise ValueError('payload pattern must bind every field')
        bound=[n.token.text for n in names if n.token.text != '_']
        if len(set(bound)) != len(bound):
            raise ValueError('duplicate payload binding')
        children=list(case.children)
        if names:
            binding=replace(case,provider=PayloadBindings(subject,ordinal,names,owner),children=[],header=(),identity=case.identity+':payload')
            children.insert(0,binding)
        chain.append(replace(case,provider=PayloadBranch(subject,ordinal),children=children))
    if not wildcard and len(seen)!=len(subject.type.variants):
        raise ValueError('non-exhaustive enum match')
    chain[-1].provider.has_condition=False
    lower_chain(node,cfg,chain,env)
    return 1


class PayloadBranch(EnumBranch):
    def condition_value(self,cfg,node,env):
        from py_compiler.syntax.block.union import UnionTag
        from py_compiler.syntax.function.calls import literal_value
        from py_compiler.syntax.generic.owned import select
        tag=cfg.emit('union-tag',cfg.context.type('i32'),(self.subject,),UnionTag(),node.span)
        return select(tag.type,'F.operator','==').expand(cfg,tag,literal_value(self.ordinal,tag.type,cfg,node.span),node.span)


class PayloadBindings:
    spelling='payload-bindings'

    def __init__(self,subject,ordinal,names,owner=None):
        self.subject,self.ordinal,self.names,self.owner=subject,ordinal,names,owner

    def lower(self,node,cfg,env,nodes,index):
        from py_compiler.syntax.block.union import UnionMember
        from py_compiler.mir.ownership.flow import Binding
        type_=self.subject.type.variants[self.ordinal]
        value=cfg.emit('union-member',type_,(self.subject,),UnionMember(self.ordinal),node.span)
        owner=self.owner or next((b for b in env.values() if cfg.values.get(b.identity) == self.subject),None) or cfg.call_result_owners.get(self.subject.identity)
        for position,(name,(field,_)) in enumerate(zip(self.names,type_.declaration.fields)):
            if name.token.text=='_':
                continue
            place=cfg.context.provider_by_name[type_.name].field_place(value,field,cfg,node.span)
            member=place.read(cfg,node.span)
            binding=Binding(node.identity+':'+str(position),name.token.text,member.type,False,'view' if member.type.binding_ownership!='copy' else 'copy',node.span)
            env[binding.name]=binding
            cfg.local_names.add(binding.name)
            cfg.values[binding.identity]=member
            cfg.body.bindings.append(binding)
            cfg.record_lifetime('declare',binding.identity,binding.name)
            if binding.ownership == 'view' and owner is not None:
                cfg.flow.loan(binding,owner)
            cfg.value_sources[member.identity]=set(cfg.value_sources.get(self.subject.identity,()))
        return 1


def retain_subject(node,term,subject,cfg,env):
    from py_compiler.syntax.generic.storage import root_binding,owns_storage
    from py_compiler.mir.ownership.flow import Binding
    if subject is None or not owns_storage(subject.type) or root_binding(term,cfg,env) is not None:
        return
    if cfg.call_result_owners.get(subject.identity) is not None:
        return
    name='<match subject '+str(subject.identity)+'>'
    binding=Binding(node.identity+':subject',name,subject.type,False,'own',node.span)
    env[name]=binding
    cfg.local_names.add(name)
    cfg.values[binding.identity]=subject
    cfg.body.bindings.append(binding)
    cfg.record_lifetime('declare',binding.identity,name)


def lower_chain(node,cfg,chain,env):
    from importlib import import_module
    if not node.match_target:
        return import_module('py_compiler.syntax.block.if').Conditional().lower_chain(cfg,chain,env)
    from dataclasses import replace
    from py_compiler.hir.thir.src.model import Edge,Terminator
    from py_compiler.mir.ownership.flow import join_expression
    from py_compiler.syntax.grammar.assignment import Assignment
    from py_compiler.syntax.complex.tensor import ArgumentValue
    target=node.match_target
    expected=None
    if len(target)>2 and target[1].text == ':':
        expected=cfg.context.type(''.join(t.text for t in target[2:-1]))
    elif len(target)==2 and target[0].text in env:
        expected=env[target[0].text].type
    result=MatchResult(expected)
    initial_values,initial_flow=dict(cfg.values),cfg.flow.fork()
    exits=[]
    for branch in chain:
        cfg.values,cfg.flow=dict(initial_values),initial_flow.fork()
        if branch.provider.has_condition:
            condition=branch.provider.condition_value(cfg,branch,env)
            yes,no=cfg.block(),cfg.block()
            cfg.current.terminator=Terminator('conditional',(Edge(yes.identity),Edge(no.identity)),condition)
            initial_values,initial_flow=dict(cfg.values),cfg.flow.fork()
            cfg.current=yes
        children=list(branch.children)
        if not children or children[-1].provider is not None:
            raise ValueError('value-producing case requires a final expression')
        final=children[-1]
        result.value=None
        if final.tokens[0].text not in ('return','throw'):
            children[-1]=replace(final,provider=result)
        cfg.scope(children,dict(env))
        cfg.flow.end_scope(cfg.live_binding_ids(env) | {b.identity for scope in cfg.active_scopes for b in scope.values()})
        if cfg.current.terminator is None:
            if result.value is None:
                raise ValueError('value-producing case requires a final expression')
            exits.append((cfg.current,dict(cfg.values),cfg.flow.fork(),result.value))
        if branch.provider.has_condition:
            cfg.current=no
    if not exits:
        return
    value=join_expression(cfg,exits,env,node.span)
    for facts in (cfg.value_dependencies,cfg.value_sources,cfg.value_origins):
        facts[value.identity]=set().union(*(facts.get(v.identity,()) for _,_,_,v in exits))
    if any(v.identity in cfg.stack_values for _,_,_,v in exits):
        cfg.stack_values.add(value.identity)
    cfg.owned_results.add(value.identity)
    # Feed the joined value through normal binding, annotation, and assignment rules.
    name='__match_value_'+str(value.identity)
    previous=cfg.grammar_captures
    cfg.grammar_captures={**previous,name:ArgumentValue(value)}
    try:
        assignment=replace(node,provider=None,children=[],tokens=(*target,target[0].with_text(name)))
        Assignment().lower(assignment,cfg,env,cfg.local_names)
    finally:
        cfg.grammar_captures=previous


class MatchResult:
    spelling='match-result'

    def __init__(self,expected):
        self.expected,self.value=expected,None

    def lower(self,node,cfg,env,nodes,index):
        from py_compiler.syntax.grammar.expression import expression
        from py_compiler.syntax.generic.storage import owns_storage,copy_storage,root_binding
        term=expression(node.tokens)
        value=cfg.expr(term,env,self.expected)
        if value is None:
            raise ValueError('value-producing case requires a value')
        if self.expected is None:
            self.expected=value.type
        if value.type != self.expected:
            raise ValueError('match arms must produce the same type')
        if owns_storage(value.type) or value.type.family == 'record':
            if value.identity not in cfg.owned_results or root_binding(term,cfg,env) is not None:
                value=copy_storage(value,cfg,node.span)
            cfg.owned_results.add(value.identity)
            cfg.value_sources[value.identity]=set()
        self.value=value
        return 1
