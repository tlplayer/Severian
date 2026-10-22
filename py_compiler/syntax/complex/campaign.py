"""Lower callable campaign syntax to a c replay harness."""
from dataclasses import dataclass, replace
from hashlib import sha256
from py_compiler.syntax.generic.block import BlockProvider
from py_compiler.syntax.complex.profile import ProfileOperation

RUNTIME = r'''
#include <stdio.h>
static FILE *campaign_input;
int64_t __sev_profile_input(void) {
    if (!campaign_input) { const char *p=getenv("SEV_CAMPAIGN_INPUT"); if (!p) abort(); campaign_input=fopen(p,"rb"); if (!campaign_input) abort(); }
    int64_t value; if(fread(&value,8,1,campaign_input)!=1) abort(); return value;
}
void __sev_profile_save(void) {
    const char *path=getenv("SEV_CAMPAIGN_PROFILE"); if (!path) abort();
    double elapsed=__sev_profile_time(), memory=__sev_profile_memory(); int64_t count=__sev_profile_allocations();
    FILE *out=fopen(path,"wb"); if(!out) abort();
    fwrite(&elapsed,8,1,out); fwrite(&memory,8,1,out); fwrite(&count,8,1,out); fclose(out);
}
void __sev_profile_difference(void) {
    const char *path=getenv("SEV_DIFFERENTIAL_RESULT"); if (!path) abort();
    FILE *out=fopen(path,"wb"); if(!out) abort();
    fputs("different\n",out); fclose(out);
}
'''


class Marker(BlockProvider):
    def __init__(self, metric, name=None):
        self.metric, self.name = metric, name

    def lower(self,node,cfg,env,nodes,index):
        if self.name:
            from py_compiler.mir.ownership.flow import Binding
            value=cfg.emit('campaign-input',cfg.context.type('int'),(),ProfileOperation('input'),node.span)
            binding=Binding(node.identity,self.name,value.type,True,'own',node.span)
            env[self.name]=binding
            cfg.local_names.add(self.name)
            cfg.body.bindings.append(binding)
            cfg.values[binding.identity]=value
        else:
            cfg.effect('profile-'+self.metric,(),ProfileOperation(self.metric),node.span)
        return 1


def expand(node,context):
    from py_compiler.syntax.block.test import TestHeader, groups
    mode=node.header.mode
    operation=mode if mode in ('fuzz','profile','differential') else 'replay'
    calls=[child for child in node.children if child.provider is None and child.tokens and child.tokens[0].text==operation]
    if not calls: return None
    if len(calls)!=1: raise ValueError('one callable campaign is allowed per test block')
    call=calls[0]; tokens=call.tokens
    if len(tokens)<4 or tokens[1].text!='(' or tokens[-1].text!=')':
        raise ValueError(mode+' requires a callable argument')
    arguments=groups(tokens[2:-1])
    targets=[]; entries=[]
    for argument in arguments:
        if len(argument)>1 and argument[1].text=='=': break
        if len(argument)!=1: raise ValueError('campaign requires a named callable')
        candidates=context.lookup_functions(argument[0].text,context.scope)
        if len(candidates)!=1: raise ValueError('campaign requires an unambiguous callable: '+argument[0].text)
        targets.append(argument[0].text); entries.append(candidates[0])
    if not entries: raise ValueError('campaign requires a named callable')
    entry=entries[0]
    fixed=operation=='replay'
    if fixed: mode='fuzz'
    if operation=='differential' or fixed and len(entries)>1:
        mode='differential'
        if len(entries)<2: raise ValueError('differential requires at least two callables')
        if entry.result in (None,'absent'): raise ValueError('differential requires callable return values')
        for other in entries[1:]:
            if [annotation for _,annotation in other.parameters]!=[annotation for _,annotation in entry.parameters]:
                raise ValueError('differential callables require matching parameter types and arity')
            if other.result!=entry.result:
                raise ValueError('differential callables require matching result types')
    elif len(entries)!=1:
        raise ValueError(operation+' requires exactly one callable')
    times=100 if mode in ('fuzz','differential') else 10
    fixture=None
    replay_cases = None
    seen_options = set()
    for option in arguments[len(entries):]:
        if len(option)<3 or option[1].text!='=' or option[0].text in seen_options:
            raise ValueError('invalid or duplicate campaign option')
        key = option[0].text
        seen_options.add(key)
        if key == 'times' and len(option)==3 and option[2].text.isdecimal() and int(option[2].text)>0:
            times=int(option[2].text)
        elif key == 'profile' and fixed and mode!='differential' and len(option)==3 and option[2].text in ('true','false'):
            mode='profile' if option[2].text=='true' else 'fuzz'
        elif key == 'fixture' and fixed:
            import ast
            fixture=ast.literal_eval(' '.join(token.text for token in option[2:]))
        elif key == 'cases':
            import ast
            replay_cases=ast.literal_eval(' '.join(token.text for token in option[2:]).replace('true','True').replace('false','False'))
            if not isinstance(replay_cases,list) or not replay_cases:
                raise ValueError('campaign replay cases must be a nonempty list of argument lists')
        else:
            raise ValueError('campaign options are times=<positive integer> and cases=<captured argument lists>')
    if fixture is not None:
        if replay_cases is not None: raise ValueError('replay accepts either cases or fixture')
        from py_compiler.packaging.campaigns import load_fixture
        replay_cases=load_fixture(node.tokens[0].lexeme.source,fixture)
    if fixed and replay_cases is None: raise ValueError('replay requires cases or fixture')
    from py_compiler.packaging.campaigns import domain,constrain
    domains={name:domain(annotation) for name,annotation in entry.parameters}
    for candidate in entries:
        names=dict(zip((name for name,_ in candidate.parameters),domains))
        def spelling(term):
            if term.kind=='name': return names.get(term.token.text,term.token.text)
            if term.kind=='literal': return term.token.text
            if term.kind=='call': return spelling(term.operands[0])+'('+','.join(spelling(x) for x in term.operands[1:])+')'
            if term.kind=='chain': return ' '.join(spelling(x) if hasattr(x,'operands') else x.text for x in term.operands)
            if term.kind=='binary': return '('+spelling(term.operands[0])+' '+term.token.text+' '+spelling(term.operands[1])+')'
            if term.kind=='unary': return term.token.text+spelling(term.operands[0])
            raise ValueError('unsupported campaign constraint expression '+term.kind)
        for guard in candidate.guards:
            constrain(domains,spelling(guard))
    from py_compiler.frontend.source.source import SourceFile
    from py_compiler.frontend.lexer.lexer import lex
    from py_compiler.frontend.parser.blocks import parse_blocks
    lines=[]; markers={}; counter=0
    def marker(metric,name=None,indent=''):
        nonlocal counter
        key='__campaign_marker_'+str(counter);counter+=1
        markers[key]=Marker(metric,name);lines.append(indent+key)
    def parameter(name,bounds,indent=''):
        if 'element' in bounds:
            length=name+'_length'; item=name+'_item'
            marker('input',length,indent)
            lines.append(indent+name+': '+bounds['type']+' = []')
            lines.append(indent+name+'_index = 0')
            lines.append(indent+'while '+name+'_index < '+length+':')
            parameter(item,bounds['element'],indent+'    ')
            lines.append(indent+'    '+name+'.append('+item+')')
            lines.append(indent+'    '+name+'_index += 1')
        else:
            raw=name+'_raw';marker('input',raw,indent)
            lines.append(indent+name+': '+bounds['type']+' = '+('bool('+raw+')' if bounds['type']=='bool' else raw))
    for index,(target,candidate) in enumerate(zip(targets,entries)):
        arguments=[]
        for position,bounds in enumerate(domains.values()):
            name='__campaign_input_'+str(index)+'_'+str(position)
            parameter(name,bounds)
            arguments.append(name)
        if mode=='profile': marker('begin')
        result='__campaign_result_'+str(index)
        lines.append((result+' = ' if candidate.result not in (None,'absent') else '')+target+'('+', '.join(arguments)+')')
    if mode=='differential':
        for index in range(1,len(entries)):
            lines.append('if __campaign_result_0 != __campaign_result_'+str(index)+':')
            marker('difference',indent='    ')
    if mode=='profile': marker('save')
    source=SourceFile(node.tokens[0].lexeme.source.path+':campaign','\n'.join(lines)+'\n')
    root,errors=parse_blocks(source,lex(source,node.syntax),node.syntax)
    if errors: raise ValueError(str(errors[0]))
    def attach(children):
        for child in children:
            child.scope=node.scope
            if child.tokens[0].text in markers: child.provider=markers[child.tokens[0].text]
            attach(child.children)
    attach(root.children)
    original=node.tokens[0].lexeme.source
    start=original.text.rfind('\n',0,node.span.start)+1
    end=original.text.find('\n',node.span.end)
    spec={'mode':mode,'function':targets[0],'times':times,'domains':domains,'parameters':list(domains),
          'result':entry.result,'complexity':entry.complexity,'label':node.header.label,
          'indent':original.text[start:node.span.start]+'    ',
          'persistence':{'source-sha256':sha256(original.text.encode()).hexdigest(),
                         'append-offset':len(original.text) if end<0 else end+1}}
    if mode=='differential': spec['functions']=targets
    if replay_cases is not None:
        from py_compiler.packaging.campaigns import validate_cases
        spec['replay-cases'] = validate_cases(spec, replay_cases)
        spec['replay-only'] = True
    children=[]
    for child in node.children:
        children.extend(root.children if child is call else [child])
    return replace(node,children=children,header=TestHeader(mode,node.header.label,node.header.clauses,
                   options=(*node.header.options,('campaign',spec))))


import unittest

class CallableCampaignTests(unittest.TestCase):
    def test_callable_lowering_records_domains_and_c_harness(self):
        from py_compiler.frontend.src.lib import compile_source
        from py_compiler.syntax.recognition import Syntax
        for mode in ('fuzz','profile'):
            result=compile_source('campaign.sev','def square(n: int) -> int with 0 <= n < 17:\n    return n*n\ntest with '+mode+' "campaign":\n    '+mode+'(square, times=3)\n',Syntax(include_tests=True))
            self.assertFalse(result.diagnostics,str(result.diagnostics))
            metadata=next(body.test for body in result.program.bodies if body.test)['campaign']
            self.assertEqual(metadata['domains']['n']['upper'],16)
            self.assertEqual(metadata['times'],3)

    def test_fixed_replay_lowering_without_campaign_block(self):
        from py_compiler.frontend.src.lib import compile_source
        from py_compiler.syntax.recognition import Syntax
        for options in ('cases=[[3]]','cases=[[3]], profile=true, times=2'):
            result=compile_source('replay.sev','def square(n: int) -> int:\n    return n*n\ntest "captured":\n    replay(square, '+options+')\n',Syntax(include_tests=True))
            self.assertFalse(result.diagnostics,str(result.diagnostics))
            metadata=next(body.test for body in result.program.bodies if body.test)['campaign']
            self.assertTrue(metadata['replay-only'])
            self.assertEqual(list(metadata['replay-cases']),[{'n':3}])

class DifferentialLoweringTests(unittest.TestCase):
    def compile(self, source):
        from py_compiler.frontend.src.lib import compile_source
        from py_compiler.syntax.recognition import Syntax
        return compile_source('differential.sev',source,Syntax(include_tests=True))

    def test_intersects_constraints_by_position_and_supports_multiple_functions(self):
        source='''def first(n: int) -> int with 0 <= n < 20:
    return n
def second(value: int) -> int with 5 <= value < 10:
    return value
def third(item: int) -> int with 6 <= item <= 8:
    return item
test with memory, differential "same domain":
    differential(first, second, third, times=7)
'''
        result=self.compile(source)
        self.assertFalse(result.diagnostics,str(result.diagnostics))
        test=next(body.test for body in result.program.bodies if body.test)
        self.assertTrue(test['memory'])
        spec=test['campaign']
        self.assertEqual(spec['mode'],'differential')
        self.assertEqual(list(spec['functions']),['first','second','third'])
        self.assertEqual((spec['domains']['n']['lower'],spec['domains']['n']['upper']),(6,8))
        self.assertEqual(spec['times'],7)

    def test_rejects_incompatible_or_missing_callables(self):
        cases=[
            ('def second(value: bool) -> int:\n    return 1','first, second','matching parameter types'),
            ('def second(value: int) -> bool:\n    return true','first, second','matching result types'),
            ('def second(value: int) -> int with value > 20:\n    return value','first, second','empty domain'),
            ('def second(value: int) -> int:\n    return value','first','at least two callables'),
        ]
        for declaration,arguments,message in cases:
            with self.subTest(message=message):
                result=self.compile('def first(n: int) -> int with 0 <= n < 10:\n    return n\n'+declaration+'\ntest with differential:\n    differential('+arguments+')\n')
                self.assertTrue(result.diagnostics)
                self.assertIn(message,str(result.diagnostics))

    def test_fixed_differential_replay_checks_both_functions_and_bounds(self):
        definitions='def first(n: int) -> int with 0 <= n < 10:\n    return n\ndef second(value: int) -> int with 2 <= value < 5:\n    return value\n'
        result=self.compile(definitions+'test "captured":\n    replay(first, second, cases=[[3]])\n')
        self.assertFalse(result.diagnostics,str(result.diagnostics))
        spec=next(body.test['campaign'] for body in result.program.bodies if body.test)
        self.assertTrue(spec['replay-only'])
        self.assertEqual(list(spec['functions']),['first','second'])
        result=self.compile(definitions+'test:\n    replay(first, second, cases=[[1]])\n')
        self.assertIn('violates parameter constraints',str(result.diagnostics))
