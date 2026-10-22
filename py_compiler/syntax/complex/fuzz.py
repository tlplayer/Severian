"""Seeded bounded fuzz campaigns specialize typed seeds into reproducible cases."""
import ast
import json
import random
import re
from pathlib import Path
import textwrap
from dataclasses import replace
from hashlib import sha256


def cases(node):
    from py_compiler.syntax.block.test import TestHeader
    seeds=[]
    for index,child in enumerate(node.children):
        tokens=child.tokens
        if child.provider or len(tokens)<5 or tokens[1].text!=':':continue
        equal=next((i for i,t in enumerate(tokens) if t.text=='='),None)
        if equal is None:continue
        type_=tokens[2].text
        if type_ not in ('string','int'):raise ValueError('fuzz generators currently require string or int seeds')
        lower,upper=-2147483648,2147483647
        if equal>3:
            constraint=' '.join(t.text for t in tokens[3:equal])
            pattern=r'\{\s*(-?\s*\d+)\s*<=\s*'+re.escape(tokens[0].text)+r'\s+and\s+'+re.escape(tokens[0].text)+r'\s*<=\s*(-?\s*\d+)\s*\}'
            matched=re.fullmatch(pattern,constraint)
            if type_!='int' or not matched:raise ValueError('fuzz integer constraint requires lower <= name and name <= upper')
            lower,upper=(int(value.replace(' ', '')) for value in matched.groups())
            if lower>upper:raise ValueError('fuzz constraints have an empty domain')
        original=ast.literal_eval(''.join(t.text for t in tokens[equal+1:]))
        if type_ == 'int' and not lower <= original <= upper:
            raise ValueError('representative fuzz seed violates its constraint')
        seeds.append((index,tokens[0].text,type_,lower,upper,original))
    if not seeds:raise ValueError('fuzz test requires a typed representative seed')
    campaign=int(sha256((node.identity+node.header.label).encode()).hexdigest()[:16],16)
    rng=random.Random(campaign)
    for iteration in range(17):
        children=list(node.children);inputs={}; checks={}
        for index,name,type_,lower,upper,original in seeds:
            if iteration==0:value=original
            elif type_=='int':value=([lower,upper,max(lower,min(upper,0))][iteration-1] if iteration<=3 else rng.randint(lower,upper))
            else:
                boundary=['','0','-1','+42','abc','9223372036854775808','-9223372036854775808',' 12','12x']
                value=boundary[iteration-1] if iteration<=len(boundary) else ''.join(rng.choice('0123456789abcdefghijklmnopqrstuvwxyz+- \t') for _ in range(rng.randrange(25)))
            inputs[name]=value
            original_node=children[index];tokens=original_node.tokens
            value_token=tokens[-1].with_text(json.dumps(value) if type_=='string' else str(value),kind='STRING' if type_=='string' else 'NUMBER')
            equal=next(i for i,t in enumerate(tokens) if t.text=='=')
            children[index]=replace(original_node,tokens=(*tokens[:3], tokens[equal], value_token))
            if equal > 3:
                check = (tokens[0].with_text('assert', kind='IDENTIFIER'),
                         tokens[3].with_text('(', kind='OPERATOR'), *tokens[4:equal-1],
                         tokens[equal-1].with_text(')', kind='OPERATOR'))
                checks[index] = replace(original_node, identity=original_node.identity + ':constraint', tokens=check)
        children = [item for index, child in enumerate(children)
                    for item in ([child, checks[index]] if index in checks else [child])]
        label=node.header.label+f' [seed {campaign}, case {iteration}: '+json.dumps(inputs,ensure_ascii=False)+']'
        options=(*node.header.options,('fuzz',{'seed':campaign,'case':iteration,'inputs':inputs,
            'strategy':'boundary cases and seeded random values', **regression(node, seeds, inputs)}))
        yield replace(node,header=TestHeader('fuzz',label,node.header.clauses,options=options),children=children), '_fuzz_'+str(iteration)


def regression(node, seeds, inputs):
    source = node.tokens[0].lexeme.source
    children = [child for child in node.children if getattr(child.provider, 'spelling', '') != 'test']
    start = source.text.rfind('\n', 0, children[0].span.start) + 1
    end = children[-1].span.end
    body = source.text[start:end]
    edits = []
    for index, name, type_, lower, upper, original in seeds:
        child = node.children[index]
        equal = next(i for i, token in enumerate(child.tokens) if token.text == '=')
        indent_start = source.text.rfind('\n', 0, child.span.start) + 1
        indent = source.text[indent_start:child.span.start]
        value = name + ': ' + type_ + ' = ' + json.dumps(inputs[name], ensure_ascii=False)
        if equal > 3:
            constraint = ' '.join(token.text for token in child.tokens[4:equal-1])
            value += '\n' + indent + 'assert(' + constraint + ')'
        edits.append((child.span.start - start, child.span.end - start, value))
    for begin, finish, value in sorted(edits, reverse=True):
        body = body[:begin] + value + body[finish:]
    body = textwrap.dedent(body)
    identity = sha256((node.header.label + '\0' + body).encode()).hexdigest()[:20]
    line_start = source.text.rfind('\n', 0, node.span.start) + 1
    indent = source.text[line_start:node.span.start] + '    '
    clauses = (' with ' + ' '.join(token.text for token in node.header.clauses)) if node.header.clauses else ''
    addition = '\n' + indent + 'test "fuzz regression ' + identity + '"' + clauses + ':\n'
    addition += textwrap.indent(body, indent + '    ') + '\n'
    end = source.text.find('\n', node.span.end)
    return {'origin-label': node.header.label, 'source-sha256': sha256(source.text.encode()).hexdigest(),
            'append-offset': len(source.text) if end < 0 else end + 1,
            'regression-id': identity, 'regression-source': addition}


def persist(root, discoveries, build_id=None, *, mode='fuzz'):
    """Append confirmed concrete tests without replacing existing source."""
    from py_compiler.packaging.coverage import artifact_slot
    from py_compiler.packaging.package import write_atomic, encode
    from py_compiler.frontend.source.source import SourceFile
    from py_compiler.frontend.lexer.lexer import lex
    from py_compiler.frontend.parser.blocks import parse_blocks
    from py_compiler.syntax.recognition import Syntax
    root = Path(root).resolve()
    if mode not in ('fuzz','differential'): raise ValueError('unknown discovery mode')
    modes={mode,*(entry.get('mode','fuzz') for entry in discoveries)}
    if not modes <= {'fuzz','differential'}: raise ValueError('unknown discovery mode')
    outputs={kind:root/'package.pkg/debug/test'/kind/artifact_slot(build_id) for kind in modes}
    grouped, results = {}, []
    for entry in discoveries:
        grouped.setdefault(entry['file'], []).append(entry)
    for filename, entries in grouped.items():
        path = (root / filename).resolve()
        result = {'file': filename, 'appended': [], 'pending': []}
        if not path.is_relative_to(root):
            raise ValueError('fuzz source escapes package')
        original = path.read_bytes().decode('utf-8')
        additions, seen = {}, set()
        for entry in entries:
            identity = entry['regression-id']
            kind=entry.get('mode','fuzz')
            if identity in seen or f'"{kind} regression {identity}"' in original:
                continue
            seen.add(identity)
            if sha256(original.encode()).hexdigest() != entry['source-sha256']:
                result['pending'].append(identity)
                continue
            additions.setdefault(entry['append-offset'], []).append(entry)
        updated = original
        for offset, cases in sorted(additions.items(), reverse=True):
            updated = updated[:offset] + ''.join(c['regression-source'] for c in cases) + updated[offset:]
            result['appended'].extend(c['regression-id'] for c in cases)
        if updated != original:
            snapshot, syntax = SourceFile(filename, updated), Syntax(include_tests=True)
            _, diagnostics = parse_blocks(snapshot, lex(snapshot, syntax), syntax)
            try:
                if diagnostics:
                    raise ValueError(str(diagnostics[0]))
                if not path.stat().st_mode & 0o222:
                    raise PermissionError('fuzz source is read-only')
                if path.read_bytes().decode('utf-8') != original:
                    raise ValueError('source changed during fuzz persistence')
                for cases in additions.values():
                    for case in cases:
                        for filename, payload in case.get('fixtures',{}).items():
                            fixture=(root/filename).resolve()
                            if not fixture.is_relative_to(outputs[case.get('mode','fuzz')].resolve()):
                                raise ValueError('fuzz fixture escapes artifact directory')
                            if fixture.exists() and fixture.read_text()!=payload:
                                raise ValueError('fuzz fixture content changed')
                            write_atomic(fixture,payload)
                mode = path.stat().st_mode & 0o777
                write_atomic(path, updated)
                path.chmod(mode)
            except (OSError, ValueError) as error:
                result['pending'].extend(result['appended'])
                result['appended'] = []
                result['error'] = str(error)
        results.append(result)
    for kind,output in outputs.items():
        cases=[entry for entry in discoveries if entry.get('mode','fuzz')==kind]
        identities={entry['regression-id'] for entry in cases}
        sources=[{**result,'appended':[i for i in result['appended'] if i in identities],
                  'pending':[i for i in result['pending'] if i in identities]} for result in results]
        write_atomic(output / 'discoveries.json', encode({'cases': cases, 'sources': sources}))
    return results


import unittest


class FuzzPersistenceTests(unittest.TestCase):
    def test_appends_nested_replay_and_deduplicates(self):
        import tempfile
        from py_compiler.frontend.src.lib import compile_source
        from py_compiler.syntax.recognition import Syntax
        source = 'test with fuzz "bounds":\n    n: int {0 <= n and n <= 10} = 5\n    assert(n <= 10)\n'
        result = compile_source('src/lib.sev', source, Syntax(include_tests=True))
        self.assertFalse(result.diagnostics, str(result.diagnostics))
        info = next(b.test['fuzz'] for b in result.program.bodies if b.test)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / 'src/lib.sev'
            path.parent.mkdir()
            path.write_text(source)
            discovery = {**info, 'file': 'src/lib.sev'}
            self.assertEqual(len(persist(root, [discovery])[0]['appended']), 1)
            saved = path.read_text()
            self.assertIn('    test "fuzz regression ', saved)
            self.assertIn('        assert(0 <= n and n <= 10)', saved)
            checked = compile_source('src/lib.sev', saved, Syntax(include_tests=True))
            self.assertFalse(checked.diagnostics, str(checked.diagnostics))
            self.assertEqual(persist(root, [discovery])[0]['appended'], [])
            self.assertEqual(path.read_text(), saved)

    def test_changed_source_is_preserved(self):
        import tempfile
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / 'lib.sev'
            path.write_text('test:\n    assert(true)\n')
            result = persist(root, [{'file': 'lib.sev', 'source-sha256': 'stale',
                'regression-id': 'case1', 'append-offset': 0, 'regression-source': 'bad'}])
            self.assertEqual(result[0]['pending'], ['case1'])
            self.assertEqual(path.read_text(), 'test:\n    assert(true)\n')
