"""Callable campaigns: typed domains, c input replay and measured growth."""
import ast
import json
import math
import random
import struct
import time
from pathlib import Path


def domain(annotation):
    if annotation == 'bool':
        return {'type': annotation, 'lower': 0, 'upper': 1}
    if annotation == 'int' or annotation.startswith(('i', 'u')) and annotation[1:].isdigit():
        bits = 64 if annotation == 'int' else int(annotation[1:])
        unsigned = annotation.startswith('u')
        if bits > 64:
            raise ValueError('campaign integer transport supports up to 64 bits')
        return {'type': annotation, 'lower': 0 if unsigned else -(1 << (bits-1)),
                'upper': (1 << (bits if unsigned else bits-1))-1}
    if annotation.startswith('list[') and annotation.endswith(']'):
        return {'type': annotation, 'lower': 0, 'upper': None, 'element': domain(annotation[5:-1])}
    raise ValueError('no campaign input generator for ' + str(annotation))


def constrain(domains, predicate):
    """Intersect comparison bounds; reject unsupported constraints rather than ignore them."""
    def target(term):
        if isinstance(term, ast.Name) and term.id in domains and 'element' not in domains[term.id]:
            return domains[term.id]
        if isinstance(term, ast.Call) and isinstance(term.func, ast.Name) and term.func.id == 'len' and len(term.args)==1 and isinstance(term.args[0], ast.Name):
            value = domains.get(term.args[0].id)
            if value is not None and 'element' in value:
                return value
        return None
    def visit(term):
        if isinstance(term, ast.BoolOp) and isinstance(term.op, ast.And):
            for value in term.values: visit(value)
            return
        if not isinstance(term, ast.Compare):
            raise ValueError('campaign constraint needs a supported comparison: ' + predicate)
        for left, operation, right in zip([term.left, *term.comparators], term.ops, term.comparators):
            bounds = target(left)
            reverse = bounds is None
            if reverse: bounds = target(right)
            if bounds is None:
                raise ValueError('campaign cannot derive this parameter constraint: ' + predicate)
            try: constant = ast.literal_eval(left if reverse else right)
            except (ValueError, TypeError):
                raise ValueError('campaign joint constraints require a generator: ' + predicate) from None
            if type(constant) is not int:
                raise ValueError('campaign bound must be an integer')
            kind = type(operation)
            if reverse: kind = {ast.Lt:ast.Gt, ast.LtE:ast.GtE, ast.Gt:ast.Lt, ast.GtE:ast.LtE, ast.Eq:ast.Eq}.get(kind)
            if kind in (ast.Gt, ast.GtE, ast.Eq):
                bounds['lower'] = max(bounds['lower'], constant + (kind is ast.Gt))
            if kind in (ast.Lt, ast.LtE, ast.Eq):
                upper = constant - (kind is ast.Lt)
                bounds['upper'] = min(bounds['upper'], upper) if bounds['upper'] is not None else upper
            if kind not in (ast.Gt, ast.GtE, ast.Lt, ast.LtE, ast.Eq):
                raise ValueError('unsupported campaign comparison: ' + predicate)
            if bounds['upper'] is not None and bounds['lower'] > bounds['upper']:
                raise ValueError('campaign constraint has an empty domain')
    visit(ast.parse(predicate, mode='eval').body)


def magnitudes(bounds):
    lo, hi = bounds['lower'], bounds['upper']
    if hi is not None and hi < 1:
        yield hi
        return
    n = 1
    while hi is None or n <= hi:
        if n >= lo: yield n
        n *= 2
    if hi is not None and hi >= lo and hi != n//2:
        yield hi


class GenerationBudgetExpired(Exception):
    pass


def value(bounds, magnitude, rng, pattern='random', deadline=None):
    lo, hi = bounds['lower'], bounds['upper']
    n = max(lo, magnitude)
    if hi is not None: n = min(n, hi)
    if 'element' not in bounds:
        return bool(n) if bounds['type']=='bool' else n
    element = bounds['element']
    values = []
    lower = max(element['lower'], -max(1,n))
    upper = min(element['upper'] if element['upper'] is not None else n, max(1,n))
    if upper < lower: upper = lower
    for index in range(n):
        if deadline is not None and index % 256 == 0 and time.monotonic() >= deadline:
            raise GenerationBudgetExpired()
        values.append(value(element, rng.randint(lower, upper), rng, deadline=deadline))
    if pattern == 'ascending': values.sort()
    elif pattern == 'descending': values.sort(reverse=True)
    elif pattern == 'duplicates': values = [value(element, 0, rng)] * n
    return values


def inputs(spec, corpus=None, deadline=None):
    rng = random.Random()
    domains = spec['domains']
    if spec.get('replay-cases'):
        for arguments in spec['replay-cases']:
            yield arguments, {'sizes':{name:len(v) if isinstance(v,list) else int(v) for name,v in arguments.items()},'pattern':'captured'}
        return
    if spec['mode'] in ('fuzz','differential'):
        for index in range(spec['times']):
            arguments = {}
            for name, bounds in domains.items():
                lo, hi = bounds['lower'], bounds['upper']
                if index == 0: n = max(lo, min(0, hi)) if hi is not None else lo
                elif index == 1: n = lo
                elif index == 2 and hi is not None: n = hi
                else:
                    # Unbounded domains grow until the campaign execution budget stops them.
                    upper = hi if hi is not None else max(lo, 1 << index)
                    n = rng.randint(lo, upper)
                arguments[name] = value(bounds, n, rng, deadline=deadline)
            if corpus and domains and index > 2 and rng.randrange(2):
                import copy
                arguments = copy.deepcopy(rng.choice(corpus))
                name = rng.choice(list(domains)); bounds = domains[name]
                if isinstance(arguments[name],list) and arguments[name] and 'element' not in bounds['element']:
                    position=rng.randrange(len(arguments[name]))
                    arguments[name][position]=rng.randint(bounds['element']['lower'],bounds['element']['upper'])
                elif 'element' not in bounds:
                    arguments[name]=rng.randint(bounds['lower'],bounds['upper'])
            yield arguments, None
    else:
        streams = {name: iter(magnitudes(bounds)) for name,bounds in domains.items()}
        current = {name: bounds['lower'] for name,bounds in domains.items()}
        while streams:
            advanced = []
            for name, stream in list(streams.items()):
                try: current[name] = next(stream); advanced.append(name)
                except StopIteration: del streams[name]
            if not advanced: break
            # Separate axes, then vary all parameters together.
            variations = [{**{k:v['lower'] for k,v in domains.items()}, name:current[name]} for name in advanced] if len(domains)>1 else []
            variations.append(dict(current))
            for sizes in variations:
                for pattern in ('random','ascending','descending','duplicates') if any('element' in d for d in domains.values()) else ('random',):
                    yield {name:value(bounds,sizes[name],rng,pattern,deadline) for name,bounds in domains.items()}, {'sizes':sizes,'pattern':pattern}


def encode_inputs(arguments):
    values = []
    def flatten(value):
        if isinstance(value,list):
            values.append(len(value))
            for item in value: flatten(item)
        else: values.append(int(value))
    for argument in arguments.values(): flatten(argument)
    return b''.join(struct.pack('=Q', n & ((1<<64)-1)) for n in values)


def reductions(arguments, domains):
    for name, original in arguments.items():
        bounds = domains[name]
        if isinstance(original, list):
            length = len(original)//2
            while length >= bounds['lower'] and length < len(original):
                yield {**arguments, name:original[:length]}
                if length == 0: break
                length //= 2
            for index, item in enumerate(original):
                if item != 0 and bounds['element']['lower'] <= 0 <= bounds['element']['upper']:
                    yield {**arguments,name:original[:index]+[0]+original[index+1:]}
        elif original != 0:
            candidate = max(bounds['lower'], min(bounds['upper'], 0))
            if candidate != original: yield {**arguments,name:candidate}


def load_fixture(source, filename):
    if not isinstance(filename,str): raise ValueError('replay fixture must be a path string')
    origin=Path(source.debug_path or source.path).resolve()
    root=next((p for p in origin.parents if (p/'package.json').is_file()),None)
    if root is None: raise ValueError('replay fixture requires a package source path')
    path=(root/filename).resolve()
    if not path.is_relative_to(root): raise ValueError('replay fixture escapes package')
    try:
        payload=json.loads(path.read_text())
    except (OSError,ValueError) as error:
        raise ValueError('cannot load replay fixture '+filename+': '+str(error)) from error
    if not isinstance(payload,dict) or payload.get('version')!=1 or not isinstance(payload.get('cases'),list) or not payload['cases']:
        raise ValueError('invalid replay fixture '+filename)
    return payload['cases']


def replay_source(function, cases, *, profile=False, times=10, slot='dev', functions=None):
    """A fixed invocation; large input data lives in a content-addressed fixture."""
    from hashlib import sha256
    payload=json.dumps({'version':1,'cases':cases},separators=(',',':'))+'\n'
    literal=json.dumps(cases)
    fixtures={}
    if len(literal)>2000:
        mode='differential' if functions else 'fuzz'
        name='package.pkg/debug/test/'+mode+'/'+slot+'/cases/'+sha256(payload.encode()).hexdigest()+'.json'
        fixtures[name]=payload
        arguments='fixture='+json.dumps(name)
    else:
        arguments='cases='+literal
    options=', profile=true, times='+str(times) if profile else ''
    return 'replay('+', '.join(functions or [function])+', '+arguments+options+')',fixtures


def captured_regression(spec, cases, reason, *, profile=False):
    from hashlib import sha256
    ordered=[[case[name] for name in spec.get('parameters',spec['domains'])] for case in cases]
    functions=spec.get('functions')
    mode='differential' if functions else 'fuzz'
    body,fixtures=replay_source(spec['function'],ordered,profile=profile,times=spec['times'],slot=spec.get('artifact-slot','dev'),functions=functions)
    identity=sha256(json.dumps([functions or spec['function'],ordered,profile,spec['times'] if profile else None]).encode()).hexdigest()[:20]
    indent=spec['indent']
    header='test with timeout('+str(max(1,round(spec['timeout']*1000)))+'ms)' if reason=='failure' and 'timeout' in spec else 'test'
    addition='\n'+indent+header+' "'+mode+' regression '+identity+'":\n'+indent+'    '+body+'\n'
    return {**spec['persistence'],'origin-label':spec['label'],'regression-id':identity,
            'regression-source':addition,'fixtures':fixtures,'inputs':cases,'reason':reason,'mode':mode}


def regression(spec, arguments, *, reason, extra=''):
    return captured_regression(spec,[arguments],reason)


def run(binary, spec, execute, timeout, env, coverage_file=None):
    """One compiled harness, fresh process/input per invocation; bounded replay/shrinking."""
    from py_compiler.packaging.coverage import read_hits
    spec = {**spec, 'timeout':timeout}
    import statistics
    path = Path(str(binary)+'.campaign-input.bin')
    metrics = Path(str(binary)+'.campaign-profile.bin')
    difference = Path(str(binary)+'.differential-result')
    records, discoveries, seen = [], [], set()
    started = time.monotonic()
    budget = timeout
    def invoke(arguments):
        ordered = {name:arguments[name] for name in spec.get('parameters',spec['domains'])}
        path.write_bytes(encode_inputs(ordered)*len(spec.get('functions',[None])))
        metrics.unlink(missing_ok=True); difference.unlink(missing_ok=True)
        if coverage_file: Path(coverage_file).unlink(missing_ok=True)
        settings = {**env,'SEV_CAMPAIGN_INPUT':str(path),'SEV_CAMPAIGN_PROFILE':str(metrics)}
        if spec['mode']=='differential': settings['SEV_DIFFERENTIAL_RESULT']=str(difference)
        if coverage_file: settings['SEV_COVERAGE_FILE'] = str(coverage_file)
        import subprocess
        try:
            result = execute(binary, timeout, settings, seekable_stdout=True)
            record = {'passed':result.returncode==0,'exit':result.returncode,'stdout':result.stdout,'stderr':result.stderr}
            if result.returncode < 0:
                import signal
                record['error'] = 'c signal '+signal.Signals(-result.returncode).name
            if spec['mode']=='differential' and result.returncode==0 and difference.exists():
                record.update(passed=False,difference=True,error='differential results differ')
        except subprocess.TimeoutExpired as failure:
            from py_compiler.packaging.testing import timeout_output
            record = {'passed':False,'exit':None,'error':f'campaign invocation exceeded its {timeout:g}s deadline',
                      'stdout':timeout_output(failure.output),'stderr':timeout_output(failure.stderr)}
        if metrics.exists() and metrics.stat().st_size==24:
            elapsed,memory,allocations = struct.unpack('=ddq',metrics.read_bytes())
            record['metrics'] = {'seconds':elapsed,'memory-bits':memory,'allocations':allocations}
        hits = set(read_hits(coverage_file)) if coverage_file and Path(coverage_file).exists() else set()
        return record,hits
    corpus=[]
    generated=iter(inputs(spec,corpus,started+budget))
    while time.monotonic()-started < budget:
        try: arguments,scale=next(generated)
        except (StopIteration, GenerationBudgetExpired): break
        samples=[]; record={}
        for iteration in range(spec['times']+3 if spec['mode']=='profile' else 1):
            record,hits=invoke(arguments)
            if not record['passed']: break
            if iteration>=3 and 'metrics' in record: samples.append(record['metrics'])
            if time.monotonic()-started>=budget: break
        record.update(inputs=arguments,scale=scale)
        if samples:
            record['measurements']={key:statistics.median(sample[key] for sample in samples) for key in samples[0]}
            record['samples']=len(samples)
        new=hits-seen; seen.update(hits)
        interesting = record.get('difference',False) if spec['mode']=='differential' else not record['passed'] or new
        if not spec.get('replay-cases') and not spec.get('replay-only') and interesting and time.monotonic()-started < 2*budget:
            confirmed,confirmed_hits=invoke(arguments)
            if confirmed['passed']==record['passed'] and confirmed.get('exit')==record.get('exit') and confirmed.get('error')==record.get('error') and confirmed.get('stdout')==record.get('stdout') and confirmed.get('stderr')==record.get('stderr') and new<=confirmed_hits:
                best=arguments
                for attempt,candidate in enumerate(reductions(best,spec['domains'])):
                    if attempt >= 32 or time.monotonic()-started>=2*budget: break
                    trial,trial_hits=invoke(candidate)
                    if trial['passed']==record['passed'] and trial.get('exit')==record.get('exit') and trial.get('error')==record.get('error') and trial.get('stdout')==record.get('stdout') and trial.get('stderr')==record.get('stderr') and (not record['passed'] or new<=trial_hits):
                        best=candidate
                discoveries.append(regression(spec,best,reason='difference' if spec['mode']=='differential' else 'coverage' if record['passed'] else 'failure'))
                corpus.append(best)
        elif spec['mode']=='differential' and record['passed'] and new:
            corpus.append(arguments)
        records.append(record)
        if not record['passed']: break
    if coverage_file:
        Path(coverage_file).write_bytes(b''.join(struct.pack('=Q', hit) for hit in sorted(seen)))
    report = {'passed':bool(records) and all(r['passed'] for r in records),'cases':records,
              'budget-seconds':budget,'budget-exhausted':time.monotonic()-started>=budget}
    if spec['mode']=='profile':
        report['growth']=growth(spec,records)
        report['passed'] = report['passed'] and not any(item['result']=='violated' for item in report['growth'])
        for metric in report['growth']:
            violating = next((o for o in metric.get('observations',[]) if o['result']=='violated'),None)
            if violating and not spec.get('replay-cases') and not spec.get('replay-only'):
                discoveries.append(growth_regression(spec, violating['replay-inputs'], metric['metric']))
        report['profile-report']=save_profile(binary,spec,report)
    return report,discoveries


import unittest

class CampaignTests(unittest.TestCase):
    def test_constraint_boundary_and_powers(self):
        domains={'values':domain('list[int]')}
        constrain(domains,'len(values) < 10000')
        self.assertEqual(list(magnitudes(domains['values'])),[1,2,4,8,16,32,64,128,256,512,1024,2048,4096,8192,9999])
        self.assertEqual(domains['values']['lower'],0)

    def test_constraints_are_not_silently_ignored(self):
        with self.assertRaises(ValueError): constrain({'n':domain('int')}, 'n != 3')
        with self.assertRaises(ValueError): constrain({'n':domain('int')}, '1 < n < 0')

    def test_c_encoding_preserves_negative_and_empty_inputs(self):
        self.assertEqual(struct.unpack('=qqq',encode_inputs({'a':[],'b':[ -7]})),(0,1,-7))


def growth(spec, records):
    """Conservative empirical comparison; timing noise never establishes a proof."""
    groups={}
    for record in records:
        scale=record.get('scale'); measured=record.get('measurements')
        if not scale or not measured or record.get('samples',0)<spec['times']: continue
        sizes=scale['sizes']
        # Each changing axis has its own series; joint changes are another series.
        axes=tuple(name for name,n in sizes.items() if n>spec['domains'][name]['lower'])
        groups.setdefault((scale['pattern'],axes),[]).append(record)
    results=[]
    for metric,contract in (('seconds','time'),('memory-bits','space')):
        rank=spec['complexity'].get(contract)
        if rank not in (0,1,2):
            results.append({'metric':metric,'result':'inconclusive','reason':'no supported polynomial bound declared'})
            continue
        observations=[]
        for (pattern,axes),series in groups.items():
            usable=[r for r in series if max(r['scale']['sizes'].values(),default=0)>0 and r['measurements'][metric]>(0.00005 if metric=='seconds' else 0)]
            if len(usable)<3: continue
            first=usable[0]; n0=max(first['scale']['sizes'].values()); y0=first['measurements'][metric]
            for last in usable[2:]:
                n1=max(last['scale']['sizes'].values()); y1=last['measurements'][metric]
                if n1<4*n0: continue
                expected=y0*(n1/n0)**rank
                observations.append({'pattern':pattern,'axes':axes,'from':n0,'to':n1,
                    'baseline':y0,'observed':y1,'limit':expected*4,
                    'result':'violated' if y1>expected*4 else 'consistent',
                    'inputs':last['inputs'],'baseline-inputs':first['inputs'],
                    'replay-inputs':[first['inputs'],usable[1]['inputs'],last['inputs']]})
        results.append({'metric':metric,'declared-exponent':rank,'observations':observations,
                        'result':'violated' if any(o['result']=='violated' for o in observations) else 'consistent' if observations else 'inconclusive'})
    return results


def save_profile(binary, spec, campaign):
    from py_compiler.packaging.package import write_atomic,encode
    from xml.sax.saxutils import escape
    directory=Path(binary).parent.parent / 'profile' / spec.get('artifact-slot','dev') / Path(binary).name
    write_atomic(directory/'results.json',encode(campaign))
    for metric in ('seconds','memory-bits','allocations'):
        series={}
        for record in campaign['cases']:
            if not record.get('measurements'): continue
            scale=record['scale']; n=max(scale['sizes'].values(),default=0)
            if n>0: series.setdefault(scale['pattern'],[]).append((n,record['measurements'][metric]))
        if not series: continue
        xmax=max(math.log2(n) for points in series.values() for n,_ in points) or 1
        ymax=max(v for points in series.values() for _,v in points) or 1
        svg=['<svg xmlns="http://www.w3.org/2000/svg" width="800" height="440" viewBox="0 0 800 440">',
             '<rect width="800" height="440" fill="white"/>',
             '<path d="M60 40V380H780" fill="none" stroke="black"/>',
             '<text x="60" y="22">'+escape(spec['function']+' — '+metric)+'</text>',
             '<text x="300" y="425">Input magnitude (log2), max '+str(2**xmax)+'</text>',
             '<text x="65" y="55">Peak '+str(ymax)+'</text>']
        for index,(pattern,points) in enumerate(series.items()):
            color=('blue','green','red','purple')[index%4]
            coords=' '.join(f'{60+math.log2(n)/xmax*700:.2f},{380-v/ymax*320:.2f}' for n,v in points)
            svg.append(f'<polyline points="{coords}" fill="none" stroke="{color}"/>')
            svg.append(f'<text x="{70+index*170}" y="405" fill="{color}">{escape(pattern)}</text>')
        write_atomic(directory/(metric+'.svg'),'\n'.join(svg)+'</svg>')
    return str(directory/'results.json')


def validate_cases(spec, cases):
    def valid(value,bounds):
        if 'element' in bounds:
            return isinstance(value,list) and bounds['lower'] <= len(value) and (bounds['upper'] is None or len(value)<=bounds['upper']) and all(valid(item,bounds['element']) for item in value)
        return isinstance(value,(int,bool)) and bounds['lower'] <= value <= bounds['upper']
    results=[]
    for case in cases:
        if not isinstance(case,list) or len(case)!=len(spec['domains']):
            raise ValueError('captured case does not match callable arity')
        arguments=dict(zip(spec.get('parameters',spec['domains']),case))
        if not all(valid(arguments[name],bounds) for name,bounds in spec['domains'].items()):
            raise ValueError('captured case violates parameter constraints')
        results.append(arguments)
    return results


def growth_regression(spec, cases, metric):
    return captured_regression(spec,cases,'complexity-'+metric,profile=True)


class CampaignReplayTests(unittest.TestCase):
    def test_captured_cases_check_domain(self):
        spec={'domains':{'n':domain('int')}}
        constrain(spec['domains'],'0 <= n < 4')
        self.assertEqual(validate_cases(spec,[[3]]),[{'n':3}])
        with self.assertRaises(ValueError): validate_cases(spec,[[4]])

    def test_growth_requires_complete_samples_and_captures_three_inputs(self):
        spec={'times':3,'domains':{'n':domain('int')},'complexity':{'space':0,'time':2}}
        records=[{'inputs':{'n':n},'samples':3,'scale':{'sizes':{'n':n},'pattern':'random'},
                  'measurements':{'seconds':n*n*.001,'memory-bits':n*n}} for n in (1,2,8)]
        report=growth(spec,records)
        self.assertEqual([r['result'] for r in report],['consistent','violated'])
        self.assertEqual(len(report[1]['observations'][0]['replay-inputs']),3)
        records[-1]['samples']=1
        self.assertEqual(growth(spec,records)[1]['result'],'inconclusive')

    def test_unbounded_generation_obeys_execution_budget(self):
        with self.assertRaises(GenerationBudgetExpired):
            value(domain('list[int]'),10**12,random.Random(),deadline=time.monotonic()-1)

    @unittest.expectedFailure
    def test_joint_parameter_constraints_pending(self):
        domains={'low':domain('int'),'high':domain('int')}
        constrain(domains,'low < high')

    @unittest.expectedFailure
    def test_float_generator_pending(self):
        self.assertEqual(domain('float')['type'],'float')

    def test_c_callable_campaign_replays_and_persists(self):
        import tempfile
        from py_compiler.packaging.testing import test_package
        source='''def checked(n: int) -> int with 0 <= n < 4:
    if n == 3:
        panic("bad input")
    return n

def work(n: int) -> int with {
    0 <= n <= 8,
    work.complexity.time := BigO.linear,
    work.complexity.space := BigO.constant,
}:
    result = 0
    i = 0
    while i < n:
        result += i
        i += 1
    return result

test with fuzz "failure replay":
    fuzz(checked, times=1, cases=[[3]])

test with profile "growing inputs":
    profile(work, times=2)
'''
        with tempfile.TemporaryDirectory(prefix='sev-callable-campaign-') as directory:
            root=Path(directory)
            (root/'lib.sev').write_text(source)
            (root/'package.json').write_text(json.dumps({'package':{'name':'campaign','version':'0.1.0'},'sources':['lib.sev']}))
            result=test_package(root,fuzz=True,profile=True,timeout=2)
            self.assertFalse(result['success'],result)
            self.assertIn('report',result,result)
            report=json.loads(Path(result['report']).read_text())
            campaigns={t['mode']:t['campaign'] for t in report['tests']}
            self.assertEqual(campaigns['fuzz']['cases'][0]['inputs'],{'n':3})
            self.assertTrue(campaigns['profile']['passed'],campaigns['profile'])
            self.assertEqual([r['scale']['sizes']['n'] for r in campaigns['profile']['cases']],[1,2,4,8])
            self.assertEqual(source,(root/'lib.sev').read_text())
            self.assertTrue((root/'package.pkg/debug/test/coverage/dev/summary.json').exists())


class ArgumentOrderTests(unittest.TestCase):
    def test_metadata_key_sorting_does_not_reorder_c_arguments(self):
        import tempfile
        import subprocess
        spec=json.loads(json.dumps({'mode':'fuzz','times':1,'parameters':['z','a'],
            'domains':{'z':domain('int'),'a':domain('int')},
            'replay-cases':[{'z':2,'a':1}]},sort_keys=True))
        observed=[]
        def execute(binary,timeout,env,**options):
            observed.append(struct.unpack('=qq',Path(env['SEV_CAMPAIGN_INPUT']).read_bytes()))
            return subprocess.CompletedProcess([],0,'','')
        with tempfile.TemporaryDirectory() as directory:
            result,_=run(Path(directory)/'test',spec,execute,1,{})
        self.assertTrue(result['passed'])
        self.assertEqual(observed,[(2,1)])


class FixedReplayTests(unittest.TestCase):
    def test_failure_and_new_coverage_do_not_discover_cases(self):
        import tempfile
        import subprocess
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            trace=root/'coverage.bin'
            calls=[]
            def execute(binary,timeout,env,**options):
                calls.append(1)
                Path(env['SEV_COVERAGE_FILE']).write_bytes(struct.pack('=Q',42))
                return subprocess.CompletedProcess([],1,'','failure')
            spec={'mode':'fuzz','times':1,'domains':{'n':domain('int')},'replay-cases':[{'n':3}]}
            result,discoveries=run(root/'test',spec,execute,1,{},trace)
            self.assertFalse(result['passed'])
            self.assertEqual(discoveries,[])
            self.assertEqual(len(calls),1)

    def test_large_fixture_persistence_and_loading(self):
        import tempfile
        from hashlib import sha256
        from py_compiler.syntax.complex.fuzz import persist
        from py_compiler.frontend.source.source import SourceFile
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            (root/'package.json').write_text('{}')
            path=root/'main.sev'
            source='test with fuzz "original":\n    fuzz(work, times=1)\n'
            path.write_text(source)
            spec={'function':'work','times':1,'domains':{'values':domain('list[int]')},
                  'indent':'    ','label':'original','persistence':{'source-sha256':sha256(source.encode()).hexdigest(),'append-offset':len(source)}}
            discovery=regression(spec,{'values':list(range(1000))},reason='coverage')
            self.assertLess(len(discovery['regression-source']),400)
            self.assertNotIn('test with fuzz',discovery['regression-source'])
            self.assertEqual(len(persist(root,[{**discovery,'file':'main.sev'}])[0]['appended']),1)
            filename=next(iter(discovery['fixtures']))
            self.assertEqual(load_fixture(SourceFile('main.sev',source,str(path)),filename),[[list(range(1000))]])
            self.assertEqual(persist(root,[{**discovery,'file':'main.sev'}])[0]['appended'],[])
            with self.assertRaisesRegex(ValueError,'cannot load replay fixture'):
                load_fixture(SourceFile('main.sev',source,str(path)),'missing.json')


class CampaignTimeoutTests(unittest.TestCase):
    def test_timeout_preserves_c_diagnostics_and_requested_deadline(self):
        import subprocess
        import tempfile
        observed = []
        def execute(binary, timeout, env, **options):
            observed.append(timeout)
            raise subprocess.TimeoutExpired(str(binary), timeout, b'last input', b'memory failure')
        spec = {'mode':'fuzz', 'times':1, 'domains':{'n':domain('int')}, 'replay-cases':[{'n':1}]}
        with tempfile.TemporaryDirectory() as directory:
            report, discoveries = run(Path(directory)/'test', spec, execute, 0.25, {})
        case = report['cases'][0]
        self.assertEqual(observed, [0.25])
        self.assertEqual(case['stdout'], 'last input')
        self.assertEqual(case['stderr'], 'memory failure')
        self.assertIn('0.25s', case['error'])
        self.assertFalse(report['passed'])

class DifferentialCampaignTests(unittest.TestCase):
    def test_only_confirmed_differences_are_saved(self):
        import subprocess
        import tempfile
        from hashlib import sha256
        source='test with differential "compare":\n    differential(first, second)\n'
        spec={'mode':'differential','function':'first','functions':['first','second'],
              'parameters':['n'],'domains':{'n':{'type':'int','lower':0,'upper':3}},
              'times':3,'indent':'    ','label':'compare',
              'persistence':{'source-sha256':sha256(source.encode()).hexdigest(),'append-offset':len(source)}}
        for outcome in ('equal','different','crash','timeout','flaky'):
            with self.subTest(outcome=outcome), tempfile.TemporaryDirectory() as directory:
                calls=[]
                def execute(binary,timeout,env,**options):
                    values=struct.unpack('=qq',Path(env['SEV_CAMPAIGN_INPUT']).read_bytes())
                    self.assertEqual(values[0],values[1])
                    calls.append(values[0])
                    Path(env['SEV_COVERAGE_FILE']).write_bytes(struct.pack('=Q',values[0]+1))
                    if outcome=='timeout': raise subprocess.TimeoutExpired(str(binary),timeout)
                    if outcome=='different' or outcome=='flaky' and len(calls)==1:
                        Path(env['SEV_DIFFERENTIAL_RESULT']).write_text('different\n')
                    return subprocess.CompletedProcess([],1 if outcome=='crash' else 0,'','')
                report,found=run(Path(directory)/'test',spec,execute,1,{},Path(directory)/'coverage')
                self.assertEqual(report['passed'],outcome=='equal')
                self.assertEqual(len(found),1 if outcome=='different' else 0)
                if found:
                    self.assertEqual(found[0]['reason'],'difference')
                    self.assertIn('replay(first, second, cases=[[0]])',found[0]['regression-source'])

    def test_differential_and_fuzz_persist_together_with_large_fixtures(self):
        import tempfile
        from hashlib import sha256
        from py_compiler.syntax.complex.fuzz import persist
        from py_compiler.frontend.source.source import SourceFile
        source='test with differential "compare":\n    differential(first, second)\n'
        spec={'mode':'differential','function':'first','functions':['first','second'],
              'domains':{'values':domain('list[int]')},'times':10,'indent':'    ','label':'compare',
              'persistence':{'source-sha256':sha256(source.encode()).hexdigest(),'append-offset':len(source)}}
        difference=regression(spec,{'values':list(range(1000))},reason='difference')
        fuzz_spec={key:value for key,value in spec.items() if key!='functions'}
        fuzz_case=regression(fuzz_spec,{'values':[]},reason='coverage')
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); (root/'package.json').write_text('{}')
            path=root/'main.sev'; path.write_text(source)
            found=[{**entry,'file':'main.sev'} for entry in (difference,fuzz_case)]
            result=persist(root,found)
            self.assertEqual(len(result[0]['appended']),2)
            self.assertEqual(result[0]['pending'],[])
            saved=path.read_text()
            self.assertIn('differential regression',saved)
            self.assertIn('fuzz regression',saved)
            filename=next(iter(difference['fixtures']))
            self.assertIn('/differential/',filename)
            self.assertEqual(load_fixture(SourceFile('main.sev',saved,str(path)),filename),[[list(range(1000))]])
            self.assertEqual(persist(root,found)[0]['appended'],[])
            self.assertEqual(path.read_text(),saved)

    def test_c_differences_persist_and_replay_without_generation(self):
        import tempfile
        from py_compiler.packaging.testing import test_package
        source='''def first(n: int) -> int with 0 <= n <= 3:
    return n
def second(value: int) -> int with 0 <= value <= 3:
    if value == 3:
        return 2
    return value
test with differential "compare":
    differential(first, second, times=3)
'''
        with tempfile.TemporaryDirectory(prefix='sev-differential-') as directory:
            root=Path(directory); path=root/'main.sev'; path.write_text(source)
            (root/'package.json').write_text(json.dumps({'package':{'name':'compare','version':'0.1.0'},'sources':['main.sev']}))
            result=test_package(root,differential=True,memory=True,coverage=True,timeout=5)
            self.assertIn('report',result,result)
            report=json.loads(Path(result['report']).read_text())
            self.assertFalse(result['success'],report)
            campaign=report['tests'][0]['campaign']
            self.assertTrue(campaign['cases'][-1]['difference'],campaign)
            self.assertEqual(campaign['cases'][-1]['inputs'],{'n':3})
            self.assertEqual(len(report['differential-persistence'][0]['appended']),1)
            saved=path.read_text()
            self.assertIn('replay(first, second, cases=[[3]])',saved)
            replay=test_package(root,memory=True,timeout=5)
            self.assertFalse(replay['success'],replay)
            self.assertEqual(replay['tests'],1,replay)
            self.assertEqual(path.read_text(),saved)

    def test_c_move_inputs_are_independent_and_equal_results_are_not_saved(self):
        import tempfile
        from py_compiler.packaging.testing import test_package
        source='''from collections import list
def first(values: move list[int]) -> move list[int] with len(values) < 4:
    if len(values) > 0:
        if values[0] == 7:
            values[0] = 8
        else:
            values[0] = 7
    return values
def second(items: move list[int]) -> move list[int] with len(items) < 4:
    if len(items) > 0:
        if items[0] == 7:
            items[0] = 8
        else:
            items[0] = 7
    return items
test with differential "independent inputs":
    differential(first, second, times=5)
test "replay independent nonempty inputs":
    replay(first, second, cases=[[[1, 2, 3]]])
'''
        collections=Path(__file__).resolve().parents[2]/'library/core/collections/collection'
        with tempfile.TemporaryDirectory(prefix='sev-differential-move-') as directory:
            root=Path(directory); path=root/'main.sev'; path.write_text(source)
            (root/'package.json').write_text(json.dumps({'package':{'name':'compare.move','version':'0.1.0'},'sources':['main.sev'],
                'dependencies':{'collections':{'path':str(collections)}}}))
            result=test_package(root,differential=True,memory=True,coverage=True,timeout=5)
            report=json.loads(Path(result['report']).read_text()) if 'report' in result else result
            self.assertTrue(result['success'],report)
            self.assertEqual(path.read_text(),source)
            self.assertEqual(report['differential-persistence'],[])
