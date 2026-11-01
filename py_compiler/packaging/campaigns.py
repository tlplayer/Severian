"""Callable campaigns: typed domains, c input replay and measured growth."""
import ast
import json
import math
import random
import struct
import time
import operator
from functools import lru_cache
from pathlib import Path


def domain(annotation):
    if annotation == 'bool':
        return {'type': annotation, 'lower': 0, 'upper': 1}
    if annotation in ('float', 'f64', 'f32', 'f16'):
        maximum = {'f16':65504.0, 'f32':float.fromhex('0x1.fffffep127')}.get(annotation, float.fromhex('0x1.fffffffffffffp1023'))
        return {'type':annotation, 'lower':-maximum, 'upper':maximum,
                'float-bits':{'f16':16, 'f32':32}.get(annotation, 64)}
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


def rounded_float(value, bounds):
    value = float(value)
    format_ = {16:'e', 32:'f', 64:'d'}[bounds['float-bits']]
    try:
        return struct.unpack('='+format_, struct.pack('='+format_, value))[0]
    except OverflowError:
        return math.copysign(math.inf, value)


def adjacent_float(value, bounds, upward):
    bits = bounds['float-bits']
    if bits == 64:
        return math.nextafter(value, math.inf if upward else -math.inf)
    format_, integer = ('e','H') if bits == 16 else ('f','I')
    if value == 0:
        encoded = 1 if upward else (1 << (bits-1)) | 1
    else:
        encoded = struct.unpack('='+integer, struct.pack('='+format_, value))[0]
        encoded += 1 if (value > 0) == upward else -1
    return struct.unpack('='+format_, struct.pack('='+integer, encoded))[0]


def comparison_bound(value, bounds, *, lower, strict=False):
    if 'float-bits' not in bounds:
        return (math.floor(value)+1 if strict else math.ceil(value)) if lower else (math.ceil(value)-1 if strict else math.floor(value))
    rounded = rounded_float(value, bounds)
    if (rounded < value if lower else rounded > value) or strict and rounded == value:
        return adjacent_float(rounded, bounds, lower)
    return rounded


def random_scalar(bounds, rng, upper=None):
    upper = bounds['upper'] if upper is None else upper
    if 'float-bits' not in bounds:
        return rng.randint(bounds['lower'], upper)
    lower = bounds['lower']
    fraction = rng.random()
    # Weighted endpoints avoid overflow in upper-lower for the full f64 range.
    candidate = lower * (1.0-fraction) + upper * fraction
    return max(lower, min(upper, rounded_float(candidate, bounds)))


@lru_cache(maxsize=256)
def constraint_expression(predicate, parameters, containers):
    """Validate the bounded integer predicate language without executing code."""
    expression = ast.parse(predicate, mode='eval').body
    if sum(1 for _ in ast.walk(expression)) > 128:
        raise ValueError('campaign constraint exceeds 128 syntax nodes')
    def check(term, locals_=()):
        if isinstance(term, ast.Constant) and (type(term.value) in (int, bool) or type(term.value) is float and math.isfinite(term.value)):
            return
        if isinstance(term, ast.Name) and term.id in (*parameters, *locals_, 'true', 'false'):
            return
        if (isinstance(term, ast.Call) and isinstance(term.func, ast.Name)
                and term.func.id in ('len', 'all', 'first', 'last') and len(term.args) == 1 and not term.keywords):
            check(term.args[0], locals_)
            return
        if isinstance(term, ast.ListComp) and len(term.generators) == 1:
            generator = term.generators[0]
            if not isinstance(generator.target, ast.Name) or generator.is_async:
                raise ValueError('campaign comprehension requires one synchronous item binding')
            check(generator.iter, locals_)
            scope = (*locals_, generator.target.id)
            check(term.elt, scope)
            for condition in generator.ifs: check(condition, scope)
            return
        if isinstance(term, ast.List):
            for child in term.elts: check(child, locals_)
            return
        if isinstance(term, ast.Subscript):
            check(term.value, locals_)
            check(term.slice, locals_)
            return
        if isinstance(term, ast.BoolOp) and isinstance(term.op, (ast.And, ast.Or)):
            for child in term.values: check(child, locals_)
            return
        if (isinstance(term, ast.UnaryOp) and
                (isinstance(term.op, ast.Not) or isinstance(term.op, (ast.USub, ast.UAdd))
                 and isinstance(term.operand, ast.Constant))):
            check(term.operand, locals_)
            return
        if isinstance(term, ast.Compare) and all(isinstance(op, (ast.Lt, ast.LtE, ast.Gt, ast.GtE, ast.Eq, ast.NotEq)) for op in term.ops):
            for child in (term.left, *term.comparators): check(child, locals_)
            return
        raise ValueError('unsupported campaign constraint: ' + predicate)
    check(expression)
    return expression


def satisfies_constraints(arguments, domains, predicates=()):
    operations = {ast.Lt:operator.lt, ast.LtE:operator.le, ast.Gt:operator.gt,
                  ast.GtE:operator.ge, ast.Eq:operator.eq, ast.NotEq:operator.ne}
    def evaluate(term, scope=None):
        scope = arguments if scope is None else scope
        if isinstance(term, ast.Constant): return term.value
        if isinstance(term, ast.Name):
            return {'true':True, 'false':False}[term.id] if term.id in ('true', 'false') else scope[term.id]
        if isinstance(term, ast.Call):
            value = evaluate(term.args[0], scope)
            if term.func.id == 'len': return len(value)
            if term.func.id == 'all': return all(value)
            index = 0 if term.func.id == 'first' else len(value) - 1
            return index, value[index]
        if isinstance(term, ast.Subscript): return evaluate(term.value, scope)[evaluate(term.slice, scope)]
        if isinstance(term, ast.List): return [evaluate(child, scope) for child in term.elts]
        if isinstance(term, ast.ListComp):
            generator = term.generators[0]
            result = []
            for value in evaluate(generator.iter, scope):
                local = {**scope, generator.target.id: value}
                if all(evaluate(condition, local) for condition in generator.ifs):
                    result.append(evaluate(term.elt, local))
            return result
        if isinstance(term, ast.UnaryOp):
            value = evaluate(term.operand, scope)
            return not value if isinstance(term.op, ast.Not) else -value if isinstance(term.op, ast.USub) else +value
        if isinstance(term, ast.BoolOp):
            values = (bool(evaluate(child, scope)) for child in term.values)
            return all(values) if isinstance(term.op, ast.And) else any(values)
        left = evaluate(term.left, scope)
        left_term = term.left
        for op, child in zip(term.ops, term.comparators):
            right = evaluate(child, scope)
            left_bounds = domains.get(left_term.id, {}) if isinstance(left_term, ast.Name) else {}
            right_bounds = domains.get(child.id, {}) if isinstance(child, ast.Name) else {}
            if 'float-bits' in left_bounds and isinstance(child, (ast.Constant, ast.UnaryOp)):
                right = rounded_float(right, left_bounds)
            if 'float-bits' in right_bounds and isinstance(left_term, (ast.Constant, ast.UnaryOp)):
                left = rounded_float(left, right_bounds)
            if not operations[type(op)](left, right): return False
            left = right
            left_term = child
        return True
    containers = tuple(name for name, bounds in domains.items() if 'element' in bounds)
    try:
        return all(bool(evaluate(constraint_expression(predicate, tuple(domains), containers))) for predicate in predicates)
    except (ZeroDivisionError, OverflowError, IndexError, TypeError):
        return False


def constrain(domains, predicate):
    """Narrow independent bounds and retain the whole predicate for joint checks."""
    constraint_expression(predicate, tuple(domains), tuple(name for name, bounds in domains.items() if 'element' in bounds))
    bindings = dict(domains)
    def collection(term):
        if isinstance(term, ast.Name): return bindings.get(term.id)
        if (isinstance(term, ast.Subscript) and isinstance(term.value, ast.Call)
                and isinstance(term.value.func, ast.Name) and term.value.func.id in ('first', 'last')
                and isinstance(term.slice, ast.Constant) and term.slice.value == 1):
            parent = collection(term.value.args[0])
            if parent is not None:
                parent['uniform-length'] = True
                return parent.get('element')
        return None
    def target(term):
        if isinstance(term, ast.Name) and term.id in bindings and 'element' not in bindings[term.id]:
            return bindings[term.id]
        if isinstance(term, ast.Call) and isinstance(term.func, ast.Name) and term.func.id == 'len' and len(term.args)==1:
            value = collection(term.args[0])
            if value is not None and 'element' in value:
                return value
        return None
    def visit(term):
        if (isinstance(term, ast.Call) and isinstance(term.func, ast.Name) and term.func.id == 'all'
                and isinstance(term.args[0], ast.ListComp)):
            comprehension = term.args[0]
            generator = comprehension.generators[0]
            parent = collection(generator.iter)
            if parent is not None and 'element' in parent and not generator.ifs:
                previous = bindings.get(generator.target.id)
                bindings[generator.target.id] = parent['element']
                visit(comprehension.elt)
                if previous is None: bindings.pop(generator.target.id)
                else: bindings[generator.target.id] = previous
            return
        if isinstance(term, ast.BoolOp) and isinstance(term.op, ast.And):
            for value in term.values: visit(value)
            return
        if not isinstance(term, ast.Compare):
            return
        for left, operation, right in zip([term.left, *term.comparators], term.ops, term.comparators):
            left_bounds, right_bounds = target(left), target(right)
            if left_bounds is not None and right_bounds is not None:
                kind = type(operation)
                if kind in (ast.Gt, ast.GtE):
                    left_bounds, right_bounds = right_bounds, left_bounds
                    kind = ast.Lt if kind is ast.Gt else ast.LtE
                if kind in (ast.Lt, ast.LtE, ast.Eq):
                    right_bounds['lower'] = max(right_bounds['lower'], comparison_bound(left_bounds['lower'], right_bounds, lower=True, strict=kind is ast.Lt))
                    if right_bounds['upper'] is not None:
                        upper = comparison_bound(right_bounds['upper'], left_bounds, lower=False, strict=kind is ast.Lt)
                        left_bounds['upper'] = min(left_bounds['upper'], upper) if left_bounds['upper'] is not None else upper
                    if kind is ast.Eq:
                        left_bounds['lower'] = max(left_bounds['lower'], comparison_bound(
                            right_bounds['lower'], left_bounds, lower=True))
                        if left_bounds['upper'] is not None:
                            upper = comparison_bound(left_bounds['upper'], right_bounds, lower=False)
                            right_bounds['upper'] = min(right_bounds['upper'], upper) if right_bounds['upper'] is not None else upper
                continue
            bounds = target(left)
            reverse = bounds is None
            if reverse: bounds = target(right)
            if bounds is None:
                continue
            try: constant = ast.literal_eval(left if reverse else right)
            except (ValueError, TypeError):
                continue
            if type(constant) not in (int, bool, float):
                continue
            if 'float-bits' in bounds:
                constant = rounded_float(constant, bounds)
            kind = type(operation)
            if reverse: kind = {ast.Lt:ast.Gt, ast.LtE:ast.GtE, ast.Gt:ast.Lt, ast.GtE:ast.LtE, ast.Eq:ast.Eq}.get(kind)
            if kind in (ast.Gt, ast.GtE, ast.Eq):
                bounds['lower'] = max(bounds['lower'], comparison_bound(constant, bounds, lower=True, strict=kind is ast.Gt))
            if kind in (ast.Lt, ast.LtE, ast.Eq):
                upper = comparison_bound(constant, bounds, lower=False, strict=kind is ast.Lt)
                bounds['upper'] = min(bounds['upper'], upper) if bounds['upper'] is not None else upper
            if bounds['upper'] is not None and bounds['lower'] > bounds['upper']:
                raise ValueError('campaign constraint has an empty domain')
    term = ast.parse(predicate, mode='eval').body
    for _ in range(len(domains) + 2):
        before = [(bounds['lower'], bounds['upper']) for bounds in domains.values()]
        visit(term)
        after = [(bounds['lower'], bounds['upper']) for bounds in domains.values()]
        if any(upper is not None and lower > upper for lower, upper in after):
            raise ValueError('campaign constraint has an empty domain')
        if before == after:
            break
    else:
        raise ValueError('campaign constraint has an empty domain (cyclic strict bounds)')
    return predicate


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


class GenerationConstraintUnsatisfied(Exception):
    pass


def value(bounds, magnitude, rng, pattern='random', deadline=None):
    lo, hi = bounds['lower'], bounds['upper']
    n = max(lo, magnitude)
    if hi is not None: n = min(n, hi)
    if 'element' not in bounds:
        if 'float-bits' in bounds:
            return max(lo, min(hi, rounded_float(n, bounds)))
        return bool(n) if bounds['type']=='bool' else n
    element = bounds['element']
    values = []
    lower = max(element['lower'], -max(1,n))
    upper = min(element['upper'] if element['upper'] is not None else n, max(1,n))
    if upper < lower: upper = lower
    shared_magnitude = random_scalar({**element, 'lower':lower}, rng, upper) if bounds.get('uniform-length') else None
    for index in range(n):
        if deadline is not None and index % 256 == 0 and time.monotonic() >= deadline:
            raise GenerationBudgetExpired()
        size = shared_magnitude if shared_magnitude is not None else random_scalar({**element, 'lower':lower}, rng, upper)
        values.append(value(element, size, rng, deadline=deadline))
    if pattern == 'ascending': values.sort()
    elif pattern == 'descending': values.sort(reverse=True)
    elif pattern == 'duplicates': values = [value(element, 0, rng)] * n
    return values


def equal_parameter_groups(domains, predicates):
    """Collect scalar equalities required by conjunctions, never optional arms."""
    parents = {name: name for name, bounds in domains.items() if 'element' not in bounds}
    def root(name):
        while parents[name] != name:
            name = parents[name]
        return name
    def visit(term):
        if isinstance(term, ast.BoolOp) and isinstance(term.op, ast.And):
            for child in term.values:
                visit(child)
        elif isinstance(term, ast.Compare):
            for left, operation, right in zip([term.left, *term.comparators], term.ops, term.comparators):
                if (isinstance(operation, ast.Eq) and isinstance(left, ast.Name)
                        and isinstance(right, ast.Name) and left.id in parents and right.id in parents):
                    parents[root(right.id)] = root(left.id)
    containers = tuple(name for name, bounds in domains.items() if 'element' in bounds)
    for predicate in predicates:
        visit(constraint_expression(predicate, tuple(domains), containers))
    groups = {}
    for name in parents:
        groups.setdefault(root(name), []).append(name)
    return tuple(tuple(names) for names in groups.values() if len(names) > 1)


def couple_equal_parameters(arguments, domains, groups, rng, preferred=None):
    """Share one representable sample across each required equality group."""
    for names in groups:
        bounds = [domains[name] for name in names]
        lower = max(bound['lower'] for bound in bounds)
        upper = min(bound['upper'] for bound in bounds)
        floats = [bound for bound in bounds if 'float-bits' in bound]
        integral = any('float-bits' not in bound for bound in bounds)
        common = dict(min(floats, key=lambda bound: bound['float-bits']) if floats else bounds[0])
        if integral:
            lower, upper = math.ceil(lower), math.floor(upper)
        common['lower'] = comparison_bound(lower, common, lower=True)
        common['upper'] = comparison_bound(upper, common, lower=False)
        if common['lower'] > common['upper']:
            raise GenerationConstraintUnsatisfied('equal campaign parameters have no common representable value')
        sample = arguments[preferred if preferred in names else names[0]]
        if not common['lower'] <= sample <= common['upper']:
            sample = random_scalar(common, rng)
        if floats:
            sample = rounded_float(sample, common)
        if integral:
            sample = int(sample)
        for name in names:
            arguments[name] = (rounded_float(sample, domains[name]) if 'float-bits' in domains[name]
                               else bool(sample) if domains[name]['type'] == 'bool' else sample)


def inputs(spec, corpus=None, deadline=None):
    rng = random.Random()
    domains = spec['domains']
    predicates = spec.get('constraints', ())
    equality_groups = equal_parameter_groups(domains, predicates)
    def accepted(arguments):
        return satisfies_constraints(arguments, domains, predicates)
    if spec.get('replay-cases'):
        for arguments in spec['replay-cases']:
            if not accepted(arguments):
                raise GenerationConstraintUnsatisfied('captured case violates joint parameter constraints')
            yield arguments, {'sizes':{name:len(v) if isinstance(v,list) else v for name,v in arguments.items()},'pattern':'captured'}
        return
    if spec['mode'] in ('fuzz','differential'):
        for index in range(spec['times']):
            for attempt in range(1024):
                if deadline is not None and time.monotonic() >= deadline:
                    raise GenerationBudgetExpired()
                arguments = {}
                for name, bounds in domains.items():
                    lo, hi = bounds['lower'], bounds['upper']
                    if attempt == 0 and index == 0: n = max(lo, min(0, hi)) if hi is not None else lo
                    elif attempt == 0 and index == 1: n = lo
                    elif attempt == 0 and index == 2 and hi is not None: n = hi
                    else:
                        # Rejections use fresh inputs and the original deadline.
                        upper = hi if hi is not None else max(lo, 1 << index)
                        n = random_scalar(bounds, rng, upper)
                    arguments[name] = value(bounds, n, rng, deadline=deadline)
                mutated_name = None
                if corpus and domains and index > 2 and rng.randrange(2):
                    import copy
                    arguments = copy.deepcopy(rng.choice(corpus))
                    name = rng.choice(list(domains)); bounds = domains[name]
                    mutated_name = name
                    if isinstance(arguments[name],list) and arguments[name] and 'element' not in bounds['element']:
                        position=rng.randrange(len(arguments[name]))
                        arguments[name][position]=random_scalar(bounds['element'],rng)
                    elif 'element' not in bounds:
                        arguments[name]=random_scalar(bounds,rng)
                couple_equal_parameters(arguments, domains, equality_groups, rng, mutated_name)
                if accepted(arguments):
                    yield arguments, None
                    break
            else:
                raise GenerationConstraintUnsatisfied('no satisfying campaign input found after 1024 attempts; provide captured cases or tighter bounds')
    else:
        streams = {name: iter(magnitudes(bounds)) for name,bounds in domains.items()}
        current = {name: bounds['lower'] for name,bounds in domains.items()}
        while streams:
            if deadline is not None and time.monotonic() >= deadline:
                raise GenerationBudgetExpired()
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
                    arguments = {name:value(bounds,sizes[name],rng,pattern,deadline) for name,bounds in domains.items()}
                    if accepted(arguments):
                        yield arguments, {'sizes':sizes,'pattern':pattern}


def encode_inputs(arguments, domains=None):
    values = []
    def flatten(value, bounds=None):
        if isinstance(value,list):
            values.append(struct.pack('=Q', len(value)))
            for item in value: flatten(item, bounds.get('element') if bounds else None)
        elif bounds and 'float-bits' in bounds or isinstance(value, float):
            values.append(struct.pack('=d', float(value)))
        else:
            values.append(struct.pack('=Q', int(value) & ((1<<64)-1)))
    for name, argument in arguments.items(): flatten(argument, domains.get(name) if domains else None)
    return b''.join(values)


def reductions(arguments, domains, predicates=()):
    return (candidate for candidate in reduction_candidates(arguments, domains)
            if satisfies_constraints(candidate, domains, predicates))


def reduction_candidates(arguments, domains):
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
        path.write_bytes(encode_inputs(ordered,spec['domains'])*len(spec.get('functions',[None])))
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
    generation_error = None
    while time.monotonic()-started < budget:
        try: arguments,scale=next(generated)
        except (StopIteration, GenerationBudgetExpired): break
        except GenerationConstraintUnsatisfied as failure:
            generation_error = str(failure)
            break
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
                for attempt,candidate in enumerate(reductions(best,spec['domains'],spec.get('constraints', ()))):
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
    report = {'passed':bool(records) and not generation_error and all(r['passed'] for r in records),'cases':records,
              'budget-seconds':budget,'budget-exhausted':time.monotonic()-started>=budget}
    if generation_error:
        report['generation-error'] = generation_error
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
        with self.assertRaises(ValueError): constrain({'n':domain('int')}, 'abs(n) != 3')
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
        if 'float-bits' in bounds:
            return isinstance(value,(int,float)) and math.isfinite(value) and bounds['lower'] <= value <= bounds['upper'] and math.isfinite(rounded_float(value,bounds))
        return isinstance(value,(int,bool)) and bounds['lower'] <= value <= bounds['upper']
    def normalize(value,bounds):
        if 'element' in bounds: return [normalize(item,bounds['element']) for item in value]
        return rounded_float(value,bounds) if 'float-bits' in bounds else value
    results=[]
    for case in cases:
        if not isinstance(case,list) or len(case)!=len(spec['domains']):
            raise ValueError('captured case does not match callable arity')
        arguments=dict(zip(spec.get('parameters',spec['domains']),case))
        if not all(valid(arguments[name],bounds) for name,bounds in spec['domains'].items()):
            raise ValueError('captured case violates parameter constraints')
        arguments = {name:normalize(value,spec['domains'][name]) for name,value in arguments.items()}
        if not satisfies_constraints(arguments, spec['domains'], spec.get('constraints', ())):
            raise ValueError('captured case violates joint parameter constraints')
        results.append(arguments)
    return results


def growth_regression(spec, cases, metric):
    return captured_regression(spec,cases,'complexity-'+metric,profile=True)


class FloatCampaignTests(unittest.TestCase):
    def test_finite_fractional_generation_and_strict_bounds(self):
        for annotation in ('float', 'f64', 'f32', 'f16'):
            domains = {'value':domain(annotation)}
            predicate = constrain(domains, '-0.5 < value < 0.5')
            spec = {'mode':'fuzz', 'times':32, 'domains':domains, 'constraints':[predicate]}
            cases = [arguments['value'] for arguments, _ in inputs(spec)]
            self.assertEqual(len(cases), 32)
            self.assertTrue(all(math.isfinite(value) and -0.5 < value < 0.5 for value in cases))
            self.assertTrue(any(value != int(value) for value in cases))

    def test_float_transport_preserves_fraction_sign_and_list_elements(self):
        domains = {'value':domain('float'), 'items':domain('list[f32]')}
        data = encode_inputs({'value':-0.0, 'items':[1.25,-2.5]}, domains)
        self.assertEqual(data[:8], struct.pack('=d', -0.0))
        self.assertEqual(struct.unpack('=Qdd', data[8:]), (2,1.25,-2.5))
        self.assertEqual(struct.unpack('=d', encode_inputs({'value':1}, {'value':domain('f32')})), (1.0,))
        values = value(domain('list[f32]'), 8, random.Random(17))
        self.assertTrue(all(isinstance(item,float) and math.isfinite(item) for item in values))

    def test_f32_replay_and_comparisons_use_native_precision(self):
        domains = {'value':domain('f32')}
        predicate = constrain(domains, 'value > 0.1')
        boundary = rounded_float(0.1, domains['value'])
        self.assertGreater(domains['value']['lower'], boundary)
        self.assertFalse(satisfies_constraints({'value':boundary}, domains, [predicate]))
        spec = {'domains':{'value':domain('f32')}}
        self.assertEqual(validate_cases(spec, [[0.1]]), [{'value':boundary}])
        for invalid in (math.nan, math.inf, -math.inf, 1e100):
            with self.assertRaisesRegex(ValueError, 'parameter constraints'):
                validate_cases(spec, [[invalid]])

    def test_full_float_domain_generation_stays_finite(self):
        for annotation in ('float', 'f32', 'f16'):
            bounds = domain(annotation)
            rng = random.Random(9)
            self.assertTrue(all(math.isfinite(random_scalar(bounds, rng)) for _ in range(100)))

    def test_native_fractional_replay_uses_mlir_decoding(self):
        import tempfile
        from py_compiler.packaging.testing import test_package
        source = '''def wide(value: float) -> float:
    if value != 1.25 and value != -2.5:
        throw Error("wide replay lost its fractional input")
    return value
def narrow(value: f32) -> f32:
    if value != f32(0.1) and value != f32(-2.5):
        throw Error("narrow replay lost its fractional input")
    return value
test "wide fractional replay":
    replay(wide, cases=[[1.25],[-2.5]])
test "narrow fractional replay":
    replay(narrow, cases=[[0.1],[-2.5]])
'''
        with tempfile.TemporaryDirectory(prefix='sev-float-campaign-') as directory:
            root = Path(directory)
            (root/'lib.sev').write_text(source)
            (root/'package.json').write_text(json.dumps({'package':{'name':'float-campaign','version':'0.1.0'},'sources':['lib.sev']}))
            result = test_package(root, timeout=2)
            self.assertTrue(result['success'], result)


class EqualParameterGenerationTests(unittest.TestCase):
    def test_full_width_equal_parameters_share_samples_and_corpus_mutations(self):
        for mode in ('fuzz', 'differential'):
            domains = {name: domain('int') for name in ('left', 'middle', 'right')}
            predicates = [constrain(domains, 'left == middle'), constrain(domains, 'middle == right')]
            specification = {'mode': mode, 'times': 32, 'domains': domains, 'constraints': predicates}
            cases = list(inputs(specification, [{'left': 42, 'middle': 42, 'right': 42}]))
            self.assertEqual(len(cases), 32)
            self.assertTrue(all(arguments['left'] == arguments['middle'] == arguments['right']
                                for arguments, _ in cases))
            self.assertGreater(len({arguments['left'] for arguments, _ in cases}), 3)

    def test_equal_parameter_groups_intersect_bounds_and_preserve_float_precision(self):
        for annotations in (('i64', 'i8'), ('f64', 'f32'), ('int', 'f32'), ('bool', 'int')):
            domains = dict(zip(('left', 'right'), (domain(annotation) for annotation in annotations)))
            predicates = [constrain(domains, 'left == right')]
            specification = {'mode': 'fuzz', 'times': 16, 'domains': domains, 'constraints': predicates}
            cases = list(inputs(specification))
            self.assertEqual(len(cases), 16)
            for arguments, _ in cases:
                self.assertEqual(arguments['left'], arguments['right'])
                for name, bounds in domains.items():
                    self.assertLessEqual(bounds['lower'], arguments[name])
                    self.assertLessEqual(arguments[name], bounds['upper'])
                    if 'float-bits' in bounds:
                        self.assertEqual(rounded_float(arguments[name], bounds), arguments[name])
                    else:
                        self.assertIsInstance(bounds['lower'], int)
                        self.assertIsInstance(bounds['upper'], int)

    def test_equalities_in_optional_boolean_arms_do_not_couple_parameters(self):
        domains = {name: domain('int') for name in ('left', 'middle', 'right')}
        self.assertEqual(equal_parameter_groups(domains, ['left == middle or middle == right']), ())
        self.assertEqual(equal_parameter_groups(domains, ['not (left == middle)']), ())
        self.assertEqual(equal_parameter_groups(domains, ['left == middle == right']),
                         (('left', 'middle', 'right'),))


class JointConstraintTests(unittest.TestCase):
    def specification(self, mode='fuzz', predicate='low < high'):
        domains = {'low':domain('int'), 'high':domain('int')}
        constraints = [constrain(domains, text) for text in
                       ('0 <= low <= 4', '0 <= high <= 4', predicate)]
        return {'mode':mode, 'times':32, 'domains':domains, 'constraints':constraints}

    def test_generated_and_mutated_inputs_satisfy_joint_constraints(self):
        for mode in ('fuzz', 'differential'):
            spec = self.specification(mode)
            cases = list(inputs(spec, [{'low':1, 'high':2}], time.monotonic()+1))
            self.assertEqual(len(cases), 32)
            self.assertTrue(all(arguments['low'] < arguments['high'] for arguments, _ in cases))

    def test_profile_axes_filter_invalid_pairs(self):
        cases = list(inputs(self.specification('profile')))
        self.assertTrue(cases)
        self.assertTrue(all(arguments['low'] < arguments['high'] for arguments, _ in cases))

    def test_chained_bounds_propagate_in_both_directions(self):
        domains = {'low':domain('int'), 'high':domain('int')}
        constrain(domains, '0 <= low < high <= 4')
        self.assertEqual((domains['low']['lower'], domains['low']['upper']), (0,3))
        self.assertEqual((domains['high']['lower'], domains['high']['upper']), (1,4))
        with self.assertRaisesRegex(ValueError, 'empty domain'):
            constrain(domains, 'low < high and high < low')

    def test_replay_and_shrinking_preserve_the_joint_domain(self):
        spec = self.specification()
        self.assertEqual(validate_cases(spec, [[1,2]]), [{'low':1, 'high':2}])
        with self.assertRaisesRegex(ValueError, 'joint parameter'):
            validate_cases(spec, [[2,1]])
        shrunk = list(reductions({'low':1, 'high':2}, spec['domains'], spec['constraints']))
        self.assertEqual(shrunk, [{'low':0, 'high':2}])
        spec['replay-cases'] = [{'low':2, 'high':1}]
        with self.assertRaises(GenerationConstraintUnsatisfied): list(inputs(spec))

    def test_joint_lengths_boolean_branches_and_inequality(self):
        domains = {'items':domain('list[int]'), 'count':domain('int')}
        predicate = constrain(domains, 'not (count == 0) and (len(items) < count or count != 2)')
        self.assertFalse(satisfies_constraints({'items':[], 'count':0}, domains, [predicate]))
        self.assertFalse(satisfies_constraints({'items':[1,2], 'count':2}, domains, [predicate]))
        self.assertTrue(satisfies_constraints({'items':[1], 'count':2}, domains, [predicate]))

    def test_impossible_predicates_have_bounded_rejection_and_report_failure(self):
        import tempfile
        spec = self.specification(predicate='low != low')
        with self.assertRaisesRegex(GenerationConstraintUnsatisfied, '1024 attempts'):
            list(inputs(spec))
        def execute(*args, **kwargs):
            self.fail('unsatisfiable inputs must not reach the executable')
        with tempfile.TemporaryDirectory() as directory:
            report, discoveries = run(Path(directory)/'test', spec, execute, 1, {})
        self.assertFalse(report['passed'])
        self.assertIn('1024 attempts', report['generation-error'])
        self.assertEqual(discoveries, [])
        with self.assertRaises(GenerationBudgetExpired):
            list(inputs(spec, deadline=time.monotonic()-1))

    def test_indexed_constraints_validate_replay_and_shrinking(self):
        domains = {'items': domain('list[int]'), 'count': domain('int')}
        predicate = 'items[0] == count'
        self.assertEqual(constrain(domains, predicate), predicate)
        specification = {'domains': domains, 'constraints': [predicate]}
        valid = {'items': [7], 'count': 7}
        self.assertTrue(satisfies_constraints(valid, domains, [predicate]))
        self.assertEqual(validate_cases(specification, [[[7], 7]]), [valid])

        for case in ([[7], 8], [[], 7]):
            with self.subTest(case=case):
                arguments = dict(zip(('items', 'count'), case))
                self.assertFalse(satisfies_constraints(arguments, domains, [predicate]))
                with self.assertRaisesRegex(ValueError, 'joint parameter'):
                    validate_cases(specification, [case])

        self.assertEqual(list(reductions(valid, domains, [predicate])), [])

    def test_unsupported_constraints_are_diagnostics(self):
        for predicate in ('abs(count) > 0', 'count.real == 1',
                          'count + 1 > 0', '__import__("os")', 'unknown < count'):
            with self.subTest(predicate=predicate), self.assertRaises(ValueError):
                constrain({'items':domain('list[int]'), 'count':domain('int')}, predicate)


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

    def test_joint_parameter_constraints_are_retained(self):
        domains={'low':domain('int'),'high':domain('int')}
        self.assertEqual(constrain(domains,'low < high'), 'low < high')

    def test_float_generator_has_a_finite_domain(self):
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


class RectangularCampaignTests(unittest.TestCase):
    def test_nested_constraints_generate_valid_rectangles(self):
        domains = {'grid': domain('list[list[int]]')}
        predicates = [
            '0 <= len(grid) <= 6',
            'all([0 <= len(row) <= 6 for row in grid])',
            'all([len(row) == len(first(grid)[1]) for row in grid])',
            'all([all([0 <= cell <= 1 for cell in row]) for row in grid])',
        ]
        for predicate in predicates:
            constrain(domains, predicate)
        spec = {'domains': domains, 'constraints': predicates, 'mode': 'fuzz', 'times': 40}
        samples = list(inputs(spec))
        self.assertEqual(len(samples), 40)
        self.assertEqual(samples[0][0]['grid'], [])
        self.assertEqual(len(samples[2][0]['grid']), 6)
        for arguments, _ in samples:
            grid = arguments['grid']
            self.assertTrue(satisfies_constraints(arguments, domains, predicates))
            self.assertTrue(all(len(row) == len(grid[0]) for row in grid))
            self.assertTrue(all(cell in (0, 1) for row in grid for cell in row))
        self.assertFalse(satisfies_constraints({'grid': [[0], [0, 1]]}, domains, predicates))
        self.assertFalse(satisfies_constraints({'grid': [[2]]}, domains, predicates))

    def test_comprehension_constraints_reject_executable_calls(self):
        with self.assertRaises(ValueError):
            constraint_expression('all([exec(cell) for cell in grid])', ('grid',), ('grid',))
