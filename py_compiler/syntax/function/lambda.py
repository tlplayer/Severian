"""Lambda expressions capture independent snapshots in the shared dynamic box."""
from py_compiler.syntax.keywords.keyword import Keyword


class Lambda(Keyword):
    def __init__(self):
        super().__init__('lambda')

    def parse_prefix(self, token, parse, tokens, cursor):
        from py_compiler.syntax.grammar.expression import Expression
        names = []
        while cursor[0] < len(tokens) and tokens[cursor[0]].text != ':':
            name = tokens[cursor[0]]
            if name.kind != 'IDENTIFIER' or name.text in names:
                raise ValueError('lambda requires distinct parameter names')
            names.append(name.text)
            cursor[0] += 1
            if cursor[0] < len(tokens) and tokens[cursor[0]].text == ',':
                cursor[0] += 1
            elif cursor[0] < len(tokens) and tokens[cursor[0]].text != ':':
                raise ValueError('lambda parameters require commas')
        if cursor[0] == len(tokens):
            raise ValueError("lambda requires ':' before its expression")
        cursor[0] += 1
        return Expression('owned', token, (self, tuple(names), parse()))

    def lower_expression(self, node, cfg, env, expected):
        _, parameters, body = node.operands
        return closure(node, parameters, body, cfg, env, expected)


def named_function(node, entries, cfg, env, expected):
    from py_compiler.syntax.grammar.expression import Expression
    candidates = [entry for entry in entries if not entry.templates and not entry.variadic]
    if expected is not None and expected.family == 'function':
        candidates = [entry for entry in candidates if len(entry.parameters) == len(expected.parameters)
            and all(annotation is not None and cfg.context.type(annotation) == type_
                    for (_, annotation), type_ in zip(entry.parameters, expected.parameters))]
    if len(candidates) != 1:
        raise ValueError('function value requires an unambiguous concrete signature')
    entry = candidates[0]
    cfg.context.compile(entry)
    from py_compiler.syntax.complex.function import callable_type
    type_ = callable_type(tuple(cfg.context.type(annotation) for _, annotation in entry.parameters), cfg.context.type(entry.result), cfg.context)
    if expected is not None and type_ != expected:
        raise ValueError('function value does not satisfy the expected callable signature')
    parameters = tuple(':argument' + str(index) for index in range(len(entry.parameters)))
    arguments = tuple(Expression('name', node.token.with_text(name, kind='IDENTIFIER')) for name in parameters)
    return closure(node, parameters, Expression('call', node.token, (node, *arguments)), cfg, env, type_)


def infer_signature(parameters, body, captures, cfg, env, expected):
    from py_compiler.syntax.generic.place import expression_type
    if expected is not None and expected.family != 'function':
        raise ValueError('lambda requires a callable type')
    if expected is not None and len(parameters) != len(expected.parameters):
        raise ValueError('lambda parameter count does not match its callable type')
    inferred = dict(zip(parameters, expected.parameters)) if expected else {}
    def infer(term, wanted=None):
        if term.kind == 'name':
            name = term.token.text
            if name in parameters:
                if wanted is not None:
                    if name in inferred and inferred[name] != wanted:
                        raise ValueError('conflicting lambda parameter types')
                    inferred[name] = wanted
                return inferred.get(name)
            if name in captures:
                return captures[name].type
        if term.kind == 'binary':
            comparison = term.token.text in ('==', '!=', '<', '<=', '>', '>=', 'in')
            boolean = term.token.text in ('and', 'or')
            hint = cfg.context.type('bool') if boolean else None if comparison else wanted
            left = infer(term.operands[0], hint)
            right = infer(term.operands[1], left or hint)
            if left is None and right is not None:
                left = infer(term.operands[0], right)
            return cfg.context.type('bool') if comparison or boolean else left or right
        if term.kind == 'unary':
            return infer(term.operands[0], cfg.context.type('bool') if term.token.text == 'not' else wanted)
        if term.kind == 'conditional':
            infer(term.operands[0], cfg.context.type('bool'))
            yes = infer(term.operands[1], wanted)
            no = infer(term.operands[2], yes or wanted)
            if yes is None and no is not None:
                infer(term.operands[1], no)
            return yes or no
        if term.kind == 'call' and term.operands[0].kind == 'name':
            entries = cfg.context.lookup_functions(term.operands[0].token.text, cfg.declaration_scope)
            concrete = [entry for entry in entries if not entry.templates and len(entry.parameters) == len(term.operands) - 1]
            if len(concrete) == 1:
                entry = concrete[0]
                for argument, (_, annotation) in zip(term.operands[1:], entry.parameters):
                    infer(argument, cfg.context.type(annotation) if annotation else None)
                return cfg.context.type(entry.result) if entry.result else wanted
        try:
            return expression_type(term, cfg, env, wanted)
        except ValueError:
            return wanted
    result = infer(body, expected.result if expected else None)
    if any(name not in inferred for name in parameters) or result is None:
        raise ValueError('lambda requires a callable annotation for unconstrained parameter or result types')
    from py_compiler.syntax.complex.function import callable_type
    return callable_type(tuple(inferred[name] for name in parameters), expected.result if expected else result, cfg.context)


def closure(node, parameters, body, cfg, env, expected):
    from hashlib import sha256
    from py_compiler.syntax.grammar.expression import Expression
    from py_compiler.syntax.generic.storage import contains_borrowed_fields
    captures, terms = {}, {}
    def collect(term):
        if not isinstance(term, Expression):
            return
        if term.kind in ('name', 'reference') and term.token.text not in parameters:
            binding = cfg.lookup(term, env)
            if binding is not None:
                if contains_borrowed_fields(binding.type, cfg.context):
                    raise ValueError('closure capture cannot retain a borrowed pointer or borrowed field')
                captures[binding.name] = binding
                terms[binding.name] = term
        for operand in term.operands:
            collect(operand)
    collect(body)
    type_ = infer_signature(parameters, body, captures, cfg, env, expected)
    from py_compiler.syntax.complex.tuple import product_type, construct_product
    capture_type = product_type(tuple(binding.type for binding in captures.values()), cfg.context)
    values = [cfg.expr(terms[name], env) for name in captures]
    snapshot = construct_product(capture_type, tuple(terms.values()), cfg, env, node.token.span, values)
    from py_compiler.syntax.dynamic.dynamic import DynamicType
    boxed = DynamicType.from_value(type_, snapshot, cfg, node.token.span)
    symbol = '__sev_closure_' + sha256((cfg.body.name + str(node.token.span.start) + type_.name + capture_type.name).encode()).hexdigest()
    helpers = getattr(cfg.context, 'closure_helpers', None)
    if helpers is None:
        helpers = cfg.context.closure_helpers = {}
    if symbol not in helpers:
        helpers[symbol] = closure_body(symbol, type_, capture_type, tuple(captures), parameters, body, cfg, node.token.span)
    from py_compiler.syntax.complex.function import CallableCode
    result = cfg.emit('memory-copy', type_, (boxed,), CallableCode(type_, helpers[symbol]), node.token.span)
    cfg.owned_results.add(result.identity)
    cfg.value_dependencies[result.identity] = set()
    cfg.value_sources[result.identity] = set()
    return result


def closure_body(symbol, type_, capture_type, captures, parameters, expression, cfg, span):
    from py_compiler.hir.thir.src.checking import Checker
    from py_compiler.hir.thir.src.model import Terminator
    from py_compiler.mir.ownership.flow import Binding
    from py_compiler.syntax.dynamic.dynamic import DynamicOperation
    from py_compiler.syntax.complex.shared import Shared, SharedOperation
    from py_compiler.syntax.complex.array import ArrayType, RecordElementPlace
    from py_compiler.syntax.function.calls import literal_value
    from py_compiler.syntax.generic.storage import copy_storage, owns_storage
    builder = Checker(cfg.source, cfg.syntax, symbol, symbol, type_.result)
    builder.context = cfg.context
    builder.block_scope, builder.declaration_scope = cfg.block_scope, cfg.declaration_scope
    builder.imported = dict(cfg.imported)
    builder.namespaces['module'] = cfg.context.global_bindings
    builder.storage = dict(cfg.context.global_storage)
    builder.fallback_scopes = [cfg.context.global_bindings]
    self_value = builder.value(type_)
    arguments = tuple(builder.value(type_) for type_ in type_.parameters)
    builder.current.parameters = (self_value, *arguments)
    self_binding = Binding(symbol + ':self', ':closure', type_, True, 'view', span)
    builder.values[self_binding.identity] = self_value
    builder.body.bindings.append(self_binding)
    local = {}
    for name, value in zip(parameters, arguments):
        binding = Binding(symbol + ':' + name, name, value.type, False, 'view', span)
        local[name] = binding
        builder.values[binding.identity] = value
        builder.body.bindings.append(binding)
    backing = ArrayType(capture_type)
    shared = Shared(backing, type_.bits)
    handle = builder.emit('dynamic-handle', shared, (self_value,), DynamicOperation('handle'), span)
    storage = builder.emit('share-data', backing, (handle,), SharedOperation('data', shared), span)
    zero = literal_value(0, cfg.context.type('index'), builder, span)
    captured = RecordElementPlace(storage, zero, capture_type).read(builder, span)
    provider = cfg.context.provider_by_name[capture_type.name]
    for index, name in enumerate(captures):
        value = provider.field_place(captured, 'item' + str(index), builder, span).read(builder, span)
        binding = Binding(symbol + ':capture:' + name, name, value.type, True, 'view', span)
        local[name] = binding
        builder.values[binding.identity] = value
        builder.body.bindings.append(binding)
    builder.active_scopes.append(local)
    result = builder.expr(expression, local, type_.result)
    if result is not None and owns_storage(result.type) and result.identity not in builder.owned_results:
        result = copy_storage(result, builder, span)
    builder.cleanup_memory(local.values(), span)
    builder.current.terminator = Terminator('finish', value=result, span=span)
    compiled = builder.finish()
    compiled.source_snapshot = None
    return compiled.freeze()


import unittest


class ClosureTests(unittest.TestCase):
    def test_c_closure_snapshot_escape_copy_and_owned_result(self):
        import json
        import tempfile
        from pathlib import Path
        from py_compiler.packaging.testing import test_package
        source = '''def increment(value: int) -> int:
    return value + 1
def apply(operation: (int)->int, value: int) -> int:
    return operation(value)
def saved(value: string) -> ()->string:
    return lambda: value
def make(offset: int) -> (int)->int:
    return lambda value: value + offset
test "named functions and typed identity":
    identity: (int)->int = lambda value: value
    assert(apply(increment, 4) == 5)
    assert(identity(42) == 42)
test "escaping captures own their snapshots":
    operation = make(3)
    assert(operation(4) == 7)
    source = "snapshot"
    first = saved(source)
    source = "changed"
    second = copy first
    drop first
    value = second()
    drop second
    assert(len(value) == 8)
test "temporary invocation arguments survive the call":
    assert(apply(make(9), 2) == 11)
'''
        with tempfile.TemporaryDirectory(prefix='sev-closures-') as directory:
            root = Path(directory)
            (root / 'src').mkdir()
            (root / 'src/lib.sev').write_text(source)
            (root / 'package.json').write_text(json.dumps({'package': {'name': 'closures', 'version': '0.1.0'}}))
            result = test_package(root)
            self.assertTrue(result['success'], result)

    def test_ambiguous_untyped_lambda_is_rejected(self):
        from py_compiler.frontend.src.lib import compile_source
        from py_compiler.syntax.recognition import Syntax
        result = compile_source('lambda.sev', 'def work():\n    identity = lambda value: value\n', Syntax())
        self.assertTrue(result.diagnostics)
        self.assertIn('annotation', str(result.diagnostics))
