"""Default trait bodies are fallbacks, never predicate candidates."""
from py_compiler.syntax.block.record import implements
from py_compiler.syntax.function.calls import select, invoke, literal_value, coerce
from py_compiler.hir.thir.src.model import Edge, Terminator
from py_compiler.mir.ownership.flow import Binding


def qualified(node):
    if node.kind == 'name':
        return node.token.text
    if node.kind == 'member':
        prefix = qualified(node.operands[0])
        return prefix + '.' + node.token.text if prefix else ''
    return ''


def resolve_default_call(callee, arguments, cfg, env, span):
    name = qualified(callee)
    direct = name.endswith('.default')
    if direct:
        name = name[:-8]
    if '.' not in name:
        return None
    namespace, method = name.rsplit('.', 1)
    context = cfg.context
    contracts = [d for d in context.declarations if d.kind == 'trait' and
                 (d.name == namespace or namespace in context.package_imports and d.name.rsplit('.', 1)[0] == namespace) and
                 any(m.name.rsplit('.', 1)[-1] == method and m.node.children and m.receiver_ownership is None for m in d.methods)]
    if not contracts:
        return None
    if len(contracts) != 1:
        raise ValueError('ambiguous trait namespace; qualify the trait name')
    contract = contracts[0]
    specs = [m for m in contract.methods if m.name.rsplit('.', 1)[-1] == method and m.node.children and m.receiver_ownership is None]
    if not specs:
        return None
    defaults = context.functions.get(contract.name + '.' + method, ())
    if not defaults:
        raise ValueError('trait default body is missing from the imported interface')
    default, values = select(defaults, arguments, cfg, env)
    cfg.deferred_call_arguments = getattr(cfg, 'deferred_call_arguments', set()) | {v.identity for v in values}
    if direct:
        return invoke(default, values, cfg, span)
    candidates = [(d, e) for d in context.declarations if implements(d, contract.name, context)
                  for e in context.functions.get(d.name + '.' + method, ())
                  if not getattr(e, 'trait_fallback', False) and
                  tuple(context.type(t) for _, t in e.parameters) == tuple(v.type for v in values)]
    constants = {op.result.identity: op.payload.value for block in cfg.body.blocks
                 for op in block.operations if op.kind == 'constant'}
    def known(entry):
        facts = {n: constants.get(v.identity) for (n, _), v in zip(entry.parameters, values)}
        answers = []
        for guard in entry.guards:
            if guard.kind == 'call' and len(guard.operands) == 2:
                callee, arg = guard.operands
                if callee.kind == 'member' and callee.token.text == 'suffix' and callee.operands[0].kind == 'name' and arg.kind == 'literal':
                    from py_compiler.syntax.symbol.forms import decode_quoted
                    value = facts.get(callee.operands[0].token.text)
                    if isinstance(value, str):
                        answers.append(value.endswith(decode_quoted(arg.token.text)))
                        continue
            return None
        return all(answers)
    states = [known(e) for _, e in candidates]
    if all(state is not None for state in states):
        matches = [pair for pair, state in zip(candidates, states) if state]
        if len(matches) > 1:
            raise ValueError('ambiguous trait dispatch: multiple matching implementations')
        if not matches:
            return invoke(default, values, cfg, span)
        declaration, entry = matches[0]
        receiver = context.provider_by_name[declaration.name].instantiate(declaration, (), cfg, env, span)
        return invoke(entry, values, cfg, span, receiver)
    candidates = [pair for pair, state in zip(candidates, states) if state is not False]
    from py_compiler.syntax.block.union import Union
    result = Union().resolve([context.type(e.result or 'absent') for _, e in candidates] + [context.type(default.result)], context)
    flags = []
    from py_compiler.syntax.complex.object import BooleanAnd, CountBoolean, Assert
    count = literal_value(0, context.type('i32'), cfg, span)
    for _, entry in candidates:
        flag = literal_value(True, context.type('bool'), cfg, span)
        bindings = {}
        for (parameter, _), value in zip(entry.parameters, values):
            binding = Binding(entry.symbol + ':guard:' + parameter, parameter, value.type, True, 'view', span)
            bindings[parameter] = binding
            cfg.values[binding.identity] = value
        for guard in entry.guards:
            test = cfg.expr(guard, bindings, context.type('bool'))
            flag = cfg.emit('and', context.type('bool'), (flag, test), BooleanAnd(), span)
        flags.append(flag)
        number = cfg.emit('count', context.type('i32'), (flag,), CountBoolean(), span)
        count = cfg.emit('binary', context.type('i32'), (count, number), '+', span)
    one = literal_value(1, context.type('i32'), cfg, span)
    valid = cfg.emit('binary', context.type('bool'), (count, one), '<=', span)
    cfg.effect('assert', (valid,), Assert('ambiguous trait dispatch: ' + contract.name + '.' + method), span)
    join = cfg.block((result,))
    for (declaration, entry), flag in zip(candidates, flags):
        chosen, following = cfg.block(), cfg.block()
        cfg.current.terminator = Terminator('conditional', (Edge(chosen.identity), Edge(following.identity)), flag)
        cfg.current = chosen
        receiver = context.provider_by_name[declaration.name].instantiate(declaration, (), cfg, env, span)
        value = invoke(entry, values, cfg, span, receiver)
        boxed = coerce(value, result, cfg, span)
        if boxed is not value and callable(getattr(value.type, 'release', None)):
            value.type.release(value, cfg, span)
        cfg.current.terminator = Terminator('jump', (Edge(join.identity, (boxed,)),))
        cfg.current = following
    value = invoke(default, values, cfg, span)
    boxed = coerce(value, result, cfg, span)
    if boxed is not value and callable(getattr(value.type, 'release', None)):
        value.type.release(value, cfg, span)
    cfg.current.terminator = Terminator('jump', (Edge(join.identity, (boxed,)),))
    cfg.current = join
    cfg.owned_results.add(join.parameters[0].identity)
    return join.parameters[0]


import unittest


class DefaultDispatchTests(unittest.TestCase):
    def test_literal_selection_default_and_different_results(self):
        from py_compiler.frontend.src.lib import compile_source
        from py_compiler.syntax.recognition import Syntax
        source = ('trait File:\n    def read(path: string) -> int:\n        return 4\n'
                  'class Special: File\n    def read(path: string) -> bool with path.suffix(".json"):\n        return File.read.default(path) == 4\n'
                  'def known() -> bool:\n    return File.read("a.json")\n'
                  'def fallback() -> int:\n    return File.read("LICENSE")\n'
                  'def dynamic(path: string) -> int | bool:\n    return File.read(path)\n')
        result = compile_source('defaults.sev', source, Syntax())
        self.assertFalse(result.diagnostics, str(result.diagnostics))

    def test_default_does_not_hide_overlapping_handlers(self):
        from py_compiler.frontend.src.lib import compile_source
        from py_compiler.syntax.recognition import Syntax
        source = ('trait File:\n    def read(path: string) -> int:\n        return 0\n'
                  'class First: File\n    def read(path: string) -> int with path.suffix(".x"):\n        return 1\n'
                  'class Second: File\n    def read(path: string) -> int with path.suffix(".x"):\n        return 2\n'
                  'def run() -> int:\n    return File.read("a.x")\n')
        result = compile_source('ambiguous.sev', source, Syntax())
        self.assertTrue(result.diagnostics)
        self.assertIn('multiple matching implementations', str(result.diagnostics))
