"""Test-local fault catalogs replay through ordinary mock specialization."""
from dataclasses import dataclass
from py_compiler.syntax.generic.block import BlockProvider
from py_compiler.syntax.grammar.expression import expression


@dataclass(frozen=True)
class Event:
    target: str
    throwing: bool
    value: object
    token: object
    identity: str


class ChaosAdd(BlockProvider):
    spelling = 'chaos'
    required_scope = 'test'

    def has_body(self, node):
        return False

    def parse_header(self, items):
        if ([t.text for t in items[:5]] != ['chaos', '.', 'add', '(', 'when']
                or len(items) < 9 or items[-1].text != ')'
                or items[5].kind != 'IDENTIFIER' or items[6].text not in ('return', 'throw')):
            raise ValueError('expected chaos.add(when function return/throw value)')
        return Event(items[5].text, items[6].text == 'throw', expression(items[7:-1]), items[0], '')

    def declare(self, node, context):
        from dataclasses import replace
        event = replace(node.header, identity=node.identity)
        context.chaos_catalog = (*getattr(context, 'chaos_catalog', ()), event)

    def lower(self, node, cfg, env, nodes, index):
        from dataclasses import replace
        cfg.chaos_events = (*getattr(cfg, 'chaos_events', ()), replace(node.header, identity=node.identity))
        return 1


def calls(nodes):
    result = set()
    for node in nodes:
        tokens = node.tokens
        result.update(token.text for token, following in zip(tokens, tokens[1:])
                      if token.kind == 'IDENTIFIER' and following.text == '(')
        result.update(calls(node.children))
    return result


def replay(node, cfg, env):
    from py_compiler.syntax.grammar.assignment import Assignment
    from py_compiler.frontend.parser.blocks import Node
    target, _, initializer = node.header
    if initializer:
        raise ValueError('chaos replay does not accept a loop initializer')
    reached, pending = set(), list(calls(node.children))
    while pending:
        name = pending.pop()
        if name in reached:
            continue
        reached.add(name)
        for entry in cfg.context.lookup_functions(name, cfg.declaration_scope):
            pending.extend(calls(entry.node.children) - reached)
    explicit = getattr(cfg, 'chaos_events', ())
    events = [event for event in (explicit or getattr(cfg.context, 'chaos_catalog', ()))
              if event.target in reached]
    if not events:
        raise ValueError('chaos loop has no reachable fault events')
    previous = getattr(cfg, 'mock_cases', {}), getattr(cfg, 'mock_scope', cfg.body.identity)
    try:
        for index, event in enumerate(events):
            entries = cfg.context.lookup_functions(event.target, cfg.declaration_scope)
            if len(entries) != 1:
                raise ValueError('chaos event requires one unambiguous callable: ' + event.target)
            entry = entries[0]
            cfg.mock_cases = {**previous[0], entry.symbol: ((), (event.throwing, event.value), event.token)}
            cfg.mock_scope = node.identity + ':event:' + str(index)
            tokens = (target, target.with_text('='), target.with_text(str(index), kind='NUMBER'))
            assignment = Node(cfg.mock_scope + ':binding', 'sentence', node.span, tokens, node.identity,
                              syntax=node.syntax, imports=node.imports, scope=node.scope)
            local = dict(env)
            Assignment().lower(assignment, cfg, local, set(), declare_local=True)
            from dataclasses import replace
            def clone(child):
                return replace(child, identity=child.identity + ':chaos:' + str(index),
                               children=[clone(nested) for nested in child.children])
            cfg.scope([clone(child) for child in node.children], local, local_names={target.text})
            cfg.body.chaos_replays = (*getattr(cfg.body, 'chaos_replays', ()),
                {'target': event.target, 'kind': 'throw' if event.throwing else 'return',
                 'event': event.identity})
    finally:
        cfg.mock_cases, cfg.mock_scope = previous
    return 1


import unittest


class ChaosTests(unittest.TestCase):
    def test_c_return_throw_transitive_catalog_and_test_isolation(self):
        import json
        import tempfile
        from pathlib import Path
        from py_compiler.packaging.testing import test_package
        source = '''def read() -> int | Error:
    return 10
def wrapper() -> int | Error:
    return read() + 1
test with chaos "results":
    chaos.add(when read return 0)
    chaos.add(when read return 20)
    count = 0
    for event in chaos:
        value = read()
        assert(value == 0 or value == 20)
        count += 1
    assert(count == 2)
test with chaos "failures":
    chaos.add(when read throw Error("denied"))
    chaos.add(when read throw Error("timeout"))
    count = 0
    for event in chaos:
        value ?= read()
        assert(value is Error)
        count += 1
    assert(count == 2)
test with chaos "transitive":
    count = 0
    for event in chaos:
        value ?= wrapper()
        assert(value is Error or value == 1 or value == 21)
        count += 1
    assert(count == 4)
test "ordinary call is unchanged":
    assert(read() == 10)
    assert(wrapper() == 11)
'''
        with tempfile.TemporaryDirectory(prefix='sev-chaos-') as directory:
            root = Path(directory)
            (root / 'src').mkdir()
            (root / 'package.json').write_text(json.dumps({'package': {'name': 'chaos', 'version': '0.1.0'}}))
            (root / 'src/lib.sev').write_text(source)
            result = test_package(root)
            self.assertTrue(result['success'], result)
