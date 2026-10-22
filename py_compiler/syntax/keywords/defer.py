"""defer owns its with-clause attachment."""
from py_compiler.syntax.keywords.keyword import Keyword


class Defer(Keyword):
    def __init__(self):
        super().__init__("defer")

    def attach(self, tokens, guards, obligations, suffix):
        from py_compiler.syntax.grammar.expression import expression
        suffix.append(expression(tokens))

    def matches(self, node):
        return bool(node.tokens) and node.tokens[0].text == 'defer'

    def lower(self, node, cfg, env, local_names):
        from dataclasses import replace
        from py_compiler.syntax.grammar.expression import expression
        if len(node.tokens) < 2 or expression(node.tokens[1:]).kind != 'call':
            raise ValueError('defer requires a callable action')
        if not cfg.deferred_actions:
            raise ValueError('defer requires an enclosing execution scope')
        action = replace(node, tokens=node.tokens[1:], provider=None)
        cfg.deferred_actions[-1][1].append((action, dict(env)))


import unittest


class DeferTests(unittest.TestCase):
    def test_c_scope_exits(self):
        import json
        import tempfile
        from pathlib import Path
        from py_compiler.packaging.testing import test_package
        source = '''trace = 0
def record(value: int):
    module.trace = module.trace * 10 + value
def work(early: bool) -> int:
    defer record(1)
    if early:
        defer record(2)
        return 7
    defer record(3)
    return 9
def loop():
    defer record(9)
    index = 0
    while index < 3:
        index += 1
        defer record(index)
        if index == 1:
            continue
        break
test "return, break, continue and caught errors clean up their scopes":
    module.trace = 0
    assert(work(true) == 7 and module.trace == 21)
    module.trace = 0
    assert(work(false) == 9 and module.trace == 31)
    module.trace = 0
    loop()
    assert(module.trace == 129)
    module.trace = 0
    defer record(9)
    throws "caught":
        defer record(1)
        throw Error("caught")
    assert(module.trace == 1)
'''
        with tempfile.TemporaryDirectory(prefix='sev-defer-') as directory:
            root = Path(directory)
            (root / 'src').mkdir()
            (root / 'src/lib.sev').write_text(source)
            (root / 'package.json').write_text(json.dumps({'package': {'name': 'deferred-actions', 'version': '0.1.0'}}))
            result = test_package(root)
            detail = Path(result['report']).read_text() if 'report' in result else result
            self.assertTrue(result['success'], detail)
