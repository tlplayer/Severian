"""Return-source constraints, solved before realization bodies are expanded.

Facts grow monotonically; only callers of a changed summary are revisited.
An empty recursive component is unresolved, not an implicitly owned result.
"""
from collections import defaultdict, deque
from dataclasses import dataclass


@dataclass(frozen=True)
class ResultContract:
    ownership: str
    sources: tuple = ()


@dataclass(frozen=True)
class Term:
    type: object
    kind: str
    arguments: tuple = ()
    target: object = None


class ReturnInference:
    def __init__(self, context):
        self.context = context
        self.entries, self.returns, self.facts = {}, {}, {}
        self.callers = defaultdict(set)

    def add(self, entry):
        key = id(entry)
        if key in self.entries:
            return key
        self.entries[key], self.facts[key] = entry, frozenset()
        context = self.context
        previous = context.scope, context.template_types, context.template_functions
        context.scope, context.template_types, context.template_functions = entry.scope, entry.type_arguments, entry.callable_arguments
        try:
            result = context.type(entry.result)
            resolved = getattr(entry, 'semantic_result', None)
            if resolved is not None:
                self.returns[key] = (tuple(Term(result, 'source', target=index) for index in resolved.sources)
                                     if resolved.ownership == 'view' else (Term(result, 'own'),))
                return key
            if result.binding_ownership == 'copy':
                self.returns[key] = (Term(result, 'own'),)
                return key
            env = {}
            for index, (name, annotation) in enumerate(entry.parameters):
                type_ = context.type(annotation)
                mode = entry.parameter_ownership.get(name, getattr(type_, 'parameter_ownership', 'view'))
                env[name] = Term(type_, 'source' if mode in ('view', 'borrow') else 'own', target=index)
            returns = []
            self.statements(entry.node.children, env, result, key, returns)
            self.returns[key] = tuple(returns)
        finally:
            context.scope, context.template_types, context.template_functions = previous
        return key

    def statements(self, nodes, env, result, caller, returns):
        from py_compiler.syntax.grammar.expression import expression
        from py_compiler.syntax.generic.block import DeclarationProvider
        for node in nodes:
            if isinstance(node.provider, DeclarationProvider):
                continue
            tokens = node.tokens
            if tokens and tokens[0].text == 'return':
                returns.append(self.expression(expression(tokens[1:]), env, result, caller))
            elif node.children:
                branch = dict(env)
                self.statements(node.children, branch, result, caller, returns)
                for name in env.keys() & branch.keys():
                    if branch[name] != env[name]:
                        env[name] = Term(env[name].type, 'join', (env[name], branch[name]))
            elif tokens and len(tokens) > 2:
                assignment = next((i for i, token in enumerate(tokens) if token.text in ('=', ':=')), None)
                if assignment is not None and tokens[0].kind == 'IDENTIFIER':
                    annotation = None
                    if tokens[1].text == ':':
                        annotation = self.context.type(''.join(t.text for t in tokens[2:assignment]))
                    env[tokens[0].text] = self.expression(expression(tokens[assignment + 1:]), env, annotation, caller)

    def expression(self, node, env, expected, caller):
        context = self.context
        if node.kind == 'name' and node.token.text in env:
            return env[node.token.text]
        if node.kind == 'unary':
            operand = self.expression(node.operands[0], env, expected, caller)
            return Term(operand.type, 'own') if node.token.text in ('copy', 'mirror', 'move') else operand
        if node.kind == 'literal':
            from py_compiler.frontend.parser.contract import Literal
            from py_compiler.hir.hir.src.program import resolve_literal
            type_, _ = resolve_literal(Literal('', node.token.span, node.token.text, node.token.kind,
                                              expected.name if expected else None), context.bound_syntax, context)
            return Term(type_, 'own')
        if node.kind in ('binary', 'chain', 'owned'):
            # Scalar operations cannot introduce a storage dependency. Full
            # operator/type validation remains with the semantic provider.
            type_ = expected or context.type('int')
            if type_.binding_ownership == 'copy':
                return Term(type_, 'own')
        if node.kind == 'call':
            from py_compiler.syntax.grammar.expression import type_spelling
            from py_compiler.syntax.complex.generic import realize
            callee, *arguments = node.operands
            explicit = ()
            if callee.kind == 'index':
                callee, arguments_node = callee.operands
                terms = arguments_node.operands if arguments_node.kind == 'template_arguments' else (arguments_node,)
                explicit = tuple(type_spelling(term) for term in terms)
            if callee.kind == 'name':
                name = callee.token.text
                entries = ((context.template_functions[name],) if name in context.template_functions else
                           context.lookup_functions(name, context.scope))
                if not entries:
                    type_ = context.type(name) if name in context.bound_syntax.types else expected
                    if type_ is not None and (name in context.bound_syntax.types or
                                             context.declaration_name(name, context.scope) in context.by_name):
                        return Term(type_, 'own')
                candidates = [entry for entry in entries if len(entry.parameters) == len(arguments)]
                values = tuple(self.expression(argument, env, None, caller) for argument in arguments)
                selected = []
                for entry in candidates:
                    concrete = realize(entry, explicit, values, context) if entry.templates else entry
                    if all(value.type == context.type(annotation) for value, (_, annotation) in zip(values, concrete.parameters)):
                        selected.append(concrete)
                if len(selected) == 1:
                    entry = selected[0]
                    type_ = context.type(entry.result)
                    if entry.infer_result_ownership:
                        target = self.add(entry)
                        self.callers[target].add(caller)
                        return Term(type_, 'call', values, target)
                    if entry.result_ownership == 'own':
                        return Term(type_, 'own')
                    contract = getattr(entry, 'semantic_result', None)
                    if contract is not None:
                        return Term(type_, 'join', tuple(values[i] for i in contract.sources))
        raise ValueError('cannot infer return ownership from this expression; use an explicit result ownership contract')

    def evaluate(self, term):
        if term.kind == 'own':
            return frozenset({('own', None)})
        if term.kind == 'source':
            return frozenset({('view', term.target)})
        if term.kind == 'join':
            return frozenset().union(*(self.evaluate(value) for value in term.arguments))
        facts = set()
        for mode, source in self.facts[term.target]:
            if mode == 'own':
                facts.add(('own', None))
            else:
                for actual_mode, actual_source in self.evaluate(term.arguments[source]):
                    facts.add(('view', actual_source) if actual_mode == 'view' else ('local', None))
        return frozenset(facts)

    def solve(self, entry):
        self.add(entry)
        pending, queued = deque(self.entries), set(self.entries)
        while pending:
            key = pending.popleft()
            queued.remove(key)
            facts = frozenset().union(*(self.evaluate(term) for term in self.returns[key]))
            if facts != self.facts[key]:
                self.facts[key] = facts
                for caller in self.callers[key] - queued:
                    queued.add(caller)
                    pending.append(caller)
        contracts = {}
        for key, facts in self.facts.items():
            modes = {mode for mode, _ in facts}
            if not modes:
                raise ValueError('recursive result ownership requires an explicit annotation')
            if 'local' in modes:
                raise ValueError('returned borrow escapes the lifetime of local storage')
            if len(modes) != 1:
                raise ValueError('generic return paths mix owned values and views; use a consistent result ownership contract')
            contracts[key] = ResultContract(next(iter(modes)), tuple(sorted(source for mode, source in facts if mode == 'view')))
        # Publish only after the entire reachable component has a solution.
        for key, contract in contracts.items():
            self.entries[key].semantic_result = contract
            self.entries[key].result_ownership = contract.ownership


def infer_result(entry, context):
    if not entry.infer_result_ownership or getattr(entry, 'semantic_result', None) is not None:
        return
    if getattr(context, 'return_inference', None) is not None:
        return
    context.return_inference = ReturnInference(context)
    try:
        context.return_inference.solve(entry)
    finally:
        context.return_inference = None


import unittest


class ReturnContractTests(unittest.TestCase):
    def compile(self, source):
        from py_compiler.frontend.src.lib import compile_source
        from py_compiler.syntax.recognition import Syntax
        return compile_source('contracts.sev', source, Syntax(), stop_after='thir')

    def test_unconstrained_recursive_ownership_requires_annotation(self):
        result = self.compile('class Point:\n    x: int\n'
                              'def cycle[T](value: T) -> T:\n    return cycle(value)\n'
                              'def work():\n    point = Point(1)\n    result = cycle(point)\n')
        self.assertTrue(result.diagnostics)
        self.assertIn('recursive result ownership requires an explicit annotation', str(result.diagnostics))

    def test_recursive_argument_permutation_tracks_actual_source(self):
        source = ('class Point:\n    x: int\n'
                  'def first[T](left: T, right: T, count: int) -> T:\n'
                  '    if count == 0:\n        return left\n'
                  '    return second(right, left, count - 1)\n'
                  'def second[T](left: T, right: T, count: int) -> T:\n'
                  '    if count == 0:\n        return right\n'
                  '    return first(right, left, count - 1)\n'
                  'def work():\n    left = Point(1)\n    right = Point(2)\n'
                  '    result = first(left, right, 2)\n')
        valid = self.compile(source + '    drop right\n    observed = result.x\n')
        self.assertFalse(valid.diagnostics, str(valid.diagnostics))
        invalid = self.compile(source + '    drop left\n    observed = result.x\n')
        self.assertTrue(invalid.diagnostics)
        self.assertIn('moved or dropped', str(invalid.diagnostics))

    def test_returning_view_of_local_through_generic_call_is_rejected(self):
        result = self.compile('class Point:\n    x: int\n'
                              'def identity[T](value: T) -> T:\n    return value\n'
                              'def invalid[T](value: T) -> T:\n'
                              '    fresh = Point(2)\n    return identity(fresh)\n'
                              'def work():\n    point = Point(1)\n    result = invalid(point)\n')
        self.assertTrue(result.diagnostics)
        self.assertIn('lifetime of local storage', str(result.diagnostics))
