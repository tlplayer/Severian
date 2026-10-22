"""Boolean expressions and their effects.

Implemented evaluation contract:
    A and B -> evaluate A once; evaluate B only when A is true.
    A or B  -> evaluate A once; evaluate B only when A is false.
    Both operands must satisfy bool, including a statically skipped operand.
    Reached effects occur left to right; skipped calls perform no mutation,
    borrow or consumption. Static ownership joins conservatively account for
    every CFG edge unless that edge has been proven unreachable.
    not x == y -> not (x == y), with x and y each evaluated once.
    a() < b() < c() -> save a(), save b(), compare; only on success call c()
    and compare the saved b(). No operand is duplicated or speculated.

Type-predicate contract (pending implementation; acceptance cases in block/if.py):
    x is T tests the active variant against a resolved type T. It reads x once
    and returns bool; it does not convert, copy, consume or extend a loan.
    This type form is distinct from the value-identity form documented for is.
    True edge: possible_types(x) intersect variants(T).
    False edge: possible_types(x) minus variants(T).
    is not and not swap the true/false facts; comparisons bind before not.
    For A and B, check B under A.true; result.true = B.true;
        result.false = join(A.false, B.false).
    For A or B, check B under A.false; result.false = B.false;
        result.true = join(A.true, B.true).
    Joins union possible types, possible moves and borrow dependencies, and
    intersect definite initialization; terminating/unreachable edges contribute
    no facts. An else after `x is int and x > 0` still sees int | string:
    the predicate may have failed OR the int may have been nonpositive.
    Refinement belongs to the tested binding/storage version. Projecting a
    variant preserves its owner and loan dependencies, not a new owning copy.
    Assignment/replacement, including through an alias or a mutating call,
    invalidates affected refinements; declared assignment types remain intact.
    A read-only call preserves facts. Consumption remains a move even when
    the consumed binding has a refined type. Calls are evaluated once, but a
    test of a temporary does not refine a later, separately evaluated call.
"""

from py_compiler.syntax.primitive.numeric.grammar import ScalarOperation
from py_compiler.hir.thir.src.model import Edge, Terminator


class Logical(ScalarOperation):
    def lower_expression(self, cfg, node, env):
        left = cfg.expr(node.operands[0], env, self.owner)
        from py_compiler.mir.ownership.flow import join_expression
        rhs, skipped = cfg.block(), cfg.block()
        values, flow = dict(cfg.values), cfg.flow.fork()
        direct = Edge(skipped.identity)
        edges = (Edge(rhs.identity), direct) if self.spelling == 'and' else (direct, Edge(rhs.identity))
        cfg.current.terminator = Terminator('conditional', edges, left)
        cfg.current = rhs
        with refinement(cfg, node.operands[0], env, self.spelling == 'and'):
            right = cfg.expr(node.operands[1], env, self.owner)
        return join_expression(cfg, [
            (skipped, values, flow, left),
            (cfg.current, dict(cfg.values), cfg.flow.fork(), right),
        ], env, node.token.span)


def conditional_expression(node, cfg, env, expected=None):
    from py_compiler.mir.ownership.flow import join_expression
    from py_compiler.syntax.function.calls import coerce
    condition, yes_node, no_node = node.operands
    condition = cfg.expr(condition, env, cfg.context.type('bool'))
    if condition.type.family != 'bool':
        raise ValueError('conditional expression requires a bool condition')
    yes_block, no_block = cfg.block(), cfg.block()
    cfg.current.terminator = Terminator('conditional', (Edge(yes_block.identity), Edge(no_block.identity)), condition)
    initial_values, initial_flow = dict(cfg.values), cfg.flow.fork()
    cfg.current = yes_block
    yes = cfg.expr(yes_node, dict(env), expected)
    if yes is None:
        raise ValueError('conditional expression requires a value on each branch')
    yes_exit = (cfg.current, dict(cfg.values), cfg.flow.fork(), yes)
    from py_compiler.syntax.generic.storage import root_binding
    def result_owner(term, value):
        owner = root_binding(term, cfg, env) or cfg.call_result_owners.get(value.identity)
        if (owner is not None and value.identity in cfg.owned_results
                and cfg.values.get(owner.identity) != value):
            return None
        return owner
    yes_owner = result_owner(yes_node, yes)
    cfg.current, cfg.values, cfg.flow = no_block, dict(initial_values), initial_flow.fork()
    no = coerce(cfg.expr(no_node, dict(env), expected or yes.type), expected or yes.type, cfg, node.token.span)
    if no is None or yes.type != no.type:
        raise ValueError('conditional expression branches must have compatible types')
    no_owner = result_owner(no_node, no)
    if yes.type.binding_ownership != 'copy' and bool(yes_owner) != bool(no_owner):
        raise ValueError('conditional memory branches mix a view and owned storage; copy the viewed branch explicitly')
    result = join_expression(cfg, [yes_exit, (cfg.current, dict(cfg.values), cfg.flow.fork(), no)], env, node.token.span)
    for facts in (cfg.value_sources, cfg.value_dependencies, cfg.value_origins):
        facts[result.identity] = facts.get(yes.identity, set()) | facts.get(no.identity, set())
    if yes.identity in cfg.stack_values or no.identity in cfg.stack_values:
        cfg.stack_values.add(result.identity)
    if yes.type.binding_ownership != 'copy' and yes_owner and no_owner:
        cfg.call_result_owners[result.identity] = yes_owner
        cfg.call_result_modes[result.identity] = 'view'
        cfg.call_result_sources[result.identity] = frozenset((cfg.flow.resolve(yes_owner.identity), cfg.flow.resolve(no_owner.identity)))
    elif yes.identity in cfg.owned_results and no.identity in cfg.owned_results:
        cfg.owned_results.add(result.identity)
    return result


def identity_expression(node, cfg, env, expected=None):
    from py_compiler.syntax.function.calls import literal_value
    from py_compiler.syntax.generic.owned import select
    from py_compiler.syntax.block.union import UnionTag
    left_node, right_node = node.operands
    if right_node.kind == 'name' and right_node.token.text == 'union' and cfg.lookup(right_node, env) is None:
        left = cfg.expr(left_node, env)
        return literal_value(left is not None and left.type.family == 'type'
                             and left.type.declaration.family == 'union',
                             cfg.context.type('bool'), cfg, node.token.span)
    target = None
    if right_node.kind == 'literal' and right_node.token.text in ('None', 'absent'):
        target = cfg.context.type(right_node.token.text)
    elif right_node.kind == 'name' and cfg.lookup(right_node, env) is None:
        target = cfg.context.type(right_node.token.text)
    left = cfg.expr(left_node, env)
    boolean = cfg.context.type('bool')
    if target is not None:
        if left is None:
            return literal_value(target.name in ('None','absent'), boolean, cfg, node.token.span)
        if left.type.family != 'union':
            return literal_value(left.type == target, boolean, cfg, node.token.span)
        candidates = target.variants if target.family == 'union' else (target,)
        tag = cfg.emit('union-tag', cfg.context.type('i32'), (left,), UnionTag(), node.token.span)
        result = literal_value(False, boolean, cfg, node.token.span)
        for index, variant in enumerate(left.type.variants):
            if variant in candidates:
                number = literal_value(index, tag.type, cfg, node.token.span)
                matches = select(tag.type, 'F.operator', '==').expand(cfg, tag, number, node.token.span)
                result = cfg.emit('identity-tag', boolean, (result, matches), BooleanOr(), node.token.span)
        return result
    right = cfg.expr(right_node, env)
    if left is None or right is None:
        return literal_value(left is right, boolean, cfg, node.token.span)
    if left.type.family == right.type.family == 'type':
        from py_compiler.syntax.complex.type import same_type
        return literal_value(same_type(left.type.declaration, right.type.declaration), boolean, cfg, node.token.span)
    if left.type != right.type:
        return literal_value(False, boolean, cfg, node.token.span)
    if left.type.family in ('integer', 'float', 'bool', 'char', 'byte', 'type', 'enum'):
        return select(left.type, 'F.operator', '==').expand(cfg, left, right, node.token.span)
    if left.type.mlir.startswith('memref<') or left.type.mlir == '!llvm.ptr':
        return cfg.emit('identity', boolean, (left, right), StorageIdentity(), node.token.span)
    raise ValueError('identity requires addressable storage or scalar values')


class BooleanOr:
    def atom(self, operation):
        from py_compiler.syntax.generic.atom import Atom
        return Atom(self, tuple(v.type for v in operation.operands), operation.result.type, ('view', 'view'), (), self)

    def render_operation(self, operation):
        from py_compiler.mlir.src.cfg import name
        left, right = operation.operands
        return [f'{name(operation.result)} = arith.ori {name(left)}, {name(right)} : i1']


class StorageIdentity(BooleanOr):
    def render_operation(self, operation):
        from py_compiler.mlir.src.cfg import name
        result = name(operation.result)
        left, right = operation.operands
        if left.type.mlir == '!llvm.ptr':
            return [f'{result} = llvm.icmp "eq" {name(left)}, {name(right)} : !llvm.ptr']
        lines = [f'{result}_left = memref.extract_aligned_pointer_as_index {name(left)} : {left.type.mlir} -> index',
                f'{result}_right = memref.extract_aligned_pointer_as_index {name(right)} : {right.type.mlir} -> index',
                f'{result}_address = arith.cmpi eq, {result}_left, {result}_right : index']
        if not getattr(left.type, 'strided', False):
            return lines + [f'{result} = arith.andi {result}_address, {result}_address : i1']
        for label, value in (('left', left), ('right', right)):
            p = result + '_' + label
            names = [p + '_base', p + '_offset', *(p + f'_size{i}' for i in range(value.type.rank)),
                     *(p + f'_stride{i}' for i in range(value.type.rank))]
            types = [f'memref<{value.type.element.mlir}>', *('index',) * (1 + 2 * value.type.rank)]
            lines.append(f'{", ".join(names)} = "memref.extract_strided_metadata"({name(value)}) : ({value.type.mlir}) -> ({", ".join(types)})')
        return lines + [f'{result}_offset = arith.cmpi eq, {result}_left_offset, {result}_right_offset : index',
                        f'{result} = arith.andi {result}_address, {result}_offset : i1']


import unittest


class LogicalOwnershipTests(unittest.TestCase):
    def test_conditional_values_and_storage_identity_lower_to_mlir(self):
        from py_compiler.frontend.src.lib import compile_source
        from py_compiler.syntax.recognition import Syntax
        from py_compiler.mlir.src.lib import lower, render
        source = ('def choose(flag:bool) -> int:\n    return 7 if flag else 9\n'
                  'def independent() -> bool:\n    a=tuple[int,3](1,2,3)\n'
                  '    b=copy a\n    return !(a is b)\n')
        result = compile_source('identity.sev', source, Syntax())
        self.assertFalse(result.diagnostics, str(result.diagnostics))
        output = render(lower(result.program))
        self.assertIn('cf.cond_br', output)
        self.assertIn('memref.extract_aligned_pointer_as_index', output)

    def test_union_identity_tests_the_active_tag(self):
        from py_compiler.frontend.src.lib import compile_source
        from py_compiler.syntax.recognition import Syntax
        from py_compiler.mlir.src.lib import lower, render
        result = compile_source('absent.sev', 'class Update:\n    name: string | None | absent\n'
                                'def work()->bool:\n    update=Update()\n'
                                '    return update.name is absent\n', Syntax())
        self.assertFalse(result.diagnostics, str(result.diagnostics))
        self.assertIn('llvm.extractvalue', render(lower(result.program)))

    def test_short_circuit_consumption_and_cleanup(self):
        from py_compiler.frontend.src.lib import compile_source
        from py_compiler.syntax.recognition import Syntax
        # A move parameter consumes the caller's binding on the RHS path.
        for operator in ('and', 'or'):
            for mode in ('view', 'move'):
                for use_after in (False, True):
                    with self.subTest(operator=operator, mode=mode, use_after=use_after):
                        source = (f'def consume(value: {mode} string) -> bool:\n    return true\n'
                                  'def work(flag: bool):\n    owner = "owned"\n'
                                  f'    result = flag {operator} consume(owner)\n')
                        if use_after:
                            source += '    observed = owner\n'
                        result = compile_source('short-circuit-ownership.sev', source, Syntax())
                        if use_after and mode == 'move':
                            self.assertTrue(result.diagnostics)
                            self.assertIn('moved', str(result.diagnostics))
                        else:
                            self.assertFalse(result.diagnostics, str(result.diagnostics))


from contextlib import contextmanager


@contextmanager
def refinement(cfg, condition, env, truth=True):
    previous = getattr(cfg, 'refinements', {})
    facts = dict(previous)
    def record(term, positive):
        if term.kind == 'unary' and term.token.text in ('not', '!'):
            record(term.operands[0], not positive)
        elif term.kind == 'binary' and ((term.token.text == 'and' and positive) or
                                       (term.token.text == 'or' and not positive)):
            for operand in term.operands:
                record(operand, positive)
        elif term.kind == 'identity' and term.token.text == 'is':
            left, right = term.operands
            binding = cfg.lookup(left, env) if left.kind in ('name', 'reference') else None
            if (binding is None or binding.type.family != 'union' or binding.identity in cfg.storage
                    or right.kind != 'name' or cfg.lookup(right, env) is not None):
                return
            target = cfg.context.type(right.token.text)
            selected = target.variants if target.family == 'union' else (target,)
            variants = [t for t in binding.type.variants if (t in selected) == positive]
            if len(variants) == 1 and variants[0].mlir:
                facts[binding.identity] = (cfg.values[binding.identity].identity, variants[0])
    record(condition, truth)
    cfg.refinements = facts
    try:
        yield
    finally:
        cfg.refinements = previous


def refined_value(binding, value, cfg, span):
    fact = getattr(cfg, 'refinements', {}).get(binding.identity)
    if fact is None or value.identity != fact[0] or value.type.family != 'union':
        return value
    from py_compiler.syntax.block.union import UnionMember
    result = cfg.emit('union-member', fact[1], (value,), UnionMember(value.type.variants.index(fact[1])), span)
    for mapping in (cfg.value_sources, cfg.value_dependencies, cfg.value_origins):
        mapping[result.identity] = set(mapping.get(value.identity, ()))
    cfg.call_result_owners[result.identity] = binding
    cfg.call_result_modes[result.identity] = 'view'
    return result
