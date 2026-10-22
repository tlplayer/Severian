"""Synchronous iterator consumers lower producer continuations to ordinary CFG."""
from dataclasses import dataclass, replace
from importlib import import_module
from py_compiler.syntax.generic.block import BlockProvider
from py_compiler.syntax.grammar.expression import expression
from py_compiler.hir.thir.src.model import Edge, Terminator
from py_compiler.mir.ownership.flow import Binding, FlowState

loop_syntax = import_module('py_compiler.syntax.block.while')


def iteration_initializer(node, initializer):
    """An initializer that reads an iteration binding runs for each item."""
    position = loop_syntax.initializer_assignment(initializer)
    if position is None:
        return False
    targets = node.header[0] if isinstance(node.header[0], tuple) else (node.header[0],)
    names = {token.text for token in targets if token.text != '_'}

    def reads_item(term):
        if term.kind in ('name', 'reference') and term.token.text in names:
            return True
        return any(reads_item(child) for child in term.operands if hasattr(child, 'kind'))

    return reads_item(expression(initializer[position + 1:]))


def initialize_for(node, initializer, cfg, env):
    if iteration_initializer(node, initializer):
        position = loop_syntax.initializer_assignment(initializer)
        if (position == 3 and initializer[0].text == 'local'
                and cfg.lookup(expression(initializer[:position]), env) is None):
            raise ValueError('an item-dependent local initializer requires an existing binding; the loop may be empty')
        return dict(env), set()
    return loop_syntax.initialize_loop(node, initializer, cfg, env)


def bindings(*scopes):
    return tuple({b.identity: b for scope in scopes for b in scope.values()}.values())


def edge(cfg, target, carried, span):
    values = tuple(cfg.values[b.identity] for b in carried)
    for source, destination in zip(values, target.parameters):
        cfg.value_sources.setdefault(destination.identity, set()).update(cfg.value_sources.get(source.identity, ()))
        if source.identity in cfg.stack_values:
            cfg.stack_values.add(destination.identity)
    cfg.current.terminator = Terminator('jump', (Edge(target.identity, values),), span=span)


@dataclass
class FrozenCapture:
    value: object
    owner: object

    @property
    def contract(self):
        return self.value.type

    def type(self, cfg, expected=None):
        return self.contract

    def binding(self, cfg):
        return self.owner

    def lower(self, cfg, expected=None):
        cfg.flow.read(self.owner)
        return self.value


class ConsumerJump:
    def __init__(self, consumer, resume, carried, flow, depth):
        self.consumer, self.resume, self.carried = consumer, resume, carried
        self.flow, self.depth = flow, depth
        self.exits = []

    def jump(self, cfg, span, *, exit_loop=False):
        consumer = self.consumer
        kept = consumer.carried if exit_loop else self.carried
        live = {b.identity for b in kept} | consumer.ambient
        depth = consumer.depth if exit_loop else self.depth
        cfg.run_defers(depth)
        cfg.cleanup_memory((b for scope in cfg.active_scopes[depth:] for b in scope.values()
                            if b.identity not in live), span)
        cfg.flow.end_scope(live)
        baseline = consumer.flow if exit_loop else self.flow
        expected = baseline.fork()
        expected.reporter = None
        expected.end_scope(live)
        if exit_loop:
            expected.moved = set(cfg.flow.moved)
        if (cfg.flow.moved != expected.moved or cfg.flow.views != expected.views
                or cfg.flow.loans != expected.loans):
            raise ValueError('iterator loop cannot change outer ownership on a continuing edge')
        edge(cfg, consumer.following if exit_loop else self.resume, kept, span)
        if exit_loop:
            consumer.exits.append((cfg.current, dict(cfg.values), cfg.flow.fork()))
        else:
            self.exits.append(cfg.flow.fork())


class Consumer:
    def __init__(self, node, cfg, env, carried, following, ambient, element):
        self.node, self.cfg, self.env = node, cfg, env
        self.carried, self.following, self.ambient = carried, following, ambient
        self.element = element
        self.flow = cfg.flow.fork()
        self.depth = len(cfg.active_scopes)
        self.captures, self.handler, self.loops = cfg.grammar_captures, cfg.yield_handler, list(cfg.loops)
        self.outer_local_names = set(cfg.local_names)
        self.exits = []
        self.active = set(cfg.active_grammars)
        self.count = 0

    def emit(self, term, producer_env, span):
        cfg = self.cfg
        value = cfg.expr(term, producer_env, self.element)
        if value.type.family == 'record' and not getattr(self, 'borrow_items', False):
            from py_compiler.syntax.generic.storage import root_binding
            if value.identity not in cfg.owned_results or root_binding(term,cfg,producer_env) is not None:
                value = value.type.copy_value(value, cfg, span)
        if value.type != self.element:
            raise ValueError('yield value does not satisfy iterator element type')
        carried = tuple(b for b in bindings(self.env, *cfg.active_scopes[self.depth:])
                        if b.identity not in cfg.storage and b.identity not in cfg.flow.moved)
        resume = cfg.block(tuple(b.type for b in carried))
        state, values = cfg.flow.fork(), dict(cfg.values)
        frame = ConsumerJump(self, resume, carried, state, len(cfg.active_scopes))
        targets = self.node.header[0] if isinstance(self.node.header[0], tuple) else (self.node.header[0],)
        name = targets[0].text if len(targets) == 1 else ':tuple-item'
        item = Binding(self.node.identity + ':item:' + str(self.count), name, value.type, False,
                       'view' if getattr(self, 'borrow_items', False) and value.type.binding_ownership != 'copy' else 'copy', span)
        self.count += 1
        body_env = dict(self.env)
        body_env[name] = item
        cfg.values[item.identity] = value
        cfg.body.bindings.append(item)
        cfg.record_lifetime('declare', item.identity, item.name)
        names = {name}
        if len(targets) > 1:
            from py_compiler.syntax.grammar.expression import Expression
            count = len(value.type.generic_arguments) if getattr(value.type.declaration if value.type.family == 'record' else None, 'template_name', '') == 'tuple' else getattr(value.type, 'count', None)
            if count != len(targets):
                raise ValueError('loop tuple pattern does not match yielded tuple arity')
            for position, token in enumerate(targets):
                if token.text == '_':
                    continue
                term = Expression('literal', token.with_text(str(position), kind='NUMBER'))
                field = value.type.element_place(value, term, cfg, body_env, span).read(cfg, span)
                binding = Binding(item.identity + ':' + str(position), token.text, field.type, False, 'view' if field.type.binding_ownership != 'copy' else 'copy', span)
                body_env[token.text] = binding
                names.add(token.text)
                cfg.values[binding.identity] = field
                cfg.body.bindings.append(binding)
                if binding.ownership == 'view':
                    cfg.flow.loan(binding, item)
        previous = cfg.grammar_captures, cfg.yield_handler, cfg.loops, cfg.active_grammars, cfg.local_names
        cfg.grammar_captures, cfg.yield_handler, cfg.loops = self.captures, self.handler, [*self.loops, frame]
        cfg.active_grammars = set(self.active)
        cfg.local_names = self.outer_local_names | names
        try:
            if iteration_initializer(self.node, self.node.header[2]):
                initializer_node = replace(self.node, identity=self.node.identity + ':yield:' + str(self.count))
                body_env, initialized = loop_syntax.initialize_loop(
                    initializer_node, self.node.header[2], cfg, body_env)
                names.update(initialized)
                names.update(cfg.local_names - self.outer_local_names)
            def clone(node):
                return replace(node, identity=node.identity + ':yield:' + str(self.count),
                               children=[clone(child) for child in node.children])
            cfg.scope([clone(n) for n in self.node.children], body_env, local_names=names)
            if cfg.current.terminator is None:
                frame.jump(cfg, span)
        finally:
            cfg.grammar_captures, cfg.yield_handler, cfg.loops, cfg.active_grammars, cfg.local_names = previous
        cfg.current, cfg.values = resume, values
        cfg.flow = FlowState.join(frame.exits) if frame.exits else state
        cfg.values.update({b.identity: v for b, v in zip(carried, resume.parameters)})


class For(BlockProvider):
    spelling = 'for'

    def lower_literal(self, node, cfg, env, source, initializer, heterogeneous=False):
        from py_compiler.syntax.grammar.expression import Expression
        values = []
        for term in source.operands:
            value = cfg.expr(term, env, values[0].type if values and not heterogeneous else None)
            if heterogeneous:
                from py_compiler.syntax.generic.storage import copy_storage, owns_storage
                if owns_storage(value.type):
                    value = copy_storage(value, cfg, term.token.span)
                    cfg.owned_results.add(value.identity)
            values.append(value)
        loop_env, local_names = initialize_for(node, initializer, cfg, env)
        if heterogeneous:
            for index, value in enumerate(values):
                name = ':tuple-iteration:' + str(index)
                owner = Binding(node.identity + name, name, value.type, True, 'own', node.span)
                loop_env[name] = owner
                local_names.add(name)
                cfg.values[owner.identity] = value
                cfg.body.bindings.append(owner)
        carried = tuple(b for b in bindings(loop_env) if b.identity not in cfg.storage)
        following = cfg.block(tuple(b.type for b in carried))
        ambient = {b.identity for scope in cfg.active_scopes for b in scope.values()} | cfg.live_binding_ids(loop_env)
        consumer = Consumer(node, cfg, loop_env, carried, following, ambient,
                            values[0].type if values else cfg.context.type('int'))
        consumer.borrow_items = heterogeneous
        originals = dict(cfg.values)
        # Capture every element before executing the first loop body, as a list
        # expression does. Consumer continuations preserve break/continue scope.
        for index, value in enumerate(values):
            if heterogeneous:
                consumer.element = value.type
            if value.type != consumer.element:
                raise ValueError('list elements must have one element type')
            owner = Binding(node.identity + ':literal:' + str(index), ':literal', value.type, True, 'view', node.span)
            cfg.values[owner.identity] = value
            capture = Expression('capture', source.token, (FrozenCapture(value, owner),))
            consumer.emit(capture, loop_env, node.span)
        if cfg.current.terminator is None:
            edge(cfg, following, carried, node.span)
            consumer.exits.append((cfg.current, dict(cfg.values), cfg.flow.fork()))
        cfg.current = following
        cfg.values = originals
        cfg.values.update({b.identity: v for b, v in zip(carried, following.parameters)})
        cfg.flow = FlowState.join([flow for _, _, flow in consumer.exits]) if consumer.exits else consumer.flow
        cfg.cleanup_memory((loop_env[n] for n in local_names), node.span)
        return 1



    def lower_stored(self, node, cfg, env, source, initializer):
        from py_compiler.syntax.generic.storage import root_binding
        from py_compiler.syntax.grammar.expression import Expression
        loop_env, local_names = initialize_for(node, initializer, cfg, env)
        owner = root_binding(source, cfg, loop_env)
        if owner is None:
            raise ValueError('stored iterator consumption requires a live binding')
        carried = tuple(b for b in bindings(loop_env) if b.identity not in cfg.storage and b.identity not in cfg.flow.moved)
        initial_values, initial_flow = dict(cfg.values), cfg.flow.fork()
        header, following = cfg.block(tuple(b.type for b in carried)), cfg.block(tuple(b.type for b in carried))
        edge(cfg, header, carried, node.span)
        cfg.current = header
        cfg.values.update({b.identity: value for b, value in zip(carried, header.parameters)})
        cursor = cfg.expr(source, loop_env)
        output = cursor.type.call_method(cursor, 'next', (), cfg, loop_env, node.span, owner)
        condition = output.type.field_place(output, 'has_value', cfg, node.span).read(cfg, node.span)
        body, exhausted = cfg.block(), cfg.block()
        cfg.current.terminator = Terminator('conditional', (Edge(body.identity), Edge(exhausted.identity)), condition)
        condition_flow = cfg.flow.fork()
        header_values = dict(cfg.values)
        ambient = {b.identity for scope in cfg.active_scopes for b in scope.values()} | cfg.live_binding_ids(loop_env)
        consumer = Consumer(node, cfg, loop_env, carried, following, ambient, cursor.type.element)
        cfg.current = body
        item = output.type.field_place(output, 'value', cfg, node.span).read(cfg, node.span)
        frozen = Binding(node.identity + ':next', ':next', item.type, True, 'view', node.span)
        cfg.values[frozen.identity] = item
        consumer.emit(Expression('capture', source.token, (FrozenCapture(item, frozen),)), loop_env, node.span)
        if cfg.current.terminator is None:
            edge(cfg, header, carried, node.span)
        consumer.exits.append((exhausted, header_values, condition_flow))
        from py_compiler.mir.ownership.flow import reconcile_owners
        reconcile_owners(cfg, consumer.exits, carried, node.span)
        moved = set.union(*(flow.moved for _, _, flow in consumer.exits))
        retained = [(b, value) for b, value in zip(carried, following.parameters) if b.identity not in moved]
        following.parameters = tuple(value for _, value in retained)
        for block, values, _ in consumer.exits:
            block.terminator = Terminator('jump', (Edge(following.identity, tuple(values[b.identity] for b, _ in retained)),))
        cfg.current, cfg.values = following, initial_values
        cfg.values.update({b.identity: value for b, value in retained})
        cfg.flow = FlowState.join([flow for _, _, flow in consumer.exits])
        cfg.cleanup_memory((loop_env[n] for n in local_names), node.span)
        return 1

    def parse_header(self, items):
        header = super().parse_header(items)
        separator = next((i for i,t in enumerate(header) if t.text == 'in'), -1)
        if separator > 1:
            names = header[:separator:2]
            if any(t.kind != 'IDENTIFIER' for t in names) or any(t.text != ',' for t in header[1:separator:2]) or separator % 2 != 1:
                raise ValueError('for requires a tuple of binding names')
            if len({t.text for t in names if t.text != '_'}) != len([t for t in names if t.text != '_']):
                raise ValueError('duplicate loop binding')
            source_tokens, initializer = loop_syntax.split_loop_initializer(header[separator+1:], self.spelling)
            return tuple(names), expression(source_tokens), initializer
        if len(header) < 3 or header[0].kind != 'IDENTIFIER' or header[1].text != 'in':
            raise ValueError('expected for name in iterable [with name = value]:')
        if len(header) >= 5 and header[-2].text == 'with' and header[-1].kind == 'IDENTIFIER':
            return header[0], expression(header[2:-2]), (header[-1],)
        source_tokens, initializer = loop_syntax.split_loop_initializer(header[2:], self.spelling)
        source = expression(source_tokens)
        if initializer and initializer[0].text == header[0].text:
            raise ValueError('iteration binding and with binding must be distinct')
        return header[0], source, initializer

    def declare(self, node, context):
        initializer = node.header[2]
        if len(initializer) == 1 and initializer[0].text != 'gpu':
            import_module('py_compiler.syntax.block.with').apply_pack(
                initializer[0].text, context, (*context.scope, node.identity), context.scope)
        super().declare(node, context)

    def lower(self, node, cfg, env, nodes, index):
        from py_compiler.syntax.grammar.expression import Expression
        if (node.header[1].kind == 'name' and node.header[1].token.text == 'chaos'
                and cfg.lookup(node.header[1], env) is None):
            from py_compiler.syntax.grammar.expansion import visible_imports
            if 'chaos' not in {item.name for item in visible_imports(node.syntax)}:
                raise ValueError('chaos replay requires test with chaos')
            from py_compiler.syntax.complex.chaos import replay
            return replay(node, cfg, env)
        if len(node.header[2]) == 1:
            if node.header[2][0].text == 'gpu':
                from py_compiler.syntax.complex.gpu import lower
                return lower(node, cfg, env)
            node = replace(node, header=(*node.header[:2], ()))
        from py_compiler.syntax.block.grammar import BoundGrammar, expression_grammar
        from py_compiler.syntax.generic.place import expression_type
        from py_compiler.syntax.generic.storage import root_binding
        _, source, initializer = node.header
        temporary_owner = None
        if source.kind == 'name' and cfg.lookup(source, env) is None:
            enum = cfg.context.types.get(cfg.context.declaration_name(source.token.text, cfg.declaration_scope))
            if enum is not None and enum.family == 'enum':
                from py_compiler.syntax.grammar.expression import Expression
                variants = tuple(Expression('member', source.token.with_text(name), (source,)) for name in enum.variants)
                source = Expression('list', source.token, variants)
        if source.kind == 'list':
            return self.lower_literal(node, cfg, env, source, initializer)
        match = expression_grammar(source, cfg, env)
        if match is not None and match.provider.result.family == 'iterator':
            provider, captures = match.provider, match.captures
            if set(captures) != {'self'}:
                raise ValueError('iterator grammar currently requires only a self capture')
            source = captures['self'].term
        else:
            if source.kind == 'call':
                from py_compiler.syntax.complex.tensor import ArgumentValue
                value = cfg.expr(source, env)
                owner = root_binding(source,cfg,env) or cfg.call_result_owners.get(value.identity)
                if owner is None:
                    temporary = '<iterator expression ' + str(value.identity) + '>'
                    owner = Binding(node.identity + ':' + temporary, temporary, value.type, False, 'own', node.span)
                    temporary_owner = owner
                    env[temporary] = owner
                    cfg.local_names.add(temporary)
                    cfg.values[owner.identity] = value
                    cfg.body.bindings.append(owner)
                    cfg.record_lifetime('declare',owner.identity,owner.name)
                source = Expression('capture',source.token,(ArgumentValue(value,owner),))
                type_ = value.type
            else:
                type_ = expression_type(source, cfg, env)
            if getattr(type_.declaration if type_.family == 'record' else None, 'template_name', '') == 'tuple':
                members = tuple(Expression('index', source.token, (source,
                    Expression('literal', source.token.with_text(str(i), kind='NUMBER'))))
                    for i in range(len(type_.generic_arguments)))
                return self.lower_literal(node, cfg, env, Expression('list', source.token, members), initializer, heterogeneous=True)
            if type_.family == 'iterator':
                return self.lower_stored(node, cfg, env, source, initializer)
            from py_compiler.syntax.complex.collection import collection_iterator
            collection_iterator(type_, cfg.context)
            candidates = [BoundGrammar(d, cfg, env) for d in cfg.context.sentence_grammars
                          if d[3] is not None and d[3].family == 'iterator'
                          and (tuple(cfg.declaration_scope[:len(d[2])]) == d[2] or
                               getattr(getattr(type_, 'declaration', None), 'template_name', '').startswith('.'.join(d[2]) + '.'))
                          and d[1] == ('iterator', '(', ('self', type_), ')')]
            if len(candidates) != 1:
                raise ValueError('for requires one applicable iterator(self) grammar')
            provider = candidates[0]
        if not cfg.block_scope.allows(provider.node.scope):
            raise ValueError('production scope cannot use a test grammar')
        if provider.node.identity in cfg.active_grammars:
            raise ValueError('recursive iterator expansion requires a frame implementation')
        owner = root_binding(source, cfg, env)
        value = cfg.expr(source, env)
        if owner is None:
            temporary = '<iterator source ' + str(value.identity) + '>'
            owner = Binding(node.identity + ':' + temporary, temporary, value.type, False, 'own', node.span)
            env[temporary] = owner
            cfg.local_names.add(temporary)
            cfg.values[owner.identity] = value
            cfg.body.bindings.append(owner)
        loop_env, local_names = initialize_for(node, initializer, cfg, env)
        loan = Binding(node.identity + ':iterator-view', ':iterator-view', owner.type, True, 'view', node.span)
        shape = provider.arrangement == ('indices', '(', ('self', value.type), ')')
        cfg.flow.begin_loan(loan.identity, cfg.flow.resolve(owner.identity), shape=shape)
        loop_env[loan.name] = loan
        cfg.values[loan.identity] = cfg.read_binding(owner, node.span)
        cfg.body.bindings.append(loan)
        carried = tuple(b for b in bindings(loop_env) if b.identity not in cfg.storage and b.identity not in cfg.flow.moved)
        following = cfg.block(tuple(b.type for b in carried))
        ambient = {b.identity for scope in cfg.active_scopes for b in scope.values()} | cfg.live_binding_ids(loop_env)
        consumer = Consumer(node, cfg, loop_env, carried, following, ambient, provider.result.element)
        original_values = dict(cfg.values)
        previous = cfg.grammar_captures, cfg.yield_handler, cfg.loops
        cfg.loops = []
        cfg.grammar_captures = {'self': FrozenCapture(value, owner)}
        cfg.yield_handler = consumer.emit
        cfg.active_grammars.add(provider.node.identity)
        # Hidden names carry consumer SSA through producer loops without name capture.
        producer_env = {':consumer:' + b.identity: b for b in loop_env.values()}
        cfg.grammar_expansions += 1
        suffix = ':iterator:' + str(cfg.grammar_expansions)
        def clone(n):
            return replace(n, identity=n.identity + suffix, children=[clone(c) for c in n.children])
        try:
            cfg.scope([clone(n) for n in provider.node.children], producer_env)
            if cfg.current.terminator is None:
                cfg.flow.end_scope(ambient)
                edge(cfg, following, carried, node.span)
                consumer.exits.append((cfg.current, dict(cfg.values), cfg.flow.fork()))
        finally:
            cfg.grammar_captures, cfg.yield_handler, cfg.loops = previous
            cfg.active_grammars.remove(provider.node.identity)
        from py_compiler.mir.ownership.flow import reconcile_owners
        reconcile_owners(cfg, consumer.exits, carried, node.span)
        moved = set.union(*(flow.moved for _, _, flow in consumer.exits)) if consumer.exits else set()
        retained = [(binding, parameter) for binding, parameter in zip(carried, following.parameters)
                    if not (callable(getattr(binding.type, 'release', None)) and binding.identity in moved)]
        following.parameters = tuple(parameter for _, parameter in retained)
        for block, values, _ in consumer.exits:
            edge_values = tuple(values[b.identity] for b, _ in retained)
            block.terminator = Terminator('jump', (Edge(following.identity, edge_values),))
        cfg.current, cfg.values = following, original_values
        cfg.values.update({b.identity: v for b, v in retained})
        cfg.flow = FlowState.join([flow for _, _, flow in consumer.exits]) if consumer.exits else consumer.flow
        cfg.flow.end_loan(loan.identity)
        cfg.cleanup_memory((loop_env[n] for n in local_names), node.span)
        if temporary_owner is not None:
            cfg.cleanup_memory((temporary_owner,), node.span)
            env.pop(temporary_owner.name, None)
            cfg.local_names.discard(temporary_owner.name)
        cfg.flow.end_scope(cfg.live_binding_ids(env) | {b.identity for scope in cfg.active_scopes for b in scope.values()})
        return 1


import unittest


class ForInitializerTests(unittest.TestCase):
    def compile(self, source):
        from py_compiler.frontend.src.lib import compile_source
        from py_compiler.syntax.recognition import Syntax
        from py_compiler.mlir.src.lib import lower, render
        result = compile_source('for-initializer.sev', source, Syntax(include_tests=False))
        self.assertFalse(result.diagnostics, str(result.diagnostics))
        for body in result.program.bodies:
            body.verify()
        return render(lower(result.program))

    def test_item_dependent_initializer_runs_for_each_literal_item(self):
        output = self.compile('''def seed(value: int) -> int:
    return value
def scan() -> int:
    total = 0
    for n in [2, 3] with minimum = seed(n):
        total += minimum
    return total
''')
        calls = [line for line in output.splitlines() if 'func.call' in line and 'seed' in line]
        self.assertEqual(len(calls), 2, output)

    def test_independent_initializer_still_runs_once(self):
        output = self.compile('''def seed() -> int:
    return 0
def scan() -> int:
    total = 0
    for n in [2, 3] with count = seed():
        count += 1
        total += count
    return total
''')
        calls = [line for line in output.splitlines() if 'func.call' in line and 'seed' in line]
        self.assertEqual(len(calls), 1, output)

    def test_empty_iteration_does_not_evaluate_item_initializer(self):
        output = self.compile('''def seed(value: int) -> int:
    return value
def scan() -> int:
    for n in [] with minimum = seed(n):
        return minimum
    return 0
''')
        calls = [line for line in output.splitlines() if 'func.call' in line and 'seed' in line]
        self.assertEqual(calls, [], output)

    def test_item_binding_shadows_outer_name_in_initializer(self):
        self.compile('''def scan() -> int:
    n = "outer"
    total = 0
    for n in [2, 3] with minimum = n + 1:
        total += minimum
    return total
''')

    def test_tuple_iteration_binds_all_fields_before_initializer(self):
        self.compile('''def scan() -> int:
    total = 0
    for n, value in [(0, 7), (1, 9)] with minimum = value + n:
        total += minimum
    return total
''')

    def test_inline_and_stored_iterators_bind_item_before_indexing(self):
        producer = '''class Indices:
    count: int
    grammar iteration["iterator", "(", self, ")"]() -> iterator[int]:
        n = 0
        while n < self.count:
            yield n
            n += 1
'''
        for source, setup in (('indices', ''), ('cursor', '    cursor = iterator(indices)\n')):
            with self.subTest(source=source):
                self.compile(producer + '''def scan() -> int:
    values = tensor[int](3)
    indices = Indices(3)
    total = 0
''' + setup + f'''    for n in {source} with minimum = values[n]:
        minimum += n
        total += minimum
    return total
''')

    def test_item_initializer_memory_is_cleaned_on_loop_exits(self):
        for transfer in ('continue', 'break', 'return 0'):
            with self.subTest(transfer=transfer):
                self.compile('''def scan() -> int:
    for n in [2, 3] with scratch = tensor[int](n):
        scratch[0] = n
        if n == 2:
            ''' + transfer + '''
    return 1
''')

    def test_item_initializer_stays_in_iteration_scope(self):
        from py_compiler.frontend.src.lib import compile_source
        from py_compiler.syntax.recognition import Syntax
        result = compile_source('for-initializer-scope.sev', '''def scan() -> int:
    for n in [2, 3] with minimum = n:
        minimum += 1
    return minimum
''', Syntax(include_tests=False))
        self.assertIn("unknown binding 'minimum'", str(result.diagnostics))

    def test_new_surrounding_local_cannot_depend_on_a_possibly_missing_item(self):
        from py_compiler.frontend.src.lib import compile_source
        from py_compiler.syntax.recognition import Syntax
        result = compile_source('for-uninitialized-local.sev', '''def scan() -> int:
    for n in [2, 3] with local.minimum = n:
        minimum += 1
    return minimum
''', Syntax(include_tests=False))
        self.assertIn('the loop may be empty', str(result.diagnostics))




    def test_c_initializer_order_frequency_and_cleanup(self):
        import json
        import tempfile
        from pathlib import Path
        from py_compiler.packaging.testing import test_package
        source = '''class Indices:
    count: int
    grammar iteration["iterator", "(", self, ")"]() -> iterator[int]:
        n = 0
        while n < self.count:
            yield n
            n += 1

class Counter:
    calls: int
    def seed(self: borrow Self, value: int) -> int:
        self.calls += 1
        return value * 10

test "dependent initializers run after binding each yielded item":
    indices = Indices(3)
    counter = Counter(0)
    total = 0
    for n in indices with seed = counter.seed(n):
        assert(seed == n * 10)
        total += seed
        seed += 1000
    assert(total == 30 and counter.calls == 3)

test "independent initializers run once and retain state":
    indices = Indices(3)
    counter = Counter(0)
    total = 0
    for n in indices with seed = counter.seed(1):
        total += seed
        seed += 1
    assert(total == 33 and counter.calls == 1)

test "empty iterators do not call item dependent initializers":
    indices = Indices(0)
    counter = Counter(0)
    for n in indices with seed = counter.seed(n):
        assert(false)
    assert(counter.calls == 0)

test "qualified local initializers remain available after an empty loop":
    indices = Indices(0)
    for n in indices with local.min_index = 7:
        assert(false)
    assert(min_index == 7)

test "nested qualified local initializers belong to the enclosing body":
    indices = Indices(3)
    total = 0
    for n in indices with minimum = n:
        for i in [0, 1] with local.min_index = n:
            min_index += i
        assert(min_index == n + 1)
        total += min_index
    assert(total == 6)

test "item dependent qualified initializers update an existing local":
    indices = Indices(3)
    selected = -1
    for n in indices with local.selected = n:
        assert(selected == n)
    assert(selected == 2)

test "stored cursor initializers index the current item":
    values = tensor[int](3)
    values[0], values[1], values[2] = 7, 11, 13
    indices = Indices(3)
    cursor = iterator(indices)
    total = 0
    for n in cursor with minimum = values[n]:
        assert(minimum == values[n])
        total += minimum
    assert(total == 31)

test "initializer allocations are released on continue and break":
    indices = Indices(100)
    visits = 0
    for n in indices with scratch = tensor[int](n + 1):
        scratch[0] = n
        visits += 1
        if n < 4:
            continue
        break
    assert(visits == 5)

def leave() -> int:
    indices = Indices(3)
    for n in indices with scratch = tensor[int](n + 1):
        scratch[0] = n + 7
        return scratch[0]
    return -1

test "initializer allocations are released on return":
    assert(leave() == 7)
'''
        with tempfile.TemporaryDirectory(prefix='sev-for-initializer-') as directory:
            root = Path(directory)
            (root / 'src').mkdir()
            (root / 'src/main.sev').write_text(source)
            (root / 'package.json').write_text(json.dumps({
                'package': {'name': 'for-initializer', 'version': '0.1.0'},
            }))
            result = test_package(root, memory=True, coverage=True)
            report = json.loads(Path(result['report']).read_text()) if result.get('report') else result
            failures = [{key: item.get(key) for key in ('label', 'stderr', 'error', 'exit', 'memory')}
                        for item in report.get('tests', ()) if not item.get('passed')]
            self.assertTrue(result['success'], failures or result)


class LiteralIterationTests(unittest.TestCase):
    def test_literal_iteration_preserves_nested_continue_and_break_targets(self):
        from py_compiler.frontend.src.lib import compile_source
        from py_compiler.syntax.recognition import Syntax
        result = compile_source('literal-iteration.sev', '''def total() -> int:
    result = 0
    for value in [0, 1, 2, 31, 32, 33, 65, 128]:
        if value == 0:
            continue
        for factor in [1, 2]:
            result += value * factor
            if factor == 2:
                break
        if value == 65:
            break
    return result
''', Syntax(include_tests=False))
        self.assertFalse(result.diagnostics, str(result.diagnostics))
        for body in result.program.bodies:
            body.verify()

    @unittest.expectedFailure
    def test_inferred_none_minimum_across_loop_backedges_pending(self):
        # SIP 0032 permits inferred reassignments. The loop must solve the
        # None/int storage contract before checking its condition and backedge.
        from py_compiler.frontend.src.lib import compile_source
        from py_compiler.syntax.recognition import Syntax
        result = compile_source('inferred-loop-minimum.sev', '''def scan() -> bool:
    for value in [3, 1, 2] with minimum = None:
        if not minimum or value < minimum:
            minimum = value
    return true
''', Syntax(include_tests=False))
        self.assertFalse(result.diagnostics, str(result.diagnostics))
        for body in result.program.bodies:
            body.verify()
