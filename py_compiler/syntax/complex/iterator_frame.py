"""Lower producer continuations to heap frame slots and ordinary MIR functions."""
from dataclasses import dataclass, replace
from hashlib import sha256
from py_compiler.hir.thir.src.model import Edge, Terminator
from py_compiler.syntax.generic.atom import Atom
from py_compiler.syntax.generic.storage import stored_type, pack_value, unpack_value, FieldPlace
from py_compiler.mlir.src.cfg import name, temporary_suffix


@dataclass(frozen=True)
class Slot:
    value: object
    storage: str
    inline: bool = False


@dataclass(frozen=True)
class FrameLayout:
    slots: tuple
    size: int
    bits: int

    @property
    def mlir(self):
        return '!llvm.struct<(' + ', '.join(slot.storage for slot in self.slots) + ')>'


@dataclass(frozen=True)
class FrameAccess:
    layout: FrameLayout
    index: int
    kind: str

    def atom(self, operation):
        return Atom(self, tuple(v.type for v in operation.operands), operation.result.type if operation.result else None,
                    tuple('view' for _ in operation.operands), ('read', 'mutation'), self)

    def render_operation(self, operation):
        tag = temporary_suffix(operation)
        slot = self.layout.slots[self.index]
        address = f'%frame_slot_{tag}_{self.index}'
        lines = [f'{address} = llvm.getelementptr {name(operation.operands[0])}[0, {self.index}] : (!llvm.ptr) -> !llvm.ptr, {self.layout.mlir}']
        if self.kind == 'load':
            if slot.inline:
                lines[0] = lines[0].replace(address + ' =', name(operation.result) + ' =')
            else:
                loaded = f'%frame_value_{tag}'
                lines.append(f'{loaded} = llvm.load {address} : !llvm.ptr -> {slot.storage}')
                if operation.result.type.family == 'record':
                    lines.append(f'{name(operation.result)} = llvm.bitcast {loaded} : !llvm.ptr to !llvm.ptr')
                else:
                    unpack_value(lines, loaded, operation.result.type, slot.storage, name(operation.result))
        else:
            value = operation.operands[1]
            stored = (pack_value(lines, name(value), value.type, slot.storage, f'%frame_packed_{tag}')
                      if slot.inline or value.type.family != 'record' else name(value))
            lines.append(f'llvm.store {stored}, {address} : {slot.storage}, !llvm.ptr')
        return lines


@dataclass(frozen=True)
class FrameInitialize:
    layout: FrameLayout
    index: int

    def atom(self, operation):
        return Atom(self, (operation.operands[0].type,), None, ('borrow',), ('mutation',), self)

    def render_operation(self, operation):
        tag = temporary_suffix(operation)
        storage = self.layout.slots[self.index].storage
        return [f'%frame_init_address_{tag} = llvm.getelementptr {name(operation.operands[0])}[0, {self.index}] : (!llvm.ptr) -> !llvm.ptr, {self.layout.mlir}',
                f'%frame_init_zero_{tag} = llvm.mlir.zero : {storage}',
                f'llvm.store %frame_init_zero_{tag}, %frame_init_address_{tag} : {storage}, !llvm.ptr']


@dataclass(frozen=True)
class FrameFree:
    def atom(self, operation):
        return Atom(self, (operation.operands[0].type,), None, ('move',), ('destruction',), self)

    def global_definitions(self):
        return ('llvm.func @free(!llvm.ptr)',)

    def render_operation(self, operation):
        return [f'llvm.call @free({name(operation.operands[0])}) : (!llvm.ptr) -> ()']


@dataclass(frozen=True)
class FrameCreate:
    layout: FrameLayout
    captures: tuple
    resume: str
    destroy: str
    result: object

    @property
    def symbol_contracts(self):
        return ((self.resume, ('!llvm.ptr',), self.result.mlir), (self.destroy, ('!llvm.ptr',), ''))

    def global_definitions(self):
        return (f'llvm.func @malloc(i{self.layout.bits}) -> !llvm.ptr',)

    def atom(self, operation):
        return Atom(self, tuple(v.type for v in operation.operands), operation.result.type,
                    tuple('view' for _ in operation.operands), ('allocation',), self)

    def render_operation(self, operation):
        from py_compiler.mlir.src.lib import symbol_ref
        tag = temporary_suffix(operation)
        frame = f'%iterator_frame_{tag}'
        lines = [f'%frame_size_{tag} = arith.constant {self.layout.size} : i{self.layout.bits}',
                 f'{frame} = llvm.call @malloc(%frame_size_{tag}) : (i{self.layout.bits}) -> !llvm.ptr',
                 f'%frame_zero_{tag} = llvm.mlir.zero : {self.layout.mlir}',
                 f'llvm.store %frame_zero_{tag}, {frame} : {self.layout.mlir}, !llvm.ptr',
                 f'%output_address_{tag} = llvm.getelementptr {frame}[0, 1] : (!llvm.ptr) -> !llvm.ptr, {self.layout.mlir}',
                 f'%output_zero_{tag} = llvm.mlir.zero : {self.result.mlir}',
                 f'%absent_tag_{tag} = arith.constant 1 : i32',
                 f'%output_none_{tag} = llvm.insertvalue %absent_tag_{tag}, %output_zero_{tag}[0] : {self.result.mlir}',
                 f'llvm.store %output_none_{tag}, %output_address_{tag} : {self.result.mlir}, !llvm.ptr']
        for index, value in zip(self.captures, operation.operands):
            slot = self.layout.slots[index]
            address = f'%capture_{tag}_{index}'
            lines.append(f'{address} = llvm.getelementptr {frame}[0, {index}] : (!llvm.ptr) -> !llvm.ptr, {self.layout.mlir}')
            stored = name(value) if value.type.family == 'record' else pack_value(lines, name(value), value.type, slot.storage, f'%packed_{tag}_{index}')
            lines.append(f'llvm.store {stored}, {address} : {slot.storage}, !llvm.ptr')
        for label, symbol, result in (('resume', self.resume, self.result.mlir), ('destroy', self.destroy, '()')):
            lines += [f'%{label}_function_{tag} = func.constant {symbol_ref(symbol)} : (!llvm.ptr) -> {result}',
                      f'%{label}_pointer_{tag} = builtin.unrealized_conversion_cast %{label}_function_{tag} : (!llvm.ptr) -> {result} to !llvm.ptr']
        type_ = operation.result.type.mlir
        lines += [f'%iterator_zero_{tag} = llvm.mlir.zero : {type_}',
                  f'%iterator_data_{tag} = llvm.insertvalue {frame}, %iterator_zero_{tag}[0] : {type_}',
                  f'%iterator_resume_{tag} = llvm.insertvalue %resume_pointer_{tag}, %iterator_data_{tag}[1] : {type_}',
                  f'{name(operation.result)} = llvm.insertvalue %destroy_pointer_{tag}, %iterator_resume_{tag}[2] : {type_}']
        return lines


def create(match, cfg):
    from py_compiler.hir.thir.src.checking import Checker
    from py_compiler.mir.ownership.flow import Binding, verify_retained_lifetimes
    from importlib import import_module
    FrozenCapture = import_module('py_compiler.syntax.block.for').FrozenCapture
    from py_compiler.syntax.function.calls import literal_value
    from py_compiler.syntax.function.layout import storage_layout, layout_of, Layout
    from py_compiler.syntax.generic.storage import copy_storage, owns_storage, release_storage
    from py_compiler.syntax.complex.object import Allocate, Invoke
    from py_compiler.syntax.block.union import UnionBox, UnionNone
    from py_compiler.syntax.function.contracts import Callable
    from py_compiler.syntax.generic.owned import select
    provider, iterator = match.provider, match.provider.result
    cfg.grammar_expansions += 1
    key = sha256((cfg.body.name + provider.node.identity + str(cfg.grammar_expansions)).encode()).hexdigest()[:24]
    producer = Checker(cfg.source, cfg.syntax, key, key, cfg.context.type('absent'))
    producer.body.source_span = provider.node.span
    producer.context, producer.block_scope = cfg.context, cfg.block_scope
    producer.declaration_scope = cfg.declaration_scope
    producer.namespaces = {'module': cfg.context.global_bindings}
    producer.storage = dict(cfg.context.global_storage)
    producer.fallback_scopes = [cfg.context.global_bindings]
    producer.active_grammars = {*cfg.active_grammars, provider.node.identity}
    parameters, arguments, owners = [], [], []
    for capture_name, capture in match.captures.items():
        owner = capture.binding(cfg)
        if owner is None:
            raise ValueError('stored iterator captures require a live binding')
        actual = capture.lower(cfg)
        formal = producer.value(actual.type)
        binding = Binding(key + ':' + capture_name, capture_name, actual.type, True, 'view', provider.node.span)
        producer.values[binding.identity] = formal
        producer.grammar_captures[capture_name] = FrozenCapture(formal, binding)
        producer.body.bindings.append(binding)
        parameters.append(formal)
        arguments.append(actual)
        owners.append(cfg.flow.resolve(owner.identity))
    producer.current.parameters = tuple(parameters)
    yields = {}
    def yielded(term, env, span):
        value = producer.expr(term, env, iterator.element)
        value = copy_storage(value, producer, span)
        resume = producer.block()
        bindings = {b.identity: b for scope in producer.active_scopes for b in scope.values()}
        live = tuple((b.identity, producer.values[b.identity]) for b in bindings.values()
                     if b.identity in producer.values and b.ownership in ('own', 'copy', 'move', 'mirror')
                     and owns_storage(b.type)
                     and (b.type.family != 'string' or producer.values[b.identity].identity in producer.owned_results))
        yields[producer.current.identity] = (value, resume.identity, live)
        producer.current.terminator = Terminator('jump', (Edge(resume.identity),))
        producer.current = resume
    producer.yield_handler = yielded
    producer.scope(provider.node.children, {})
    if producer.current.terminator is None:
        producer.current.terminator = Terminator('finish')
    source = producer.body
    source.verify()
    # Analyze the connected producer CFG before yield edges become returns.
    # Each resume state restores exactly the facts proven at its own yield.
    ownership = verify_retained_lifetimes(source)
    optional = iterator.result(cfg.context)
    values = {v.identity: v for block in source.blocks for v in (*block.parameters, *(op.result for op in block.operations if op.result))}
    definitions = {op.result.identity: op for block in source.blocks for op in block.operations if op.result}
    slot_values = [Slot(None, 'i32'), Slot(None, optional.mlir)]
    index_by_value = {}
    layouts = [Layout(4, 4), layout_of(optional, cfg.context)]
    for identity, value in sorted(values.items()):
        op = definitions.get(identity)
        inline = value.type.family == 'record' and op is not None and (
            isinstance(op.payload, Allocate) or bool(getattr(op.payload, 'result_layout', '')))
        storage = stored_type(value.type, cfg.context) if inline or value.type.family != 'record' else '!llvm.ptr'
        index_by_value[identity] = len(slot_values)
        slot_values.append(Slot(value, storage, inline))
        layouts.append(storage_layout(value.type, cfg.context) if inline or value.type.family != 'record'
                       else Layout(cfg.context.type('usize').bits // 8, cfg.context.type('usize').bits // 8))
    size, alignment = 0, 1
    for layout in layouts:
        size = (size + layout.alignment - 1) // layout.alignment * layout.alignment + layout.size
        alignment = max(alignment, layout.alignment)
    layout = FrameLayout(tuple(slot_values), (size + alignment - 1) // alignment * alignment, cfg.context.type('usize').bits)
    pointer = cfg.context.types['pointer']
    def helper(label, result):
        entry = Callable(provider.node, '__iterator_' + label + '_' + key, (('frame', 'pointer'),), result.name, ())
        entry.scope = provider.scope
        cfg.context.add_callable(entry, None)
        builder = Checker(cfg.source, cfg.syntax, entry.symbol, entry.symbol, result)
        builder.body.source_span = provider.node.span
        builder.context, builder.block_scope = cfg.context, cfg.block_scope
        builder.body.declaration = entry.name
        frame = builder.value(pointer)
        builder.current.parameters = (frame,)
        return entry, builder, frame
    resume_entry, resume, frame = helper('resume', optional)
    def load(builder, frame, value, span):
        return builder.emit('frame-load', value.type, (frame,), FrameAccess(layout, index_by_value[value.identity], 'load'), span)
    def store(builder, frame, index, value, span):
        builder.effect('frame-store', (frame, value), FrameAccess(layout, index, 'store'), span)
    span = provider.node.span
    previous = resume.emit('frame-output', optional, (frame,), FrameAccess(layout, 1, 'load'), span)
    if optional.release:
        optional.release(previous, resume, span)
    empty = resume.emit('union-none', optional, (), UnionNone(), span)
    store(resume, frame, 1, empty, span)
    state = resume.emit('frame-state', cfg.context.type('i32'), (frame,), FrameAccess(layout, 0, 'load'), span)
    blocks = [resume.block() for _ in source.blocks]
    starts = {0: 0, **{index + 1: point[1] for index, point in enumerate(yields.values())}}
    states = {block: index + 1 for index, block in enumerate(yields)}
    suspended = {states[block]: ownership[block] for block in yields if block in ownership}
    for number, target in starts.items():
        if number and number not in suspended:
            continue
        following = resume.block()
        restored = resume.block() if number else blocks[target]
        expected = literal_value(number, state.type, resume, span)
        condition = select(state.type, 'F.operator', '==').expand(resume, state, expected, span)
        resume.current.terminator = Terminator('conditional', (Edge(restored.identity), Edge(following.identity)), condition)
        if number:
            resume.current = restored
            resume.effect('ownership-restore', (), suspended[number], span)
            resume.current.terminator = Terminator('jump', (Edge(blocks[target].identity),))
        resume.current = following
    resume.current.terminator = Terminator('finish', value=empty)
    def transfer(edge):
        incoming = [load(resume, frame, v, span) for v in edge.arguments]
        for actual, formal in zip(incoming, source.blocks[edge.target].parameters):
            store(resume, frame, index_by_value[formal.identity], actual, span)
        resume.current.terminator = Terminator('jump', (Edge(blocks[edge.target].identity),))
    for original, block in zip(source.blocks, blocks):
        resume.current = block
        for op in original.operations:
            if isinstance(op.payload, Allocate):
                resume.effect('frame-initialize', (frame,), FrameInitialize(layout, index_by_value[op.result.identity]), op.span)
                continue  # Reused slots start with no initialized fields.
            operands = tuple(load(resume, frame, v, op.span) for v in op.operands)
            payload = op.payload
            if isinstance(payload, FieldPlace):
                payload = replace(payload, receiver=operands[0])
            if hasattr(payload, 'plan') and payload.plan is not None:
                if payload.plan.location == 'stack':
                    raise ValueError('stored iterator cannot retain explicit stack allocation across suspension')
                payload = replace(payload, plan=replace(payload.plan, location='heap'))
            if op.result:
                value = resume.emit(op.kind, op.result.type, operands, payload, op.span)
                store(resume, frame, index_by_value[op.result.identity], value, op.span)
            else:
                resume.effect(op.kind, operands, payload, op.span)
        if original.identity in yields:
            value, _, _ = yields[original.identity]
            item = load(resume, frame, value, span)
            output = resume.emit('union', optional, (item,), UnionBox(0), span)
            store(resume, frame, 1, output, span)
            following = literal_value(states[original.identity], state.type, resume, span)
            store(resume, frame, 0, following, span)
            resume.current.terminator = Terminator('finish', value=output)
        elif original.terminator.kind == 'jump':
            transfer(original.terminator.edges[0])
        elif original.terminator.kind == 'conditional':
            condition = load(resume, frame, original.terminator.value, span)
            left, right = resume.block(), resume.block()
            resume.current.terminator = Terminator('conditional', (Edge(left.identity), Edge(right.identity)), condition)
            for bridge, edge in zip((left, right), original.terminator.edges):
                resume.current = bridge
                transfer(edge)
        elif original.terminator.kind == 'panic':
            resume.current.terminator = original.terminator
        else:
            exhausted = literal_value(-1, state.type, resume, span)
            store(resume, frame, 0, exhausted, span)
            resume.current.terminator = Terminator('finish', value=empty)
    resume_entry.body = resume.finish()
    cfg.context.bodies.append(resume_entry.body)
    destroy_entry, destroy, frame = helper('destroy', cfg.context.type('absent'))
    state = destroy.emit('frame-state', state.type, (frame,), FrameAccess(layout, 0, 'load'), span)
    output = destroy.emit('frame-output', optional, (frame,), FrameAccess(layout, 1, 'load'), span)
    if optional.release:
        optional.release(output, destroy, span)
    for block, (_, _, live) in yields.items():
        if block not in ownership:
            continue
        chosen, following = destroy.block(), destroy.block()
        expected = literal_value(states[block], state.type, destroy, span)
        condition = select(state.type, 'F.operator', '==').expand(destroy, state, expected, span)
        destroy.current.terminator = Terminator('conditional', (Edge(chosen.identity), Edge(following.identity)), condition)
        destroy.current = chosen
        destroy.effect('ownership-restore', (), ownership[block], span)
        facts = ownership[block]
        released = set()
        pending = {identity: value for identity, value in live
                   if identity in facts.initialized and identity not in facts.moved}
        while pending:
            ready = [identity for identity in pending if not any(
                identity in sources and owner in pending
                for (owner, _), sources in facts.retained)]
            if not ready:
                raise ValueError('cyclic stored borrows prevent iterator destruction')
            value = pending.pop(ready[-1])
            if value.identity not in released:
                released.add(value.identity)
                restored = load(destroy, frame, value, span)
                release_storage(restored, destroy, span)
        destroy.current.terminator = Terminator('jump', (Edge(following.identity),))
        destroy.current = following
    destroy.effect('frame-free', (frame,), FrameFree(), span)
    destroy_entry.body = destroy.finish()
    cfg.context.bodies.append(destroy_entry.body)
    operation = FrameCreate(layout, tuple(index_by_value[p.identity] for p in parameters), resume_entry.symbol, destroy_entry.symbol, optional)
    result = cfg.emit('memory-copy', iterator, tuple(arguments), operation, span)
    cfg.value_dependencies[result.identity] = set(owners)
    cfg.value_sources[result.identity] = set().union(*(cfg.value_sources.get(a.identity, set()) for a in arguments))
    cfg.owned_results.add(result.identity)
    return result


import unittest


class IteratorFrameOwnershipTests(unittest.TestCase):
    source = '''class Counter:
    count: int
    grammar iterate["iterator", "(", self, ")"]() -> iterator[int]:
        consumed = 7
        drop consumed
        while position < self.count with position = 0:
            yield position
            step = 1
            yield position + step
            position += 1

test "each continuation restores its own initialized locals":
    counter = Counter(2)
    cursor = iterator(counter)
    assert(next(cursor).value == 0)
    assert(next(cursor).value == 1)
    assert(next(cursor).value == 1)
    assert(next(cursor).value == 2)
    assert(not next(cursor).has_value)
'''

    def test_resume_facts_are_specific_to_each_yield(self):
        from py_compiler.frontend.src.lib import compile_source
        from py_compiler.syntax.recognition import Syntax
        from py_compiler.mir.ownership.flow import LifetimeEvent, OwnershipRestore
        result = compile_source('frame-ownership.sev', self.source, Syntax())
        self.assertFalse(result.diagnostics, str(result.diagnostics))
        body = next(b for b in result.program.bodies if b.declaration.startswith('__iterator_resume_'))
        declarations = {op.payload.slot: op.payload.owner for block in body.blocks for op in block.operations
                        if isinstance(op.payload, LifetimeEvent) and op.payload.kind == 'declare'}
        restored = [op.payload for block in body.blocks for op in block.operations
                    if isinstance(op.payload, OwnershipRestore)]
        self.assertEqual(len(restored), 2)
        for facts in restored:
            self.assertIn(declarations['position'], facts.initialized)
            self.assertIn(declarations['consumed'], facts.moved)
            self.assertNotIn(declarations['consumed'], facts.initialized)
        self.assertNotIn(declarations['step'], restored[0].initialized)
        self.assertIn(declarations['step'], restored[1].initialized)

    def test_native_multiple_resumes(self):
        import json
        import tempfile
        from pathlib import Path
        from py_compiler.testing import test_package
        with tempfile.TemporaryDirectory(prefix='sev-frame-ownership-') as directory:
            root = Path(directory)
            (root / 'src').mkdir()
            (root / 'src/lib.sev').write_text(self.source)
            (root / 'package.json').write_text(json.dumps({'package': {'name': 'frame-ownership', 'version': '0.1.0'}}))
            result = test_package(root)
            self.assertTrue(result['success'], result)


class SuspendedCleanupTests(unittest.TestCase):
    def test_suspended_cleanup_keeps_owned_record_fields(self):
        from py_compiler.frontend.src.lib import compile_source
        from py_compiler.syntax.recognition import Syntax
        from py_compiler.syntax.complex.array import MemoryOperation
        source = """class Pair:
    left: array[int]
    right: array[int]
class Counter:
    count: int
    grammar iterate["iterator", "(", self, ")"]() -> iterator[int]:
        pair = Pair(array[int,1](), array[int,2]())
        taken = move pair.left
        yield self.count
        drop taken
        yield self.count

def work():
    counter = Counter(2)
    cursor = iterator(counter)
    next(cursor)
    drop cursor
"""
        result = compile_source('suspended-cleanup.sev', source, Syntax())
        self.assertFalse(result.diagnostics, str(result.diagnostics))
        body = next(b for b in result.program.bodies if b.declaration.startswith('__iterator_destroy_'))
        releases = [op for block in body.blocks for op in block.operations
                    if isinstance(op.payload, MemoryOperation) and op.payload.kind == 'release']
        self.assertGreaterEqual(len(releases), 3)
