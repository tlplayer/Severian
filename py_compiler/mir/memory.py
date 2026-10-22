"""Resolve local placement and scoped arena/pool slabs before MIR registration."""
from dataclasses import dataclass, replace
from py_compiler.syntax.complex.array import MemoryOperation, RecordMemory
from py_compiler.syntax.complex.tensor import TensorOperation


@dataclass(frozen=True)
class Slab:
    name: str
    size: int
    alignment: int


@dataclass(frozen=True)
class EndLocal:
    def atom(self, operation):
        from py_compiler.syntax.generic.atom import Atom
        return Atom(self, tuple(v.type for v in operation.operands), None, ('move',), (), self)

    def render_operation(self, operation):
        return []


def resolve(body):
    operations = [op for block in body.blocks for op in block.operations]
    constants = {op.result.identity: op.payload.value for op in operations if op.kind == 'constant'}
    allocations = {op.result.identity: op for op in operations if isinstance(op.payload, (MemoryOperation, RecordMemory)) and op.payload.kind == 'allocate' and op.payload.plan}
    roots = {key: {key} for key in allocations}
    fresh = {op.result.identity for op in operations if op.result and op.result.type.family in ('array', 'pointer') and op.kind in ('call', 'memory-copy')}
    fresh.update(value.identity for value in body.blocks[0].parameters if value.type.family in ('array', 'pointer'))
    from py_compiler.syntax.generic.storage import FieldPlace
    # Owned field buffers are separate allocations, not the record's backing
    # slab. Their destructors must not be replaced by an end-of-slab marker.
    fresh.update(op.result.identity for op in operations
                 if op.result and op.kind == 'read' and isinstance(op.payload, FieldPlace)
                 and (op.result.type.family == 'string' or callable(getattr(op.result.type, 'release', None))))
    fresh.update(op.result.identity for op in operations if op.result
                 and isinstance(op.payload, RecordMemory) and op.payload.kind == 'load'
                 and op.result.type.family != 'record')
    roots.update({key: {key} for key in fresh})
    changed = True
    while changed:
        changed = False
        for block in body.blocks:
            for op in block.operations:
                if op.result and op.kind != 'memory-copy' and op.result.type.binding_ownership != 'copy' and op.result.identity not in allocations and op.result.identity not in fresh:
                    inherited = set().union(*(roots.get(v.identity, set()) for v in op.operands))
                    if inherited != roots.get(op.result.identity, set()):
                        roots[op.result.identity] = inherited; changed = True
            for edge in block.terminator.edges:
                for actual, formal in zip(edge.arguments, body.blocks[edge.target].parameters):
                    inherited = roots.get(formal.identity, set()) | roots.get(actual.identity, set())
                    if inherited != roots.get(formal.identity, set()):
                        roots[formal.identity] = inherited; changed = True
    escapes = set()
    for block in body.blocks:
        if block.terminator.value:
            escapes.update(roots.get(block.terminator.value.identity, ()))
        for op in block.operations:
            from py_compiler.syntax.generic.storage import FieldPlace
            owns_field = isinstance(op.payload, FieldPlace) and op.kind == 'write' and callable(getattr(op.payload.type, 'release', None))
            from py_compiler.syntax.complex.shared import SharedOperation
            shared_create = isinstance(op.payload, SharedOperation) and op.payload.kind == 'create'
            if op.kind in ('call', 'share-create') or owns_field or shared_create:
                for value in op.operands:
                    escapes.update(roots.get(value.identity, ()))
    def repeated(start):
        pending = [edge.target for edge in body.blocks[start].terminator.edges]
        visited = set()
        while pending:
            current = pending.pop()
            if current == start:
                return True
            if current not in visited:
                visited.add(current)
                pending.extend(edge.target for edge in body.blocks[current].terminator.edges)
        return False
    cyclic_allocations = {op.result.identity for block in body.blocks if repeated(block.identity)
                          for op in block.operations if op.result and op.result.identity in allocations}
    # Every incoming allocation of one descriptor must use a compatible release.
    # In particular, the pre-loop allocation cannot become stack storage while
    # its replacement on a backedge remains heap storage.
    joins = [roots.get(value.identity, set()) for block in body.blocks for value in block.parameters]
    forced_heap = set(escapes) | cyclic_allocations | fresh
    for identity, op in allocations.items():
        count = constants.get(op.operands[0].identity)
        size = op.payload.stride if isinstance(op.payload, RecordMemory) else max(1, (op.result.type.element.bits + 7) // 8)
        if op.payload.plan.location == 'heap' or not isinstance(count, int) or not 0 <= count * size <= 4096:
            forced_heap.add(identity)
    changed = True
    while changed:
        changed = False
        for incoming in joins:
            if incoming & forced_heap and not incoming <= forced_heap:
                forced_heap.update(incoming)
                changed = True
    for incoming in joins:
        explicit_stack = {identity for identity in incoming if identity in allocations and allocations[identity].payload.plan.location == 'stack'}
        if explicit_stack and (incoming - explicit_stack) & forced_heap:
            raise ValueError('explicit stack storage cannot join a heap owner')
    replacements, local, regions = {}, set(), []
    groups = {}
    for identity, op in allocations.items():
        plan = op.payload.plan
        count = constants.get(op.operands[0].identity)
        size = op.payload.stride if isinstance(op.payload, RecordMemory) else max(1, (op.result.type.element.bits + 7) // 8)
        if plan.strategy in ('arena', 'pool'):
            if identity in escapes:
                raise ValueError('function arena/pool allocation cannot escape or transfer to shared ownership')
            if not isinstance(count, int) or count < 0 or identity in cyclic_allocations:
                raise ValueError('function arena/pool requires fixed counts outside repeated allocation sites')
            groups.setdefault((plan.strategy, plan.alignment), []).append((identity, op, count, size))
        elif plan.location == 'infer':
            placement = 'stack' if identity not in forced_heap and isinstance(count, int) and 0 <= count * size <= 4096 else 'heap'
            replacements[identity] = replace(op.payload, plan=replace(plan, location=placement))
            if placement == 'stack':
                local.add(identity)
        elif plan.location == 'stack':
            local.add(identity)
    positions = {id(op): index for index, op in enumerate(operations)}
    for (strategy, alignment), entries in groups.items():
        region = f'memory_region_{len(regions)}'
        cursor, slots = 0, []
        for identity, op, count, size in entries:
            required = max(1, count * size)
            start = positions[id(op)]
            end = max((positions[id(candidate)] for candidate in operations if any(identity in roots.get(v.identity, ()) for v in candidate.operands)), default=start)
            available = (next((slot for slot in slots if slot[2] < start and slot[1] >= required), None)
                         if strategy == 'pool' and len(body.blocks) == 1 else None)
            if available:
                offset = available[0]; available[2] = end
            else:
                offset = (cursor + alignment - 1) // alignment * alignment
                cursor = offset + required
                slots.append([offset, required, end])
            if isinstance(op.payload, RecordMemory):
                replacements[identity] = replace(op.payload, plan=replace(op.payload.plan, location='heap'), region=region, region_offset=offset)
            else:
                replacements[identity] = MemoryOperation('allocate', replace(op.payload.plan, location='heap'), region, offset)
            local.add(identity)
        regions.append(Slab(region, cursor, alignment))
    body.memory_regions = tuple(regions)
    for block in body.blocks:
        for index, op in enumerate(block.operations):
            payload = replacements.get(op.result.identity) if op.result else None
            if isinstance(op.payload, (MemoryOperation, TensorOperation)) and op.payload.kind == 'release' and roots.get(op.operands[0].identity, set()) and roots[op.operands[0].identity] <= local:
                payload = EndLocal()
            if payload is not None:
                updated = replace(op, payload=payload)
                block.operations[index] = replace(updated, atom=payload.atom(updated))
    return roots, local


def verify_memory(body):
    """Check the plan's fixed point and every elided deallocation's provenance."""
    from py_compiler.mir.verification import snapshot
    candidate = replace(body, blocks=[replace(block, operations=list(block.operations)) for block in body.blocks])
    roots, local = resolve(candidate)
    if snapshot(candidate) != snapshot(body):
        raise ValueError('memory obligations are unresolved or the placement plan is stale')
    slabs = {slab.name: slab for slab in body.memory_regions}
    if len(slabs) != len(body.memory_regions):
        raise ValueError('duplicate memory region')
    for slab in slabs.values():
        if slab.size < 0 or slab.alignment <= 0 or slab.alignment & (slab.alignment - 1):
            raise ValueError('invalid memory region layout')
    for block in body.blocks:
        for operation in block.operations:
            payload = operation.payload
            if isinstance(payload, EndLocal):
                sources = roots.get(operation.operands[0].identity, set())
                # Shared handles end after their counter/backing release; those
                # are not array placement decisions.
                if operation.kind != 'share-end' and (not sources or not sources <= local):
                    raise ValueError('local cleanup has no local allocation proof')
            if isinstance(payload, (MemoryOperation, RecordMemory)) and payload.kind == 'allocate' and payload.plan:
                if payload.plan.location not in ('stack', 'heap'):
                    raise ValueError('unresolved allocation placement')
                if payload.region and payload.region not in slabs:
                    raise ValueError('allocation has no memory region')


import unittest


class PlacementTests(unittest.TestCase):
    def test_c_regions_temporaries_and_loop_replacement(self):
        import json
        import tempfile
        from pathlib import Path
        from py_compiler.packaging.testing import test_package
        source = '''class Buffer:
    data: array[int]
    def size(self: view Self) -> int:
        return len(self.data)
    def values(self: view Self) -> view array[int]:
        return self.data

test "temporary owners retain saved views":
    assert(Buffer(array[int,2]()).size() == 2)
    saved = Buffer(array[int,3]()).values()
    assert(len(saved) == 3)

test "loop replacement reaches continue and break with live storage":
    owner = array[int,1]()
    position = 0
    while position < 5:
        drop owner
        owner = array[int,2]()
        owner[0] = position
        position += 1
        if position < 4:
            continue
        break
    assert(owner[0] == 3)
    assert(len(owner) == 2)

test with compiler "a backedge still requires a replacement owner":
    reject "loop cannot consume":
        def invalid():
            owner = array[int,1]()
            while true:
                drop owner
                continue

test "arena destroys initialized records and retains inner owned buffers":
    unsafe:
        slots = allocate[array[Buffer],strategy=memory.arena](2)
        source = Buffer(array[int,1]())
        source.data[0] = 42
        slots[1] = source
        saved = copy slots[1]
        drop slots
        assert(saved.data[0] == 42)

test "pool keeps simultaneously live branch allocations separate":
    unsafe:
        a = allocate[int,strategy=memory.pool](8)
        a[0] = 42
        if true:
            b = allocate[int,strategy=memory.pool](8)
            b[0] = 7
            assert(a[0] == 42)
        assert(a[0] == 42)
'''
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'src').mkdir()
            (root / 'src/lib.sev').write_text(source)
            (root / 'package.json').write_text(json.dumps({'package': {'name': 'memory-regressions', 'version': '0.1.0'}}))
            result = test_package(root)
            self.assertTrue(result['success'], result)

    def test_record_regions_preserve_owned_field_cleanup(self):
        from py_compiler.frontend.src.lib import compile_source
        from py_compiler.syntax.recognition import Syntax
        from py_compiler.mlir.src.lib import lower, render, verify_c
        for strategy in ('arena', 'pool'):
            source = ('class Buffer:\n    data: array[int]\n'
                      f'def work(flag: bool) -> int:\n    unsafe:\n        values = allocate[array[Buffer],strategy=memory.{strategy}](2)\n'
                      '        if flag:\n            values[0] = Buffer(array[int,1]())\n'
                      '        values[1] = Buffer(array[int,2]())\n        return 42\n')
            result = compile_source('record-region.sev', source, Syntax())
            self.assertFalse(result.diagnostics, str(result.diagnostics))
            text = render(lower(result.program))
            self.assertIn('memref.view %memory_region_0', text)
            self.assertIn('read of uninitialized array element', text)
            self.assertGreaterEqual(text.count('memref.dealloc'), 3)
            verify_c(text)

    def test_branches_have_distinct_pool_slots(self):
        from py_compiler.frontend.src.lib import compile_source
        from py_compiler.syntax.recognition import Syntax
        source = ('def work(flag: bool) -> int:\n    unsafe:\n'
                  '        a = allocate[int,strategy=memory.pool](8)\n'
                  '        if flag:\n            b = allocate[int,strategy=memory.pool](8)\n            b[0] = 7\n'
                  '        a[0] = 42\n        return a[0]\n')
        result = compile_source('branch-region.sev', source, Syntax())
        self.assertFalse(result.diagnostics, str(result.diagnostics))
        body = next(body for body in result.program.bodies if body.declaration == 'work')
        self.assertEqual(body.memory_regions[0].size, 128)

    def test_local_stack_and_escaping_heap(self):
        from py_compiler.frontend.src.lib import compile_source
        from py_compiler.syntax.recognition import Syntax
        from py_compiler.mlir.src.lib import lower, render
        for ending, allocator in (('return a[0]', 'memref.alloca'), ('return a', 'memref.alloc(')):
            result = compile_source('place.sev', f'def work():\n    unsafe:\n        a = allocate[int]()\n        {ending}\n', Syntax())
            self.assertFalse(result.diagnostics, str(result.diagnostics))
            self.assertIn(allocator, render(lower(result.program)))

    def test_arena_and_pool_reuse(self):
        from py_compiler.frontend.src.lib import compile_source
        from py_compiler.syntax.recognition import Syntax
        from py_compiler.mlir.src.lib import lower, render, verify_c
        for strategy, size in (('arena', 128), ('pool', 64)):
            result = compile_source('region.sev', f'def work():\n    unsafe:\n        a = allocate[int,strategy=memory.{strategy}](8)\n        drop a\n        b = allocate[int,strategy=memory.{strategy}](8)\n', Syntax())
            self.assertFalse(result.diagnostics, str(result.diagnostics))
            body = next(b for b in result.program.bodies if b.declaration == 'work')
            self.assertEqual(body.memory_regions[0].size, size)
            text = render(lower(result.program))
            self.assertEqual(text.count('memref.dealloc'), 1)
            verify_c(text)
