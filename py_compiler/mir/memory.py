"""Resolve local placement and scoped arena/pool slabs before MIR registration."""
from dataclasses import dataclass, replace
from py_compiler.syntax.complex.array import MemoryOperation


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
    allocations = {op.result.identity: op for op in operations if isinstance(op.payload, MemoryOperation) and op.payload.kind == 'allocate' and op.payload.plan}
    roots = {key: {key} for key in allocations}
    changed = True
    while changed:
        changed = False
        for block in body.blocks:
            for op in block.operations:
                if op.result and op.kind != 'memory-copy' and op.result.type.binding_ownership != 'copy' and op.result.identity not in allocations:
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
            if op.kind in ('call', 'share-create') or owns_field:
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
    replacements, local, regions = {}, set(), []
    groups = {}
    for identity, op in allocations.items():
        plan = op.payload.plan
        count = constants.get(op.operands[0].identity)
        size = max(1, (op.result.type.element.bits + 7) // 8)
        if plan.strategy in ('arena', 'pool'):
            if identity in escapes:
                raise ValueError('function arena/pool allocation cannot escape or transfer to shared ownership')
            if not isinstance(count, int) or count < 0 or len(body.blocks) != 1:
                raise ValueError('function arena/pool requires fixed counts in a straight-line region')
            groups.setdefault((plan.strategy, plan.alignment), []).append((identity, op, count, size))
        elif plan.location == 'infer':
            placement = 'stack' if identity not in escapes and identity not in cyclic_allocations and isinstance(count, int) and 0 <= count * size <= 4096 else 'heap'
            replacements[identity] = MemoryOperation('allocate', replace(plan, location=placement))
            if placement == 'stack':
                local.add(identity)
    positions = {id(op): index for index, op in enumerate(operations)}
    for (strategy, alignment), entries in groups.items():
        region = f'memory_region_{len(regions)}'
        cursor, slots = 0, []
        for identity, op, count, size in entries:
            required = max(1, count * size)
            start = positions[id(op)]
            end = max((positions[id(candidate)] for candidate in operations if any(identity in roots.get(v.identity, ()) for v in candidate.operands)), default=start)
            available = next((slot for slot in slots if slot[2] < start and slot[1] >= required), None) if strategy == 'pool' else None
            if available:
                offset = available[0]; available[2] = end
            else:
                offset = (cursor + alignment - 1) // alignment * alignment
                cursor = offset + required
                slots.append([offset, required, end])
            replacements[identity] = MemoryOperation('allocate', replace(op.payload.plan, location='heap'), region, offset)
            local.add(identity)
        regions.append(Slab(region, cursor, alignment))
    body.memory_regions = tuple(regions)
    for block in body.blocks:
        for index, op in enumerate(block.operations):
            payload = replacements.get(op.result.identity) if op.result else None
            if isinstance(op.payload, MemoryOperation) and op.payload.kind == 'release' and roots.get(op.operands[0].identity, set()) & local:
                payload = EndLocal()
            if payload is not None:
                updated = replace(op, payload=payload)
                block.operations[index] = replace(updated, atom=payload.atom(updated))


import unittest


class PlacementTests(unittest.TestCase):
    def test_local_stack_and_escaping_heap(self):
        from py_compiler.frontend.src.lib import compile_source
        from py_compiler.syntax.recognition import Syntax
        from py_compiler.mlir.src.lib import lower, render
        for ending, allocator in (('return a[0]', 'memref.alloca'), ('return a', 'memref.alloc(')):
            result = compile_source('place.sev', f'def work():\n    a = allocate[int]()\n    {ending}\n', Syntax())
            self.assertFalse(result.diagnostics, str(result.diagnostics))
            self.assertIn(allocator, render(lower(result.program)))

    def test_arena_and_pool_reuse(self):
        from py_compiler.frontend.src.lib import compile_source
        from py_compiler.syntax.recognition import Syntax
        from py_compiler.mlir.src.lib import lower, render, verify_native
        for strategy, size in (('arena', 128), ('pool', 64)):
            result = compile_source('region.sev', f'def work():\n    a = allocate[int,strategy=memory.{strategy}](8)\n    drop a\n    b = allocate[int,strategy=memory.{strategy}](8)\n', Syntax())
            self.assertFalse(result.diagnostics, str(result.diagnostics))
            body = next(b for b in result.program.bodies if b.declaration == 'work')
            self.assertEqual(body.memory_regions[0].size, size)
            text = render(lower(result.program))
            self.assertEqual(text.count('memref.dealloc'), 1)
            verify_native(text)
