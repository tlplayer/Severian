"""Typed destruction recipes become executable control flow only in MIR."""
from dataclasses import dataclass, replace


@dataclass(frozen=True)
class Cleanup:
    recipe: object

    def atom(self, operation):
        from py_compiler.syntax.generic.atom import Atom
        return Atom(self, tuple(v.type for v in operation.operands), None,
                    ('move',), ('cleanup',), self)

    def render_operation(self, operation):
        raise ValueError('unresolved cleanup obligation at target boundary')


def defer(value, release, checker, span):
    from py_compiler.hir.thir.src.checking import Checker
    from py_compiler.hir.thir.src.model import Terminator
    count = getattr(checker, 'cleanup_count', 0)
    checker.cleanup_count = count + 1
    identity = checker.body.name + ':cleanup:' + str(count)
    recipe = Checker(checker.source, checker.syntax, identity, identity, checker.context.type('absent'))
    recipe.context = checker.context
    recipe.unsafe_depth = checker.unsafe_depth
    recipe.body.source_span = span
    formal = recipe.value(value.type)
    recipe.current.parameters = (formal,)
    if value.identity in checker.stack_values:
        recipe.stack_values.add(formal.identity)
    release(formal, recipe, span)
    recipe.current.terminator = Terminator('finish')
    checker.effect('cleanup', (value,), Cleanup(recipe.body.freeze()), span)


def materialize(body):
    """Consume each obligation once; a second invocation leaves the CFG intact."""
    from py_compiler.mir.cfg.cfg import Value, Operation, ExecutionBlock, Edge, Terminator
    from py_compiler.mir.lowering import lower_body
    from py_compiler.syntax.generic.storage import FieldPlace
    next_value = max((v.identity for b in body.blocks for v in
                      (*b.parameters, *(op.result for op in b.operations if op.result))), default=-1) + 1
    while True:
        found = next(((block, position, op) for block in body.blocks
                      for position, op in enumerate(block.operations) if isinstance(op.payload, Cleanup)), None)
        if found is None:
            return
        block, position, obligation = found
        recipe = lower_body(obligation.payload.recipe, prepare=False)
        recipe.verify()
        mapping = {recipe.blocks[0].parameters[0].identity: obligation.operands[0]}
        for region in recipe.blocks:
            for old in (*region.parameters, *(op.result for op in region.operations if op.result)):
                if old.identity not in mapping:
                    mapping[old.identity] = Value(next_value, old.type)
                    next_value += 1
        def value(old):
            return mapping[old.identity] if old is not None else None
        def operation(old):
            payload = old.payload
            if isinstance(payload, FieldPlace):
                payload = replace(payload, receiver=value(payload.receiver))
            new = Operation(old.kind, value(old.result), tuple(value(v) for v in old.operands), payload, old.span)
            return replace(new, atom=payload.atom(new))
        if len(recipe.blocks) == 1 and recipe.blocks[0].terminator.kind == 'finish':
            block.operations[position:position + 1] = [operation(op) for op in recipe.blocks[0].operations]
            continue
        continuation = ExecutionBlock(len(body.blocks), (), block.operations[position + 1:], block.terminator)
        body.blocks.append(continuation)
        indices = {region.identity: len(body.blocks) + index for index, region in enumerate(recipe.blocks)}
        block.operations = block.operations[:position]
        block.terminator = Terminator('jump', (Edge(indices[0]),))
        for region in recipe.blocks:
            exit_ = region.terminator
            if exit_.kind == 'finish':
                exit_ = Terminator('jump', (Edge(continuation.identity),))
            else:
                exit_ = replace(exit_, value=value(exit_.value), edges=tuple(
                    Edge(indices[edge.target], tuple(value(v) for v in edge.arguments)) for edge in exit_.edges))
            body.blocks.append(ExecutionBlock(indices[region.identity],
                () if region.identity == 0 else tuple(value(v) for v in region.parameters),
                [operation(op) for op in region.operations], exit_))


import unittest


class CleanupTests(unittest.TestCase):
    def test_cleanup_is_deferred_and_materialization_is_idempotent(self):
        from py_compiler.frontend.src.lib import compile_source
        from py_compiler.syntax.recognition import Syntax
        from py_compiler.mir.lowering import lower_body
        from py_compiler.mir.verification import snapshot
        source = 'class Buffer:\n    data: array[int]\ndef work():\n    values = array[Buffer,2]()\n'
        result = compile_source('cleanup.sev', source, Syntax(), stop_after='thir')
        self.assertFalse(result.diagnostics, str(result.diagnostics))
        typed = next(b for b in result.program.typed_modules[0].bodies if b.declaration == 'work')
        self.assertTrue(any(isinstance(op.payload, Cleanup) for b in typed.blocks for op in b.operations))
        body = lower_body(typed, prepare=False)
        materialize(body)
        body.verify()
        first = snapshot(body)
        materialize(body)
        self.assertEqual(first, snapshot(body))
        self.assertFalse(any(isinstance(op.payload, Cleanup) for b in body.blocks for op in b.operations))
