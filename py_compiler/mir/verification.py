"""Only the MIR verifier can publish a current, executable ReadyMir proof."""
from dataclasses import dataclass, fields, is_dataclass
from collections.abc import Mapping
from hashlib import sha256


def lifetime_contract(callable_):
    body = getattr(callable_, 'body', callable_)
    return (getattr(callable_, 'parameters', ()), getattr(callable_, 'result', None),
            getattr(callable_, 'parameter_ownership', {}),
            getattr(callable_, 'receiver_ownership', None),
            getattr(callable_, 'result_ownership', 'own'),
            getattr(body, 'result_sources', ()), getattr(body, 'retained_result_sources', ()),
            tuple((destination, tuple(sources)) for destination, sources in
                  getattr(body, 'retained_parameter_sources', ())),
            getattr(body, 'result_layout', ''))


def snapshot(value):
    """Capture semantics, including mutable providers, without source/AST cycles."""
    from py_compiler.syntax.function.contracts import Callable
    active = set()
    def visit(item):
        if item is None or isinstance(item, (str, int, bool, float, bytes)):
            return item
        kind = type(item).__module__ + '.' + type(item).__qualname__
        if id(item) in active:
            return ('recursive', kind)
        active.add(id(item))
        try:
            if isinstance(item, Callable):
                return (kind, item.symbol, visit(lifetime_contract(item)))
            if isinstance(item, Mapping):
                return tuple(sorted(((visit(k), visit(v)) for k, v in item.items()), key=repr))
            if isinstance(item, (list, tuple)):
                return tuple(visit(v) for v in item)
            if isinstance(item, (set, frozenset)):
                return tuple(sorted((visit(v) for v in item), key=repr))
            # Type declarations contain source grammar and parent scope graphs.
            # The resolved type's semantic identity is independent of those graphs.
            if hasattr(item, 'family') and hasattr(item, 'mlir'):
                names = ('name', 'family', 'mlir', 'bits', 'signed', 'variants', 'element',
                         'generic_arguments', 'location', 'owns_fields', 'binding_ownership',
                         'parameter_ownership')
                return (kind, tuple((name, visit(getattr(item, name, None))) for name in names))
            ignored = {'source_snapshot', 'source_span', 'span', 'origin', 'node', 'lowering_position'}
            if is_dataclass(item):
                return (kind, tuple((f.name, visit(getattr(item, f.name))) for f in fields(item)
                                    if f.name not in ignored))
            if hasattr(item, '__dict__'):
                return (kind, tuple((name, visit(value)) for name, value in sorted(vars(item).items())
                                    if name not in ignored and not name.startswith('_')))
            return (kind, str(item))
        finally:
            active.remove(id(item))
    return sha256(repr(visit(value)).encode()).hexdigest()


def dependency_contracts(body):
    from py_compiler.syntax.function.contracts import Callable
    from py_compiler.mir.cleanup import Cleanup
    contracts = {}
    for block in body.blocks:
        for operation in block.operations:
            if isinstance(operation.payload, Cleanup):
                contracts.update(dependency_contracts(operation.payload.recipe))
            atom = getattr(operation, 'atom', None) or operation.payload
            owner = getattr(atom, 'owner', None)
            dependency = getattr(operation.payload, 'dependency', None)
            if isinstance(owner, Callable):
                dependency = owner
            if dependency is not None:
                contracts[dependency.symbol] = snapshot(lifetime_contract(dependency))
    return tuple(sorted(contracts.items()))


def artifact_contracts(bodies, program_bodies):
    """Include reachable lifetime contracts even when machine code is unchanged."""
    local = {body.name: body for body in program_bodies}
    pending, contracts = list(bodies), {}
    while pending:
        body = pending.pop()
        if body.name in contracts:
            continue
        require_ready(body)
        dependencies = dependency_contracts(body)
        contracts[body.name] = (snapshot(lifetime_contract(body)), dependencies)
        for block in body.blocks:
            for operation in block.operations:
                symbol = getattr(operation.payload, 'symbol', None)
                if symbol in local and symbol not in contracts:
                    pending.append(local[symbol])
    return tuple(sorted(contracts.items()))


_ISSUER = object()


@dataclass(frozen=True, init=False)
class ReadyMir:
    body: object
    revision: str

    def __init__(self, body, issuer=None):
        if issuer is not _ISSUER:
            raise ValueError('ReadyMir must be issued by the MIR verifier')
        object.__setattr__(self, 'body', body)
        object.__setattr__(self, 'revision', snapshot(body))

    def verify(self, body):
        if body is not self.body or snapshot(body) != self.revision:
            raise ValueError('stale ReadyMir: body or callable contract changed; repeat semantic verification')


def require_ready(body):
    ready = getattr(body, '_ready_mir', None)
    if not isinstance(ready, ReadyMir):
        raise ValueError('target lowering requires verifier-issued ReadyMir')
    ready.verify(body)


def verify_semantics(body):
    from py_compiler.mir.ownership.flow import verify_retained_lifetimes
    body.verify()
    verify_retained_lifetimes(body)


def prepare_body(body):
    from py_compiler.mir.cleanup import materialize
    from py_compiler.mir.memory import resolve, verify_memory
    # A changed callee requires THIR to re-establish the caller's borrow facts.
    dependencies = dependency_contracts(body)
    previous = getattr(body, '_checked_dependencies', body.call_contracts)
    if previous is None:
        previous = dependencies
    if previous != dependencies:
        raise ValueError('callable contract changed; repeat THIR checking')
    # Replaying a proof must start from the checked obligations. A mutated,
    # already materialized body may have lost a destructor, so cannot serve as
    # its own evidence that cleanup was complete.
    if getattr(body, '_ready_mir', None) is not None:
        require_ready(body)
    body._checked_dependencies = dependencies
    body._ready_mir = None
    verify_semantics(body)
    materialize(body)
    verify_semantics(body)
    resolve(body)
    verify_semantics(body)
    verify_memory(body)
    body._ready_mir = ReadyMir(body, _ISSUER)
    return body._ready_mir


def verify_target_clone(original, clone):
    """Constant symbol renaming is a target transform, with a fresh proof."""
    require_ready(original)
    from dataclasses import replace
    from py_compiler.hir.hir.src.program import Constant
    if len(original.blocks) != len(clone.blocks):
        raise ValueError('target clone may only rename constant symbols')
    normalized = []
    for before, after in zip(original.blocks, clone.blocks):
        if len(before.operations) != len(after.operations):
            raise ValueError('target clone may only rename constant symbols')
        operations = []
        for old, new in zip(before.operations, after.operations):
            if isinstance(old.payload, Constant) and isinstance(new.payload, Constant):
                payload = replace(new.payload, identity=old.payload.identity)
                new = replace(new, payload=payload, atom=payload.atom(new))
            operations.append(new)
        normalized.append(replace(after, operations=operations))
    if snapshot(replace(clone, blocks=normalized)) != snapshot(original):
        raise ValueError('target clone may only rename constant symbols')
    from py_compiler.mir.cleanup import Cleanup
    if any(isinstance(op.payload, Cleanup) for block in clone.blocks for op in block.operations):
        raise ValueError('target transform introduced unresolved cleanup')
    from py_compiler.mir.memory import verify_memory
    verify_semantics(clone)
    verify_memory(clone)
    clone._checked_dependencies = dependency_contracts(clone)
    clone._ready_mir = ReadyMir(clone, _ISSUER)


import unittest


class ReadinessTests(unittest.TestCase):
    def compile(self):
        from py_compiler.frontend.src.lib import compile_source
        from py_compiler.syntax.recognition import Syntax
        result = compile_source('ready.sev', 'def leaf(value: int) -> int:\n    return value + 1\n'
                                'def caller() -> int:\n    return leaf(41)\n', Syntax())
        self.assertFalse(result.diagnostics, str(result.diagnostics))
        return result.program

    def test_constructor_is_restricted(self):
        with self.assertRaisesRegex(ValueError, 'issued'):
            ReadyMir(None)

    def test_cfg_change_invalidates_existing_lir(self):
        from dataclasses import replace
        from py_compiler.lir.src.lib import lower
        program = self.compile()
        lir = lower(program, 'x86_64-unknown-linux-gnu', 64)
        body = next(b for b in program.bodies if b.declaration == 'caller')
        body.blocks[0].terminator = replace(body.blocks[0].terminator, kind='panic', value=None)
        with self.assertRaisesRegex(ValueError, 'stale ReadyMir'):
            lir.text()

    def test_changed_callee_contract_requires_rechecking_thir(self):
        from types import SimpleNamespace
        from py_compiler.lir.src.lib import lower
        program = self.compile()
        body = next(b for b in program.bodies if b.declaration == 'caller')
        call = next(op.payload for b in body.blocks for op in b.operations if op.kind == 'call')
        call.dependency.body = SimpleNamespace(result_sources=(0,))
        with self.assertRaisesRegex(ValueError, 'stale ReadyMir'):
            lower(program, 'x86_64-unknown-linux-gnu', 64)
        with self.assertRaisesRegex(ValueError, 'repeat THIR'):
            prepare_body(body)

    def test_unverified_body_cannot_bypass_lir_through_mlir(self):
        from dataclasses import replace
        from py_compiler.mlir.src.lib import lower
        program = self.compile()
        unverified = replace(program.bodies[-1])
        with self.assertRaisesRegex(ValueError, 'requires verifier-issued'):
            lower(replace(program, bodies=(unverified,)))

    def test_thir_cannot_be_reused_after_contract_change(self):
        from types import SimpleNamespace
        from py_compiler.mir.lowering import lower_body
        program = self.compile()
        typed = next(b for b in program.typed_modules[0].bodies if b.declaration == 'caller')
        call = next(op.payload for b in typed.blocks for op in b.operations if op.kind == 'call')
        call.dependency.body = SimpleNamespace(result_sources=(0,))
        with self.assertRaisesRegex(ValueError, 'repeat THIR'):
            lower_body(typed)

    def test_removed_cleanup_cannot_receive_a_fresh_proof(self):
        from py_compiler.frontend.src.lib import compile_source
        from py_compiler.syntax.recognition import Syntax
        result = compile_source('release.sev', 'def work():\n    values = array[int,2]()\n', Syntax())
        self.assertFalse(result.diagnostics, str(result.diagnostics))
        body = next(b for b in result.program.bodies if b.declaration == 'work')
        prepare_body(body)  # Unchanged verified MIR is safe to verify again.
        release = next((b, op) for b in body.blocks for op in b.operations if op.kind == 'deallocate')
        release[0].operations.remove(release[1])
        with self.assertRaisesRegex(ValueError, 'stale ReadyMir'):
            prepare_body(body)

    def test_target_clone_cannot_remove_code(self):
        from dataclasses import replace
        program = self.compile()
        body = next(b for b in program.bodies if b.declaration == 'caller')
        clone = replace(body, blocks=[replace(b, operations=[]) for b in body.blocks])
        with self.assertRaisesRegex(ValueError, 'only rename constant'):
            verify_target_clone(body, clone)

    def test_ownership_is_checked_after_both_transformations(self):
        from unittest.mock import patch
        from dataclasses import replace
        from py_compiler.frontend.source.source import Diagnostic
        from py_compiler.mir.lowering import lower_body
        from py_compiler.mir.cfg.cfg import Operation
        from py_compiler.mir.ownership.flow import LifetimeEvent
        from py_compiler.mir.cleanup import materialize
        from py_compiler.mir.memory import resolve
        typed = self.compile().typed_modules[0].bodies[-1]
        for target, transform in (('py_compiler.mir.cleanup.materialize', materialize),
                                  ('py_compiler.mir.memory.resolve', resolve)):
            with self.subTest(pass_=target):
                def corrupt(body):
                    transform(body)
                    for kind in ('declare', 'move', 'read'):
                        payload = LifetimeEvent(kind, 'bad-owner', '', frozenset())
                        operation = Operation('lifetime', None, (), payload, body.source_span)
                        body.blocks[0].operations.append(replace(operation, atom=payload.atom(operation)))
                with patch(target, side_effect=corrupt), self.assertRaisesRegex(Diagnostic, 'moved'):
                    lower_body(typed)
