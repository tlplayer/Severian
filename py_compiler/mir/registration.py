"""Group typed realizations under interfaces/modules and label their Atoms."""
from dataclasses import dataclass
from py_compiler.hir.thir.src.lib import identity


@dataclass(frozen=True)
class LabeledAtom:
    label: str
    submodule: str
    callable: str
    block: int
    position: int
    atom: object


@dataclass(frozen=True)
class MirSubmodule:
    typed: object
    label: str
    dependencies: tuple
    atoms: tuple


@dataclass(frozen=True)
class MirModule:
    identity: str
    interface: str
    submodules: tuple
    bodies: tuple


def register(typed):
    labels = {sub.identity: '__sev_submodule_' + sub.identity for sub in typed.submodules}
    if len(labels) != len(typed.submodules):
        raise ValueError('duplicate THIR realization at MIR handoff')
    owned = [callable_.body for sub in typed.submodules for callable_ in sub.callables]
    if (len({id(body) for body in owned}) != len(owned)
            or len({id(body) for body in typed.bodies}) != len(typed.bodies)
            or {id(body) for body in owned} != {id(body) for body in typed.bodies}):
        raise ValueError('MIR body membership disagrees with typed submodules')
    groups = []
    from py_compiler.mir.lowering import lower_body
    bodies = {}
    for submodule in typed.submodules:
        if any(dependency not in labels for dependency in submodule.dependencies):
            raise ValueError('MIR dependency has no typed submodule')
        atoms = []
        for callable_ in submodule.callables:
            if (not callable_.body.blocks
                    or tuple(value.type for value in callable_.body.blocks[0].parameters) != callable_.parameters
                    or callable_.body.result_type != callable_.result):
                raise ValueError('MIR callable signature changed after THIR')
            body = lower_body(callable_.body)
            bodies[id(callable_.body)] = body
            for block in body.blocks:
                for position, operation in enumerate(block.operations):
                    if operation.atom is None:
                        raise ValueError('MIR registration requires resolved Atoms')
                    operation.atom.verify(operation)
                    label = '__sev_atom_' + identity((submodule.identity, callable_.identity, block.identity, position))
                    atoms.append(LabeledAtom(label, labels[submodule.identity], callable_.body.name,
                                             block.identity, position, operation.atom))
        groups.append(MirSubmodule(submodule, labels[submodule.identity],
                                   tuple(labels[d] for d in submodule.dependencies), tuple(atoms)))
    return MirModule(typed.identity, typed.interface, tuple(groups), tuple(bodies[id(body)] for body in typed.bodies))


def interfaces(modules):
    """A package may group several source modules beneath one interface."""
    result = {}
    for module in modules:
        result.setdefault(module.interface, []).append(module)
    return {key: tuple(value) for key, value in result.items()}


import unittest


class RegistrationTests(unittest.TestCase):
    def test_lir_rejects_missing_empty_body(self):
        from dataclasses import replace
        from py_compiler.hir.thir.src.lib import ThirTests
        from py_compiler.lir.src.lib import lower
        result = ThirTests().compile()
        self.assertFalse(result.diagnostics, str(result.diagnostics))
        initializer = result.program.bodies[0]
        self.assertFalse(any(block.operations for block in initializer.blocks))
        incomplete = replace(result.program, bodies=result.program.bodies[1:])
        with self.assertRaisesRegex(ValueError, 'body membership'):
            lower(incomplete, 'x86_64-unknown-linux-gnu', 64)

    def test_mir_rejects_missing_body_and_changed_signature(self):
        from dataclasses import replace
        from py_compiler.hir.thir.src.lib import ThirTests
        result = ThirTests().compile(stop_after='thir')
        self.assertFalse(result.diagnostics, str(result.diagnostics))
        typed = result.program.typed_modules[0]
        with self.assertRaisesRegex(ValueError, 'body membership'):
            register(replace(typed, bodies=typed.bodies[1:]))
        selected = next(sub for sub in typed.submodules if any(call.name == 'hash' for call in sub.callables))
        changed = replace(selected, callables=tuple(replace(call, parameters=()) for call in selected.callables))
        submodules = tuple(changed if sub is selected else sub for sub in typed.submodules)
        with self.assertRaisesRegex(ValueError, 'signature changed'):
            register(replace(typed, submodules=submodules))

    def test_missing_typed_dependency_is_rejected(self):
        from dataclasses import replace
        from py_compiler.hir.thir.src.lib import ThirTests
        result = ThirTests().compile(stop_after='thir')
        self.assertFalse(result.diagnostics, str(result.diagnostics))
        typed = result.program.typed_modules[0]
        changed = replace(typed.submodules[0], dependencies=('missing',))
        with self.assertRaisesRegex(ValueError, 'dependency'):
            register(replace(typed, submodules=(changed, *typed.submodules[1:])))

    def test_hash_groups_and_lir_keep_mir_atom_labels(self):
        from py_compiler.hir.thir.src.lib import ThirTests
        from py_compiler.lir.src.lib import lower
        result = ThirTests().compile(interface='geometry')
        self.assertFalse(result.diagnostics, str(result.diagnostics))
        modules = result.program.mir_modules
        self.assertEqual(tuple(interfaces(modules)), ('geometry',))
        hashes = [sub for sub in modules[0].submodules if any(c.name == 'hash' for c in sub.typed.callables)]
        self.assertEqual(len(hashes), 3)
        self.assertEqual(len({sub.label for sub in hashes}), 3)
        expected = tuple(atom for module in modules for sub in module.submodules for atom in sub.atoms)
        lir = lower(result.program, 'x86_64-unknown-linux-gnu', 64)
        self.assertEqual(lir.atoms, expected)
        self.assertEqual(len({atom.label for atom in expected}), len(expected))

    def test_lir_rejects_a_thir_checkpoint(self):
        from py_compiler.hir.thir.src.lib import ThirTests
        from py_compiler.lir.src.lib import lower
        result = ThirTests().compile(stop_after='thir')
        self.assertFalse(result.diagnostics, str(result.diagnostics))
        with self.assertRaisesRegex(ValueError, 'MIR registration'):
            lower(result.program, 'x86_64-unknown-linux-gnu', 64)
