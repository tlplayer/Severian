"""Construct executable MIR exclusively from checked THIR bodies."""
from dataclasses import fields, replace


def lower_body(typed, *, prepare=True):
    from py_compiler.hir.thir.src.model import TypedBody, thaw_data
    from py_compiler.mir.cfg.cfg import Body, ExecutionBlock, Value, Operation, Edge, Terminator
    if not isinstance(typed, TypedBody):
        raise ValueError('MIR requires a finalized typed body')
    typed.verify()
    values = {}
    def value(source):
        if source is None:
            return None
        existing = values.get(source.identity)
        if existing is None:
            existing = values[source.identity] = Value(source.identity, source.type)
        elif existing.type != source.type:
            raise ValueError('typed value changes type at MIR boundary')
        return existing
    body = Body(typed.identity, typed.name, typed.source, typed.result_type)
    for description in fields(Body):
        if description.name != 'blocks':
            data = thaw_data(getattr(typed, description.name))
            if description.name in ('bindings', 'storage'):
                data = list(data)
            setattr(body, description.name, data)
    for region in typed.blocks:
        operations = []
        for expression in region.operations:
            operation = Operation(expression.kind, value(expression.result),
                                  tuple(value(operand) for operand in expression.operands),
                                  expression.payload, expression.span)
            atom = expression.payload.atom(operation)
            atom.verify(operation)
            contract = expression.contract
            if (atom.inputs, atom.result, atom.ownership, atom.effects) != (
                    contract.inputs, contract.result, contract.ownership, contract.effects):
                raise ValueError('operation contract changed after type checking')
            operations.append(replace(operation, atom=atom))
        exit_ = region.terminator
        terminator = Terminator(exit_.kind,
            tuple(Edge(edge.target, tuple(value(argument) for argument in edge.arguments)) for edge in exit_.edges),
            value(exit_.value), exit_.message, exit_.span)
        body.blocks.append(ExecutionBlock(region.identity, tuple(value(parameter) for parameter in region.parameters),
                                          operations, terminator))
    if prepare:
        from py_compiler.mir.verification import prepare_body
        prepare_body(body)
    return body


import unittest


class LoweringTests(unittest.TestCase):
    def test_qualified_scopes_and_inferred_method_parameter(self):
        result = self.compile('x = 0\nclass foo:\n    x = 1\n    def add(a):\n        local.x: local int = 2 + self.x\n        return local.x + a\n    def outer() -> int:\n        return module.x\n')
        self.assertFalse(result.diagnostics, "\n".join(map(str, result.diagnostics)))
        add = next(body for body in result.program.bodies if body.declaration == "foo.add")
        self.assertEqual([v.type.name for v in add.blocks[0].parameters], ["foo", "i64"])
        self.assertEqual(add.result_type.name, "i64")
        from py_compiler.syntax.generic.storage import FieldPlace, GlobalPlace
        self.assertTrue(any(isinstance(o.payload, FieldPlace) for b in add.blocks for o in b.operations))
        outer = next(body for body in result.program.bodies if body.declaration == "foo.outer")
        self.assertTrue(any(isinstance(o.payload, GlobalPlace) for b in outer.blocks for o in b.operations))

    def test_local_qualifier_never_falls_back_to_global(self):
        result = self.compile('x = 0\ndef example() -> int:\n    return local.x\n')
        self.assertTrue(result.diagnostics)
        self.assertIn("unknown binding", str(result.diagnostics[0]))

    def test_scope_qualifiers_are_not_block_keywords(self):
        for spelling in ("module", "local", "block"):
            result = self.compile(f"{spelling}:\n    unimplemented\n")
            self.assertTrue(result.diagnostics)

    def test_unimplemented_emits_a_source_located_panic(self):
        from py_compiler.mlir.src.lib import lower, render
        result = self.compile("def pending() -> int:\n    unimplemented\n")
        self.assertFalse(result.diagnostics, str(result.diagnostics))
        body = result.program.bodies[-1]
        self.assertEqual(body.blocks[0].terminator.kind, "panic")
        self.assertEqual(body.blocks[0].terminator.message, "unimplemented at blocks.sev:2:5")
        ir = render(lower(result.program))
        self.assertIn("cf.assert", ir)
        self.assertIn("llvm.unreachable", ir)

    def compile(self, text):
        from py_compiler.frontend.src.lib import compile_source
        from py_compiler.syntax.recognition import Syntax
        return compile_source("blocks.sev", text, Syntax())

    def returned_constant(self, result):
        self.assertFalse(result.diagnostics, "\n".join(map(str, result.diagnostics)))
        body = result.program.bodies[-1]
        returned = body.blocks[-1].terminator.value
        return next(o.payload.value for b in body.blocks for o in b.operations
                    if o.result == returned and o.kind == "constant")

    def test_scalar_constant_binding_is_a_snapshot(self):
        result = self.compile("def example() -> int:\n    x = 1\n    y := x\n    x += 1\n    return y\n")
        self.assertEqual(self.returned_constant(result), 1)

    def test_default_and_explicit_views_follow_the_binding(self):
        for prefix in ("", "view "):
            result = self.compile(f'def example() -> string:\n    text = "old"\n    alias := {prefix}text\n    text = "new"\n    return alias\n')
            self.assertEqual(self.returned_constant(result), "new")

    def test_explicit_copy_does_not_follow_rebinding(self):
        result = self.compile('def example() -> string:\n    text = "old"\n    snapshot := copy text\n    text = "new"\n    return snapshot\n')
        self.assertEqual(self.returned_constant(result), "old")

    def test_view_handle_can_move_without_moving_owner(self):
        result = self.compile('def example() -> string:\n    text = "old"\n    alias := text\n    next_alias := move alias\n    text = "new"\n    return next_alias\n')
        self.assertEqual(self.returned_constant(result), "new")

    def test_rebinding_constant_is_rejected(self):
        result = self.compile('def example():\n    text = "old"\n    alias := text\n    alias = "new"\n')
        self.assertTrue(result.diagnostics)
        self.assertIn("constant", str(result.diagnostics[0]))

    def test_annotations_are_preserved_for_constructed_and_computed_values(self):
        result = self.compile('def example() -> i32:\n    x: i32 = u8(1)\n    return x\n')
        self.assertFalse(result.diagnostics, str(result.diagnostics))
        body = next(b for b in result.program.bodies if b.declaration == 'example')
        self.assertEqual(next(b for b in body.bindings if b.name == 'x').type.name, 'i32')
        result = self.compile('def example():\n    x: i32 = 1 == 1\n')
        self.assertTrue(result.diagnostics)
        self.assertIn('bool does not satisfy i32', str(result.diagnostics))

    def test_conditional_join_has_real_edges_and_parameters(self):
        result = self.compile('def choose(flag: bool) -> int:\n    x = 1\n    if flag:\n        x = 2\n    elif not flag:\n        x = 3\n    else:\n        x = 4\n    return x\n')
        self.assertFalse(result.diagnostics, str(result.diagnostics))
        body = result.program.bodies[-1]
        body.verify()
        self.assertEqual(sum(b.terminator.kind == "conditional" for b in body.blocks), 2)
        self.assertIn(body.blocks[-1].terminator.value, body.blocks[-1].parameters)

    def test_view_reads_joined_source_not_its_original_value(self):
        result = self.compile('def example(flag: bool) -> string:\n    text = "old"\n    alias := text\n    if flag:\n        text = "new"\n    return alias\n')
        self.assertFalse(result.diagnostics, str(result.diagnostics))
        body = result.program.bodies[-1]
        # Join parameters follow outer declaration order: flag, text, alias.
        self.assertEqual(body.blocks[-1].terminator.value, body.blocks[-1].parameters[1])

    def test_move_on_one_branch_rejects_joined_read(self):
        result = self.compile('def example(flag: bool) -> string:\n    text = "owned"\n    if flag:\n        taken := move text\n    return text\n')
        self.assertTrue(result.diagnostics)
        self.assertIn("moved", str(result.diagnostics[0]))

    def test_local_scope_does_not_leak_names(self):
        result = self.compile('def example() -> int:\n    if true:\n        hidden = 1\n    return hidden\n')
        self.assertTrue(result.diagnostics)
        self.assertIn("unknown binding", str(result.diagnostics[0]))

    def test_local_constant_can_shadow_without_rebinding_outer(self):
        result = self.compile('def example() -> int:\n    x := 1\n    if true:\n        x := 2\n    return x\n')
        self.assertFalse(result.diagnostics, str(result.diagnostics))
        body = result.program.bodies[-1]
        self.assertEqual(body.blocks[-1].terminator.value, body.blocks[-1].parameters[0])

    def test_class_trait_enum_contracts(self):
        result = self.compile('trait Named:\n    name: string\nclass Person: Named:\n    name: string\nenum Colour:\n    Red\n    Blue\n')
        self.assertFalse(result.diagnostics, str(result.diagnostics))
        self.assertEqual([d.kind for d in result.program.declarations], ["trait", "class", "enum"])
        bad = self.compile('trait Named:\n    name: string\nclass Person: Named:\n    name: int\n')
        self.assertTrue(bad.diagnostics)

    def test_control_flow_passes_native_dialect_verification(self):
        import shutil
        from py_compiler.mlir.src.lib import verifier_path, verify_native, render, lower
        if not shutil.which(verifier_path()):
            self.skipTest("install mlir-opt for native dialect verification")
        result = self.compile('def choose(flag: bool) -> int:\n    x = 1\n    if flag and true:\n        x += 2\n    else:\n        x = 4\n    return x\n')
        self.assertFalse(result.diagnostics, str(result.diagnostics))
        verify_native(render(lower(result.program)))

    def test_scope_storage_and_unimplemented_pass_native_verification(self):
        import shutil
        from py_compiler.mlir.src.lib import verifier_path, verify_native, render, lower
        if not shutil.which(verifier_path()):
            self.skipTest("install mlir-opt for native dialect verification")
        result = self.compile('x = 0\nclass foo:\n    x = 1\n    def add(a):\n        local.x: local int = 2 + self.x\n        return local.x + a\n    def outer() -> int:\n        return module.x\n    def pending() -> int:\n        unimplemented\n')
        self.assertFalse(result.diagnostics, "\n".join(map(str, result.diagnostics)))
        verify_native(render(lower(result.program)))


class AtomPipelineTests(unittest.TestCase):
    def test_returned_array_copy_does_not_borrow_its_input(self):
        program = self.compile('def snapshot(values: array[int]) -> array[int]:\n'
                               '    result = copy values\n    return result\n')
        body = next(body for body in program.bodies if body.declaration == 'snapshot')
        self.assertEqual(body.result_sources, ())
        self.assertTrue(any(operation.kind == 'memory-copy' for block in body.blocks for operation in block.operations))

    def compile(self, text):
        from py_compiler.frontend.src.lib import compile_source
        from py_compiler.syntax.recognition import Syntax
        result = compile_source("atoms.sev", text, Syntax())
        self.assertFalse(result.diagnostics, str(result.diagnostics))
        return result.program

    def test_compound_assignment_reaches_typed_atoms(self):
        program = self.compile("def example() -> int:\n    a = 1\n    a += 2\n    return a\n")
        operations = [o for body in program.bodies for block in body.blocks for o in block.operations]
        self.assertTrue(operations)
        self.assertTrue(all(o.atom is not None for o in operations))
        from py_compiler.lir.src.lib import lower
        self.assertIn("arith.addi", lower(program, "x86_64-unknown-linux-gnu", 64).text())

    def test_enum_owns_ordering_without_integer_family(self):
        program = self.compile("def example() -> bool:\n    return BigO.constant < BigO.linear\n")
        operation = next(o for body in program.bodies for block in body.blocks for o in block.operations if o.kind == "scalar")
        self.assertEqual(operation.atom.owner.family, "enum")
        self.assertEqual(operation.atom.ownership, ("view", "copy"))

    def test_user_enum_uses_its_own_variants(self):
        self.compile("enum Colour:\n    Red\n    Blue\ndef example() -> bool:\n    return Colour.Red < Colour.Blue\n")
