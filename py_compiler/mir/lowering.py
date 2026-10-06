"""Resolve executable sentences into typed SSA, explicit CFG edges and ownership facts."""
from hashlib import sha256
from dataclasses import replace
from functools import wraps
from py_compiler.frontend.source.source import Diagnostic
from py_compiler.syntax.sentence.syntax import expression, OWNERSHIP
from py_compiler.frontend.parser.contract import Literal
from py_compiler.hir.hir.src.program import Constant, resolve_literal
from py_compiler.mir.cfg.cfg import Body, ExecutionBlock, Value, Operation, Edge, Terminator
from py_compiler.mir.ownership.flow import Binding, FlowState


def located(method):
    @wraps(method)
    def call(self, node, *args, **kwargs):
        try:
            return method(self, node, *args, **kwargs)
        except ValueError as failure:
            origin = node[0] if isinstance(node, list) else node
            span = origin.token.span if hasattr(origin, "token") else origin.span
            raise Diagnostic("MIR", str(failure), self.source, span) from failure
    return call


class Builder:
    def __init__(self, source, syntax, identity, name, result):
        self.source, self.syntax = source, syntax
        self.body = Body(identity, name, source.path, result)
        self.next_value = 0
        self.current = self.block()
        self.flow = FlowState()
        self.values = {}
        self.local_names = set()
        self.namespaces = {}
        self.fallback_scopes = []
        self.storage = {}
        self.storage_factory = None
        self.declared_nodes = set()

    def value(self, type_):
        value = Value(self.next_value, type_)
        self.next_value += 1
        return value

    def block(self, types=()):
        block = ExecutionBlock(len(self.body.blocks), tuple(self.value(t) for t in types))
        self.body.blocks.append(block)
        return block

    def emit(self, kind, type_, operands, payload, span):
        result = self.value(type_)
        self.current.operations.append(Operation(kind, result, tuple(operands), payload, span))
        return result

    def effect(self, kind, operands, payload, span):
        self.current.operations.append(Operation(kind, None, tuple(operands), payload, span))

    def lookup(self, node, env):
        if node.kind == "reference":
            provider = self.syntax.scope_providers[node.token.text]
            return provider.bindings(self, env).get(node.operands[0].text)
        for scope in (env, *self.fallback_scopes):
            if node.token.text in scope:
                return scope[node.token.text]
        return None

    def read_binding(self, binding, span):
        identity = self.flow.resolve(binding.identity)
        if identity in self.storage:
            return self.storage[identity].read(self, span)
        return self.values[identity]

    def infer(self, binding, type_, env):
        previous = self.values[binding.identity].type
        if previous is not None:
            if type_ and previous != type_:
                raise ValueError("conflicting parameter type constraints")
            type_ = previous
        if type_ is None:
            raise ValueError(f"type of {binding.name!r} requires an annotation or expected type")
        updated = replace(binding, type=type_)
        value = self.values[binding.identity]
        value = Value(value.identity, type_)
        self.values[binding.identity] = value
        self.body.blocks[0].parameters = tuple(value if p.identity == value.identity else p for p in self.body.blocks[0].parameters)
        self.body.bindings = [updated if b.identity == binding.identity else b for b in self.body.bindings]
        env[binding.name] = updated
        return updated

    def live_binding_ids(self, env):
        return {b.identity for scope in (env, *self.namespaces.values()) for b in scope.values()}

    @located
    def expr(self, node, env, expected=None):
        token = node.token
        if node.kind == "construct":
            type_ = self.syntax.types.get(token.text)
            if type_ is None:
                raise ValueError(f"constructor {token.text!r} has no executable provider")
            if expected and type_ != expected:
                raise ValueError(f"constructor {type_.name} does not satisfy {expected.name}")
            return self.expr(node.operands[0], env, type_)
        if node.kind == "literal" or (node.kind == "unary" and token.text in ("-", "+") and node.operands[0].kind == "literal"):
            literal_token = node.operands[0].token if node.kind == "unary" else token
            spelling = (token.text if node.kind == "unary" else "") + literal_token.text
            type_, value = resolve_literal(Literal("", token.span, spelling, literal_token.kind, expected.name if expected else None), self.syntax)
            if not type_.mlir:
                raise ValueError("unit is a no-result contract, not a stored runtime value")
            identity = self.body.identity + ":literal:" + str(self.next_value)
            constant = Constant(identity, None, type_, value, token.span, self.source.path, None)
            return self.emit("constant", type_, (), constant, token.span)
        if node.kind in ("name", "reference"):
            binding = self.lookup(node, env)
            if binding is None:
                raise ValueError(f"unknown binding {token.text!r}")
            if binding.type is None:
                binding = self.infer(binding, expected, env)
            self.flow.read(binding)
            if expected and binding.type != expected:
                raise ValueError(f"{binding.type.name} does not satisfy {expected.name}")
            return self.read_binding(binding, token.span)
        if node.kind == "unary":
            if token.text in OWNERSHIP:
                raise ValueError("ownership modifiers must apply to the entire binding initializer")
            operand = self.expr(node.operands[0], env, expected)
            if token.text == "not" and operand.type.name == "bool":
                return self.emit("not", operand.type, (operand,), None, token.span)
            raise ValueError(f"unary {token.text!r} lacks a provider for {operand.type.name}")
        if node.kind == "binary":
            operator = token.text
            if operator in ("and", "or"):
                return self.short_circuit(node, env)
            comparison = operator in ("==", "!=", "<", "<=", ">", ">=")
            left = self.expr(node.operands[0], env, None if comparison else expected)
            right = self.expr(node.operands[1], env, left.type)
            if left.type != right.type or left.type.family not in ("integer", "float", "bool", "char", "byte"):
                raise ValueError("operator requires matching scalar providers")
            comparison = operator in ("==", "!=", "<", "<=", ">", ">=")
            if not comparison and left.type.family in ("bool", "char"):
                raise ValueError("arithmetic is not defined by this scalar provider")
            result = self.syntax.types["bool"] if comparison else left.type
            return self.emit("binary", result, (left, right), operator, token.span)
        raise ValueError("unresolved expression")

    def short_circuit(self, node, env):
        left = self.expr(node.operands[0], env, self.syntax.types["bool"])
        rhs, join = self.block(), self.block((left.type,))
        direct = Edge(join.identity, (left,))
        edges = (Edge(rhs.identity), direct) if node.token.text == "and" else (direct, Edge(rhs.identity))
        self.current.terminator = Terminator("conditional", edges, left)
        self.current = rhs
        right = self.expr(node.operands[1], env, left.type)
        self.current.terminator = Terminator("jump", (Edge(join.identity, (right,)),))
        self.current = join
        return join.parameters[0]

    @located
    def sentence(self, node, env, local_names):
        for provider in self.syntax.sentence_providers:
            if provider.matches(node):
                return provider.lower(node, self, env, local_names)
        raise ValueError("sentence has no registered syntax provider")

    def scope(self, nodes, env, declarations=False, local_names=None):
        local_names = set() if local_names is None else local_names
        previous = self.local_names
        self.local_names = local_names
        index = 0
        try:
            while index < len(nodes):
                node = nodes[index]
                if self.current.terminator:
                    raise ValueError("unreachable sentence after a terminating operation")
                if node.provider is None:
                    self.sentence(node, env, local_names)
                    index += 1
                else:
                    index += node.provider.lower(node, self, env, nodes, index)
        finally:
            self.local_names = previous

    def finish(self):
        if self.body.result_type is None:
            self.body.result_type = self.syntax.types["unit"]
        if self.current.terminator is None:
            if self.body.result_type.mlir:
                raise ValueError("not every path returns the declared result")
            self.current.terminator = Terminator("finish")
        self.body.verify()
        return self.body


def prepare(source, root, syntax):
    from py_compiler.hir.hir.src.declarations import DeclarationContext
    context = DeclarationContext(source, syntax, Builder)
    context.declared_nodes = {node.identity for node in root.children if node.provider}
    for node in root.children:
        if node.provider:
            context.active_provider = node.provider
            node.provider.declare(node, context)
    initializer = Builder(source, syntax, root.identity, "__sev_init_" + source.identity, syntax.types["unit"])
    initializer.declared_nodes = context.declared_nodes
    root.provider.configure(initializer, context)
    initializer.scope(root.children, context.global_bindings)
    context.initial_flow = initializer.flow.fork()
    context.resolve()
    return (initializer.finish(), *context.bodies), tuple(context.declarations)


import unittest


class LoweringTests(unittest.TestCase):
    def test_qualified_scopes_and_inferred_method_parameter(self):
        result = self.compile('x = 0\nclass foo:\n    x = 1\n    def add(a):\n        local.x: local int = 2 + self.x\n        return local.x + a\n    def outer() -> int:\n        return global.x\n')
        self.assertFalse(result.diagnostics, "\n".join(map(str, result.diagnostics)))
        add = next(body for body in result.program.bodies if body.declaration == "foo.add")
        self.assertEqual([v.type.name for v in add.blocks[0].parameters], ["pointer", "i64"])
        self.assertEqual(add.result_type.name, "i64")
        from py_compiler.syntax.type.storage import FieldPlace, GlobalPlace
        self.assertTrue(any(isinstance(o.payload, FieldPlace) for b in add.blocks for o in b.operations))
        outer = next(body for body in result.program.bodies if body.declaration == "foo.outer")
        self.assertTrue(any(isinstance(o.payload, GlobalPlace) for b in outer.blocks for o in b.operations))

    def test_local_qualifier_never_falls_back_to_global(self):
        result = self.compile('x = 0\ndef example() -> int:\n    return local.x\n')
        self.assertTrue(result.diagnostics)
        self.assertIn("unknown binding", str(result.diagnostics[0]))

    def test_scope_qualifiers_are_not_block_keywords(self):
        for spelling in ("global", "local", "block"):
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
        for initializer in ("u8(1)", "1 == 1"):
            result = self.compile(f"def example():\n    x: i32 = {initializer}\n")
            self.assertTrue(result.diagnostics)

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
        result = self.compile('x = 0\nclass foo:\n    x = 1\n    def add(a):\n        local.x: local int = 2 + self.x\n        return local.x + a\n    def outer() -> int:\n        return global.x\n    def pending() -> int:\n        unimplemented\n')
        self.assertFalse(result.diagnostics, "\n".join(map(str, result.diagnostics)))
        verify_native(render(lower(result.program)))
