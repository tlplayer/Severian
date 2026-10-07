"""SIP-0030: typed construction, pure rendering, native dialect verification."""
from dataclasses import dataclass
from decimal import Decimal
import json
import subprocess
import shutil
from py_compiler.hir.hir.src.program import Constant, Program


def quoted(value):
    # MLIR strings use escaped UTF-8 bytes, not JSON's Unicode escape syntax.
    return '"' + ''.join(chr(b) if 32 <= b < 127 and b not in (34, 92) else f"\\{b:02X}"
                         for b in value.encode("utf-8")) + '"'


@dataclass(frozen=True)
class MlirFunction:
    symbol: str
    constant: Constant


@dataclass(frozen=True)
class MlirProgram:
    functions: tuple[MlirFunction, ...]
    bodies: tuple = ()
    declarations: tuple = ()

    def verify(self):
        symbols = set()
        for function in self.functions:
            if function.symbol in symbols:
                raise ValueError("duplicate MLIR symbol")
            symbols.add(function.symbol)
            c = function.constant
            if not callable(getattr(c.type, "render_constant", None)):
                raise ValueError(f"missing MLIR provider for {c.type.name}")
            if not c.type.mlir and not c.type.no_result:
                raise ValueError("value has no MLIR representation")
        for body in self.bodies:
            if body.name in symbols:
                raise ValueError("duplicate executable symbol")
            symbols.add(body.name)
            body.verify()


def lower(program):
    # Constant materializers are implementation symbols, not source functions.
    from hashlib import sha256
    result = MlirProgram(tuple(MlirFunction("__sev_constant_" + sha256(c.identity.encode()).hexdigest(), c)
                              for c in program.constants), program.bodies, program.declarations)
    result.verify()
    return result


def render(program):
    program.verify()
    globals_, functions = [], []
    declarations = {place.identity: place for body in program.bodies for place in body.storage}
    globals_.extend("  " + place.declaration() for place in declarations.values())
    for function in program.functions:
        c, symbol = function.constant, function.symbol
        type_ = c.type.mlir
        definitions, body = c.type.render_constant(c.value, symbol)
        globals_.extend('  ' + line for line in definitions)
        body.append(f"func.return %value : {type_}" if type_ else "func.return")
        result = f" -> {type_}" if type_ else ""
        attributes = f" attributes {{sev.type = {quoted(c.type.name)}, sev.source = {quoted(c.source)}, sev.scalar_start = {c.span.start} : i64, sev.scalar_end = {c.span.end} : i64}}"
        functions.append(f"  func.func @{symbol}(){result}{attributes} {{\n" +
                         "\n".join("    " + line for line in body) + "\n  }")
    from py_compiler.mlir.src.cfg import render_body
    from py_compiler.syntax.generic.atom import ExternalSymbol
    external = {}
    defined = {body.name for body in program.bodies} | {f.symbol for f in program.functions}
    for body in program.bodies:
        for block in body.blocks:
            for operation in block.operations:
                implementation = operation.atom.implementation if operation.atom else None
                if isinstance(implementation, ExternalSymbol):
                    signature = (tuple(t.mlir for t in operation.atom.inputs), operation.atom.result.mlir if operation.atom.result else "")
                    if implementation.symbol in defined:
                        raise ValueError("external symbol conflicts with a local definition")
                    if implementation.symbol in external and external[implementation.symbol] != signature:
                        raise ValueError("conflicting external atom signatures")
                    external[implementation.symbol] = signature
    for symbol, (inputs, result) in sorted(external.items()):
        suffix = " -> " + result if result else ""
        functions.append(f"  func.func private @{symbol}({', '.join(inputs)}){suffix}")
    functions.extend(render_body(body) for body in program.bodies)
    contracts = [{"name": d.name, "kind": d.kind, "fields": d.fields, "variants": d.variants, "traits": d.traits}
                 for d in program.declarations]
    attributes = " attributes {sev.declarations = " + quoted(json.dumps(contracts)) + "}" if contracts else ""
    return "module" + attributes + " {\n" + "\n".join(globals_ + functions) + "\n}\n"


def verifier_path():
    for executable in ("mlir-opt", "mlir-opt-21", "mlir-opt-20"):
        found = shutil.which(executable)
        if found:
            return found
    return "mlir-opt"


def verify_native(text, executable=None):
    executable = executable or verifier_path()
    completed = subprocess.run([executable], input=text, text=True, capture_output=True, timeout=60)
    if completed.returncode:
        raise ValueError("MLIR verifier rejected output:\n" + completed.stderr)


import unittest
from py_compiler.frontend.source.source import Span
from py_compiler.syntax.prelude import type_definitions


class MlirTests(unittest.TestCase):
    def test_render_is_pure_and_retains_semantic_types(self):
        types = type_definitions(64)
        constants = tuple(Constant(name, name, types[name], value, Span("s", 0, 1), "x.sev", "=")
                          for name, value in (("u8", 255), ("char", 128512), ("byte", 4), ("string", "λ")))
        target = lower(Program(constants, ()))
        text = render(target)
        self.assertEqual(text, render(target))
        self.assertIn('sev.type = "byte"', text)
        self.assertIn("memref<2xi8>", text)
        self.assertIn("arith.constant 128512 : i32", text)

    def test_all_declared_representations_with_native_verifier(self):
        import shutil
        if not shutil.which(verifier_path()):
            self.skipTest("install mlir-opt to verify emitted dialect contracts")
        values = {"string": "😀", "bool": True, "char": 955, "absence": None,
                  "pointer": 0, "integer": 1, "float": Decimal("1.5"), "byte": 4}
        types = {t.name: t for t in type_definitions(64).values() if t.family in values}
        constants = tuple(Constant(t.name, t.name, t, values[t.family], Span("s", 0, 1), "x.sev", "=")
                          for t in types.values())
        verify_native(render(lower(Program(constants, ()))))

    def test_empty_string_and_nonzero_pointer_with_native_verifier(self):
        if not shutil.which(verifier_path()):
            self.skipTest("install mlir-opt to verify emitted dialect contracts")
        types = type_definitions(64)
        constants = (Constant("empty", "empty", types["string"], "", Span("s", 0, 2), "x.sev", "="),
                     Constant("address", "address", types["pointer"], 4096, Span("s", 3, 7), "x.sev", "="))
        verify_native(render(lower(Program(constants, ()))))
