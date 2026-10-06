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
            if c.type.family not in {"integer", "float", "bool", "char", "string", "pointer", "absence", "unit", "byte"}:
                raise ValueError(f"missing MLIR provider for {c.type.name}")
            if c.type.family != "unit" and not c.type.mlir:
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
        type_, family = c.type.mlir, c.type.family
        body = []
        if family == "string":
            data = list(c.value.encode("utf-8"))
            static = f"memref<{len(data)}xi8>"
            global_name = symbol + "_bytes"
            # memref.global visibility is a string attribute, unlike func.func's keyword.
            globals_.append(f'  memref.global "private" constant @{global_name} : {static} = dense<{json.dumps(data)}>')
            body += [f"%storage = memref.get_global @{global_name} : {static}",
                     f"%value = memref.cast %storage : {static} to {type_}"]
        elif family == "absence" or (family == "pointer" and c.value == 0):
            body.append(f"%value = llvm.mlir.zero : {type_}")
        elif family == "pointer":
            body += [f"%address = arith.constant {c.value} : i{c.type.bits}",
                     f"%value = llvm.inttoptr %address : i{c.type.bits} to {type_}"]
        elif family != "unit":
            if family == "float":
                # Preserve decimal precision until MLIR/APFloat selects target rounding.
                value = str(c.value)
                if "." not in value:
                    parts = value.upper().split("E")
                    value = parts[0] + ".0" + ("E" + parts[1] if len(parts) == 2 else "")
            else:
                value = str(int(c.value))
            body.append(f"%value = arith.constant {value} : {type_}")
        body.append(f"func.return %value : {type_}" if type_ else "func.return")
        result = f" -> {type_}" if type_ else ""
        attributes = f" attributes {{sev.type = {quoted(c.type.name)}, sev.source = {quoted(c.source)}, sev.scalar_start = {c.span.start} : i64, sev.scalar_end = {c.span.end} : i64}}"
        functions.append(f"  func.func @{symbol}(){result}{attributes} {{\n" +
                         "\n".join("    " + line for line in body) + "\n  }")
    from py_compiler.mlir.src.cfg import render_body
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
from py_compiler.syntax.primitive.catalog import primitives


class MlirTests(unittest.TestCase):
    def test_render_is_pure_and_retains_semantic_types(self):
        types = primitives()
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
        values = {"string": "😀", "bool": True, "char": 955, "absence": None, "unit": None,
                  "pointer": 0, "integer": 1, "float": Decimal("1.5"), "byte": 4}
        types = {t.name: t for t in primitives().values()}
        constants = tuple(Constant(t.name, t.name, t, values[t.family], Span("s", 0, 1), "x.sev", "=")
                          for t in types.values())
        verify_native(render(lower(Program(constants, ()))))

    def test_empty_string_and_nonzero_pointer_with_native_verifier(self):
        if not shutil.which(verifier_path()):
            self.skipTest("install mlir-opt to verify emitted dialect contracts")
        types = primitives()
        constants = (Constant("empty", "empty", types["string"], "", Span("s", 0, 2), "x.sev", "="),
                     Constant("address", "address", types["pointer"], 4096, Span("s", 3, 7), "x.sev", "="))
        verify_native(render(lower(Program(constants, ()))))
