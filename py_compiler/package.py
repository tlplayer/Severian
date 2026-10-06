"""Package publication: content-addressed IR, interface and metadata; index committed last."""
from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
from pathlib import Path
import json
import os
import re
import subprocess
import tempfile
from py_compiler.frontend.src.lib import compile_source
from py_compiler.hir.hir.src.program import Program
from py_compiler.mlir.src.lib import verify_native, verifier_path
from py_compiler.lir.src.lib import lower, compile_object
from py_compiler.syntax.recognition import Syntax


def encode(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n"


def digest(value):
    return sha256(value.encode()).hexdigest()


def write_atomic(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, delete=False) as stream:
        stream.write(text)
        temporary = stream.name
    os.replace(temporary, path)


def build(root, target="x86_64-unknown-linux-gnu", pointer_bits=64, jobs=4, mlir_opt=None):
    root = Path(root).resolve()
    output = root / "package.pkg"
    log = output / "debug/build/log.txt"
    def report(message, failure=False):
        write_atomic(log, message)
        write_atomic(output / "debug/build/error.txt", message if failure else "")
        write_atomic(output / "debug/build/warn.txt", "")
    report("Building with py_compiler\n")
    try:
        return _build(root, output, target, pointer_bits, jobs, mlir_opt, report)
    except (OSError, ValueError, subprocess.TimeoutExpired) as failure:
        report(str(failure) + "\n", True)
        raise


def _build(root, output, target, pointer_bits, jobs, mlir_opt, report):
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", target):
        raise ValueError("target must be a safe target-triple path component")
    if jobs < 1:
        raise ValueError("jobs must be positive")
    architecture = target.split("-", 1)[0]
    required_width = {"x86_64": 64, "aarch64": 64, "riscv64": 64,
                      "i386": 32, "i686": 32, "arm": 32, "wasm32": 32}.get(architecture)
    if required_width and required_width != pointer_bits:
        raise ValueError(f"{target} requires --pointer-bits {required_width}")
    syntax = Syntax(pointer_bits)
    syntax.types  # Validate target representation before reading source.
    manifest_text = (root / "package.json").read_text(encoding="utf-8")
    manifest = json.loads(manifest_text)
    if not isinstance(manifest, dict):
        raise ValueError("package.json must contain an object")
    package = manifest.get("package", {})
    if not isinstance(package, dict) or not package.get("name") or not package.get("version"):
        raise ValueError("package.json requires package.name and package.version")
    if not all(isinstance(package[field], str) for field in ("name", "version")):
        raise ValueError("package.name and package.version must be strings")
    if manifest.get("dependencies"):
        raise ValueError("dependency linking is outside the primitive milestone")
    files = sorted((root / "src").rglob("*.sev"))
    if not files:
        raise ValueError("package has no src/**/*.sev sources")
    snapshots = []
    for path in files:
        if not path.resolve().is_relative_to(root):
            raise ValueError("source symlink escapes the package")
        snapshots.append((path.relative_to(root).as_posix(), path.read_bytes().decode("utf-8")))
    lock = (root / "package.lock").read_text(encoding="utf-8") if (root / "package.lock").exists() else encode({"version": 1, "packages": []})
    # Executor.map preserves source order while independent source units compile concurrently.
    with ThreadPoolExecutor(max_workers=jobs) as executor:
        results = list(executor.map(lambda pair: compile_source(pair[0], pair[1], syntax), snapshots))
    errors = [diagnostic for result in results for diagnostic in result.diagnostics]
    if errors:
        report("\n\n".join(map(str, errors)) + "\n", True)
        return {"success": False, "diagnostics": len(errors), "sample": [str(e) for e in errors[:5]],
                "log": str(output / "debug/build/log.txt")}
    program = Program(tuple(c for r in results for c in r.program.constants),
                      tuple(s for r in results for s in r.program.submodules),
                      tuple(b for r in results for b in r.program.bodies),
                      tuple(d for r in results for d in r.program.declarations))
    lir = lower(program, target, pointer_bits)
    target_program = lir.mlir
    ir = lir.text()
    # Publishing an artifact requires the actual dialect verifier, not only our structural checks.
    verify_native(ir, mlir_opt)
    native = compile_object(lir, mlir_opt)
    compiler_root = Path(__file__).parent
    compiler_id = digest(encode([(p.relative_to(compiler_root).as_posix(), p.read_text())
                                for p in sorted(compiler_root.rglob("*.py"))]))
    content_id = digest(encode([manifest_text, lock, snapshots]))
    build_id = digest(encode([content_id, compiler_id, target, pointer_bits]))
    abi_id = digest(encode([target, pointer_bits, "py-primitive-abi-v1"]))
    identity = {"name": package["name"], "version": package["version"], "content-id": content_id}
    ir_path = f"artifacts/{target}/{build_id}/ir/package.mlir"
    object_path = f"artifacts/{target}/{build_id}/object/package.o"
    llvm_path = f"artifacts/{target}/{build_id}/ir/package.ll"
    lowered_path = f"artifacts/{target}/{build_id}/ir/package.llvm.mlir"
    interface_path = f"package.pkgi/severian/{build_id}/lib.sevi"
    realization_path = f"metadata/realizations/{build_id}.json"
    declarations, symbols = [], []
    for function in target_program.functions:
        c = function.constant
        # File-qualified exports avoid conflating independent module scopes.
        name = f"{c.source}::{c.name}" if c.name else c.identity
        contract = f"{c.name or '<literal>'}: {c.type.name} {c.binding_operator or ''}".strip()
        symbols.append({"id": c.identity, "name": name, "kind": "constant", "type-id": c.type.name,
                        "exported": c.name is not None, "entry": function.symbol,
                        "entry-kind": "constant-materializer", "source": "source/" + c.source})
        if c.name:
            declarations.append({"name": name, "symbol-id": c.identity, "source": "source/" + c.source,
                                 "contract": contract})
    for declaration in program.declarations:
        declarations.append({"name": declaration.name, "symbol-id": declaration.identity,
                             "source": "source/" + declaration.source,
                             "contract": encode({"kind": declaration.kind, "fields": declaration.fields,
                                                 "variants": declaration.variants, "traits": declaration.traits})})
    for body in program.bodies:
        if body.declaration:
            parameters = ", ".join(f"{binding.name}: {binding.type.name}"
                                   for binding in body.bindings[:len(body.blocks[0].parameters)])
            declarations.append({"name": f"{body.source}::{body.declaration}", "symbol-id": body.identity,
                                 "source": "source/" + body.source,
                                 "contract": f"def {body.declaration}({parameters}) -> {body.result_type.name}",
                                 "complexity": body.complexity, "test": body.test})
        symbols.append({"id": body.identity, "name": body.name, "kind": "execution-body", "entry": body.name,
                        "complexity": body.complexity, "test": body.test,
                        "bindings": [{"id": b.identity, "name": b.name, "type": b.type.name,
                                      "constant": b.constant, "ownership": b.ownership} for b in body.bindings]})
    semantic_id = digest(encode(declarations))
    interface = {"format": "severian.interface", "version": 1, "package": identity,
                 "semantic-id": semantic_id, "metadata": {"build-id": build_id, "path": realization_path},
                 "declarations": declarations}
    interface_text = encode(interface)
    realization = {"format": "py_compiler.realization", "version": 1, "package": identity,
                   "build-id": build_id, "state": "compiled-object", "compiler": compiler_id,
                   "target": target, "abi": abi_id, "pointer-bits": pointer_bits,
                   "interface": {"path": interface_path, "checksum": digest(interface_text)},
                   "artifacts": [{"path": ir_path, "kind": "mlir", "checksum": digest(ir)}],
                   "dependencies": list(native.dependencies), "native-artifacts": [{"path": object_path, "kind": "object", "checksum": sha256(native.object_bytes).hexdigest()}]}
    realization["artifacts"].extend([{"path": llvm_path, "kind": "llvm-ir", "checksum": digest(native.llvm_ir)},
                                     {"path": lowered_path, "kind": "llvm-dialect", "checksum": digest(native.llvm_dialect)},
                                     *realization["native-artifacts"]])
    realization["initializers"] = [body.name for body in program.bodies if not body.declaration]
    # No native layout is claimed before target data-layout conversion.
    layouts = {"abi": abi_id, "state": "representation-only", "native-layouts": [],
               "representations": [{"type": t.name, "mlir": t.mlir, "bits": t.bits}
                                   for t in {t.name: t for t in syntax.types.values()}.values()]}
    pending = {ir_path: ir, llvm_path: native.llvm_ir, lowered_path: native.llvm_dialect, interface_path: interface_text, realization_path: encode(realization),
               f"metadata/symbols/{build_id}.json": encode(symbols),
               f"metadata/dependencies/{build_id}.json": encode([{"submodule": sub.identity, "declarations": sub.declarations, "dependencies": sub.dependencies} for sub in program.submodules]),
               f"metadata/layouts/{abi_id}.json": encode(layouts),
               f"build/{build_id}/{target}/artifacts.json": encode(realization["artifacts"]),
               "source/package.json": manifest_text, "source/package.lock": lock}
    pending.update({"source/" + path: text for path, text in snapshots})
    for path, text in pending.items():
        write_atomic(output / path, text)
    object_destination = output / object_path
    object_destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=object_destination.parent, delete=False) as stream:
        stream.write(native.object_bytes)
        temporary = stream.name
    os.replace(temporary, object_destination)
    # Publish the interface index only after every artifact has been written.
    index_path = output / "package.pkgi/index.json"
    index = json.loads(index_path.read_text()) if index_path.exists() else {"interfaces": [], "binding-sets": []}
    if not isinstance(index.get("interfaces"), list):
        raise ValueError("existing interface index uses an incompatible schema")
    entries = [entry for entry in index["interfaces"] if entry.get("build-id") != build_id]
    entries.append({"build-id": build_id, "path": interface_path, "semantic-id": semantic_id})
    index["interfaces"] = entries
    write_atomic(index_path, encode(index))
    report(f"Built {len(snapshots)} source files; {len(program.constants)} primitive values\nMLIR: {ir_path}\n")
    return {"success": True, "build-id": build_id, "ir": str(output / ir_path), "object": str(output / object_path),
            "log": str(output / "debug/build/log.txt")}


import unittest


class PackageTests(unittest.TestCase):
    def test_independent_errors_do_not_publish(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "src").mkdir()
            (root / "package.json").write_text(encode({"package": {"name": "example", "version": "0.1.0"}}))
            (root / "src/a.sev").write_text("x: u8 = 256\n")
            (root / "src/b.sev").write_text("class X:\n")
            result = build(root)
            self.assertFalse(result["success"])
            self.assertEqual(result["diagnostics"], 2)
            self.assertFalse((root / "package.pkg/package.pkgi/index.json").exists())

    def test_verified_package_has_reproducible_identity(self):
        import shutil
        if not shutil.which(verifier_path()):
            self.skipTest("install mlir-opt for package publication test")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "src").mkdir()
            (root / "package.json").write_text(encode({"package": {"name": "geometry", "version": "0.1.0"}}))
            (root / "src/constants.sev").write_text("size: i32 = 42\ntext = \"λ\"\n")
            first, second = build(root), build(root)
            self.assertEqual(first["build-id"], second["build-id"])
            index = json.loads((root / "package.pkg/package.pkgi/index.json").read_text())
            self.assertEqual(len(index["interfaces"]), 1)
            self.assertTrue(Path(first["ir"]).is_file())
