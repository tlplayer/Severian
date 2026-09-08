#!/usr/bin/env python3
"""Thirty acceptance gates for the source-language compiler migration.

Run with the already-built sev_compiler binary. No test rebuilds it, invokes
the Rust compiler on a subject, skips an unsupported feature, or marks a
migration failure as expected. See migration/README.md for the IR contract.
"""
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import textwrap
import unittest

from bootstrap_mlir import ROOT, tool


COMPILER = Path(os.environ.get(
    "SEVERIAN_SOURCE_COMPILER", ROOT / "sev_compiler/target/host/dev/bin/sev_compiler"
))


def source(text):
    return textwrap.dedent(text).strip() + "\n"


def contract(name="Fuse", symbol="<~>", precedence=7, associativity="Left",
             body="return left * 10 + right", parent="G", extra=""):
    """An ordinary source G implementation, not a host-side operator registry."""
    return source(f"""
        trait {name}: {parent}:
            symbol: Y = {symbol}
            arity: int = 2
            precedence: int = {precedence}
            associativity: Associativity = {associativity}
            left: Operand = Value[int]
            right: Operand = Value[int]
            result: T = int
            evaluation: Evaluation = LeftToRight
            effects: Effects = Pure
            overflow: Overflow | absent = absent
            underflow: Underflow | absent = absent
            control_flow: ControlFlow | absent = absent
            {extra}

            def semantic(left: int, right: int) -> int:
                {body}

        extend int:
            operator {symbol}[G:{name}](right: Self) -> Self
    """)


class MigrationCase(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="sev-migration-")
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        self.binary_digest = hashlib.sha256(COMPILER.read_bytes()).hexdigest()

    def tearDown(self):
        self.assertEqual(hashlib.sha256(COMPILER.read_bytes()).hexdigest(),
                         self.binary_digest, "a source-only test rebuilt the compiler")

    def write(self, text, name="subject.sev"):
        path = self.directory / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(source(text))
        return path

    def invoke(self, arguments, *, cwd=None):
        return subprocess.run(list(map(str, arguments)), cwd=cwd or self.directory,
                              capture_output=True, text=True, timeout=180)

    def succeeds(self, arguments):
        result = self.invoke(arguments)
        self.assertEqual(result.returncode, 0,
                         f"{arguments}\n{result.stdout}\n{result.stderr}")
        return result.stdout

    def compile(self, text, *, emit="mlir", name="subject.sev", sysroot=ROOT):
        path = self.write(text, name)
        return self.succeeds([COMPILER, "test", path, "--emit", emit,
                              "--sysroot", sysroot])

    def native(self, text, *, expected="", name="subject.sev", sysroot=ROOT):
        mlir = self.compile(text, name=name, sysroot=sysroot)
        emitted = self.directory / (Path(name).stem + ".mlir")
        lowered = emitted.with_suffix(".llvm.mlir")
        llvm = emitted.with_suffix(".ll")
        executable = emitted.with_suffix(".exe")
        emitted.write_text(mlir)
        self.succeeds([tool("SEVERIAN_MLIR_OPT", "mlir-opt-21"), emitted,
                        "--verify-each", "--lift-cf-to-scf",
                        "--buffer-deallocation-pipeline=private-function-dynamic-ownership",
                        "--convert-bufferization-to-memref", "--convert-scf-to-cf",
                        "--convert-arith-to-llvm", "--convert-cf-to-llvm",
                        "--finalize-memref-to-llvm", "--convert-func-to-llvm", "--convert-ub-to-llvm",
                        "--reconcile-unrealized-casts", "-o", lowered])
        output = self.succeeds([tool("SEVERIAN_MLIR_TRANSLATE", "mlir-translate-21"),
                               "--mlir-to-llvmir", lowered])
        llvm.write_text(output)
        self.succeeds([tool("SEVERIAN_CLANG", "clang-21"), llvm, "-o", executable])
        self.assertEqual(self.succeeds([executable]), expected)
        return mlir

    def rejects(self, text, diagnostic, *, name="rejected.sev", sysroot=ROOT):
        result = self.invoke([COMPILER, "check", self.write(text, name),
                              "--sysroot", sysroot])
        self.assertGreater(result.returncode, 0, "expected a diagnostic, not success or a crash")
        self.assertRegex(result.stderr, r"error: E[0-9]+:")
        self.assertRegex(result.stderr, diagnostic)
        self.assertNotRegex(result.stderr, r"(?i)(segmentation fault|stack overflow|panic)")

    def agent_ir(self, text, **kwargs):
        ir = json.loads(self.compile(text, emit="agent-ir", **kwargs))
        self.assertEqual(ir["schema_version"], 1)
        self.assertEqual(ir["stage"], "cfg")
        definitions = ir["definitions"]
        ids = [definition["id"] for definition in definitions]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertTrue(all(isinstance(identity, str) and identity for identity in ids))
        return ir

    def definition(self, ir, name, kind=None):
        found = [d for d in ir["definitions"]
                 if d["name"].split(".")[-1] == name and (kind is None or d["kind"] == kind)]
        self.assertEqual(len(found), 1, (name, kind, found))
        return found[0]


class Gate1Baseline(MigrationCase):
    def test_01_scalar_behavior(self):
        self.native('''
            test:
                assert((7 + 5) * 2 - 4 == 20)
                assert(1.25 + 1.25 == 2.5)
                assert(-7 + 9 == 2)
        ''')

    def test_02_callable_record_behavior(self):
        self.native('''
            class Counter:
                value: int
                def add(amount: int) -> int:
                    value = value + amount
                    return value
            def recurse(value: int) -> int:
                if value == 0:
                    return 0
                return 1 + recurse(value - 1)
            test:
                counter = Counter(40)
                assert(counter.add(recurse(2)) == 42)
                assert(counter.value == 42)
        ''')

    def test_03_import_alias_and_member(self):
        self.write('''
            type Amount = int
            class Box:
                value: Amount
                def get() -> Amount:
                    return value
        ''', "models.sev")
        self.native('''
            import "models.sev" as models
            test:
                assert(models.Box(42).get() == 42)
        ''')

    def test_04_deterministic_output(self):
        program = 'test:\n    assert(21 + 21 == 42)\n'
        self.assertEqual(self.compile(program), self.compile(program))

    def test_05_active_pipeline_is_inspectable(self):
        ir = self.agent_ir('''
            def answer(value: int) -> int:
                return value + value
            test:
                assert(answer(21) == 42)
        ''')
        definition = self.definition(ir, "answer", "function")
        body = next(f for f in ir["functions"] if f["definition"] == definition["id"])
        self.assertIn(body["entry"], [b["id"] for b in body["blocks"]])
        self.assertTrue(body["blocks"])
        self.assertTrue(all("terminator" in block for block in body["blocks"]))


class Gate2Definitions(MigrationCase):
    def test_01_inherited_g_contract(self):
        program = "trait ArithmeticContract: G:\n    pass\n" + contract(parent="ArithmeticContract")
        self.native(program + '\ntest:\n    assert(4 <~> 2 == 42)\n')
        ir = self.agent_ir(program)
        base = self.definition(ir, "ArithmeticContract")
        derived = self.definition(ir, "Fuse", "grammar")
        self.assertIn(base["id"], derived["bases"])

    def test_02_counterfeit_g_does_not_grant_capability(self):
        self.rejects('''
            trait G:
                pass
            trait Counterfeit: G:
                def semantic():
                    cfg.block()
        ''', r"(?i)(reserved.*G|compiler.*G|grammar capability)")

    def test_03_cfg_capability_cannot_escape(self):
        attempts = [
            "def ordinary():\n    cfg.block()\n",
            "def ordinary():\n    create = cfg.block\n    create()\n",
            "def helper():\n    cfg.block()\ndef ordinary():\n    helper()\n",
            "class Box:\n    value: int\n    operator +[G:Add](right: Self) -> Self:\n        cfg.block()\n        return self\n",
            '@mlir("cf.br")\ndef escaped()\ndef ordinary():\n    escaped()\n',
        ]
        for index, attempt in enumerate(attempts):
            with self.subTest(attempt=index):
                self.rejects(attempt, r"(?i)(grammar capability|compiler context)",
                             name=f"escape_{index}.sev")

    def test_04_compiler_words_are_contextual(self):
        self.native('''
            class Ordinary:
                effects: int
                overflow: int
                def read() -> int:
                    return effects + overflow
            def evaluate_once(value: int) -> int:
                return value
            test:
                assert(evaluate_once(Ordinary(40, 2).read()) == 42)
        ''')

    def test_05_absent_is_preserved_and_required_fields_are_checked(self):
        program = contract(extra="short_circuit: bool | absent = absent")
        unspecified = self.definition(self.agent_ir(program), "Fuse", "grammar")["metadata"]
        explicit = self.definition(self.agent_ir(program.replace(
            "short_circuit: bool | absent = absent", "short_circuit: bool | absent = false")),
            "Fuse", "grammar")["metadata"]
        self.assertIsNone(unspecified["overflow"])
        self.assertIsNone(unspecified["short_circuit"])
        self.assertIs(explicit["short_circuit"], False)
        self.rejects(program.replace("precedence: int = 7",
                                     "precedence: int | absent = absent"),
                     r"(?i)(precedence.*required|required.*precedence)")


class Gate3SourceSyntax(MigrationCase):
    def test_01_imported_new_operator_without_rebuild(self):
        self.write(contract(), "syntax.sev")
        self.native('import "syntax.sev"\ntest:\n    assert(4 <~> 2 == 42)\n')

    def test_02_longest_symbol_and_word_boundaries(self):
        self.write(contract("Short", "<~>") + "\n" + contract("Long", "<~>=!", body="return left + right")
                   + "\n" + contract("Word", "fuse"), "symbols.sev")
        self.native('''
            import "symbols.sev"
            test:
                fuse_count = 2
                assert(4 <~> fuse_count == 42)
                assert(40 <~>=! 2 == 42)
                assert(4 fuse fuse_count == 42)
        ''')

    def test_03_source_precedence_and_associativity(self):
        self.native(contract(precedence=7) + '\ntest:\n    assert(1 <~> 2 * 3 == 16)\n', name="low.sev")
        self.native(contract(precedence=9) + '\ntest:\n    assert(1 <~> 2 * 3 == 36)\n', name="high.sev")
        self.native(contract(associativity="Left") + '\ntest:\n    assert(1 <~> 2 <~> 3 == 123)\n', name="left.sev")
        self.native(contract(associativity="Right") + '\ntest:\n    assert(1 <~> 2 <~> 3 == 33)\n', name="right.sev")

    def test_04_rename_and_respell_contract(self):
        self.native(contract("UnrelatedName", "<!~>") + '\ntest:\n    assert(4 <!~> 2 == 42)\n')
        self.rejects(contract("UnrelatedName", "<!~>").replace(
            "[G:UnrelatedName]", "[G:MissingContract]"), r"(?i)(unknown.*MissingContract|unresolved.*MissingContract)")

    def test_05_conflicting_imports_are_diagnosed(self):
        self.write(contract("First", precedence=7), "first.sev")
        self.write(contract("Second", precedence=9), "second.sev")
        self.rejects('import "first.sev"\nimport "second.sev"\n',
                     r"(?i)(conflicting.*syntax|ambiguous.*symbol)")


class Gate4SemanticExecution(MigrationCase):
    def test_01_edit_semantic_body_without_rebuild(self):
        self.write('def combine(left: int, right: int) -> int:\n    return left * 10 + right\n', "helper.sev")
        declaration = 'import "helper.sev"\n' + contract(body="return combine(left, right)")
        self.native(declaration + '\ntest:\n    assert(4 <~> 2 == 42)\n')
        self.write('def combine(left: int, right: int) -> int:\n    return left + right * 10\n', "helper.sev")
        self.native(declaration + '\ntest:\n    assert(4 <~> 2 == 24)\n')

    def test_02_generic_requirements_select_source_implementations(self):
        self.native('''
            trait Addable:
                operator +[G:Add](other: Self) -> Self
            class Box: Addable:
                amount: int
                operator +[G:Add](other: Self) -> Self:
                    return Box(amount + other.amount)
            def twice[T: Addable](value: T) -> T:
                return value + value
            test:
                assert(twice(21) == 42)
                assert(twice(2.5) == 5.0)
                assert(twice(Box(7)).amount == 14)
        ''')

    def test_03_compound_assignment_evaluates_place_and_rhs_once(self):
        self.native('''
            def index() -> int:
                print("index")
                return 0
            def rhs() -> int:
                print("rhs")
                return 5
            test:
                values = [37]
                values[index()] += rhs()
                assert(values[0] == 42)
        ''', expected="index\nrhs\n")

    def test_06_compound_field_update_preserves_implementation_and_receiver(self):
        self.native('''
            class Value:
                amount: int
                operator +[G:Add](other: Self) -> Self:
                    print("update")
                    return Value(amount + other.amount + 1)
            class Holder:
                value: Value
            def receiver(holder: Holder) -> Holder:
                print("receiver")
                return holder
            def rhs() -> Value:
                print("rhs")
                return Value(4)
            test:
                holder = Holder(Value(37))
                receiver(holder).value += rhs()
        ''', expected="receiver\nrhs\nupdate\n")

    def test_07_indexed_update_evaluates_receiver_once(self):
        self.native('''
            def receiver(values: list[int]) -> list[int]:
                print("receiver")
                return values
            def index() -> int:
                print("index")
                return 0
            def rhs() -> int:
                print("rhs")
                return 5
            test:
                values = [37]
                receiver(values)[index()] += rhs()
                assert(values[0] == 42)
        ''', expected="receiver\nindex\nrhs\n")

    def test_04_assignment_requires_a_writable_place(self):
        self.rejects('test:\n    value = 40\n    value += 2\n',
                     r"(?i)(immutable|writ[ae]ble|mutable place)")
        self.rejects('test:\n    (20 + 20) += 2\n',
                     r"(?i)(writ[ae]ble|place|assignment target)", name="temporary.sev")

    def test_05_lazy_operands_and_user_truth(self):
        self.native('''
            class Flag:
                enabled: bool
                operator if[G:If](self) -> bool:
                    print("truth")
                    return enabled
            def right() -> bool:
                print("right")
                return true
            test:
                if Flag(true):
                    print("body")
                assert((false and right()) == false)
                assert(true or right())
        ''', expected="truth\nbody\n")


class Gate5CanonicalCfg(MigrationCase):
    def test_01_nested_control_flow(self):
        self.native('''
            def accumulate(limit: int) -> int:
                total := 0
                i := 0
                while i < limit:
                    i += 1
                    if i == 3:
                        continue
                    match i:
                        case 5:
                            total += 20
                        case 7:
                            return total + 100
                        case _:
                            if i % 2 == 0:
                                total += i
                            else:
                                total += 1
                return total
            test:
                assert(accumulate(2) == 3)
                assert(accumulate(5) == 27)
                # Contributions: 1 + 2 + 4 + 20 + 6, then the early return adds 100.
                assert(accumulate(10) == 133)
        ''')

    def test_02_source_grammar_constructs_a_branch(self):
        self.native('''
            trait Unless: G:
                symbol: Y = unless
                arity: int = 2
                left: Operand = Value[bool]
                right: Operand = Body
                result: T | absent = absent
                precedence: int | absent = absent
                control_flow: ControlFlow = Branch
                def semantic(condition: Ex, body: B):
                    value = lower(condition)
                    execute = cfg.block()
                    exit = cfg.block()
                    cfg.branch(value, exit, execute)
                    cfg.at(execute)
                    lower(body)
                    cfg.goto(exit)
                    cfg.at(exit)
            test:
                value := 0
                unless false:
                    value = 42
                unless true:
                    value = 99
                assert(value == 42)
        ''')

    def test_03_all_edges_have_resolved_grammar_provenance(self):
        ir = self.agent_ir('''
            def choose(value: int) -> int:
                while value > 0:
                    if value == 2:
                        return 42
                    break
                return 7
            test:
                assert(choose(2) == 42)
        ''')
        grammars = {d["id"] for d in ir["definitions"] if d["kind"] == "grammar"}
        self.assertTrue(grammars)
        branching = False
        for function in ir["functions"]:
            blocks = {block["id"]: block for block in function["blocks"]}
            self.assertIn(function["entry"], blocks)
            for block in blocks.values():
                terminator = block["terminator"]
                self.assertIn(terminator["grammar"], grammars)
                self.assertIn("source", terminator)
                branching |= len(terminator["successors"]) > 1
                for edge in terminator["successors"]:
                    self.assertIn(edge["target"], blocks)
                    self.assertEqual(len(edge["arguments"]), len(blocks[edge["target"]]["parameters"]))
                self.assertFalse(any(op["kind"] in {"If", "Loop", "Choose"}
                                     for op in block["operations"]))
        self.assertTrue(branching)

    def test_04_custom_grammar_cannot_emit_two_terminators(self):
        self.rejects('''
            trait Broken: G:
                symbol: Y = broken
                arity: int = 1
                left: Operand = Body
                control_flow: ControlFlow = Branch
                def semantic(body: B):
                    target = cfg.block()
                    cfg.goto(target)
                    cfg.goto(target)
                    cfg.at(target)
                    lower(body)
            test:
                broken:
                    print("unreachable compilation")
        ''', r"(?i)(already terminated|second terminator|after.*terminator)")

    def test_05_error_paths_are_typed_and_preserve_identity(self):
        self.native('''
            class InvalidValue: Error:
                value: int
            def checked(value: int) -> int | InvalidValue:
                if value < 0:
                    throw InvalidValue(value)
                return value
            def forwarded(value: int) -> int | InvalidValue:
                return checked(value)
            test:
                result ?= forwarded(-7)
                if result is InvalidValue:
                    assert(result.value == -7)
                else:
                    assert(false)
                assert(forwarded(42) == 42)
        ''')
        self.rejects('def invalid():\n    throw 42\n', r"(?i)(throw.*Error|Error.*throw)")


    def test_06_error_identity_and_guarded_projection(self):
        self.native('''
            class First: Error:
                value: int
            class Second: Error:
                value: int
            def checked(value: int) -> int | First | Second:
                if value < 0:
                    throw First(value)
                if value == 0:
                    throw Second(value)
                return value
            def forwarded(value: int) -> int | First | Second:
                return checked(value)
            test:
                first ?= forwarded(-7)
                if first is Second:
                    assert(false)
                if first is First:
                    assert(first.value == -7)
                else:
                    assert(false)
                second ?= forwarded(0)
                if second is Second:
                    assert(second.value == 0)
                else:
                    assert(false)
                assert(forwarded(42) == 42)
        ''')
        self.rejects('''
            class First: Error:
                value: int
            def checked() -> int | First:
                throw First(7)
            test:
                result ?= checked()
                assert(result.value == 7)
        ''', r"(?i)(unknown field|narrow|variant)", name="unguarded.sev")
        self.rejects('''
            class LooksLikeError:
                value: int
            def invalid():
                throw LooksLikeError(42)
        ''', r"(?i)(throw.*Error|Error.*throw)", name="not-error.sev")


class Gate6LibraryAndRetirement(MigrationCase):
    def test_01_actual_integer_and_float_sources(self):
        for filename in ("int.sev", "float.sev"):
            real = ROOT / "sev_compiler/universal/primitive" / filename
            # Relative locators are part of the language's existing import syntax.
            relative = os.path.relpath(real, self.directory)
            with self.subTest(library=filename):
                self.native(f'import "{relative}"\n' + source('''
                    test:
                        small: i8 = 21
                        assert(small + small == 42)
                        assert(2.5 + 2.5 == 5.0)
                '''), name=filename)

    def test_02_collection_and_string_protocols(self):
        self.native('''
            test:
                values = [1, 2, 3]
                total := 0
                if values:
                    for value in values:
                        total += value
                assert(total == 6)
                if "λ":
                    total += 1
                if "":
                    assert(false)
                assert(total == 7)
        ''')

    def test_03_numeric_conversion_contracts(self):
        self.native('''
            test:
                narrow: i8 = 42
                wide: i64 = i64(narrow)
                assert(wide == 42)
                assert(i8(300, mode="lossy") == 44)
                assert(int(3.75, mode="lossy") == 3)
                assert(float(42) == 42.0)
        ''')

    def test_04_removing_source_contract_disables_the_operation(self):
        ir = self.agent_ir('test:\n    assert(21 + 21 == 42)\n')
        grammar = self.definition(ir, "Add", "grammar")
        origin = grammar["source"]
        relative = Path(origin["path"])
        if relative.is_absolute():
            relative = relative.relative_to(ROOT)
        self.assertTrue(relative.is_relative_to("sev_compiler/universal"))
        overlay = self.directory / "sysroot"
        (overlay / "sev_compiler").mkdir(parents=True)
        shutil.copytree(ROOT / "sev_compiler/universal", overlay / "sev_compiler/universal",
                        ignore=shutil.ignore_patterns("target", ".git"))
        (overlay / "library").symlink_to(ROOT / "library", target_is_directory=True)
        declaration_file = overlay / relative
        content = declaration_file.read_bytes()
        start, end = origin["start"], origin["end"]
        self.assertLess(start, end)
        self.assertLessEqual(end, len(content))
        declaration_file.write_bytes(content[:start] + content[end:])
        self.rejects('test:\n    value = 21 + 21\n',
                     r"(?i)(unknown|unresolved|missing|unregistered|requires).*([+]\b|Add|operator|grammar)|(?i:expected.*operator)",
                     sysroot=overlay)

    def test_06_conditional_expression_executes_source_grammar(self):
        overlay = self.directory / "conditional-sysroot"
        (overlay / "sev_compiler").mkdir(parents=True)
        shutil.copytree(ROOT / "sev_compiler/universal", overlay / "sev_compiler/universal")
        (overlay / "library").symlink_to(ROOT / "library", target_is_directory=True)
        path = overlay / "sev_compiler/universal/grammar/contracts.sev"
        original = path.read_text()
        before, conditional = original.split("trait Conditional: G:", 1)
        self.assertIn("cfg.branch(value, taken, otherwise)", conditional)
        path.write_text(before + "trait Conditional: G:" + conditional.replace(
            "cfg.branch(value, taken, otherwise)", "cfg.branch(value, otherwise, taken)", 1))
        self.native('''
            def left() -> int:
                assert(false)
                return 7
            def right() -> int:
                return 42
            test:
                assert((left() if true else right()) == 42)
        ''', sysroot=overlay)

    def test_05_legacy_dispatch_and_parallel_topology_are_retired(self):
        forbidden = {
            "frontend/lexer/src/token/mod.sev": r"\b(?:PlusEqual|MinusEqual|StarEqual|SlashEqual|PercentEqual)\b",
            "frontend/parser/src/statement/mod.sev": r"\b(?:PlusEqual|MinusEqual|StarEqual)\b|symbol\s*\+=\s*[\"']=[\"']",
            "frontend/semantic/src/callable.sev": r"\bscalar_(?:callable_)?operation\s*\(",
            "transforms/mir/src/callable.sev": r"case\s+(?:If|Loop|Choose):|Operation\.(?:If|Loop|Choose)\(",
            "transforms/mlir/src/emit/callable.sev": r"case\s+(?:If|Loop|Choose):|\bscalar_(?:callable_)?operation\s*\(",
        }
        for relative, pattern in forbidden.items():
            with self.subTest(path=relative):
                path = ROOT / "sev_compiler" / relative
                if path.exists():
                    self.assertNotRegex(path.read_text(), pattern)


if __name__ == "__main__":
    unittest.main(verbosity=2)
