#!/usr/bin/env python3
"""Build the source compiler, then verify and execute its emitted MLIR."""
import os
from pathlib import Path
import shutil
import subprocess


ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = ROOT / "sev_compiler/target/acceptance"
SEED = Path(os.environ.get("SEVERIAN_BIN", ROOT / "target/debug/sev"))
SANITIZE = os.environ.get("SEVERIAN_SANITIZE", "0") == "1"


def tool(variable, default):
    selected = os.environ.get(variable, default)
    resolved = shutil.which(selected)
    if not resolved:
        raise RuntimeError(f"required tool is unavailable: {selected}")
    return resolved


def run(arguments, *, output=None, succeeds=True, cwd=ROOT):
    result = subprocess.run(
        [str(argument) for argument in arguments], cwd=cwd,
        capture_output=True, text=True, timeout=180,
    )
    if output:
        output.write_text(result.stdout)
    if (result.returncode == 0) != succeeds:
        raise AssertionError(
            f"unexpected exit {result.returncode}: {' '.join(map(str, arguments))}\n"
            f"{result.stdout}\n{result.stderr}"
        )
    return result


def main():
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    opt = tool("SEVERIAN_MLIR_OPT", "mlir-opt-21")
    translate = tool("SEVERIAN_MLIR_TRANSLATE", "mlir-translate-21")
    clang = tool("SEVERIAN_CLANG", "clang-21")
    compiler = ROOT / "sev_compiler/target/host/dev/bin/sev_compiler"
    if not os.environ.get("SEVERIAN_SKIP_BUILD"):
        run([SEED, "build"], cwd=ROOT / "sev_compiler")
    string_source = ROOT / "sev_compiler/universal/primitive/string/core.sev"
    io_source = ROOT / "library/system/io/src/text.sev"
    scalar_tests = ROOT / "sev_compiler/frontend/semantic/src/scalar/tests"
    string_import = f'import "{os.path.relpath(string_source, ARTIFACTS)}" as utf8\n'
    cases = {
        "int_add": "left: i32 = 1\nright: i32 = 2\nsum: i32 = left + right\nassert(sum == 3)\n",
        "arithmetic": "left: i32 = 7\nright: i32 = 5\nanswer: i32 = (left + right) * 2 - 4\nassert(answer == 20)\n",
        "negative": "value: i32 = -7\nanswer: i32 = value + 9\nassert(answer == 2)\n",
        "false_assertion": "left: i32 = 1\nright: i32 = 2\nassert(left + right == 4)\n",
        "main": "def add(value: i32) -> i32:\n    return value + 1\ndef main() -> i32:\n    assert(add(41) == 42)\n    return 0\n",
        "main_status": "def main() -> i32:\n    return 7\n",
        "unit_main": "def main():\n    assert(1 == 1)\n",
        "default_argument": "def answer(value: i32 = 3) -> i32:\n    return value\nassert(answer() == 3)\nassert(answer(value=4) == 4)\n",
        "float_arithmetic": "def scale(value: f64, factor: f64 = 2.0) -> f64:\n    return value * factor\nassert(scale(1.25) == 2.5)\nassert(0.25 + 0.5 == 0.75)\nassert(-0.5 < 0.0)\nassert(1.0 >= 0.5)\nassert(0.5 <= 0.5)\n",
        "unsigned_bytes": "high: u8 = 255\nlow: u8 = 1\nassert(high > low)\nassert(low < high)\nassert(high >= low)\nassert(low <= high)\n",
        "byte_bounds": string_import + 'utf8.byte_at("a", 1)\n',
        "negative_byte": string_import + 'utf8.byte_at("a", -1)\n',
        "character_bounds": string_import + 'utf8.character_at("λ", 1)\n',
        "empty_character": string_import + 'utf8.character_at("", 0)\n',
        "decode_continuation": string_import + 'utf8.decode("λ", 1)\n',
        "cast_i8_overflow": 'i8(128)\n',
        "cast_i8_underflow": 'i8(-129, checked)\n',
        "cast_u8_negative": 'u8(-1)\n',
        "cast_u8_overflow": 'u8(256)\n',
        "cast_i64_upper_float": 'int(9223372036854775808.0)\n',
        "cast_i64_lower_float": 'int(-9223372036854777856.0)\n',
        "cast_i8_float_overflow": 'i8(128.0, lossy)\n',
        "cast_float_nan": 'int(0.0 / 0.0)\n',
        "cast_float_infinity": 'int(1.0 / 0.0)\n',
        "cast_float_negative_infinity": 'int(-1.0 / 0.0)\n',
        "imported_output": f'import "{os.path.relpath(io_source, ARTIFACTS)}" as io\nio.print("library output")\n',
    }
    inputs = {}
    for name, source in cases.items():
        source_path = ARTIFACTS / f"{name}.sev"
        source_path.write_text(source)
        inputs[name] = (source_path, "build")
    inputs.update({
        # Prerequisite order, not directory numbering.
        "example_hello": (ROOT / "docs/examples/00-getting-started/01-hello.sev", "build"),
        "example_functions": (ROOT / "docs/examples/02-functions/01-basic/01-basic-functions.sev", "build"),
        "example_conditional": (ROOT / "docs/examples/02-functions/02-control-flow/07-conditional-expression.sev", "test"),
        "example_variables": (ROOT / "docs/examples/00-getting-started/02-variables.sev", "build"),
        "example_printing": (ROOT / "docs/examples/00-getting-started/03-printing.sev", "build"),
        "example_primitives": (ROOT / "docs/examples/01-types/01-basic/01-primitives.sev", "build"),
        "example_constants": (ROOT / "docs/examples/01-types/01-basic/00-constants.sev", "build"),
        "example_inference": (ROOT / "docs/examples/01-types/01-basic/02-inference.sev", "build"),
        "example_signatures": (ROOT / "docs/examples/02-functions/01-basic/02-signatures.sev", "build"),
        "example_conversion": (ROOT / "docs/examples/01-types/01-basic/03-conversion.sev", "build"),
        "example_conversion_tests": (ROOT / "docs/examples/01-types/01-basic/03-conversion.sev", "test"),
        "numeric_conversion": (ROOT / "sev_compiler/universal/primitive/numeric/conversion.sev", "test"),
        "numeric_conversion_build": (ROOT / "sev_compiler/universal/primitive/numeric/conversion.sev", "build"),
        "example_compiler_tests": (ROOT / "docs/examples/03-testing/02-with-tests/08-compile.sev", "test"),
        "printing": (io_source, "test"),
        "example_math": (ROOT / "docs/examples/05-building/src/math.sev", "build"),
        "example_clamp": (ROOT / "docs/examples/03-testing/01-basics/01-ordinary-and-named.sev", "test"),
        "scalar_functions": (scalar_tests / "functions.sev", "test"),
        "expression_values": (scalar_tests / "values.sev", "test"),
        "packs": (scalar_tests / "packs.sev", "test"),
        "string_core": (string_source, "test"),
        "string_core_build": (string_source, "build"),
        "string_format": (string_source.with_name("format.sev"), "test"),
        "char_encoding": (ROOT / "sev_compiler/universal/primitive/char/encoding.sev", "test"),
        "char_utf8": (ROOT / "sev_compiler/universal/primitive/char/utf8.sev", "test"),
    })
    test_selection = ARTIFACTS / "test_selection.sev"
    test_selection.write_text("def main():\n    return\ntest:\n    assert(false)\n")
    inputs["build_excludes_tests"] = (test_selection, "build")
    inputs["false_test"] = (test_selection, "test")
    test_main = ARTIFACTS / "test_main.sev"
    test_main.write_text("def main():\n    assert(false)\ntest:\n    assert(true)\n")
    inputs["test_excludes_main"] = (test_main, "test")
    expected_stdout = {
        "example_hello": "hello, severian\n",
        "example_functions": "large\n",
        "example_primitives": "10\n1000000\n0.5\ntrue\na\n",
        "example_constants": "3\n3.1415926\n",
        "example_inference": "10:int\n0.5:float\ntrue:bool\nseverian:string\n",
        "example_signatures": "width: 24\n",
        "example_conversion": "10\n0.5\n10.5\ntrue\nseverian\n10.5!\n",
        "printing": (
            "\ncount 42 true λ 0.5 None\na|b!next\nonly end\n"
            "values: 42:0.5:false:a\n"
            'aλ😀z\n\nquote: " slash: \\ tab:\tend\n'
            "-9223372036854775808 9223372036854775807\n255 -128 😀 中\n"
            "3.1415926 1.2345678901234567 5e-324 1.7976931348623157e+308 -0\n"
            "nan inf -inf\n"
        ),
        "example_variables": "Hello, World!\n",
        "example_printing": (
            "You can print in global scope before main starts.\n"
            "Call functions in global or allocate values like python call:one\n"
            "Call functions in global or allocate values like python call:two\n"
            "Or inside main\n"
        ),
        "expression_values": "aλ😀\n",
        "packs": "ba12\n1!;two!;false!;\n7;8;\n{42}\ntint\nc12\nf12\np10.5\n",
        "imported_output": "library output\n",
    }
    runtime_failures = {
        "cast_i8_overflow", "cast_i8_underflow", "cast_u8_negative", "cast_u8_overflow",
        "cast_i64_upper_float", "cast_i64_lower_float", "cast_i8_float_overflow",
        "cast_float_nan", "cast_float_infinity", "cast_float_negative_infinity",
        "false_assertion", "false_test", "main_status", "byte_bounds",
        "negative_byte", "character_bounds", "empty_character", "decode_continuation",
    }
    function_counts = {}
    for name, (source_path, command) in inputs.items():
        emitted = ARTIFACTS / f"{name}.mlir"
        run([compiler, command, "--emit", "mlir", source_path, "--sysroot", ROOT], output=emitted)
        text = emitted.read_text()
        assert "module {" in text and '"func.return"' in text, "expected executable MLIR"
        function_counts[name] = sum(line.lstrip().startswith("func.func ") for line in text.splitlines())
        if name in {"example_math", "example_clamp", "scalar_functions"}:
            assert "__sev_scalar_" in text, "source functions must survive lowering"
        if name in {"example_clamp", "scalar_functions"}:
            assert '"func.call"' in text and '"scf.if"' in text
        verified = ARTIFACTS / f"{name}.verified.mlir"
        run([opt, "--verify-each", emitted, "-o", verified])
        llvm_mlir = ARTIFACTS / f"{name}.llvm.mlir"
        run([opt, emitted, "--buffer-deallocation-pipeline=private-function-dynamic-ownership",
             "--convert-bufferization-to-memref", "--convert-scf-to-cf", "--convert-arith-to-llvm",
             "--convert-cf-to-llvm", "--finalize-memref-to-llvm", "--convert-func-to-llvm",
             "--reconcile-unrealized-casts", "-o", llvm_mlir])
        llvm_ir = ARTIFACTS / f"{name}.ll"
        run([translate, "--mlir-to-llvmir", llvm_mlir], output=llvm_ir)
        assert "__sev_string_from_" not in llvm_ir.read_text()
        assert "__sev_io_" not in llvm_ir.read_text()
        executable = ARTIFACTS / name
        run([clang, llvm_ir, "-o", executable])
        result = run([executable], succeeds=name not in runtime_failures)
        if name in expected_stdout:
            assert result.stdout == expected_stdout[name], repr(result.stdout)
            assert '"memref.global"' in text and '"memref.load"' in text
            assert "@putchar" in llvm_ir.read_text()
        if name == "main_status":
            assert result.returncode == 7, "preserve the source main's exit status"
        if name in {"expression_values", "printing", "string_core", "string_format"} and SANITIZE:
            sanitized = ARTIFACTS / f"{name}_asan"
            run([clang, "-fsanitize=address", llvm_ir, "-o", sanitized])
            checked = run([sanitized])
            assert checked.stdout == expected_stdout.get(name, "")
            print(f"PASS: {name} under AddressSanitizer/LeakSanitizer", flush=True)
        print(f"PASS: {name} (source -> MLIR -> native)", flush=True)
    # Symbols use numeric callable IDs. Compare the same subject in both modes
    # to prove tests are absent from build IR, rather than merely uncalled.
    assert function_counts["false_test"] == function_counts["build_excludes_tests"] + 1
    assert function_counts["string_core"] == function_counts["string_core_build"] + 4
    assert function_counts["numeric_conversion"] == function_counts["numeric_conversion_build"] + 40
    rejected = {
        "numeric_mode": ('int(1.5, checked)\n', "required numeric policy"),
        "numeric_unknown_mode": ('int(1, unknown)\n', "unknown numeric conversion mode"),
        "numeric_keyword": ('int(value=1)\n', "must be positional"),
        "numeric_arity": ('int()\n', "requires a value"),
        "numeric_mode_expression": ('int(1, true)\n', "must be a policy name"),
        "compiler_reject_accepts": ('test with compiler "wrong expectation":\n    reject:\n        value = 1\n', "compiler test expectation failed"),
        "compiler_accept_rejects": ('test with compiler "wrong expectation":\n    accept:\n        value = missing\n', "compiler test expectation failed"),
        "unknown_name": ("answer: i32 = missing + 1\n", "unknown name missing"),
        "wrong_type": ("answer: i32 = true\n", "boolean cannot initialize an integer"),
        "overflow": ("answer: i8 = 128\n", "integer literal is outside"),
        "unsupported": ("def answer[T](value: T) -> T:\n    return value\n", "outside the scalar bootstrap subset"),
        "malformed": ("answer: i32 =\n", "expected an expression"),
        "missing_return": ("def answer() -> i32:\n    value = 3\n", "without returning a value"),
        "branch_missing_return": ("def answer(value: bool) -> i32:\n    if value:\n        return 1\n", "without returning a value"),
        "wrong_return": ("def answer() -> i32:\n    return true\n", "boolean cannot initialize an integer"),
        "wrong_argument": ("def answer(value: i32) -> i32:\n    return value\nanswer(true)\n", "boolean cannot initialize an integer"),
        "wrong_arity": ("def answer(value: i32) -> i32:\n    return value\nanswer()\n", "wrong argument count"),
        "duplicate_function": ("def answer():\n    return\ndef answer():\n    return\n", "duplicate or reserved function"),
        "global_capture": ("value: i32 = 1 + 2\ndef answer() -> i32:\n    return value\n", "unknown name value"),
        "mutable_global_capture": ('value := "one"\nvalue = "two"\ndef answer() -> string:\n    return value\n', "unknown name value"),
        "unreachable": ("def answer() -> i32:\n    return 1\n    return 2\n", "statement after unconditional return"),
        "wrong_condition": ("def answer() -> i32:\n    if 1:\n        return 1\n    return 0\n", "integer cannot initialize bool"),
        "unit_binding": ("def nothing():\n    return\nvalue = nothing()\n", "unit calls cannot initialize"),
        "wrong_string": ('value: i32 = "hello"\n', "expected scalar type"),
        "mixed_string_add": ('value = "a" + 1\n', "integer cannot initialize text"),
        "immutable_update": ('value = "a"\nvalue += "b"\n', "cannot reassign immutable"),
        "unknown_update": ("value += 1\n", "cannot update unknown"),
        "duplicate_mutable": ("value := 1\nvalue := 2\n", "duplicate scalar binding"),
        "update_type": ('value := 1\nvalue = "two"\n', "expected scalar type"),
        "conditional_type": ('value = "one" if true else 2\n', "integer cannot initialize text"),
        "conditional_condition": ("value = 1 if 2 else 3\n", "integer cannot initialize bool"),
        "conditional_missing_else": ("value = 1 if true\n", "conditional expression requires else"),
        "bad_interpolation": ('value = "a"\nprint(f"{value junk}")\n', "unexpected token in formatted string interpolation"),
        "bad_default": ("def answer(value: int = 1 + 2) -> int:\n    return value\n", "defaults must be literals"),
        "default_type": ('def answer(value: int = "bad") -> int:\n    return value\n', "expected scalar type"),
        "pack_default": ('def show(*values: V = 1):\n    return\n', "variadic packs cannot have defaults"),
        "pack_duplicate": ('def show(*values: V, values: int = 1):\n    return\n', "duplicate parameter"),
        "unknown_keyword": ('print("a", typo="!")\n', "unknown keyword argument"),
        "duplicate_keyword": ('print("a", sep=" ", sep="|")\n', "duplicate argument"),
        "positional_after_keyword": ('print(sep="|", "a")\n', "positional argument follows keyword"),
        "bad_separator": ('print("a", sep=1)\n', "no overload accepts"),
        "bad_flush": ('print("a", flush="yes")\n', "expected scalar type"),
        "bad_stream": ('print("a", file=1)\n', "integer cannot initialize"),
        "bad_exponent": ('value = 1.0e+\n', "floating exponent requires digits"),
        "format_specifier": ('print(f"{1.0:.2f}")\n', "unexpected token in formatted string"),
        "segment_count": ('@mlir("memref.alloc", operand_segments="2,0")\ndef allocate(size: index) -> string\n', "account for every parameter"),
        "segment_spelling": ('@mlir("memref.alloc", operand_segments="junk,1")\ndef allocate(size: index) -> string\n', "invalid MLIR operand segment size"),
        "duplicate_boundary_argument": ('@mlir("memref.alloc", operand_segments="1,0", operand_segments="1,0")\ndef allocate(size: index) -> string\n', "duplicate boundary argument"),
        "extra_boundary_argument": ('@mlir("arith.extui", typo="ignored")\ndef widen(value: u8) -> int\n', "unsupported boundary argument"),
        "multiple_c_symbols": ('@c("putchar", symbol="different")\ndef put(value: i32) -> i32\n', "exactly one symbol"),
        "character_count": ("value = 'ab'\n", "one Unicode scalar"),
        "empty_character_literal": ("value = ''\n", "one Unicode scalar"),
        "bad_escape": ('value = "\\q"\n', "unsupported literal escape"),
        "nul_escape": ('value = "\\0"\n', "unsupported literal escape"),
        "unterminated_string": ('value = "hello\n', "unterminated string literal"),
        "byte_overflow": ("value: u8 = 256\n", "outside u8"),
        "cyclic_import": ('import "cyclic_import.sev"\n', "cyclic source import"),
        "duplicate_alias": (string_import + string_import, "duplicate import alias"),
        "boundary_body": ('@mlir("arith.extui")\ndef convert(value: u8) -> int:\n    return 1\n', "cannot have source bodies"),
        "c_string_abi": ('@c(symbol="puts")\ndef puts(value: string) -> i32\n', "scalar ABI types"),
        "conflicting_external": ('@c(symbol="putchar")\ndef wrong(value: int) -> i32\n', "conflicting external parameter"),
    }
    for name, (source, diagnostic) in rejected.items():
        path = ARTIFACTS / f"{name}.sev"
        path.write_text(source)
        mode = "test" if name.startswith("compiler_") else "build"
        result = run([compiler, mode, "--emit", "mlir", path, "--sysroot", ROOT], succeeds=False)
        assert result.returncode > 0, "a crash is not a diagnostic"
        assert "error:" in result.stderr and "module {" not in result.stdout
        assert diagnostic in result.stderr, result.stderr
        print(f"PASS: {name} rejected with a diagnostic", flush=True)
    # The source tree is selected explicitly when invoked outside the repository.
    # A bare filename also exercises relative imports without a parent component.
    relocated = run(
        [compiler, "build", "--emit", "mlir", "imported_output.sev", "--sysroot", ROOT],
        cwd=ARTIFACTS,
    )
    assert relocated.stdout == (ARTIFACTS / "imported_output.mlir").read_text()
    print("PASS: source imports and explicit sysroot outside the repository")
    print(f"PASS: {len(inputs) + len(rejected) + 1 + 4 * int(SANITIZE)} compiler acceptance checks")
    print(f"Compiler acceptance artifacts: {ARTIFACTS}")


if __name__ == "__main__":
    main()
