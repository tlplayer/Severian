# Validation record

Validated from `/home/tplayer/Documents/Severian` on 2026-08-11.

## Upstream baseline

- Source commit: `d23e16e80c7bcc1103be34f77f823b4bc6cddc78`
- `labgob` and `labrpc` Go tests pass.
- `go test ./...` cannot build the retained launchers because the repository does
  not contain `diskv`, `lockservice`, `pbservice`, or `viewservice`.
- Later lab packages contain course placeholders. For example, the versioned KV
  suite fails against the unimplemented server and the Raft source contains
  `Your code here` sections.

## Severian result

`./docs/lab/distributed_systems/run_labs.sh` completed successfully:

- 9 sources passed parsing, resolution, type checking, and ownership checking;
- 26 native test blocks compiled and passed;
- 9 native `main` programs compiled and executed;
- `sev lint docs/lab/distributed_systems` reported 0 warnings.

`cargo test --workspace --all-targets` also passed, including the permanent
`distributed_systems_labs_compile_and_execute_natively` integration test.

No frontend interpreter or host-language Severian evaluator is involved. Each
test and program passes through the normal HIR, MIR, MLIR, LLVM, and native
runtime path.

## Compiler findings fixed by the labs

The native runs exposed and now guard these lowering issues:

- class fields retain their primitive/collection ABI types when read;
- constructor and class-method calls coerce boxed field values to declared
  parameter types;
- class methods receive top-level constants in their lowering environment;
- user class methods named `join` or two-argument `get` are not mistaken for
  collection/map built-ins;
- membership against a boxed collection field is lowered as collection
  membership rather than scalar comparison.

The naming pass also established one policy owner: semantic analysis accepts
style-migration spellings, while `sev lint` enforces and fixes snake_case. Its
UPPER_SNAKE_CASE handling now preserves already-correct constants.

