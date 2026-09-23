# Semantic graph V2 migration

The ordered contract and conditional retirement ledger live in
[SIP-0008 review](../../../docs/sip/0008-v2-implementation-review.md).
No retirement candidate has been removed by this implementation.

## Ownership boundary

Package resolution, module/library dependencies and source membership belong to
`library/package`. The compiler owns semantic relationships inside a submodule.
Cross-submodule references must be projected through package-owned memberships;
the compiler must not derive a competing package graph from its source loader.
This boundary follows the user's clarification during implementation.

## Implementation status

This is a P1 foundation, not completion of P1–P8. DefId assignment still uses
revision-local ordinals/positions in existing producers. Persistent keys,
containment/region/sentence integration, typed `with`, proof/dispatch/refinement,
source-free interfaces and ObjectUnit planning remain required. The definition
graph is a checked projection of the authoritative declarations and CFG calls;
it does not own a second executable body or ownership analysis.

## Entry-point capability ledger

| Entry | Current owner | Regression owner / retirement gate |
| --- | --- | --- |
| Standalone check/build/test; MLIR and Agent IR | `driver/src/pipeline/source.sev` | `migration.py`, `cfg.py`, `v2_architecture.py`, `bootstrap_mlir.py` |
| Package build/test and prelude/archive reuse | `driver/src/package_compiler.sev`, package session/pipeline | `artifact_layout.py`, package interface tests, P5 source-free gates |
| Library exports and ABI contracts | `pipeline/package_interface.sev`, MLIR callable emitter | `artifact_layout.py`, `runtime_helper_contracts.py`, P5/P6 |
| Editor/documentation inspection | `pipeline/editor.sev` | Existing editor tests; persistent-identity migration still required |
| `Compiler.analyze_source/analyze_file` | `pipeline/mod.sev` | Semantic/ownership tests; P7 must preserve diagnostics |
| `Compiler.compile_mir_to_mlir` and provider registration | `pipeline/mod.sev`, `compile/src/planner.sev` | Compile-provider tests; P6 CFG region extraction required before retirement |
| Alternate backend/native artifact emission | Backend interfaces and older lowering path | Backend/provider suites; active CLI coverage alone cannot authorize retirement |
| Rust seed and compiler rebuild | `rust_compiler/` | Bootstrap suites; retirement is a separate self-hosting milestone |

## Reproducible baseline

Source revision: `541bf73e0813b0a4efb4f1ed4b8715aa089155f8`.
Source compiler SHA-256:
`2b5a4ca0a6d33cece42445a67c3b7a417df4cce965c0511510f785d543c94e46`.
Sysroot: `/home/tplayer/Documents/Severian`.
Native tools: `mlir-opt-21`, `mlir-translate-21`, `clang-21`.

Fresh `migration.py -v`: 34 tests, 9 failure reports (including separate int/float
subtests), 605.942 seconds. Failures: five Agent IR identity assertions, indexed
receiver evaluation, actual int and float source libraries, and conditional
grammar mutation. Full log: `/tmp/sev-v2/baseline-migration.log` for this session.

A fresh Agent IR probe reproduced the reported duplicate identity:
`format.StringRepresentation`, `primitive_format.StringRepresentation` and
`string_methods.format.StringRepresentation` all name the same source declaration
and `0:1:284`. These are aliases, not three distinct definitions. The probe had
no dangling calls with the old ordinal encoding.

Historical September 8 results in RESULTS.md are not the current baseline.
Candidate validation is recorded after rebuilding below.

## Bootstrap correction and validation

The new native graph tests exposed an existing seed package-ordering defect:
disconnected format-provider packages could sort after the requested root, so
`sev_rust test` selected YAML's test package instead. Those earlier zero-exit
invocations are **not** counted as graph-test passes. Package ordering now visits
disconnected packages before the requested root while retaining dependency order.
The Rust module suite passes all eight tests, including the new root-preservation
regression. The Rust seed rebuilt successfully in 65.004 seconds under the
180-second / 6 GB guard.

Initial source-compiler release and development builds hit the 180-second
ceiling. Development-stage measurements were 89.209 seconds semantic analysis
and 79.218 seconds MIR, before native code generation completed. These attempts
are failed resource gates, not successful compiler builds.

## Package membership contract

The package library selects the owning resolved package by longest containing
root. Standalone files use the nearest manifest, matching existing package
policies. Without an explicit mapping, each source file is a compatibility
submodule. A package can group files without changing compiler code:

```json
{
  "package": {
    "name": "compiler-frontend",
    "version": "0.1.0",
    "metadata": {
      "semantic": {
        "module": "frontend",
        "submodules": {
          "analysis": ["src/definitions.sev", "src/callable.sev"],
          "syntax": ["src/parser.sev"]
        }
      }
    }
  }
}
```

Multiple owners for a source diagnose. Length-delimited unit keys contain the
package identity, module and submodule, separately from content hashes. Explicit
membership survives a file move when its member path is updated; default file
membership intentionally does not. This does not yet make declaration DefIds
persistent. Initializer order remains unchanged.

The compiler checks declaration/call references and emits local SCCs. It delegates
cross-submodule edges to `package.semantic_dependencies`, which uses the existing
`package.dependency` graph library. Agent IR includes package-owned unit metadata
and edges. Its flat declaration list is an inspection index; it does not impose
one global semantic owner or global SCC schedule.
