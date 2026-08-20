# Compiler bootstrap

`compiler/bootstrap` is the only component allowed to load the Severian core declarations needed to start compilation.

## Pipeline

```text
library/core/primitives/*.sev
  -> real Severian lexer
  -> real Severian parser
  -> AST declarations
  -> declaration validation
  -> UniversalContextBuilder
  -> UniversalContext
```

Bootstrap must use the same lexer and parser as user programs. It may not scan source with `find`, `contains`, `match_indices`, line prefixes, regular expressions, or formatting-dependent block splitting.

## Minimal kernel

The Rust bootstrap kernel may know:

- How to traverse parsed declarations.
- The schema required to describe a primitive.
- How to convert parsed properties and signatures into typed universal definitions.
- Validation rules for duplicate names, duplicate literal defaults, invalid representations, and invalid operator signatures.

It may not hardcode:

- Primitive names such as `int`, `i32`, `f32`, `string`, or `bytes`.
- Which primitive supports `+` or another operator.
- Declaration order.
- Integer or pointer width for the active target.
- A manual list of literal categories.

## Loading

The driver loads the core context once:

```text
Driver
  -> Bootstrap::load_core()
  -> Arc<UniversalContext>
  -> semantic, ownership, MIR, lowering, interface, backend orchestration
```

No downstream crate calls bootstrap or reopens primitive files.

## Required tests

- Comments and formatting changes do not change meaning.
- Declaration reordering does not change stable IDs.
- Duplicate names fail.
- Multiple defaults for one literal kind fail.
- Missing required metadata fails at the declaration span.
- Invalid operator signatures fail before semantic analysis of user code.
- Every primitive source file is parsed by the real parser.
- The loaded context is reused across a full compile.

## End state

When compiled `.pkg` or `.pkgi` interfaces are available, bootstrap may load a validated interface cache. The source parser remains the correctness reference and regeneration path.
