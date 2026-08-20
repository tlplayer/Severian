# Compiler boundaries

Boundaries connect the compiler to external formats, tools, runtimes, and systems. They consume compiler-owned models and do not define Severian language semantics.

## Responsibilities

- `driver`: composition root and pipeline orchestration.
- `interface`: `.pkg` and `.pkgi` encoding, decoding, validation, and compatibility.
- `backend`: backend contract, artifact production, and external tool invocation.
- `abi`: external calling and data-layout contracts.
- `ffi`: foreign ownership, lifetime, conversion, and safety contracts.
- `xxi`: source-facing external language declarations and imports.

## External interface pipeline

```text
@c / @rust source declarations
        ↓ XXI
language, provider, symbol, source type contracts
        ↓ FFI
ownership, lifetime, conversion, ABI selection
        ↓ ABI
concrete target layout and argument/return classification
```

No layer may skip downward: XXI does not lay out records, and ABI never reads
source declarations or semantic `TypeId`s.

## Driver rule

The driver is the only component that constructs the complete compile context:

```text
source inputs
core UniversalContext
ABI Target
package/interface dependencies
backend selection
```

It passes references through the pipeline. Global reload functions are prohibited.

## Boundary rule

A boundary may convert from a compiler model into an external model. It may not become the canonical owner of `TypeId`, `PrimitiveId`, operators, literal defaults, or primitive definitions.

## Failure rule

Unsupported external capabilities must produce an explicit error. A boundary must not silently substitute a different integer width, float format, ABI, calling convention, or runtime ownership model.
