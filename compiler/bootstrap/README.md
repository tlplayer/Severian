# Compiler bootstrap

`compiler/bootstrap` assembles the initial universal context. Primitive axioms
are installed directly by `compiler/universal`; bootstrap then loads extensible
source-defined compiler protocols.

## Pipeline

```text
compiler/universal/primitive
  -> TypeContextBuilder
  -> compiler protocol sources
  -> real Severian lexer/parser
  -> protocol declaration validation
  -> UniversalContext
```

Bootstrap must use the same lexer and parser as user programs for source-defined
protocols. It may not reinterpret primitive definitions from library source.

## Minimal kernel

The Rust bootstrap kernel may know:

- How to traverse parsed declarations.
- How to install the compiler-owned universal primitive schema.
- How to convert parsed compiler protocols into stable universal routes.
- Validation rules for duplicate protocol declarations and invalid routes.

Primitive names, operators, representations, and literal defaults may be known
only by `compiler/universal/primitive`; bootstrap itself must not duplicate them.

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

- Universal primitive identities are stable.
- Exactly one default exists per literal kind that requires a default.
- Operator signatures resolve through universal tests.
- Compiler protocol comments and formatting do not change meaning.
- The loaded context is reused across a full compile.

## End state

When compiled `.pkg` or `.pkgi` interfaces are available, bootstrap may load a
validated compiler-protocol cache. Primitive axioms remain owned by universal.
