# Behavioral symmetry tests

Syntax agreement is not semantic agreement. A symmetry case runs a small
Severian program and an independent Python or Rust program, then compares their
normalized observable result. The reference is an oracle for that case only;
it does not define the whole Severian language.

## Current cases

| Case | Severian surface | Reference | Comparison | Result |
| --- | --- | --- | --- | --- |
| `scalar operators` | precedence, integer remainder, `u64` bitwise operations | Python | exact stdout `14`, `1`, `15` | passing |
| `ordinary generics` | constrained generic body, explicit applications to `int` and `i64` | Rust | exact stdout `14`, `42` | passing |

Run them with:

```bash
sev run docs/api --bin api-symmetry
```

The runner itself is Severian. Python and Rust appear only in the independent
reference programs under `docs/api/symmetry/`; neither validates API records.

## What a case must specify

Each behavioral case should identify:

1. Stable API IDs covered.
2. Input values and concrete representations.
3. Observable output, error, mutation, or ownership result.
4. Reference language and any intentional semantic adaptation.
5. Normalization applied before comparison.
6. A weakness ID when Severian cannot yet reach the comparison point.

Output equality is appropriate for pure scalar functions. Tensor cases should
compare shape, representation, values, and tolerances. Error cases should
compare the error category and relevant payload rather than implementation
stack text. Ownership cases need a compile-success/compile-failure oracle, not
runtime output.

## Safety policy

Cases run sequentially. Each child is pinned to one CPU, limited to 512 MiB of
virtual memory, and bounded by a 30-second timeout. This keeps LLVM/MLIR from
creating a thread pool sized to the entire machine under a tight address-space
limit. The runner stops a failing case cleanly and reports its name; it does not
launch retries in parallel.

## Semantic differences to make explicit

- Python integers are unbounded while Severian/Rust fixed-width integers are
  not. Integer symmetry cases must choose values inside the tested width or
  explicitly compare overflow behavior.
- Python and Rust floating-point formatting can differ. Floating cases should
  compare parsed values using an API-declared tolerance.
- Rust overflow behavior depends on build profile unless checked operations are
  selected explicitly.
- Python ownership is not an oracle for Severian moves and borrows. Use Rust or
  compile-outcome fixtures for those contracts.
- Host tensor libraries may infer shape or dtype at runtime. A reference may
  validate numeric results, but it must not weaken Severian's static shape and
  representation contract.

## Missing symmetry coverage

The present slice does not yet compare characters/strings, error propagation,
measured quantities, shape constraints, ownership failures, concurrency, or
tensor operations. Tensor symmetry should proceed structurally: elementwise,
reduction, rank-2 and batched matmul under the same operation ID, views/layout,
gather/scatter, conversion, then `StorageViewAbi` specialization and launch.
Those omissions are test gaps, not implied correctness.
