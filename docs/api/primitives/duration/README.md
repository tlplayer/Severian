# `duration`

API ID: `primitive.measured`; Universal path:
`universal.primitive.duration`.

## Representation

`duration` is a measured semantic type represented as f64 canonical seconds.
Suffixes `ns`, `us`, `ms`, `s`, `min`, `hr`, and `day` normalize at semantic
analysis.

## Source semantics

Same-type sign, addition/subtraction, equality, and ordering are defined.
`duration / duration` returns `float`. A machine `float / duration` returns
`frequency`.

```sev
def fraction(elapsed: duration, budget: duration) -> float:
    return elapsed / budget
```

## ABI and lowering

Lowering uses f64 after unit normalization. OS/native APIs using integer
nanoseconds require explicit checked conversion and rounding.

## Tensor

`duration` is not a tensor element. Time-series tensors should specify numeric
storage plus a unit/schema contract.

## Current weakness

Monotonic versus wall-clock meaning, infinity/negative policy, and native time
conversion rounding are not fully specified here.
