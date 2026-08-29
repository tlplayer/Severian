# Appendix A — generic terms and notation

These terms apply to every record and symmetry test under `docs/api/`.

This appendix reserves metavariables used by all API records. They describe
contracts; they are not necessarily literal Severian identifiers.

| Term | Meaning |
| --- | --- |
| `T`, `U` | A source or IR type. `T` is conventionally the primary operand type. |
| `O` | An operation identity or operation class. It does **not** mean output. |
| `R` | Result type. |
| `V` | Runtime value. |
| `E` | Error type or error union member. |
| `D` | One dimension value or symbolic dimension expression. |
| `S...` | Ordered shape sequence of zero or more dimensions. |
| `A...` | Ordered axis sequence. |
| `K` | Primitive element kind, such as signed integer or IEEE float. |
| `B` | Element bit width. |
| `L` | Storage layout/stride contract. |
| `P` | Execution placement, such as CPU or GPU. |
| `G` | Concrete target architecture/device generation. |
| `N` | Count, rank, or arity when constrained to a non-negative integer. |

## Function notation

`foo(T, O)` in prose means that `foo` is parameterized by a type `T` and an
operation identity `O`. Severian source generics use brackets:

```sev
def foo[T, O](value: T) -> R:
    ...
```

Records use `type_params = ["T", "O", "R"]` and put relationships in
`constraints`. A metavariable appearing in a parameter or result must be
declared in `type_params` unless it is a concrete language type.

## Tensor notation

```text
Tensor[T, S...]
Tensor[T, D0, D1, ...]
Tensor[T, ?]
```

- `T` is element type data.
- `S...` is shape data, not part of an operation ID.
- A named `D` preserves equality wherever that dimension is reused.
- `?` is an anonymous runtime dimension and does not imply equality with any
  other `?`.
- Unranked and rank zero are distinct: `Tensor[T, ?rank]` is not `Tensor[T]`.

Generic specialization resolves relationships such as `T = bf16` or
`D = 128`. Runtime specialization supplies unresolved concrete shape, stride,
layout, placement, and target data. Neither mechanism creates source functions
such as `matmul_rank4` or `load_bf16`.

## Operation notation

Structural operations use one identity plus data:

```text
Elementwise(O, T, S...)
Reduce(O, A..., T, S...)
Matmul(T, lhs_shape, rhs_shape, batch_dimensions, contraction_dimensions)
```

`O` selects behavior inside a structural class (`Add`, `Relu`, `Sum`, and so
on). Dtype, rank, axes, and batch dimensions remain fields in the IR/ABI.

## Constraint notation

| Form | Meaning |
| --- | --- |
| `T: Trait` | `T` satisfies a trait. |
| `T: Add[U, R]` | adding `T` and `U` produces `R`. |
| `D0 = D1` | dimensions must unify. |
| `D % K = 0` | dimension divisibility constraint. |
| `rank(S) = N` | shape rank is statically known as `N`. |
| `A in axes(S)` | axis is valid for the shape. |
| `P supports O` | selected placement has a legal lowering for the operation. |

## Status vocabulary

- `implemented`: accepted, lowered, and backed by listed conformance evidence.
- `partial`: a real path exists but at least one documented contract dimension
  or backend is missing.
- `experimental`: behavior exists but compatibility is not promised.
- `specified`: normative contract exists without a complete implementation.
- `unavailable`: deliberately rejected or not yet implemented.
- `deprecated`: accepted temporarily and scheduled for removal.
