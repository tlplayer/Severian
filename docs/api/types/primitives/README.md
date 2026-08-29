# Primitive types

Primitive types are registered once in the universal type system. Their names
select a `PrimitiveRepresentation`; they do not select dtype-named functions or
backend symbols.

Family pages explain shared rules; every registered primitive also has an exact
contract folder containing `README.md` and `conformance.sev`.

| Family | Primitive folders |
| --- | --- |
| Boolean/character | [`bool`](bool/README.md), [`char`](char/README.md) |
| Signed integers | [`int`](int/README.md), [`i8`](i8/README.md), [`i16`](i16/README.md), [`i32`](i32/README.md), [`i64`](i64/README.md), [`i128`](i128/README.md), [`isize`](isize/README.md) |
| Unsigned integers | [`u8`](u8/README.md), [`u16`](u16/README.md), [`u32`](u32/README.md), [`u64`](u64/README.md), [`u128`](u128/README.md), [`usize`](usize/README.md) |
| Floating point | [`float`](float/README.md), [`f8e4m3fn`](f8e4m3fn/README.md), [`f8e5m2`](f8e5m2/README.md), [`f16`](f16/README.md), [`bf16`](bf16/README.md), [`f32`](f32/README.md), [`f64`](f64/README.md), [`f80`](f80/README.md), [`f128`](f128/README.md) |
| Text/storage/control | [`string`](string/README.md), [`Error`](Error/README.md), [`bytes`](bytes/README.md), [`None`](None/README.md), [`unit`](unit/README.md), [`args`](args/README.md) |
| Measured | [`data_size`](data_size/README.md), [`duration`](duration/README.md), [`data_rate`](data_rate/README.md), [`frequency`](frequency/README.md), [`percentage`](percentage/README.md), [`temperature`](temperature/README.md), [`voltage`](voltage/README.md), [`current`](current/README.md), [`power`](power/README.md) |

Family references remain available for
[signed integers](signed-integers.md),
[unsigned integers](unsigned-integers.md),
[floating point](floating-point.md), [text](text.md), and
[measured values](measured.md).

The authoritative registry is `compiler/universal/src/primitive/mod.rs`. The
machine index is [`../language/primitives/families.toml`](../language/primitives/families.toml).

## Invariants

- A primitive name maps to one representation record.
- Width and format are data in that record.
- `f16`, `bf16`, and `f32` never require different operation identities.
- Backend legality is checked after source type resolution and before emission.
- A registered type may be semantically valid even when a target lacks a legal
  instruction sequence; the backend must reject that pair explicitly.
- `api/check.sev` derives the primitive folder names from Universal and
  compiles every folder's `conformance.sev`; a new registry entry therefore
  fails documentation validation until its full folder exists.
