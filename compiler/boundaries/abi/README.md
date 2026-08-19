# severian-abi

Compiler-side model for binary interfaces.

## Boundary

This crate owns:

- calling-convention identities and properties;
- concrete ABI-safe scalar, pointer, array, record, union, enum, function, resource, and opaque types;
- generic ABI schemas;
- type/const/address-space schema parameters;
- schema composition and instantiation;
- ownership/lifetime/parameter-mode metadata;
- target-dependent concrete layout;
- structural ABI validation.

This crate does **not** own:

- FFI symbol lookup or dynamic loading;
- package/build/link resolution;
- Tensor, Data, DataFrame, network, Python, XLA, Triton, NCCL, MPI, etc.;
- semantic language types;
- HIR/MIR/MLIR lowering.

## Core invariant

Generic schemas may contain unresolved parameters:

```text
View[T, Space]
Array[T, N]
LibraryDefinedDescriptor[T, Space]
```

`AbiType` and `AbiSignature` may not.

```text
Interface/library ABI schema
        ↓ instantiate
Concrete AbiType / AbiSignature
        ↓ validate
Target layout
        ↓ lowering/codegen
```

This lets a tensor library describe, for example, a dense tensor descriptor as an ordinary record of pointers/rank/shape/strides while the ABI compiler remains unaware that the record represents a Tensor.

## Extensibility rule

Adding a library/runtime concept should normally add an ABI schema, not an `AbiType` enum variant.

Good:

```text
library/tensor defines DenseTensor[T, Space]
library/data defines ColumnView[T]
library/xla defines BufferHandle
```

Bad:

```text
AbiType::Tensor
AbiType::DataFrame
AbiType::XlaBuffer
```
