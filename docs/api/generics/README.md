# Generics and shapes

Severian has three generic parameter kinds. Keeping them distinct prevents
dtype, rank, and runtime extent from being collapsed into names or accidental
monomorphizations.

| Section | Parameter kind | API record | Status |
| --- | --- | --- | --- |
| [Type parameters](type-parameters.md) | `T: type` | `generic.function` | partial across dependency interfaces |
| [Dimension parameters](dimensions.md) | `D: dim` | `generic.dimension` | implemented core model |
| [Shape packs](shapes.md) | `S...: shape` / source `*S: tensor.Dim` | `generic.shape` | partial conformance |
| [Type application](type-application.md) | `callee[T](...)` | `generic.type_application` | partial cross-package support |

The machine records are in
[`../language/generics/core.toml`](../language/generics/core.toml). Generic
notation is defined in [`../APPENDIX.md`](../APPENDIX.md).

## Separation rule

Generic substitution resolves facts such as `T → bf16` and symbolic shape
relations. Runtime kernel specialization resolves concrete device facts such
as shape `[1, 16, 512, 128]`, strides, and target architecture. Neither process
creates dtype- or rank-named source functions.
