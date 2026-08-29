# Tensor conversion

API ID: `tensor.convert`

Convert has source and target `PrimitiveRepresentation` fields. Every legal
pair uses this one operation identity.

```sev
import tensor

def compute_wide[T: tensor.TensorElement, *S: tensor.Dim](
    value: Tensor[T, *S],
) -> Tensor[f32, *S]:
    return tensor.to_f_32[T](value)
```

Rounding, saturation, sign extension, truncation, and invalid conversion are
properties of the selected representation pair. Storage conversion and compute
accumulation policy are distinct decisions.

Current weakness: the exhaustive i8–i128, u8–u128, and f8–f128 backend matrix
is not complete.
