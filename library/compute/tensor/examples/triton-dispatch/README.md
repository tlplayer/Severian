# Triton dispatch smoke test

This executable validates the production GPU boundary:

```text
Tensor operations
  -> FusionRegion
  -> runtime shape/stride specialization
  -> Severian TTIR
  -> pinned native Triton donor
  -> HSACO/PTX
  -> Severian GPU runtime
  -> device execution
```

The dependent `add` and `relu` dispatches keep their structural identities.
Element type, rank, shape, strides, and target architecture are data in the
graph and kernel specialization; none are encoded in operation or function
names.
