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

Build the native donor component once, then build/run the ordinary Severian
package:

```sh
compiler/boundaries/triton/native/build-native.sh
cargo build -p severian-driver -p severian-tensor-jit-provider
target/debug/sev run library/compute/tensor/examples/triton-dispatch
```

The first command is incremental and returns immediately while the existing
bridge matches its ABI sources. `sev build` stages both runtime libraries next
to the generated executable; application builds do not invoke CMake or compile
Triton.
