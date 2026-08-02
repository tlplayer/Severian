# Ranked tensors and linalg

This example keeps contiguous tensor storage and shape metadata behind the
`tensor` package. `rankedMatmul`, `rankedAdd`, and `rankedRelu` invoke MLIR
`linalg.matmul` and `linalg.generic` kernels through generated C ABI wrappers.
User code selects mathematical operations, not SIMD or GPU implementation
details.
