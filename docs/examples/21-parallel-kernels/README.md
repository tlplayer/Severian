# Parallel kernel placement

This example executes the same fused dense+bias+ReLU kernel under `simd`,
`simt`, and `gpu` placement requests. `fuse` is orthogonal to placement, so it
can be attached to a vector or device task without becoming a decorator.

All three paths currently execute through the native pthread CPU fallback. The
generated MLIR still distinguishes them with `severian_parallel`,
`severian_fusion`, and `severian_device_fallback` attributes. That makes the
example executable today and gives future MLIR vector, GPU, and fusion passes a
stable operation-local contract to consume.

`denseReluKernel` is written as `Relu(add(matVec(...), bias))`. Because the SIMD
and GPU spawns request `fuse`, the compiler rewrites that graph to the
single-pass `fusedDenseRelu` reference kernel. The otherwise-equivalent SIMT
function omits `fuse` and therefore remains an unfused control. Other operation
graphs are currently left unchanged.
