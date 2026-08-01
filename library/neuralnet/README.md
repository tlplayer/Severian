# neuralnet

`neuralnet` builds layers from the portable `tensor` package instead of
owning another matrix implementation. The first slice includes affine dense
projection, ReLU activation, and the activation Jacobian used during gradient
propagation.

Keeping differentiation-oriented operations in `tensor` means a future AD
pass can lower the same definitions to scalar CPU loops, MLIR vector kernels,
or GPU kernels. The package deliberately contains no hidden CUDA-specific API;
target choice is expressed through tensor execution-policy decorators at the
calling kernel.
