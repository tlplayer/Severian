# tensor

`tensor` provides portable, shape-oriented kernels used by numerical and
machine-learning packages. `Tensor[T]` preserves the declared element type and
carries runtime rank, shape, and stride metadata. Slices, reshapes, transposes,
and permutations use views when possible; materialization is an explicit
aliasing boundary. SafeTensor parameters remain read-only mmap-backed storage
at their native byte width and are decoded by kernels on demand.

The typed ML surface includes broadcasting arithmetic, batched matrix
multiplication, reshape and permutation, gather, concatenation, repetition,
last-axis mean and softmax, exact exponential/log/tanh/SILU operations, RMS and
layer-normalization primitives, and rotary position encoding. These are enough
to express transformer attention and MLP blocks as normal `.sev` functions.

`rankedAdd` implements trailing-axis broadcasting, and `rankedSum` is an MLIR
reduction. Runtime shape checks reject incompatible broadcasts, tensor products,
and malformed slices before entering a kernel.

## Lowering direction

Tensor calls are recognized as typed universal operations. Their shapes and
scalar arguments flow through Severian IR into emitted MLIR, which binds the
host implementation through the tensor runtime ABI. Native builds then use the
LLVM host pipeline. This keeps the model graph visible to the compiler instead
of delegating inference to another framework or command-line program.
Severian decorators
import a package's syntax symbols; they are not Python-style wrappers or
execution-policy annotations. The `parallel` package enables task-local `simd`,
`simt`, and `gpu` requests for library kernels. Existing compatible list-based
activation chains fuse automatically; direct tensor-dialect bufferization and
typed GPU kernels are the next compiler stages.

The API keeps mathematical behavior independent of execution placement. The
current implementation is CPU-first; StableHLO/ROCm placement can refine the
same operations without rewriting model architecture code.
