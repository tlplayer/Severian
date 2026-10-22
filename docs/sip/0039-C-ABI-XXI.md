# SIP-0039: C ABI Contracts and XXI Boundaries

Status: Implementing
Type: Library | Compiler | Interop
Tags: abi | xxi | c | mlir | ownership

## Appendix

- `system.abi`: target-dependent C representation and calling contracts.
- `system.xxi`: checked wrappers over external calls, status and transfer limits.
- `@c`: compiler declaration of an external C symbol; calls require `unsafe`.
- `CAbi`: the Python compiler's implementation of the same target contract.
- `Target`: architecture and operating system; pointer width and C long width.
- `Layout`: representation size and alignment in bytes.
- ABI compatibility describes a binary calling boundary. It does not make pointers portable between processes or CPU and GPU address spaces.

## Goal(s)

- One path: declaration -> checked MIR -> MLIR call -> LLVM -> linked C symbol.
- C source belongs to a library/provider. Never generate a second program to implement a Severian expression.
- Select layout and argument extension from the compilation target, never the build host.
- Keep C spelling, symbol identity, types, ownership and error handling explicit.
- `system.xxi` depends on `system.abi`; neither depends on a compiler AST package.

## Examples & Problems

```sev
@c(symbol="example_add")
def add(left: i32, right: i32) -> i32

def answer() -> i32:
    unsafe:
        return add(20, 22)
```

```text
add(i32, i32) -> i32
  -> typed MLIR call to example_add
  -> existing object compilation and linking

C source files supplied by xxi.c.sources
  -> compiled as external library inputs
  -> no generated expression-specific C wrappers

x86_64 Linux / aarch64 Linux / x86_64 Darwin / aarch64 Darwin
  -> 64-bit pointers, 64-bit C long
x86_64 Windows / aarch64 Windows
  -> 64-bit pointers, 32-bit C long
unknown target or mismatched pointer width -> diagnostic
```

Supported direct values are bool, 8/16/32/64-bit integers, f32/f64 and raw pointers. Unit is a result only. Signed/unsigned narrow argument and result extension follows the target C ABI. Use explicit widths; Severian `int` is not a portable spelling of C `int` or C `long`.

The existing string-input adapter remains an explicit MLIR conversion to a temporary NUL-terminated buffer, valid only for the duration of the call. C must not retain it. Returned C strings require an explicit pointer and ownership wrapper. Aggregates by value, varargs, callbacks, and 32-bit targets are rejected until their contracts and regressions exist.

C declarations remain callable only from `unsafe`. XXI checks C status codes, counts, progress, and budgets after the call; it cannot prove that arbitrary C respects pointers or lifetimes.

MLIR's C API is a compiler binding for constructing/verifying IR. It is separate from the generated program's C ABI calls. GPU driver calls must use the same MLIR call and dependency path; kernel compilation and transfer scheduling must not be hidden in generated C.

## Tests

```text
same i8 declaration, different targets
  x86_64 Linux -> sign extension attributes
  aarch64 Linux -> no x86 extension attributes
  Windows -> Windows C convention

void parameter -> reject
record by value -> reject
f16/f128 -> reject
C string return -> reject without ownership adapter
C call outside unsafe -> reject
C call inside unsafe -> MLIR call, no generated C source

xxi.count(-1, 10) -> error
xxi.transfer(11, 10) -> error
xxi.transfer(0, 10, true) -> error (no progress)
xxi.account(budget(2, 10), 11) -> error
```

Regression tests follow the implementation in each `.sev`/`.py` file. Cross-target IR checks need no GPU. Linking/execution is a separate host-specific check; foreign-target objects are never executed on the build host.

References:

- [LLVM parameter attributes](https://llvm.org/docs/LangRef.html#parameter-attributes).
- [Microsoft x64 ABI conventions](https://learn.microsoft.com/en-us/cpp/build/x64-software-conventions).

## GPU/SIMD ownership and completion convention

- Borrowed input: C can read during the call; it cannot retain, resize, or free it.
- Owned output: Severian allocates host output and retains its allocator/destructor.
- Device storage: the GPU provider owns it and releases it with HIP, never host `free`.
- Captures: scalar copies in MLIR stack slots remain alive until the synchronous call returns.
- Completion: return means kernel execution and host output copy completed. A successful launch alone is not completion.
- Failure: the provider attempts all acquired-resource releases, preserves the first failure, and returns a status. The current compiler adapter reports failure with `cf.assert` after provider cleanup; recoverable GPU errors are not implemented.
- Unsafe C: an ABI cannot stop arbitrary C retaining a pointer, writing outside an extent, or using an incompatible allocator. Raw `@c` remains unsafe. These guarantees apply to the shipped synchronous provider.

```sev
# Implemented for contiguous rank-one f32/f64 tensors with equal lengths.
# Both return a fresh host tensor; input aliases are allowed because inputs are read-only.
def gpu_multiply(a: tensor[f64], b: tensor[f64]) -> tensor[f64]:
    with gpu:
        return a * b

def simd_multiply(a: tensor[f64], b: tensor[f64]) -> tensor[f64]:
    with simd:
        return a * b
```

```text
with gpu: a * b
  -> check shape and byte extent before allocation
  -> snapshot inputs into private contiguous host staging
  -> main program MLIR contains gpu.module and host MLIR adapter
  -> ROCDL serializes the GPU module in that compilation pipeline
  -> gpu.binary storage becomes an LLVM dialect constant in the same module
  -> MLIR calls the fixed system/gpu/c/rocm.cpp HIP boundary
  -> private stream: upload -> launch -> completion -> download
  -> release events, device buffers, module, stream, argument storage
  -> return -> release host staging -> expose owned host result

with simd: a * b
  -> shape check -> MLIR vector operations + scalar tail -> owned host result

empty tensors -> no driver allocation or kernel launch
length mismatch -> fail before output/device allocation
same tensor used twice -> independent read-only snapshot
unsupported element type/rank/stride -> diagnostic, no silent CPU fallback
```

No expression-specific C file or object is generated. The fixed provider contains HIP calls and resource cleanup; arithmetic lives in MLIR. The compiler resolves this provider dependency so callers do not import HIP or a collection of vendor libraries. SIMD needs no external provider.

The initial policy is deliberately synchronous and copy-based: no concurrent host/device ownership and no persistent cache requiring dirty flags. Future asynchronous execution must return an owning completion token that retains all buffers; reads, mutation, resize, and destruction must wait or be rejected. Merely recording a `pending` boolean in a copyable library object would not enforce this contract. Persistent device buffers, cancellation, zero-copy host registration, and recoverable device errors require additional ownership and failure-path work.

Regression coverage is inline in `py_compiler/syntax/complex/gpu.py`: both placement spellings compile, GPU arithmetic remains visible in the main MLIR module, rendering cannot invoke a C compiler, serialized images remain in MLIR, and unsupported types fail. `RocmExecutionTests` is the explicit hardware suite; it must pass on ROCm before hardware support is considered validated.

- [HIP stream synchronization](https://rocm.docs.amd.com/projects/HIP/en/docs-10.1.0/how-to/hip_runtime_api/asynchronous.html).
- [MLIR GPU module serialization](https://mlir.llvm.org/docs/Dialects/GPU/#module-serialization).
