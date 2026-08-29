# Rust port status

The donor snapshot is immutable reference code. Production Rust lives in
`compiler/boundaries/triton`.

Ported and tested:

- AMD `gfx` parsing and ISA-family classification.
- AMD wave size, LDS capacity, LDS transfer, cluster, atomic, and scaled
  conversion capability policy.
- Linear-layout basis validation and GF(2)/XOR layout application, including
  identity, broadcast, and swizzled layouts.
- Severian FusionRegion to structural TTIR construction.

Still behind `TritonCompiler`:

- TTIR canonicalization and combination rewrites.
- TritonGPU layout selection and conversion rewrites.
- Reduction and contraction scheduling.
- AMD/NVIDIA target-dialect lowering.
- LLVM optimization, object emission, and HSACO/PTX packaging.

The last item is a binary-toolchain boundary. Porting orchestration and passes
to Rust removes CMake and C++ from Severian application builds; eliminating
LLVM itself would require a separate machine-code emitter rather than a source
translation of Triton.
