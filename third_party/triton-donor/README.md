# Triton donor snapshot

This directory is a source-only snapshot of Triton revision
`8957b9aac23e526fb1252c7c3b592e6f43c175c8`.

Imported paths:

- `include/` and `lib/`: Triton core IR, layout, optimization, and lowering
  sources.
- `third_party/amd/`: AMD dialect, optimization, and target lowering sources.
- `LICENSE`: the donor MIT license.

The snapshot is reference material for the Rust implementation in
`compiler/boundaries/triton`. It is not a Cargo build input and Severian must
not compile these C++ sources as part of an application build.

Porting rule: copy behavior and tests into Rust in reviewable structural units,
record the donor source path in the Rust module, and keep element type, rank,
shape, layout, and target architecture as IR data. Do not translate donor
class or pass names into tensor operation identities.

LLVM machine-code emission is deliberately outside this source snapshot. A
Rust orchestration layer does not make LLVM or MLIR themselves pure Rust; a
distributed compiler component remains necessary until Severian owns a native
AMD/NVIDIA machine-code emitter.
