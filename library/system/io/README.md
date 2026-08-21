# io

Composable stream contracts shared by files, sockets, codecs, archives, and
process pipes. `Reader`, `Writer`, `Seeker`, and `Closer` use structural trait
conformance, while `MemoryStream` provides a deterministic reference
implementation with atomic failures, explicit EOF, seeking, truncation, and
zero-filled sparse writes.

The package also owns standard-stream output and the bootstrap native boundary
for `print`. Core's prelude only re-exports that declaration; neither semantic
analysis nor any compiler IR recognizes `print` specially. The current `puts`
provider is deliberately isolated in [`src/lib.sev`](src/lib.sev) and can be
replaced by the complete stream implementation without changing the prelude.
