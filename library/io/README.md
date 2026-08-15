# io

Composable stream contracts shared by files, sockets, codecs, archives, and
process pipes. `Reader`, `Writer`, `Seeker`, and `Closer` use structural trait
conformance, while `MemoryStream` provides a deterministic reference
implementation with atomic failures, explicit EOF, seeking, truncation, and
zero-filled sparse writes.
