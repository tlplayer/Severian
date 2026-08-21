# Core compile protocol

This source package owns the ordinary Severian `Compiler` and
`CompileType[C]` marker traits. It contains no Rust routing policy. Bootstrap
resolves declarations implementing the protocol into stable universal
`CompilerId` values until the protocol implementation is self-hosted.
