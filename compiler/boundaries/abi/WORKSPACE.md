# Workspace placement

Place this directory at:

```text
compiler/boundaries/abi/
```

Update the root workspace member from the old path:

```toml
"compiler/abi",
```

to:

```toml
"compiler/boundaries/abi",
```

The interface boundary should depend on ABI by path:

```toml
severian-abi = { path = "../abi" }
```

Dependency direction:

```text
abi
 ↑
interface
 ↑
semantic / lowering
```

`severian-abi` must not depend on interface, semantic, HIR, MIR, MLIR, FFI, or runtime crates.
