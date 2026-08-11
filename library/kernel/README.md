# kernel

`kernel` is a deterministic, hosted laboratory for operating-system policy. It
implements physical-page ownership, virtual mappings with W^X enforcement,
capability checks, process scheduling, a small VFS, syscall dispatch, interrupt
classification, and concurrent syscall workers.

The package deliberately does not claim to be a freestanding kernel. It runs on
the native Severian runtime so the compiler can exercise ownership and
concurrency invariants now. A boot protocol, freestanding ABI, linker script,
interrupt stubs, architecture context switching, and device drivers remain
requirements for a bootable target.

The complete executable laboratory is in `docs/lab/operating_system`.
