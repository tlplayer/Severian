# Operating-system kernel laboratory

The bootable follow-on is an original, clean-room Unix-like teaching kernel.
Its staged implementation and assurance gates are documented in
[`UNIX_LAB_ROADMAP.md`](UNIX_LAB_ROADMAP.md). This direction uses public
university lab requirements as a syllabus and does not copy another kernel's
source or license into Severian.

This example builds the `platform` and `kernel` packages first, consumes their
artifacts, and links a native executable that composes:

- transactional physical-page allocation and process teardown;
- per-process virtual mappings with alignment, ownership, and W^X checks;
- capability-gated VFS and syscall operations;
- bounded concurrent syscall workers with deterministic reply replay;
- priority-ordered round-robin scheduling;
- timer, page-fault, and keyboard interrupt classification;
- a cross-subsystem kernel audit; and
- a bounds-guarded unsafe boot-image inspection boundary.

Build and execute it:

```sh
cd docs/lab/operating_system
sev build
./target/debug/operating-system-lab
sev compile-tests main.sev -o /tmp/severian-os-tests
/tmp/severian-os-tests
```

## What this proves

The example executes OS policy and invariants through Severian's real ownership
checker, native classes and collections, pthread tasks, bounded channels, and
MLIR/LLVM lowering. It is useful for finding compiler bugs involving aliasing,
runtime cloning, dynamic fields, task ABI types, and deterministic shutdown.

## What keeps it from booting

This remains a hosted kernel laboratory. A genuinely bootable target requires a
freestanding compiler/runtime mode, target triples without libc, a boot protocol
and linker script, architecture-specific startup and context switching,
interrupt-entry assembly, volatile/MMIO operations, atomics, allocators that do
not call the host, panic behavior without stdio, and hardware drivers. Those are
explicit target requirements rather than behavior simulated by this example.
