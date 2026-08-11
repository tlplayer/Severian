# Clean-room Unix laboratory roadmap

This roadmap replaces the Linux-port direction. Severian will implement a small
Unix-like operating system from its own interfaces and tests. No source,
headers, test programs, or license text from another kernel is copied into this
repository.

The progression follows the public teaching structure of MIT 6.1810's xv6 labs,
with Stanford's Pintos projects used as an independent completeness check:

- MIT: <https://pdos.csail.mit.edu/6.1810/2025/schedule.html>
- Stanford: <https://web.stanford.edu/class/archive/cs/cs140/cs140.1088/projects/>

Their source licenses permit reuse with notice requirements. This repository's
license must remain the sole license here, so Severian uses the lab requirements
as a behavioral syllabus and writes original implementations and tests.

## Target

The first bootable target is `riscv64-unknown-none` under QEMU `virt`. RISC-V
keeps the architecture boundary small and matches the teaching material without
making Severian depend on xv6 code. Architecture-specific assembly is limited to
entry, trap return, and context switching; everything else should be Severian.

The eventual project layout is:

```text
docs/lab/operating_system/
├── package.toml
├── boot/riscv64/
├── src/
│   ├── memory.sev
│   ├── process.sev
│   ├── scheduler.sev
│   ├── syscall.sev
│   ├── trap.sev
│   ├── filesystem.sev
│   └── console.sev
├── user/
└── tests/
```

## Milestones

### 0. Hosted invariants

Keep the existing `kernel` package as the executable specification. Allocation,
mapping, capabilities, scheduling, syscalls, interrupts, and VFS tests define
behavior that the freestanding kernel must preserve.

Exit criteria:

- all hosted kernel tests pass under `sev test` and `sev memory`;
- mutation testing kills boundary changes in allocation and mapping checks;
- every public kernel operation has branch coverage.

### 1. Freestanding boot and console

Add a freestanding runtime profile, linker layout, RISC-V entry shim, stack,
zeroed BSS, UART output, panic path, and QEMU runner.

Exit criteria:

- QEMU prints one deterministic boot record and exits through the test device;
- no libc, host allocator, pthread, filesystem, or stdio symbol is linked;
- the boot image and linker map are reproducible.

### 2. Physical and virtual memory

Implement page allocation, reference counts, Sv39 page tables, kernel/user
address separation, permission checks, guarded user copies, and teardown.

Exit criteria:

- exhaustion is transactional and double-free is detected;
- unmapped, non-user, write-protected, and execute-protected accesses fault;
- page-table teardown returns every owned frame exactly once.

### 3. Traps and system calls

Implement supervisor trap entry/return, timer and external interrupts, user
fault handling, a stable syscall-number table, and checked argument transfer.

Exit criteria:

- malformed pointers cannot read or write kernel memory;
- unknown syscalls fail without corrupting the process;
- register state is preserved across a timer interrupt and syscall round trip.

### 4. Processes and scheduling

Implement process creation, address-space cloning, executable loading, wait,
exit, sleep/wakeup, pipes, and preemptive round-robin scheduling.

Exit criteria:

- orphan and zombie cleanup is deterministic;
- blocking does not lose wakeups;
- concurrent pipe readers/writers survive stress and forced preemption.

### 5. Copy-on-write

Replace eager address-space copying with read-only shared mappings, reference
counts, and write-fault copying.

Exit criteria:

- read-only executable pages never become writable;
- the last reference frees a physical page exactly once;
- fork followed by exec allocates substantially fewer pages than eager copying.

### 6. File system and recovery

Implement block caching, inodes, directories, path traversal, file descriptors,
and a bounded write-ahead log before adding larger files and symbolic links.

Exit criteria:

- crash injection at every logged write recovers to the old or new transaction;
- concurrent create/unlink does not leak blocks or expose freed inodes;
- path traversal rejects invalid components and handles root correctly.

### 7. Contention and multicore safety

Boot multiple harts, split allocator and block-cache contention, and validate
lock ordering, interrupt state, and sleep-lock boundaries.

Exit criteria:

- race instrumentation reports no unsynchronized shared mutation;
- lock-order tests cannot construct a cycle;
- multicore stress preserves allocator, process-table, and cache invariants.

### 8. Memory mapping and device I/O

Add lazy file-backed mappings, unmap/writeback behavior, and one small virtio
device path. Networking follows only after the memory and interrupt contracts
are stable.

Exit criteria:

- partial unmap and dirty-page writeback behave deterministically;
- device rings enforce ownership and descriptor bounds;
- malformed device completion data cannot escape the driver boundary.

## Assurance policy

Each milestone must ship with native QEMU tests, host-side black-box tests,
coverage thresholds, mutation operators for its invariants, and memory checking
where the host runner can instrument shared code. A milestone is incomplete if
it only boots or only reaches high coverage; its tests must also kill mutations
in permission, bounds, reference-count, and synchronization logic.
