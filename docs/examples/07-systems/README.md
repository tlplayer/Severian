# Systems examples

This directory freezes Severian's system boundary from unsafe memory through
portable IO. These are language and library contracts, not permission for the
compiler to grow operating-system special cases.

The dependency direction is:

```text
source declarations
        ↓
      XXI                 @c, @rust, and provider selection
        ↓
      FFI                 ownership, lifetime, and value conversion
        ↓
      ABI                 target layout and call classification
        ↓
platform-native providers
        ↓
system libraries          process, filesystem, files, IO, network, environment
```

The compiler owns syntax, semantic identities, and the boundary machinery.
Libraries own policy and user-facing behavior. In particular, `print` is an IO
library function that the prelude may later re-export; it is not syntax, HIR,
MIR, LIR, or a backend intrinsic.

## Sections

| Directory | Contract |
| --- | --- |
| [`01-unsafe`](01-unsafe/) | Explicit unsafe scopes and auditable boundary operations |
| [`02-memory`](02-memory/) | Managed allocation, raw allocation, layout, and deterministic cleanup |
| [`03-pointers`](03-pointers/) | Pointer formation, indexing, arithmetic, nullability, and casts |
| [`04-ffi`](04-ffi/) | XXI declarations mapped through FFI and ABI |
| [`05-platform`](05-platform/) | Neutral target facts and explicit platform capabilities |
| [`06-process`](06-process/) | Arguments, spawning, waiting, status, termination, and pipes |
| [`07-filesystem`](07-filesystem/) | Paths, directories, metadata, and filesystem mutations |
| [`08-files`](08-files/) | Typed file access, mapping, and locking |
| [`09-io`](09-io/) | Readers, writers, streams, standard handles, and printing |
| [`10-network`](10-network/) | Addressing, DNS, sockets, and HTTP layering |
| [`11-environment`](11-environment/) | Process environment reads, mutation, and snapshots |

## Rules shared by every example

- Safe wrappers expose structured errors and do not require `unsafe` at call
  sites.
- Raw pointers, unchecked layout conversions, and direct foreign calls remain
  visibly `unsafe`.
- Resource ownership is explicit. Handles close once, moving a handle
  invalidates the source, and borrowed data cannot outlive its owner.
- Byte sizes and byte offsets are data-unit values such as `0B`, `4KiB`, or
  `1MiB`, never dimensionless integers. Collection and pointer element indices
  remain dimensionless because they count elements rather than bytes.
- Platform differences are selected through providers and capabilities rather
  than target-name conditionals spread through user code.
- Tests that mutate process-global state, the filesystem, or the network are
  integration tests and must isolate and clean up their resources.
- No example relies on an `extern` keyword. A resolved language attribute such
  as `@c` already declares the boundary.

## Prelude boundary

The prelude imports ordinary declarations from system libraries:

```sev
import print from io
```

The native declaration belongs to the selected `io` provider, not to core.
Bootstrap currently loads that provider source beside the prelude until package
import resolution can follow the re-export itself. Compiler IR continues to
represent the result as an ordinary resolved call.
