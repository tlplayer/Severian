# Domain implementation migration

The compiler owns language meaning and IR transformation. A package owns code
whose only purpose is to execute a domain operation. The deciding question is:

> Does the compiler need to understand this operation to preserve language
> semantics or optimize IR?

If yes, it may remain compiler-owned. If it merely needs to execute, it belongs
in a library behind a constrained semantic contract or typed foreign call.

## Ownership boundary

Compiler-owned:

- parsing, types, ownership, traits, operators, and semantic attributes;
- `with`/`without` sequencing and trait-registry construction;
- HIR, MIR, optimization, generic ABI/FFI lowering, and code generation;
- generic concurrency semantics and tensor-aware analysis/orchestration;
- backend selection, legality, and cross-operation optimization.

Library-owned:

- OS calls and platform layouts;
- serializers, codecs, regex engines, compression, crypto, and hashing;
- file/image/audio/serializer/driver implementation registries;
- profiling collectors and hardware/vendor discovery;
- model architectures and checkpoint conventions;
- backend capability and operation-description data.

Tensor remains privileged: packages describe capabilities and mappings, while
the compiler owns tensor IR, selection, legality, fusion, and lowering.

## Migration ledger

| Area | Current state | Next boundary |
|---|---|---|
| Generic C ABI/FFI | Complete first slice | Add Rust ABI through the same `ForeignCall` |
| File text reads | Package-owned `c-v1` provider | Migrate writes, binary handles, mapping, locks, and directory operations separately |
| Regex | Package-owned POSIX provider | Add an optional compiled-pattern handle without changing compiler lowering |
| Math/random/environment/process/network | Package-owned providers | Preserve architecture guards while expanding platform coverage |
| Data-format dispatch | Closed dispatch generated from reachable `Reader` trait metadata | Extend the same `registry[T]` primitive to image/audio/serializer/driver contracts |
| JSON/CSV | Parsers and encoders are source-owned by `json` and `csv`; the runtime exposes only generic dynamic-value primitives | Grow codec coverage in packages without compiler lowering changes |
| YAML/TOML/base64/compression | Library or incomplete | Keep all engines and codecs package-owned |
| Hash/crypto/TLS | SHA/MD5 and OpenSSL details still in compiler/platform | Move algorithms/providers into `hash`, `crypto`, and `tls` packages |
| Filesystem/path/directory | Many POSIX shims remain in the compiler bridge | Move one coherent handle/operation family at a time into `file`, `path`, and `os` |
| Time/profiling | Clock calls and collectors remain platform-specific | Keep metric semantics in compiler; put collectors in profile packages |
| Threads/channels | Language scheduling plus pthread implementation are coupled | Retain task/channel MIR; move pthread/Windows execution providers behind runtime ABI |
| Vendor discovery | ROCm/CUDA tools and paths are compiler-known | Resolve driver/toolkit capabilities through driver packages and package resolution |
| XLA/PJRT discovery | ROCm plugin paths are compiler-known | Let the XLA package provide plugin artifacts and discovery metadata |
| Models/checkpoints | Generic model IR exists; Safetensors loading remains under compiler/XLA | Move checkpoint formats and architecture plans into model/storage packages |
| String/collection algorithms | Large compiler runtime catalog | Retain representation, allocation, bounds, and atomic primitives; migrate ordinary algorithms |

## Guardrails

- Generic lowering must not match package names, provider symbols, file
  extensions, model architectures, or vendor installation paths.
- Package manifests own native sources, targets, include paths, and system
  libraries. Packages never receive mutable compiler internals.
- A migrated provider gains an architecture test that rejects its old symbols,
  headers, and implementation calls under compiler lowering/backend/platform.
- A migration is not complete until its native acceptance test traverses
  source package → `ForeignCall` → generic lowering → package provider.
- Hardware intrinsics remain possible, but they are selected from typed
  capabilities rather than used as homes for standard algorithms.

## Recommended order

1. Finish the remaining file/path/directory operation families.
2. Move hash, TLS, HTTP, and codec implementations.
3. Separate pthread execution from generic task/channel MIR.
4. Move ROCm/CUDA/PJRT discovery into driver/XLA packages.
5. Move checkpoint loaders and any model-branded lowering descriptions into
   model and storage packages.
6. Audit string and collection helpers, retaining only representation-critical
   runtime primitives.
