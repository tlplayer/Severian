# Library catalog

The catalog is grouped for discovery, while package imports stay flat.

| Area | Packages | Initial status |
| --- | --- | --- |
| Language foundation | `prelude`, `option`, `result`, `iteration` | language design |
| Core data | `boolean`, `collections`, `text`, `bytes` | `boolean` started |
| Numerics | `math`,  `probability`, `statistics`, `random` | `math`, `tensor`, `probability` experimental |
| Machine learning | `model`, `model.neuralnet`, `autodiff`, `optimization` | `tensor` is the canonical container; `model` owns the public machine-learning hierarchy |
| Text processing | `regex`, `unicode`, `format` | `regex` native baseline implemented |
| Data formats | `json`, `csv`, `base64`, `binary` | `json` scalar/list baseline implemented |
| Files and I/O | `io`, `file`, `path` | `io`, `file` native baseline implemented |
| Time and environment | `time`, `environment`, `process`, `system` | `system` name reserved; interfaces planned |
| Concurrency | `sync`, `task`, `channel` | language/runtime design |
| Parallel computing | `parallel`, `distributed` | placement/fusion contracts and local execution experimental; device runtimes planned |
| Networking | `network`, `http`, `url` | `network` bind/loopback baseline implemented |
| Observability | `log`, `trace`, `metrics` | `log` native sinks implemented |
| Security | `hash`, `crypto`, `tls` | provider policy required |
| Data and storage | `pql`, `storage`, `database`, `compression`, `archive` | PQL validation, extensible storage plans, and executable database server experimental |
| Virtualization | `vm`, `container`, `hypervisor` | typed plans and native host discovery experimental |
| Operating systems | `kernel` | hosted kernel laboratory with memory, process, VFS, syscall, interrupt, and scheduler invariants experimental |
| Orchestration | `orchestrator` | scheduling and reconciliation baseline experimental |
| Development | `testing`, `benchmark`, `profile` | language design |

## Admission stages

Packages move through explicit stages:

1. `planned`: the scope and name are reserved in this catalog.
2. `interface-pending`: the package exists, but its typed ABI is not yet
   expressible or connected.
3. `runtime-pending`: the typed interface exists, but its runtime symbols are
   not all implemented yet.
4. `experimental`: callable implementation and Severian tests exist; APIs may
   still change.
5. `stable`: documented behavior, failures, ownership, and compatibility are
   maintained.

Package names use complete words. In particular, Severian uses `system`, not
the clipped Python spelling `sys`. Acronyms remain acceptable when they are the
established name of a domain rather than a shortened ordinary word.

Native capabilities use typed `native(\"symbol\") def ...` declarations inside explicit `unsafe:` blocks
in the `platform` package. Standard-library source imports that package instead of
relying on a compiler-invented namespace. Package acceptance requires native
compilation, execution, and exact output validation.
