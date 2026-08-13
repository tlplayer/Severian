# Library catalog

The catalog is grouped for discovery, while package imports stay flat.

| Area | Packages | Initial status |
| --- | --- | --- |
| Language foundation | `core`, `prelude`, `option`, `result`, `iteration` | `core` experimental |
| Core data | `list`, `map`, `set`, `string`, `boolean`, `bytes`, `data` | collection, string, and tabular APIs experimental |
| Numerics | `math`, `tensor`, `probability`, `statistics`, `random` | `math`, `tensor`, `probability`, and `random` experimental |
| Visualization | `graphics`, `plot` | deterministic headless canvas and charts experimental; windows and GPU backends planned |
| Machine learning | `model`, `model.neuralnet`, `autodiff`, `optimization` | `tensor` is the canonical container; `model` owns the public machine-learning hierarchy |
| Text processing | `regex`, `unicode`, `format` | `regex` native baseline implemented |
| Data formats | `json`, `csv`, `yaml`, `base64`, `binary` | format packages own codecs and documents; `file.read()` provides extension dispatch |
| Files and I/O | `io`, `file`, `path`, `os` | typed contents in `file`; namespace operations and metadata in `os` |
| Time and environment | `time`, `environment`, `process`, `system` | clock, environment, and process APIs experimental; `system` reserved |
| Concurrency | `sync`, `task`, `channel` | language/runtime design |
| Parallel computing | `parallel`, `distributed` | placement/fusion contracts and local execution experimental; device runtimes planned |
| Networking | `network`, `http`, `url` | network and HTTP/1 client APIs experimental |
| Observability | `log`, `logging`, `trace`, `metrics` | logging sinks experimental |
| Security | `hash`, `crypto`, `tls` | `hash` native baseline implemented; provider policy required for cryptography |
| Data and storage | `pql`, `storage`, `database`, `mysql`, `compression`, `archive` | PQL validation, SQLite/database server, and native MariaDB/MySQL clients experimental |
| Virtualization | `vm`, `container`, `hypervisor` | typed plans and native host discovery experimental |
| Operating systems | `os`, `kernel` | host filesystem metadata plus a hosted kernel laboratory experimental |
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
