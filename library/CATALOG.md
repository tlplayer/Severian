# Library catalog

The catalog is grouped for discovery, while package imports stay flat.

| Area | Packages | Initial status |
| --- | --- | --- |
| Language foundation | `core`, `prelude`, `option`, `result`, `iteration` | `core` experimental |
| Core data | `collections.traits`, `list`, `set`, `map`, `deque`, `queue`, `heap`, `string`, `boolean`, `bits`, `bytes`, `data` | traits and list experimental; set, map, deque, queue, and heap planned |
| Numerics | `math`, `tensor`, `statistics`, `random` | `math`, `tensor`, and `random` experimental |
| Visualization | `graphics`, `plot` | deterministic headless canvas and charts experimental; windows and GPU backends planned |
| Machine learning | `model`, `model.neuralnet`, `autodiff`, `optimization` | `tensor` is the canonical container; `model` owns the public machine-learning hierarchy |
| Text processing | `regex`, `unicode`, `format` | `regex` native baseline implemented |
| Data formats | `json`, `csv`, `yaml`, `base64`, `binary` | format packages own codecs and documents; `file.read()` provides extension dispatch |
| Files and I/O | `io`, `file`, `path`, `os` | typed contents in `file`; namespace operations and metadata in `os` |
| Foreign interfaces | `ffi` | stable C ABI views, handles, and output parameters experimental |
| Time and environment | `time`, `environment`, `process` | clock, environment, and process APIs experimental |
| Concurrency | `sync`, `task`, `channel` | language/runtime design |
| Parallel computing | `parallel`, `distributed` | placement/fusion contracts and local execution experimental; device runtimes planned |
| Networking | `network`, `tls`, `http`, `url` | dual-stack byte streams, verified TLS, and streaming HTTP/1.1 clients experimental |
| Observability | `log`, `trace`, `metrics` | logging sinks experimental |
| Security | `hash`, `crypto` | `hash` native baseline implemented; general cryptography provider policy remains planned |
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

Package names use complete words. Acronyms remain acceptable when they are the
established name of a domain rather than a shortened ordinary word.

Foreign capabilities use typed `@c(symbol = "...")` declarations. Compiler
operations instead use `@mlir`, `@xla`, or a policy such as
`@compile(mlir, stablehlo, xla)`; those declarations never enter ABI
resolution. ABI vocabulary comes from `ffi`; each domain package owns its
declarations and providers. Package acceptance requires native compilation,
execution, and exact output validation.
