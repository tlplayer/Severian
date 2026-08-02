# Library catalog

The catalog is grouped for discovery, while package imports stay flat.

| Area | Packages | Initial status |
| --- | --- | --- |
| Language foundation | `prelude`, `option`, `result`, `iteration` | language design |
| Core data | `boolean`, `collections`, `text`, `bytes` | `boolean` started |
| Numerics | `math`, `matrix`, `tensor`, `probability`, `statistics`, `random` | `math`, `matrix`, `tensor`, `probability` experimental |
| Machine learning | `models`, `neuralnet`, `autodiff`, `optim` | `models`, `neuralnet` experimental |
| Text processing | `regex`, `unicode`, `format` | `regex` native baseline implemented |
| Data formats | `json`, `csv`, `base64`, `binary` | `json` scalar/list baseline implemented |
| Files and I/O | `io`, `file`, `path` | `io`, `file` native baseline implemented |
| Time and environment | `time`, `environment`, `process` | planned |
| Concurrency | `sync`, `task`, `channel` | language/runtime design |
| Parallel computing | `parallel`, `distributed` | placement/fusion contracts and local execution experimental; device runtimes planned |
| Networking | `network`, `http`, `url` | `network` bind/loopback baseline implemented |
| Observability | `log`, `trace`, `metrics` | `log` native sinks implemented |
| Security | `hash`, `crypto`, `tls` | provider policy required |
| Data and storage | `database`, `compression`, `archive` | planned |
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

Native capabilities use typed `native(\"symbol\") def ...` declarations in the
`platform` package. Standard-library source imports that package instead of
relying on a compiler-invented namespace. Package acceptance requires native
compilation, execution, and exact output validation.
