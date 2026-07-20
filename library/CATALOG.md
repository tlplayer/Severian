# Library catalog

The catalog is grouped for discovery, while package imports stay flat.

| Area | Packages | Initial status |
| --- | --- | --- |
| Language foundation | `prelude`, `option`, `result`, `iteration` | language design |
| Core data | `boolean`, `collections`, `text`, `bytes` | `boolean` started |
| Numerics | `math`, `probability`, `statistics`, `random` | `math`, `probability` started |
| Text processing | `regex`, `unicode`, `format` | `regex` interface pending |
| Data formats | `json`, `csv`, `base64`, `binary` | `json` interface pending |
| Files and I/O | `io`, `file`, `path` | `file` interface pending |
| Time and environment | `time`, `environment`, `process` | planned |
| Concurrency | `sync`, `task`, `channel` | language/runtime design |
| Networking | `network`, `http`, `url` | `network` interface pending |
| Observability | `log`, `trace`, `metrics` | `log` interface pending |
| Security | `hash`, `crypto`, `tls` | provider policy required |
| Data and storage | `database`, `compression`, `archive` | planned |
| Development | `testing`, `benchmark`, `profile` | language design |

## Admission stages

Packages move through explicit stages:

1. `planned`: the scope and name are reserved in this catalog.
2. `interface-pending`: the package exists, but its typed ABI is not yet
   expressible or connected.
3. `experimental`: callable implementation and Severian tests exist; APIs may
   still change.
4. `stable`: documented behavior, failures, ownership, and compatibility are
   maintained.

The next compiler feature needed by runtime-backed packages is a real typed
foreign/runtime declaration. Until then, hard-coded semantic return types are
compatibility scaffolding, not library implementations.

