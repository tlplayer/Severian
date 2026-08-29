# System libraries

Stable IDs: `library.system*`, `library.file`, and `library.process`.

System packages cover CLI parsing, containers, devices, drivers, environment,
files, I/O, OCI, orchestration, OS queries, paths, platforms, and processes.
`path` is a value-level path abstraction; `file` performs filesystem effects.
`device` describes devices to programs; compiler `TargetSpec` owns compile-time
target decisions. `driver` is an OS/runtime integration package and is not the
Severian compiler driver.

Container and OCI packages describe related but distinct layers: OCI is the
format/runtime contract, while container is the user-facing lifecycle. The
orchestrator coordinates resources and therefore composes process, device,
network, and container effects.

File and process exports are source-checked today. Other system packages are
manifest-complete but require per-symbol export contracts.
