# container

Container specifications cover images, processes, Linux namespaces, cgroup
limits, filesystem mounts, virtual networks, and persistent storage. Severian
produces a deterministic create plan while `platform.hostContainerBackend()`
checks whether the host exposes namespace and cgroup facilities.

The current native boundary discovers capabilities; applying a privileged plan
is intentionally separate and requires an authorized host executor.
