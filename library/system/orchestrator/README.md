# orchestrator

An executable desired-state control-plane baseline. The package validates
workloads, schedules replicas, reconciles observed state, describes health
policy, service discovery, rolling updates, and persistent-volume claims, and
stores controller history by executing against the `database` package.

Controllers currently produce deterministic actions. Applying those actions to
a host is a separate privileged executor boundary, so unit and native tests do
not mutate the machine running the compiler.

The package also exposes deterministic single-node inference policy for bounded
queue pressure, worker readiness/completion health, and worker recovery labels.
The executable vertical example in
`docs/examples/27-inference-orchestrator` applies that policy to native worker
tasks and tensor-backed model execution.
