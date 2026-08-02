# orchestrator

An executable desired-state control-plane baseline. The package validates
workloads, schedules replicas, reconciles observed state, describes health
policy, service discovery, rolling updates, and persistent-volume claims, and
stores controller history through the shared storage/PQL layer.

Controllers currently produce deterministic actions. Applying those actions to
a host is a separate privileged executor boundary, so unit and native tests do
not mutate the machine running the compiler.
