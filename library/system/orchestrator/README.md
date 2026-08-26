# orchestrator

An executable desired-state control-plane baseline. The package validates
workloads, schedules replicas, reconciles observed state, describes health
policy, service discovery, rolling updates, and persistent-volume claims.

Controllers currently produce deterministic actions. Applying those actions to
a host is a separate privileged executor boundary, so unit and native tests do
not mutate the machine running the compiler.

The package also exposes resource-aware services, portable accelerator claims,
CDI allocation, readiness-gated routing, replica health, and restart actions.
The executable vertical example in `docs/labs/02-orchestrator-llm` applies that
policy to three containerized tensor-backed model workers.
