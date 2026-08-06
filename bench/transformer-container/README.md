# Host versus OCI transformer benchmark

This benchmark runs the exact same compiled Severian transformer executable on
the host and inside a resource-limited OCI container. It reports two different
quantities:

- Median inference time inside the persistent process after 20 warm-up passes.
- Median whole-process time on the host and cold container-run time in OCI.

The benchmark performs 500 encoder passes per sample and rejects
non-deterministic output or any host/container checksum and shape mismatch.
The tiny fixed workload emphasizes language/runtime overhead; it is not a
large-model throughput result.

Run the complete comparison with Podman or Docker:

```sh
python3 bench/transformer-container/run.py
```

Run only the compiled host executable when an OCI runtime is unavailable:

```sh
python3 bench/transformer-container/run.py --host-only
```

The runner invokes `sev build` first, so `model` and `model.neuralnet` artifacts
are built and consumed before the benchmark executable is linked.
