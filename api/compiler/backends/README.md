# Backends

API ID: `compiler.backend.cpu_mlir`

CPU MLIR, GPU MLIR, StableHLO/XLA, and machine routes consume typed structural IR. Backend capability is separate from language identity; GPU regions produce launcher calls rather than fake CPU tensor bodies.

```sev
def backend_subject(left: f32, right: f32) -> f32:
    return left + right
```

Current weakness: optimized GPU scheduling and complete launcher execution are still partial.
