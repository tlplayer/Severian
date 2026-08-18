Covers tasks, channels, async, await, with gpu 

| Construct          | Meaning                                 | Typical target  |
| ------------------ | --------------------------------------- | --------------- |
| `with simd`        | one operation across vector lanes       | CPU AVX/NEON    |
| `with simt`        | many logical threads executing a kernel | GPU             |
| `with parallel`    | independent work across workers         | multicore CPU   |
| `with tasks`       | dynamic independent jobs                | CPU/thread pool |
| `with distributed` | work across processes/nodes             | cluster         |
| `with gpu`         | choose GPU placement/backend            | GPU             |
