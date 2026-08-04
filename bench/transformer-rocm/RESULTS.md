# Measured transformer ROCm result

Measured 2026-08-04 on the installed AMD Navi 32 GPU with 10 measured
iterations after 2 warmups. Severian targeted `gfx1101`; the comparison used
PyTorch `2.9.1+rocm7.2.1` with HIP `7.2.53211`.

The complete encoder and one reverse-mode/SGD step passed against PyTorch:
forward maximum absolute error was `2.72e-14`, and the updated FFN weight error
was `3.61e-16`.

| Warm operation | Severian | PyTorch | Severian / PyTorch |
| --- | ---: | ---: | ---: |
| Encoder inference | 13.892 ms | 0.383 ms | 36.28x |
| Forward + reverse-mode autodiff + SGD | 93.658 ms | 0.728 ms | 128.63x |

Severian's figure is the per-step mean from its in-process interval; PyTorch's
figure is the median of individually synchronized samples. This tiny graph is
dominated by launch overhead. Severian currently launches one block per output,
uses one thread per block, creates and synchronizes a stream for every tensor
operation, and allocates managed memory without reuse. The reusable autodiff
traversal also materializes gradient tensors as separate launches. The result
establishes a real executable baseline, but it is not yet a competitive GPU
kernel schedule.
