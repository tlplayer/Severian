# Severian Popcorn benchmarks

This directory is the optimization feedback loop for Severian-generated GPU
kernels. A benchmark run keeps the source, backend selection, raw TTIR, Triton
compiler output, exact Popcorn submission, command, and remote result close
together so performance regressions can be traced back to generated code.

## Boundary

Popcorn requires every submission to be one Python file. Severian's generated
file is transport and ABI glue only:

```text
kernel.sev
    -> sev kernel emit
    -> kernel.ttir
    -> generated submission.py embeds kernel.ttir
    -> triton.compile(kernel.ttir)
    -> compiled PTX or HSACO
    -> Popcorn correctness and timing harness
```

There is no Torch implementation, handwritten Triton kernel, or fallback
calculation inside the submission. `task` types and tensors belong to the
external Popcorn protocol; computation is performed by the Severian-generated
TTIR.

## Prepare and inspect

From the repository root:

```sh
python3 benchmarks/popcorn/run.py vectorsum_v2
```

This builds `sev` and writes ignored experiment artifacts under
`benchmarks/popcorn/vectorsum_v2/build/`:

- `reduction_sum.ttir`: exact compiler output sent to Triton;
- `submission.py`: single-file Popcorn submission;
- `inspection.txt`: backend choice, operation, parameter, and result details;
- `build.json`: hashes, generation time, compile time, target, and commands.

If a Python environment with Triton is available, validate and retain every
lowered stage before using remote GPU time:

```sh
python3 benchmarks/popcorn/run.py vectorsum_v2 \
  --triton-python benchmarks/inference/.venv/bin/python
```

The additional target-specific directory, such as
`build/compiled/cuda-sm_80/`, contains TTIR, TTGIR, LLVM IR, PTX/AMDGCN, the
GPU binary, and `compilation.json`. This compilation does not need a local GPU
when an explicit target is used. Keeping targets separate prevents an older
architecture's binary from being mistaken for the current experiment.

## Run through Popcorn

Install and authenticate the official Popcorn CLI once. It installs either
`popcorn` or `popcorn-cli`:

```sh
popcorn register discord
```

Correctness only:

```sh
python3 benchmarks/popcorn/run.py vectorsum_v2 --mode test --gpu A100
```

Unranked timing run:

```sh
python3 benchmarks/popcorn/run.py vectorsum_v2 --mode benchmark --gpu A100
```

Profiling:

```sh
python3 benchmarks/popcorn/run.py vectorsum_v2 --mode profile --gpu A100
```

`--mode leaderboard` is supported but must be requested explicitly. The
default mode only prepares local artifacts and never consumes remote compute or
changes a leaderboard.

Each remote run creates an ignored timestamped directory such as:

```text
vectorsum_v2/results/20260813T200000Z-A100-benchmark/
    build.json
    command.json
    local_compilation/
        compilation.json
        kernel.ptx
        kernel.cubin
    reduction_sum.ttir
    submission.py
    popcorn.json
```

That snapshot is the evidence for an optimization: the exact generated kernel
and the timing result cannot drift apart. `local_compilation/` records the
locally compiled stages when `--triton-python` was supplied; the remote
Popcorn result remains the authority for the selected competition GPU.

Popcorn's `--benchmark-index` is only meaningful for `--mode profile` on its
special `B200_Brev` target. It does not select one shape from an ordinary
benchmark run, so the harness rejects that misleading combination.

## Adding a competition kernel

Create `benchmarks/popcorn/<problem>/` containing:

- `kernel.sev`: Severian implementation;
- `adapter.py`: Popcorn tuple/output ABI mapping only;
- `benchmark.json`: leaderboard, entry, default GPU, and GPU-to-target mapping.

The adapter may call the generated `launch(...)` function. It must not contain
an alternate kernel, framework calculation, or correctness fallback.

The initial `vectorsum_v2` manifest maps A100, H100, B200, and L4 to concrete
CUDA architectures. Unknown Popcorn GPUs require an explicit Severian target,
for example:

```sh
python3 benchmarks/popcorn/run.py vectorsum_v2 \
  --mode benchmark --gpu NEW_GPU --target cuda:sm_120
```
