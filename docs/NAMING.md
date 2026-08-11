# Severian naming

Naming communicates semantic role. Snake case is the default, Pascal case marks
a concrete named type, and conventional scientific notation is preserved only
where its spelling is already widely recognized.

| Role | Rule | Examples | Diagnostic |
| --- | --- | --- | --- |
| variables, parameters, fields | `snake_case` | `hidden_state`, `token_count`, `x` | `N001` |
| functions and methods | `snake_case` | `load_model`, `matrix_multiply` | `N002` |
| types, classes, traits, variants | `PascalCase` | `Tensor`, `ModelConfig` | `N003` |
| constants | `UPPER_SNAKE_CASE` | `MAX_TOKENS`, `DEFAULT_PORT` | `N004` |
| packages, modules, import aliases | `snake_case` | `safe_tensor`, `model_runtime` | `N005` |
| decorators | `snake_case` | `@tensor`, `@parallel` | `N006` |

Short conventional names remain valid for coordinates, indices, and generic
types: `x`, `y`, `z`, `i`, `j`, `k`, `T`, `K`, and `V`.

Coordinate accessors have one intentionally narrow exception: `getX`, `getY`,
`getZ`, `setX`, `setY`, and `setZ`. The exception does not extend to names such
as `getHiddenState`, which becomes `get_hidden_state`.

Named scientific constructs use their canonical spellings: `ReLU`, `GELU`,
`SiLU`, `LSTM`, `GRU`, `RMSNorm`, `LayerNorm`, `Softmax`, and `Conv2D`.
Functional forms remain lowercase: `relu`, `gelu`, `softmax`, and
`cross_entropy`.

The technical registry is deliberately small: `BERT`, `GPT`, `CUDA`, `ROCm`,
`MLIR`, `XLA`, `StableHLO`, and `PJRT`. For acronym-derived concrete names, one
leading acronym may fuse with the following word, while adjacent acronym
concepts require explicit boundaries:

```text
HTTPServer       -> httpserver
HTTPRPCServer    -> http_rpc_server
XLAGPUExecutable -> xla_gpu_executable
```

Ordinary words are not clipped. The package is `system`, not `sys`; any future
implementation-block syntax is reserved as `implement`, never `impl`. `elif`,
legacy `else if`, and `Severian.toml` are compatibility spellings covered by
`N007` during their migration windows.
