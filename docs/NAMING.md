# Severian naming

Naming communicates semantic role. Snake case is the default, Pascal case marks
a concrete named type, and conventional scientific notation is preserved only
where its spelling is already widely recognized.

| Role | Rule | Examples | Diagnostic |
| --- | --- | --- | --- |
| variables, parameters, fields | `snake_case` | `hidden_state`, `token_count`, `x` | `N001` |
| functions and methods | `snake_case` | `load_model`, `matrix_multiply` | `N002` |
| types, classes, traits, variants | `PascalCase` | `Tensor`, `TensorShape`, `HttpServer` | `N003` |
| constants | `SCREAMING_SNAKE_CASE` | `MAX_TOKENS`, `DEFAULT_PORT` | `N004` |
| packages, modules, import aliases | `snake_case` | `safe_tensor`, `model_runtime` | `N005` |
| decorators | `snake_case` | `@tensor`, `@parallel` | `N006` |

Prefer one clear word when it conveys the meaning. Use `snake_case` when a word
boundary removes ambiguity. Short conventional names remain valid for
coordinates, indices, and generic types: `x`, `y`, `z`, `i`, `j`, `k`, `T`,
`K`, and `V`.

There are no coordinate-accessor exceptions. Fixed fields use ordinary member
access, while a variable selects a field through the dynamic object API:

```sev
point.x
point.x = 10

axis = "x"
value = point.get(axis)
point.set(axis, 10)
```

`getX`, `setX`, and other camelCase spellings may remain callable for external
compatibility, but `sev lint` reports them in Severian source. `get` and `set`
are not reflection escape hatches: field names must be strings, `set` requires
a changeable receiver, and compile-time-known fields retain their declared
types.

Named scientific constructs use their canonical spellings: `ReLU`, `GELU`,
`SiLU`, `LSTM`, `GRU`, `RMSNorm`, `LayerNorm`, `Softmax`, and `Conv2D`.
Functional forms remain lowercase: `relu`, `gelu`, `softmax`, and
`cross_entropy`.

The technical registry is deliberately small: `BERT`, `GPT`, `CUDA`, `ROCm`,
`MLIR`, `XLA`, `StableHLO`, and `PJRT`. PascalCase types use readable title-case
acronyms, such as `HttpServer`. In ordinary snake-case names, one leading
acronym may fuse with the following semantic word, while adjacent acronym
concepts require explicit boundaries:

```text
HTTPServer    -> httpserver
HTTPTPSServer -> http_tps_server
XLAGPUClient  -> xla_gpu_client
```

Ordinary words are not clipped: use `statement`, `expression`, `platform`, and
`configuration` instead of arbitrary abbreviations. The package is `system`,
not `sys`; any future implementation-block syntax is reserved as `implement`,
never `impl`. Conditional chains use `elif`; legacy `else if` and
`else <condition>:` spellings are compatibility forms covered by `N007` during
their migration windows. The sole package manifest name is `package.toml`.
