# harness

`harness` combines a loaded `model.LoadedModel`, prompt options, and the
canonical `data.Data` context into an `InferencePlan`.

```sev
import harness
import model

loaded = model.load("owner/model")
service = harness.wrap(loaded, "Answer from the supplied records.")
request = harness.plan(service, "Who owns the compiler?")
```

Placement and backend begin as `auto`. Installed compiler components choose
the actual CPU, SIMD, GPU, StableHLO/XLA, or other execution route. Runtime
providers implement formats or architecture families through
`harness.Runtime`; they do not implement individual checkpoint IDs.

`harness.with_data` accepts `data.Data` directly. `harness.with_storage` reads
the same type from a `storage.Connection`, so retrieval does not introduce a
second dataframe representation.
