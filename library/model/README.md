# model

`model` is Severian's model-artifact boundary. Application code imports it
directly:

```sev
import model

loaded = model.load("HuggingFaceTB/SmolLM2-135M-Instruct")
local = model.load("models/custom/model.safetensors")
remote = model.load("https://example.test/model.safetensors")
```

One acquisition path handles local files, local directories, direct URLs,
`hf://owner/repository`, and Hugging Face repository names. A checkpoint name
does not select custom inference code. Configuration parsing and architecture
lowering consume the resulting `model.Artifact` after it has been acquired.

Artifacts default to `target/models/<model>/<revision>`. Callers can pass an
explicit cache root, and immutable revisions can be supplied with
`model.reference`.
