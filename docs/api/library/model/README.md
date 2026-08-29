# Model libraries

Stable IDs: `library.model`, `library.model.higgs_audio_v2`, and
`library.model.omnivoice`.

Model packages compose tensor primitives into architectures. Higgs Audio v2 is
a codec/model dependency; OmniVoice is an executable model example. Qwen
decoder layers, attention, RoPE, RMSNorm, SiLU, and Softmax must remain `.sev`
compositions and must not acquire dedicated compiler operation IDs.

The end-to-end relationship is:

`checkpoint + tokenizer/audio input -> model graph -> structural tensor IR ->
specialized CPU/GPU artifact -> samples -> codec/container output`.

Checkpoint identity, tokenizer/sampler policy, tensor execution, codec decode,
and WAV writing are separate validation boundaries. Good-sounding audio is a
valuable use test, not a replacement for deterministic numeric and shape
checks. The fully unranked executable path remains partial.
