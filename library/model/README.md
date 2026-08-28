# model

`model` is Severian's typed model-artifact and module-composition package.
Application code imports it directly:

```sev
import model

loaded = model.load("HuggingFaceTB/SmolLM2-135M-Instruct")
local = model.load("models/custom/model.safetensors")
remote = model.load("https://example.test/model.safetensors")
```

One acquisition path handles local files, local directories, direct URLs,
`hf://owner/repository`, and Hugging Face repository names. A checkpoint name
does not select custom inference code. Architecture code is ordinary typed
`.sev` code composed from `tensor` operations; the compiler lowers that graph
through Severian IR and MLIR to the selected host backend.

Artifacts default to `target/models/<model>/<revision>`. Callers can pass an
explicit cache root, and immutable revisions can be supplied with
`model.reference`.

## Typed modules

`Module[T]` is the dtype-generic inference contract used by model composition:

```sev
trait Module[T]:
    def forward(input: tensor.Tensor[T]) -> tensor.Tensor[T]
```

`Linear`, `Affine`, `Embedding`, and `RmsNorm` implement or build on that
contract. The generic functional `linear[T]` remains available for other
checkpoint dtypes. `StateDict` provides lazy, scoped SafeTensor access, so an
architecture names its own parameters instead of teaching the loader about
every possible model family.

## OmniVoice CPU graph

The package contains the checkpoint-backed OmniVoice logits graph: the exact
28-layer Qwen3 configuration used by `k2-fsa/OmniVoice`, grouped-query
attention, Q/K RMS normalization, RoPE, SwiGLU MLP blocks, residuals, and the
eight audio-codebook output heads. The input side includes text lookup,
codebook-offset audio lookup, reduction across codebooks, and text/audio mask
mixing. The generation configuration and shifted iterative unmask schedule are
also represented as typed Severian values.

```sev
import model
import tensor

voice = model.load_omnivoice("/models/OmniVoice")
embeddings = tensor.f32(tensor.ranked(values, [batch, sequence, 1024]))
logits = voice.forward(embeddings) # [batch, 8, sequence, 1025]
assert(voice.close())
```

Construction validates the global tensors and every transformer layer against
the architecture before returning. SafeTensor parameters stay mmap-backed and
are read lazily by CPU kernels, so loading does not duplicate the checkpoint in
an expanded framework tensor format.

This is native Severian execution. It does not invoke Python, PyTorch, ONNX, or
an external inference command. Remote artifact acquisition currently uses the
system download boundary; local inference does not.

The logits graph is not yet the complete text-to-WAV pipeline. Full OmniVoice
synthesis additionally requires the tokenizer input builder, the iterative
masked-codebook update/sampling kernels, and the Higgs Audio V2 codec
encoder/decoder. Those
components remain separate architecture modules rather than being hidden
behind `load_omnivoice`.

## Compiler acceptance ladder

The executable compiler golden is
`docs/examples/08-numerics/16-qwen-voice-golden.sev`; its pinned-asset contract
is the adjacent TOML manifest. Run it with:

```text
sev check docs/examples/08-numerics/16-qwen-voice-golden.sev
sev test docs/examples/08-numerics/16-qwen-voice-golden.sev
sev check docs/examples/08-numerics/16-qwen-voice-golden.sev --emit mlir
```

The golden calls the ranked model APIs from this package rather than carrying
a second Qwen implementation. It covers projection, RMSNorm, RoPE, batched
softmax attention, SwiGLU, decoder residuals, audio-head reshaping, and—when
the pinned SmolLM2 fixture is installed—a real `load[bf16]` StorageView read.
The compiler test also checks that compute artifacts are ranked and that no
Matmul/load symbol encodes dtype or rank.

Ranked APIs preserve dimensions that are already known by the program.
Existing `Tensor[T]` model classes intentionally remain valid: when those
classes erase rank or receive genuinely dynamic storage, they cross the small
runtime Tensor-JIT boundary instead of pretending an opaque pointer is a
builtin MLIR tensor. Completing that executable JIT launcher and the
tokenizer/sampler/codec stages is required before the manifest's WAV quality
acceptance can run.
