he intended final usage is simply:

sev llm_server_torture.sev

Then:

curl http://127.0.0.1:8080/health

and:

curl -X POST http://127.0.0.1:8080/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{"prompt":"Explain why the sky is blue in one sentence.","max_tokens":64}'

The file deliberately requires the following to become real:

http.download() with HTTPS, redirects and streaming ~1 GB downloads.
Binary file support, directories, existence and file sizes.
Typed JSON object decoding.
Safetensors loading.
Qwen tokenizer loading and chat-template tokenization.
BF16 tensors; Severian's current documented tensor execution is still primarily f64.
Qwen2 operators: embedding, RMSNorm, RoPE, GQA, SiLU/SwiGLU, batched matmul, masking, KV cache, vocabulary projection and argmax.
model.compileCausalLM(..., "xla", ...) → StableHLO → XLA → PJRT.
Separate compiled prefill and single-token decode execution.
Move semantics for the KV cache.
Persistent compiled executables across requests.
An actual concurrent HTTP server backed by Severian's runtime/netpoll.
JSON responses and request limits.
A warmup inference before the server reports ready.

This is intentionally ahead of the current library. Severian presently lists http as planned, network as only a bind/loopback baseline, and model as experimental.

The important rule for the agent should be:

DO NOT modify llm_server_torture.sev to make the test easier.

DO NOT shell out to Python, PyTorch, Transformers, llama.cpp, vLLM,
ONNX Runtime, curl, or another inference server.

Implement whatever Severian compiler, runtime, library, XLA, PJRT,
tensor, tokenizer, safetensors, filesystem, HTTP, and networking
functionality is missing until:

    sev llm_server_torture.sev

downloads Qwen2.5-0.5B-Instruct, loads it into Severian tensors,
compiles the model through Severian -> StableHLO -> XLA -> PJRT,
performs a real warmup inference, starts the HTTP server, accepts a
prompt, executes the compiled model, and returns generated text.

Do not replace model execution with fixtures, canned strings, mocks,
or hard-coded responses.

That should expose cracks across almost the entire stack instead of allowing individual compiler components to look complete in isolation. The existing sev program.sev workflow is already intended to compile to a temporary native executable and execute it, so this extends that same acceptance philosophy to the ML/runtime stack. 