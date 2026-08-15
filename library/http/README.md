# HTTP

`http` provides HTTP/1.1 over native TCP and certificate-verified TLS. It
supports DNS hostnames, IPv4 and IPv6, response-status validation, absolute and
same-origin relative redirects, `Content-Length`, chunked transfer encoding,
and bodies delimited by connection close.

`get`, `post`, `put`, and `delete` return response text for compatibility.
Large bodies should use `download`, which transfers 64 KiB chunks directly to
an atomic destination file:

```sev
import http

_downloaded = http.download(
    "https://huggingface.co/openai-community/gpt2/resolve/main/config.json",
    "models/gpt2-config.json",
)
```

Downloads follow at most ten redirects, reject HTTPS-to-HTTP downgrades, reject
non-success final status codes, verify TLS certificates and hostnames, and
remove their temporary output on failure. The runtime never invokes `curl` or
another process.
