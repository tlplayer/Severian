#!/usr/bin/env python3
"""Common cold-start and streaming measurement for local GPU model servers."""

from __future__ import annotations

import json
import subprocess
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

CHECKPOINT_REVISION = "14d7620ba47cf51be0b176e14e27e38a34d4ff88"


@dataclass(frozen=True)
class ServerSpec:
    framework: str
    command: list[str]
    health_url: str
    generate_url: str
    request: dict
    response_kind: str


def _request(url: str, body: dict | None = None, timeout: float = 5.0):
    data = None if body is None else json.dumps(body).encode()
    request = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="GET" if data is None else "POST",
    )
    return urllib.request.urlopen(request, timeout=timeout)


def wait_ready(process: subprocess.Popen, url: str, timeout: float) -> int:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            stdout, stderr = process.communicate()
            raise RuntimeError(
                f"server exited during load ({process.returncode})\n{stdout}\n{stderr}"
            )
        try:
            with _request(url) as response:
                if 200 <= response.status < 300:
                    return time.monotonic_ns()
        except (OSError, urllib.error.HTTPError):
            pass
        time.sleep(0.1)
    raise TimeoutError(f"server did not become ready at {url}")


def _json_events(response):
    """Accept SSE (`data:`) and SGLang's newline/NUL-delimited JSON streams."""
    pending = b""
    while True:
        chunk = response.read(1)
        if not chunk:
            break
        if chunk not in (b"\n", b"\0"):
            pending += chunk
            continue
        line, pending = pending.strip(), b""
        if line.startswith(b"data:"):
            line = line[5:].strip()
        if not line or line == b"[DONE]":
            continue
        yield json.loads(line)
    if pending.strip():
        yield json.loads(pending)


def _event_token_ids(event: dict, kind: str, tokenizer) -> list[int]:
    if kind == "sglang":
        metadata = event.get("meta_info", {})
        ids = metadata.get("output_token_ids") or event.get("output_ids")
        if ids is not None:
            return [int(token) for token in ids]
        return tokenizer.encode(event.get("text", ""), add_special_tokens=False)
    choices = event.get("choices", [])
    if not choices:
        return []
    choice = choices[0]
    token_ids = choice.get("token_ids")
    if token_ids is not None:
        return [int(token) for token in token_ids]
    text = choice.get("text", "")
    if not text:
        text = choice.get("delta", {}).get("content", "")
    return tokenizer.encode(text, add_special_tokens=False)


def _gpu_snapshot() -> dict:
    command = ["rocm-smi", "--showproductname", "--showmeminfo", "vram", "--json"]
    result = subprocess.run(command, text=True, capture_output=True, timeout=10)
    if result.returncode != 0:
        raise RuntimeError(f"rocm-smi failed: {result.stderr}")
    devices = json.loads(result.stdout)
    if not devices:
        raise RuntimeError("rocm-smi reported no AMD GPU")
    key = sorted(devices)[0]
    values = devices[key]
    used = next(
        int(value)
        for name, value in values.items()
        if "VRAM Total Used Memory" in name
    )
    model = next(
        str(value)
        for name, value in values.items()
        if "Card Series" in name or "Card Model" in name
    )
    return {"gpu_model": model, "gpu_index": 0, "vram_used_bytes": used}


class _VramMonitor:
    def __init__(self, initial: int):
        self.peak = initial
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        while not self._stop.wait(0.05):
            try:
                self.peak = max(self.peak, _gpu_snapshot()["vram_used_bytes"])
            except (OSError, RuntimeError, subprocess.SubprocessError):
                pass

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, *_):
        self._stop.set()
        self._thread.join()


def benchmark_server(spec: ServerSpec, tokenizer, output_tokens: int, timeout: float) -> dict:
    process_start = time.monotonic_ns()
    before = _gpu_snapshot()
    process = subprocess.Popen(spec.command, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        model_ready = wait_ready(process, spec.health_url, timeout)
        loaded = _gpu_snapshot()
        request_started = time.monotonic_ns()
        generated: list[int] = []
        first_token = None
        with _VramMonitor(loaded["vram_used_bytes"]) as monitor:
            with _request(spec.generate_url, spec.request, timeout=timeout) as response:
                for event in _json_events(response):
                    event_ids = _event_token_ids(event, spec.response_kind, tokenizer)
                    if event_ids and first_token is None:
                        first_token = time.monotonic_ns()
                    # Some APIs send the full prefix on every event; others send deltas.
                    if len(event_ids) >= len(generated) and event_ids[: len(generated)] == generated:
                        generated = event_ids
                    else:
                        generated.extend(event_ids)
        request_ended = time.monotonic_ns()
        if first_token is None:
            raise RuntimeError("server stream contained no generated token")
        if len(generated) < output_tokens:
            raise RuntimeError(f"server returned {len(generated)} tokens, expected {output_tokens}")
        generated = generated[:output_tokens]
        decode_ns = request_ended - first_token
        return {
            "framework": spec.framework,
            "checkpoint_revision": CHECKPOINT_REVISION,
            "gpu_model": loaded["gpu_model"],
            "gpu_index": loaded["gpu_index"],
            "process_start_ns": process_start,
            "model_ready_ns": model_ready,
            "first_token_ns": first_token,
            "load_ns": model_ready - process_start,
            "ttft_ns": first_token - request_started,
            "decode_ns": decode_ns,
            "decode_tokens_per_second": (output_tokens - 1) * 1e9 / decode_ns,
            "gpu_memory_before_bytes": before["vram_used_bytes"],
            "gpu_memory_after_load_bytes": loaded["vram_used_bytes"],
            "peak_vram_bytes": monitor.peak,
            "generated_token_ids": generated,
        }
    finally:
        process.terminate()
        try:
            process.wait(timeout=20)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()


def load_record(inputs: Path, length: str) -> dict:
    return json.loads(inputs.read_text())[length]
