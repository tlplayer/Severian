"""Budget native compiler commands, including their child processes (Linux).

prlimit enforces an inherited address-space ceiling before exec. A watchdog
also bounds elapsed time and sampled aggregate RSS. GNU time supplies native
CPU and peak-RSS measurements; profiling never invokes an interpreter on Sev.
"""
import math
import os
from pathlib import Path
import signal
import subprocess
import tempfile
import time


DEFAULT_MEMORY = 3_000_000_000
DEFAULT_TIMEOUT = 60.0
MAX_CAPTURE = 64 * 1024 * 1024


def session_processes(session):
    result = {}
    for entry in Path('/proc').iterdir():
        if not entry.name.isdigit():
            continue
        try:
            fields = (entry / 'stat').read_text().rsplit(')', 1)[1].split()
            # stat fields 3 onward: state, ppid, pgrp, session, ... rss.
            if int(fields[3]) == session:
                result[int(entry.name)] = int(fields[21]) * os.sysconf('SC_PAGE_SIZE')
        except (OSError, ValueError, IndexError):
            continue
    return result


def kill_session(session):
    processes = session_processes(session)
    try:
        os.killpg(session, signal.SIGKILL)
    except ProcessLookupError:
        pass
    for pid in processes:
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass


def run(arguments, *, cwd=None, timeout=None, memory_bytes=None, env=None,
        stdout=None, stderr=None, metrics_path=None):
    """Return CompletedProcess plus a .resources dict; limits fail nonzero.

    Environment budgets cap the caller's requested limits. Run serially when
    profiling: per-process AS limits and sampled tree RSS are different metrics.
    """
    configured_time = float(os.environ.get('SEVERIAN_TEST_TIMEOUT_SECONDS', DEFAULT_TIMEOUT))
    configured_memory = int(os.environ.get('SEVERIAN_TEST_MEMORY_BYTES', DEFAULT_MEMORY))
    seconds = min(float(timeout), configured_time) if timeout is not None else configured_time
    memory = min(int(memory_bytes), configured_memory) if memory_bytes is not None else configured_memory
    if not math.isfinite(seconds) or seconds <= 0 or memory <= 0:
        raise ValueError('resource limits must be positive and finite')
    command = list(map(str, arguments))
    if not command:
        raise ValueError('resource guard requires a command')
    with tempfile.TemporaryDirectory(prefix='sev-resources-') as temporary:
        directory = Path(temporary)
        metrics = Path(metrics_path).resolve() if metrics_path else directory / 'resources.txt'
        metrics.unlink(missing_ok=True)
        output_path = Path(stdout).resolve() if stdout else directory / 'stdout'
        error_path = Path(stderr).resolve() if stderr else directory / 'stderr'
        # Keep a native timeout alive even if the Python supervisor is killed.
        wrapped = ['timeout', '--kill-after=2s', f'{seconds:g}s', '/usr/bin/time', '-f',
                   'user_seconds=%U\nsystem_seconds=%S\npeak_rss_kib=%M\nexit_status=%x',
                   '-o', str(metrics), 'prlimit', f'--as={memory}', '--core=0', '--', *command]
        started = time.monotonic()
        peak_tree = 0
        limit = None
        with output_path.open('wb') as output, error_path.open('wb') as errors:
            process = subprocess.Popen(wrapped, cwd=cwd, env=env, stdout=output,
                                       stderr=errors, start_new_session=True)
            try:
                while process.poll() is None:
                    resident = sum(session_processes(process.pid).values())
                    peak_tree = max(peak_tree, resident)
                    if resident > memory:
                        limit = 'memory'
                    elif time.monotonic() - started >= seconds:
                        limit = 'timeout'
                    if limit:
                        kill_session(process.pid)
                        break
                    time.sleep(0.05)
                process.wait()
            finally:
                kill_session(process.pid)
                process.wait()
        if limit is None and process.returncode in (124, 137) and time.monotonic() - started >= seconds:
            limit = 'timeout'
        measured = dict(seconds=round(time.monotonic() - started, 6),
                        timeout_seconds=seconds, memory_bytes=memory,
                        sampled_tree_peak_rss_bytes=peak_tree, limit=limit)
        child_signal = None
        if metrics.exists():
            for line in metrics.read_text().splitlines():
                if line.startswith('Command terminated by signal '):
                    child_signal = int(line.rsplit(' ', 1)[1])
                key, separator, value = line.partition('=')
                if separator and key in ('user_seconds', 'system_seconds', 'peak_rss_kib'):
                    measured[key] = int(value) if key == 'peak_rss_kib' else float(value)
        code = (124 if limit == 'timeout' else 125) if limit else process.returncode
        if not limit and child_signal is not None:
            # GNU time encodes signals as 128+N; retain subprocess semantics.
            code = -child_signal
        if limit:
            with error_path.open('a') as errors:
                errors.write(f'\nresource guard: {limit} limit exceeded '
                             f'({seconds:g}s, {memory} bytes)\n')
        captured = []
        for path, supplied in ((output_path, stdout), (error_path, stderr)):
            if supplied:
                captured.append('')
            else:
                with path.open('rb') as stream:
                    data = stream.read(MAX_CAPTURE + 1)
                if len(data) > MAX_CAPTURE:
                    code = code or 125
                    data = data[:MAX_CAPTURE] + b'\nresource guard: captured output limit exceeded\n'
                captured.append(data.decode(errors='replace'))
        result = subprocess.CompletedProcess(command, code, *captured)
        result.resources = measured
        return result


if __name__ == '__main__':
    import argparse
    import json
    import sys

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--timeout', type=float, default=DEFAULT_TIMEOUT)
    parser.add_argument('--memory-bytes', type=int, default=DEFAULT_MEMORY)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('command', nargs=argparse.REMAINDER)
    arguments = parser.parse_args()
    command = arguments.command
    if command[:1] == ['--']:
        command = command[1:]
    if not command:
        parser.error('a command is required after --')
    # Explicit environment caps still win over command-line budgets.
    os.environ.setdefault('SEVERIAN_TEST_TIMEOUT_SECONDS', str(arguments.timeout))
    os.environ.setdefault('SEVERIAN_TEST_MEMORY_BYTES', str(arguments.memory_bytes))
    arguments.report.parent.mkdir(parents=True, exist_ok=True)
    result = run(command, timeout=arguments.timeout, memory_bytes=arguments.memory_bytes,
                 metrics_path=arguments.report.with_suffix('.resources.txt'))
    arguments.report.write_text(json.dumps(dict(command=command, exit_code=result.returncode,
                                                **result.resources), indent=2) + '\n')
    sys.stdout.write(result.stdout)
    sys.stderr.write(result.stderr)
    sys.exit(result.returncode if result.returncode >= 0 else 128 - result.returncode)
