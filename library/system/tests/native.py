#!/usr/bin/env python3
"""Exercise the package-owned OS provider independently of either compiler."""
import os
from pathlib import Path
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[1]
with tempfile.TemporaryDirectory(prefix='sev-system-native-') as temporary:
    executable = Path(temporary) / 'host'
    subprocess.run([os.environ.get('CC', 'cc'), '-std=c17', '-Wall', '-Wextra',
                    '-Werror', str(ROOT / 'tests/host.c'),
                    str(ROOT / 'extern/posix/filesystem.c'),
                    str(ROOT / 'extern/posix/process.c'), '-lm', '-o', str(executable)], check=True)
    subprocess.run([executable], check=True, timeout=30)
