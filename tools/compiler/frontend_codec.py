#!/usr/bin/env python3
"""Compatibility launcher for the Severian-owned frontend codec recipe."""
import os
from pathlib import Path
import subprocess

if __name__ == '__main__':
    root = Path(__file__).resolve().parents[2]
    recipe = root / 'sev_compiler/build/frontend_codec.sev'
    compiler = os.environ.get('SEVERIAN_GENERATOR_COMPILER') or str(root / 'bin/sev_rust')
    environment = {**os.environ, 'SEVERIAN_ACTIVE_GENERATOR': str(recipe),
                   'SEVERIAN_SYSROOT': str(root)}
    raise SystemExit(subprocess.call([compiler, str(recipe)], env=environment))
