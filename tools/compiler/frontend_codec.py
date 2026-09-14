#!/usr/bin/env python3
"""Compatibility entry for the package-owned frontend codec build recipe."""
from pathlib import Path
import runpy

if __name__ == '__main__':
    runpy.run_path(str(Path(__file__).resolve().parents[2] /
                      'sev_compiler/build/frontend_codec.py'), run_name='__main__')
