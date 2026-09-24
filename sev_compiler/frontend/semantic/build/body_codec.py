#!/usr/bin/env python3
"""Generate the body-only archive used by demand-driven semantic loading."""
from pathlib import Path
import runpy

root = Path(__file__).resolve().parents[4]
codec = runpy.run_path(str(root / 'sev_compiler/build/frontend_codec.py'))
codec['generate'](root=root, bodies=True)
