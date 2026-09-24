from pathlib import Path
import runpy

root = Path(__file__).resolve().parents[4]
runpy.run_path(str(root / "sev_compiler/build/frontend_codec.py"))["generate"](root, interface=True)
