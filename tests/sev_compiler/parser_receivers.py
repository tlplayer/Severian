#!/usr/bin/env python3
"""Check source-parser receiver compatibility with the Rust bootstrap."""
from pathlib import Path
import sys

from parser_alias_extensions import main


if __name__ == "__main__":
    sys.exit(main(Path(__file__).with_suffix(".sev")))
