#!/usr/bin/env bash
set -euo pipefail

extension_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$extension_dir"

if command -v code >/dev/null 2>&1; then
  editor=code
elif command -v codium >/dev/null 2>&1; then
  editor=codium
else
  echo "Neither 'code' nor 'codium' is on PATH." >&2
  exit 1
fi

if ! command -v npx >/dev/null 2>&1; then
  echo "npx is required to package the extension." >&2
  exit 1
fi

rm -f severian-language.vsix
npx --yes @vscode/vsce package --out severian-language.vsix
"$editor" --install-extension severian-language.vsix --force

echo "Installed Severian Language. Reload the editor window, then open a .sev file."
