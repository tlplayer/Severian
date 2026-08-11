#!/usr/bin/env bash
set -euo pipefail

extension_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$extension_dir"

if command -v codium >/dev/null 2>&1; then
  editor=codium
  default_extensions_dir="${HOME:?}/.vscode-oss/extensions"
elif command -v code >/dev/null 2>&1; then
  editor=code
  default_extensions_dir="${HOME:?}/.vscode/extensions"
else
  echo "Neither 'code' nor 'codium' is on PATH." >&2
  exit 1
fi
user_extensions_dir="${SEVERIAN_EXTENSIONS_DIR:-$default_extensions_dir}"

if ! command -v python3 >/dev/null 2>&1; then
  echo "python3 is required to read the extension manifest." >&2
  exit 1
fi

read -r publisher extension_name version < <(
  python3 -c 'import json, pathlib; package = json.loads(pathlib.Path("package.json").read_text()); print(package["publisher"], package["name"], package["version"])'
)
install_path="$user_extensions_dir/$publisher.$extension_name-$version"

mkdir -p "$user_extensions_dir"
if [[ -L "$install_path" ]]; then
  unlink "$install_path"
elif [[ -e "$install_path" ]]; then
  echo "Refusing to replace non-link extension path: $install_path" >&2
  exit 1
fi
ln -s "$extension_dir" "$install_path"

echo "Linked Severian Language for $editor: $install_path -> $extension_dir"
echo "Reload the editor window, then open a .sev file."
