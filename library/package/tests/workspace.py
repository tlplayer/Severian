#!/usr/bin/env python3
"""Validate the library workspace against real manifests and dependency paths."""
from pathlib import Path
import tomllib

ROOT = Path(__file__).resolve().parents[2]
workspace = tomllib.loads((ROOT / 'package.toml').read_text())['workspace']['members']
assert len(workspace) == len(set(workspace))
actual = {str(p.parent.relative_to(ROOT)) for p in ROOT.rglob('package.toml')
          if p.parent != ROOT and 'package.pkg' not in p.parts}
assert set(workspace) == actual, (set(workspace) - actual, actual - set(workspace))
for member in workspace:
    manifest = ROOT / member / 'package.toml'
    document = tomllib.loads(manifest.read_text())
    assert document.get('package', {}).get('name', '').strip(), manifest
    for alias, dependency in document.get('dependencies', {}).items():
        if not isinstance(dependency, dict) or 'path' not in dependency:
            continue
        target = manifest.parent / dependency['path'] / 'package.toml'
        assert target.is_file(), (manifest, alias, target)
        target_name = tomllib.loads(target.read_text())['package']['name']
        assert target_name == dependency.get('package', alias), (manifest, alias, target_name)
print(f'library workspace: {len(workspace)} manifests and dependency paths passed')
