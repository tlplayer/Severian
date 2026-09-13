#!/usr/bin/env python3
"""Validate the library workspace against real manifests and dependency paths."""
from pathlib import Path
import json

def read_manifest(path):
    return json.loads("\n".join(line for line in path.read_text().splitlines() if not line.lstrip().startswith("//")))

ROOT = Path(__file__).resolve().parents[2]
workspace = read_manifest(ROOT / 'package.json')['workspace']['members']
assert len(workspace) == len(set(workspace))
actual = {str(p.parent.relative_to(ROOT)) for p in ROOT.rglob('package.json')
          if p.parent != ROOT and 'package.pkg' not in p.parts}
assert set(workspace) == actual, (set(workspace) - actual, actual - set(workspace))
for member in workspace:
    manifest = ROOT / member / 'package.json'
    document = read_manifest(manifest)
    assert document.get('package', {}).get('name', '').strip(), manifest
    for alias, dependency in document.get('dependencies', {}).items():
        if not isinstance(dependency, dict) or 'path' not in dependency:
            continue
        target = manifest.parent / dependency['path'] / 'package.json'
        assert target.is_file(), (manifest, alias, target)
        target_name = read_manifest(target)['package']['name']
        assert target_name == dependency.get('package', alias), (manifest, alias, target_name)
print(f'library workspace: {len(workspace)} manifests and dependency paths passed')
