#!/usr/bin/env python3
"""Stage and publish the prelude source packages in dependency order.

Recipes refer to the existing canonical sources. Staging copies and rewrites
imports, so a release never depends on the producer's compiler checkout.
"""
import argparse
import json
import os
from pathlib import Path
import re
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[2]
RECIPES = ROOT / 'library/prelude'
IMPORT = re.compile(r'(?m)^(import (?:\*|[A-Za-z_][A-Za-z_0-9]*) from )"([^"]+)"')


def stage(destination):
    recipe = json.loads((RECIPES/'packages.json').read_text())
    groups = recipe['packages']
    owners = {}
    for group, entries in groups.items():
        for entry in entries:
            owners[(ROOT/entry['source']).resolve()] = group

    def visit(path, group):
        for match in IMPORT.finditer(path.read_text()):
            child = (path.parent/match[2]).resolve()
            if not child.is_relative_to(ROOT) or not child.is_file():
                raise ValueError(f'prelude import escapes the source tree: {child}')
            if child not in owners:
                owners[child] = group
                visit(child, group)
    for path, group in list(owners.items()): visit(path, group)
    dependencies = {g: set() for g in groups}
    native = {g: set() for g in groups}
    symbols = {g: set() for g in groups}
    for source, group in owners.items():
        symbols[group].update(re.findall(r'@c\(\s*symbol\s*=\s*"([^"]+)"', source.read_text()))
    copied_assets = set()
    def asset(source, group):
        source = source.resolve()
        if (source, group) in copied_assets: return
        if not source.is_relative_to(ROOT) or not source.is_file():
            raise ValueError(f'native prelude asset escapes the source tree: {source}')
        copied_assets.add((source, group))
        target = destination/group/source.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        for header in re.findall(r'^\s*#\s*include\s*"([^"]+)"', source.read_text(), re.M):
            asset(source.parent/header, group)
    for source, group in sorted(owners.items()):
        def rewrite(match):
            child = (source.parent/match[2]).resolve()
            owner = owners[child]
            if owner == group: return match[0]
            dependencies[group].add(owner)
            return match[1] + json.dumps('package:' + owner + '/' + str(child.relative_to(ROOT)))
        target = destination/group/source.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(IMPORT.sub(rewrite, source.read_text()))
        for parent in source.parents:
            manifest = parent/'package.json'
            if manifest.is_file():
                metadata = json.loads(manifest.read_text())
                for relative in metadata.get('xxi', {}).get('c', {}).get('sources', []):
                    provider = (parent/relative).resolve()
                    text = provider.read_text()
                    if any(re.search(r'\b'+re.escape(symbol)+r'\s*\(', text) for symbol in symbols[group]):
                        native[group].add(str(provider.relative_to(ROOT)))
                        asset(provider, group)
                break
            if parent == ROOT: break
    for group, entries in groups.items():
        (destination/group/'lib.sev').write_text(''.join(
            'import ' + name + ' from ' + json.dumps(entry['source']) + (' as '+entry['alias'] if entry.get('alias') else '')+'\n'
            for entry in entries for name in entry.get('names', ['*'])))
    facade = destination/'prelude'
    facade.mkdir(parents=True, exist_ok=True)
    (facade/'lib.sev').write_text(''.join('import * from "package:'+g+'"\n' for g in groups))
    (facade/'prelude.toml').write_text((ROOT/'sev_compiler/universal/prelude.toml').read_text())
    dependencies['prelude'] = set(groups)
    for group in dependencies:
        manifest = {
            'package': {'name': 'sev-prelude' + ('' if group == 'prelude' else '-'+group),
                        'version': recipe['version'], 'edition': '2026',
                        'metadata': {'prelude': {'role': 'root' if group == 'prelude' else 'provider', 'format': 1}}},
            'lib': {'path': 'lib.sev'},
            'dependencies': {g: {'package': 'sev-prelude-'+g, 'path': '../'+g, 'version': recipe['version']}
                             for g in sorted(dependencies[group])},
        }
        if native.get(group):
            manifest['xxi'] = {'c': {'sources': sorted(native[group])}}
        (destination/group/'package.json').write_text(json.dumps(manifest, indent=2)+'\n')
    order = []
    def ordered(group, active=()):
        if group in active: raise ValueError('prelude package cycle: '+str(active+(group,)))
        if group in order: return
        for dep in sorted(dependencies[group]): ordered(dep, active+(group,))
        order.append(group)
    ordered('prelude')
    return [destination/g for g in order]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage-only', action='store_true')
    parser.add_argument('--compiler', type=Path, default=ROOT/'sev_compiler/package.pkg/host/dev/bin/sev_compiler')
    parser.add_argument('--directory', type=Path, default=RECIPES/'package.pkg/staged')
    args = parser.parse_args()
    packages = stage(args.directory.resolve())
    if not args.stage_only:
        prelude = packages[-1]
        resolution = subprocess.run([str(args.compiler), 'resolve', str(prelude), '--sysroot', str(ROOT)],
                                    check=True, text=True, capture_output=True)
        (prelude/'package.lock').write_text(resolution.stdout)
        # Validate the entire graph before reserving any immutable version.
        subprocess.run([str(args.compiler), '__precompile-prelude', str(prelude),
                        str(prelude/'package.pkg/validation'), str(ROOT)], check=True)
    for package in packages:
        print(package, flush=True)
        if not args.stage_only:
            subprocess.run([str(args.compiler), 'publish', str(package), '--sysroot', str(ROOT)], check=True)


if __name__ == '__main__': main()
