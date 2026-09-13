#!/usr/bin/env python3
"""JSON package publication, resolution and immutable reuse through the CLI."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[4]
COMPILER = Path(os.environ.get('SEVERIAN_SOURCE_COMPILER', ROOT / 'sev_compiler/package.pkg/host/dev/bin/sev_compiler'))


def snapshot(root):
    return {str(path.relative_to(root)): (hashlib.sha256(path.read_bytes()).hexdigest(), path.stat().st_mtime_ns)
            for path in root.rglob('*') if path.is_file()}


with tempfile.TemporaryDirectory(prefix='sev-json-publication-') as temporary:
    root = Path(temporary)
    env = {**os.environ, 'SEVERIAN_REGISTRY': str(root / 'registry'),
           'SEVERIAN_HOME': str(root / 'home'), 'SEVERIAN_SYSROOT': str(ROOT)}

    def sev(*arguments, cwd=root):
        result = subprocess.run([str(COMPILER), *map(str, arguments)], cwd=cwd,
                                env=env, capture_output=True, text=True, timeout=300)
        assert result.returncode == 0, (arguments, result.stdout, result.stderr)
        return result.stdout.strip()

    producer = root / 'answer'
    producer.mkdir()
    (producer / 'package.json').write_text('// provider\n' + json.dumps({
        'package': {'name': 'answer', 'version': '0.1.0', 'export': ['answer']},
        'lib': {'path': 'lib.sev'},
    }))
    (producer / 'lib.sev').write_text('def answer() -> int:\n    return 42\n')
    archive = Path(sev('publish', '--local', cwd=producer).splitlines()[-1])
    release = archive.parent
    assert archive.read_bytes().startswith(b'SEVPKG')
    assert json.loads((release / 'metadata/package.json').read_text())['package']['name'] == 'answer'
    json.loads((release / 'metadata/package.lock').read_text())
    json.loads((release / 'metadata/publication.json').read_text())
    json.loads((root / 'registry/answer/0.1.0/index.json').read_text())
    before = snapshot(release)
    assert Path(sev('publish', '--local', cwd=producer).splitlines()[-1]) == archive
    assert snapshot(release) == before
    sev('new', 'consumer')
    consumer = root / 'consumer'
    sev('add', 'answer@0.1.0', cwd=consumer)
    (consumer / 'src/main.sev').write_text('import answer\ndef main():\n    print(answer.answer())\n')
    assert sev('run', cwd=consumer).splitlines()[-1] == '42'
    assert snapshot(release) == before
    json.loads((consumer / 'package.json').read_text())
    json.loads((consumer / 'package.lock').read_text())
print('JSON package publication and immutable consumption: passed')
