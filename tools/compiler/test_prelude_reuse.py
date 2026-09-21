#!/usr/bin/env python3
"""Exercise published prelude semantics with fresh source-compiler processes."""
import json
import os
from pathlib import Path
import subprocess
import tempfile

from prelude_packages import ROOT, stage


def main():
    compiler = Path(os.environ.get('SEVERIAN_SOURCE_COMPILER', ROOT/'sev_compiler/package.pkg/host/dev/bin/sev_compiler'))
    with tempfile.TemporaryDirectory(prefix='sev-prelude-reuse-') as temporary:
        root = Path(temporary)
        environment = {**os.environ, 'SEVERIAN_REGISTRY': str(root/'registry'), 'SEVERIAN_HOME': str(root/'home'),
                       'SEVERIAN_SYSROOT': str(ROOT)}
        def sev(*arguments, succeeds=True, timing=None):
            env = dict(environment)
            if timing: env['SEVERIAN_TIMINGS'] = str(root/timing)
            result = subprocess.run([str(compiler), *map(str, arguments)], cwd=root, env=env,
                                    text=True, capture_output=True, timeout=300)
            assert (result.returncode == 0) == succeeds, (arguments, result.stdout, result.stderr)
            return result
        first = root/'first.sev'
        first.write_text('assert(20 + 22 == 42)\n')
        sev(first, timing='before.tsv')
        packages = stage(root/'staged')
        for package in packages:
            print('publishing', package.name, flush=True)
            sev('publish', package)
        sev(first, timing='after.tsv')
        second = root/'second.sev'
        second.write_text('''class Counted:
    value: int
    def doubled() -> int:
        return value * 2

def identity[T](value: T) -> T:
    return value

test:
    assert(Counted(21).doubled() == 42)
    assert(identity(42) == 42)
    assert(len([1, 2, 3]) == 3)
    assert("hello".upper() == "HELLO")
    assert(round(1.5) == 2.0)
''')
        sev('test', second, timing='test.tsv')
        assert 'prelude-reused\t' in (root/'test.tsv').read_text()
        assert 'compiling ' not in sev('test', second).stderr
        probe = ROOT/'library/prelude/package.pkg/acceptance/consumer.sev'
        probe.parent.mkdir(parents=True, exist_ok=True)
        probe.write_text('assert(20 + 22 == 42)\n')
        try:
            sev(probe, timing='library.tsv')
            assert 'prelude-reused\t' in (root/'library.tsv').read_text()
        finally:
            probe.unlink()
        consumer = root/'consumer'
        consumer.mkdir()
        consumer_manifest = {
            'package': {'name': 'prelude-consumer', 'version': '0.1.0'},
            'bin': [{'name': 'consumer', 'path': 'main.sev'}],
            'dependencies': {
                'runtime': {'package': 'sev-prelude-runtime', 'version': 'latest'},
                'storage': {'package': 'core.storage', 'path': os.path.relpath(ROOT/'library/core/storage', consumer)},
            },
            'test': {'coverage': False},
            'lint': {'enabled': False},
        }
        (consumer/'package.json').write_text(json.dumps(consumer_manifest))
        (consumer/'main.sev').write_text('import * from "package:runtime/library/core/storage/statistics/src/lib.sev" as stats\n'
                                         'import * from "package:storage/statistics/src/lib.sev" as local_stats\n'
                                         'test:\n    assert(stats.allocation_count() >= u64(0))\n'
                                         '    assert(local_stats.allocation_count() >= u64(0))\n')
        sev('test', consumer)
        consumer_manifest['prelude'] = {'exclude': ['round']}
        (consumer/'package.json').write_text(json.dumps(consumer_manifest))
        (consumer/'main.sev').write_text('def round() -> int:\n    return 42\n'
                                         'test:\n    assert(round() == 42)\n')
        sev('test', consumer/'main.sev')
        for name, diagnostic in [('missing', 'unknown prelude exclusion'), ('string', 'requires extracting compiler intrinsic')]:
            consumer_manifest['prelude'] = {'exclude': [name]}
            (consumer/'package.json').write_text(json.dumps(consumer_manifest))
            result = sev('check', consumer/'main.sev', succeeds=False)
            assert diagnostic in result.stderr, result.stderr
        rejected = root/'reserved.sev'
        rejected.write_text('def max() -> int:\n    return 42\n')
        assert 'reserved prelude function' in sev('check', rejected, succeeds=False).stderr
        manifest = packages[-1]/'package.json'
        document = json.loads(manifest.read_text())
        document['package']['version'] = '0.1.1'
        manifest.write_text(json.dumps(document, indent=2)+'\n')
        sev('publish', packages[-1])
        assert 'compiling ' in sev(first).stderr
        inputs = sorted((root/'package.pkg/build').glob('*/inputs'), key=lambda p: p.stat().st_mtime_ns)
        assert 'sev-prelude/0.1.1/' in inputs[-1].read_text()
        print('source prelude timings:\n'+(root/'before.tsv').read_text(), flush=True)
        print('published prelude timings:\n'+(root/'after.tsv').read_text(), flush=True)
        print('published prelude test timings:\n'+(root/'test.tsv').read_text(), flush=True)
    print('published prelude reuse, generic/class isolation, reservations, and latest selection: passed')


if __name__ == '__main__': main()
