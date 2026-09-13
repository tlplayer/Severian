#!/usr/bin/env python3
"""Real-process cache contracts: warm invocations must never enter the compiler."""
import hashlib
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[3]
COMPILER = Path(os.environ.get('SEVERIAN_SOURCE_COMPILER', ROOT / 'sev_compiler/package.pkg/host/dev/bin/sev_compiler'))


class Incremental(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix='sev-incremental-')
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.env = {**os.environ, 'SEVERIAN_SYSROOT': str(ROOT),
                    'SEVERIAN_HOME': str(self.root / 'home')}
        self.env.pop('SEVERIAN_REGISTRY', None)
        self.counter = 0

    def invoke(self, *arguments, cwd=None, warm=False, succeeds=True):
        self.counter += 1
        timings = self.root / f'timings-{self.counter}'
        result = subprocess.run([str(COMPILER), *map(str, arguments)], cwd=cwd or self.root,
                                env={**self.env, 'SEVERIAN_TIMINGS': str(timings)},
                                text=True, capture_output=True, timeout=240)
        self.assertEqual(result.returncode == 0, succeeds, result.stdout + result.stderr)
        if warm:
            self.assertFalse(list(self.root.glob(timings.name + '*')),
                             'warm invocation entered the compiler/prelude pipeline')
        return result

    def package(self, name, library=False):
        root = self.root / name
        (root / 'src').mkdir(parents=True)
        target = '[lib]\npath = "src/lib.sev"' if library else f'[[bin]]\nname = "{name}"\npath = "src/main.sev"'
        (root / 'package.toml').write_text(f'[package]\nname = "{name}"\nversion = "0.1.0"\n{target}\n')
        (root / 'src' / ('lib.sev' if library else 'main.sev')).write_text(
            'def answer() -> int:\n    return 42\n' if library else 'def main():\n    print(42)\n')
        return root

    @staticmethod
    def records(root):
        return {str(p.relative_to(root)): (hashlib.sha256(p.read_bytes()).hexdigest(), p.stat().st_mtime_ns)
                for p in root.rglob('*') if p.is_file() and p.name != 'lock'}

    def test_standalone_reuse_restore_invalidation_and_corruption(self):
        source = self.root / 'main.sev'
        source.write_text('def main():\n    print(42)\n')
        self.assertEqual(self.invoke(source).stdout.strip(), '42')
        self.assertEqual(self.invoke(source, warm=True).stdout.strip(), '42')
        self.invoke('build', source, '-o', self.root / 'different', warm=True)
        self.assertTrue(os.access(self.root / 'different', os.X_OK))
        (self.root / 'different').write_text('damaged output')
        self.invoke('build', source, '-o', self.root / 'different', warm=True)
        self.assertEqual(subprocess.check_output([self.root / 'different'], text=True).strip(), '42')
        source.write_text(source.read_text().replace('42', '43'))
        self.assertEqual(self.invoke(source).stdout.strip(), '43')
        self.assertEqual(self.invoke(source, warm=True).stdout.strip(), '43')
        for record in (self.root / 'package.pkg/build/units').glob('*/inputs'):
            record.write_text('broken')
        self.assertEqual(self.invoke(source).stdout.strip(), '43')
        self.invoke(source, warm=True)
        self.invoke('build', source, '--emit', 'mlir')
        self.invoke('build', source, '--emit', 'mlir', warm=True)
        self.invoke('test', source)
        self.invoke('test', source, warm=True)

    def test_dependency_cache_survives_consumer_changes_and_other_consumers(self):
        dependency = self.package('dep', library=True)
        self.invoke('build', cwd=dependency)
        cache = dependency / 'package.pkg/build/units'
        before = self.records(cache)
        self.invoke('build', cwd=dependency, warm=True)
        for name in ['first', 'second']:
            consumer = self.package(name)
            with (consumer / 'package.toml').open('a') as manifest:
                manifest.write('[dependencies]\ndep = { path = "../dep" }\n')
            source = consumer / 'src/main.sev'
            source.write_text('import dep\ndef main():\n    print(dep.answer())\n')
            self.assertEqual(self.invoke('run', cwd=consumer).stdout.strip(), '42')
            self.assertEqual(self.records(cache), before, 'dependency rebuilt for a consumer')
            self.invoke('build', cwd=consumer, warm=True)
            source.write_text(source.read_text().replace('dep.answer()', 'dep.answer() + 1'))
            self.assertEqual(self.invoke('run', cwd=consumer).stdout.strip(), '43')
            self.assertEqual(self.records(cache), before, 'consumer edit rebuilt dependency')
        source = dependency / 'src/lib.sev'
        source.write_text(source.read_text().replace('42', '50'))
        self.assertEqual(self.invoke('run', cwd=consumer).stdout.strip(), '51')
        self.assertNotEqual(self.records(cache), before)
        self.invoke('build', cwd=consumer, warm=True)

    def test_default_registry_and_named_local_publication(self):
        producer = self.package('published', library=True)
        self.invoke('publish', 'published', '--local', cwd=producer)
        release = self.root / 'home/registry/packages/published/0.1.0'
        self.assertTrue(release.is_dir())
        before = self.records(release)
        consumer = self.package('unrelated')
        self.invoke('add', 'published@0.1.0', cwd=consumer, warm=True)
        (consumer / 'src/main.sev').write_text('import published\ndef main():\n    print(published.answer())\n')
        self.assertEqual(self.invoke('run', cwd=consumer).stdout.strip(), '42')
        self.invoke('build', cwd=consumer, warm=True)
        self.assertEqual(self.records(release), before)
        self.assertFalse(list(release.rglob('build')))
        outside = self.root / 'elsewhere'
        outside.mkdir()
        subject = outside / 'main.sev'
        subject.write_text('import published\ndef main():\n    print(published.answer())\n')
        self.assertEqual(self.invoke(subject, cwd=outside).stdout.strip(), '42')
        self.invoke(subject, cwd=outside, warm=True)


if __name__ == '__main__':
    unittest.main(verbosity=2)
