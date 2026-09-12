"""Verify default native artifact destinations and explicit output overrides."""
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]


class ArtifactLayout(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix='sev-artifact-layout-')
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.binaries = {
            'rust': Path(os.environ.get('SEVERIAN_NATIVE_RUST_COMPILER', ROOT / 'package.pkg/release/sev')),
            'source': Path(os.environ.get('SEVERIAN_NATIVE_SOURCE_COMPILER', ROOT / 'sev_compiler/package.pkg/host/dev/bin/sev_compiler')),
        }

    def fixture(self, compiler, name):
        root = self.root / compiler / name
        root.mkdir(parents=True)
        (root / 'subject.sev').write_text('test "layout":\n    assert(true)\n')
        return root

    def invoke(self, compiler, root, *arguments):
        result = subprocess.run([str(self.binaries[compiler]), *map(str, arguments)],
                                cwd=root, env={**os.environ, 'SEVERIAN_SYSROOT': str(ROOT)},
                                capture_output=True, text=True, timeout=60)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        return result

    def test_default_profiles_are_under_debug(self):
        for compiler in self.binaries:
            with self.subTest(compiler=compiler):
                root = self.fixture(compiler, 'profile')
                self.invoke(compiler, root, 'test', 'subject.sev', '--profile')
                reports = list((root / 'package.pkg/debug/profiles').glob('*/report.json'))
                self.assertEqual(len(reports), 1)
                self.assertEqual(json.loads(reports[0].read_text())['exit_code'], 0)
                self.assertFalse((root / 'package.pkg/profiles').exists())

    def test_single_directory_and_package_tests_use_debug(self):
        for compiler in self.binaries:
            for mode in ('single', 'directory', 'package'):
                with self.subTest(compiler=compiler, mode=mode):
                    root = self.fixture(compiler, mode)
                    if mode == 'package':
                        (root / 'package.toml').write_text('[package]\nname = "layout"\n\n'
                                                         '[[bin]]\nname = "layout"\npath = "subject.sev"\n')
                    self.invoke(compiler, root, 'test', *(['subject.sev'] if mode == 'single' else []))
                    tests = root / 'package.pkg/debug/tests'
                    executables = [p for p in tests.rglob('*') if p.is_file() and os.access(p, os.X_OK)]
                    self.assertTrue(executables, str(tests))
                    self.assertTrue(all('bin' in p.relative_to(tests).parts for p in executables))
                    self.assertFalse((root / 'package.pkg/tests').exists())
                    self.assertFalse((root / 'package.pkg/host/dev/tests').exists())

    def test_source_build_run_and_intermediates_have_separate_locations(self):
        for command in ('build', 'run'):
            with self.subTest(command=command):
                root = self.fixture('source', command)
                (root / 'subject.sev').write_text('def main():\n    print("layout")\n')
                self.invoke('source', root, command, 'subject.sev')
                artifacts = root / 'package.pkg'
                if command == 'build':
                    self.assertTrue(os.access(artifacts / 'bin/subject', os.X_OK))
                else:
                    self.assertEqual(len(list((artifacts / 'cache/run').glob('*/bin/subject'))), 1)
                self.assertTrue(list((artifacts / 'cache/native').glob('*/module.ll')))
                self.assertTrue(all(p.is_dir() for p in artifacts.iterdir()))
                self.assertFalse(list((artifacts / 'bin').glob('*.mlir')))

    def test_source_explicit_output_stays_exact(self):
        root = self.fixture('source', 'explicit')
        destination = root / 'chosen' / 'program'
        self.invoke('source', root, 'build', 'subject.sev', '-o', destination)
        self.assertTrue(os.access(destination, os.X_OK))
        self.assertFalse(destination.with_suffix('.ll').exists())


if __name__ == '__main__':
    unittest.main(verbosity=2)
