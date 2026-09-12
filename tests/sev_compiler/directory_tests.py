"""Batch-test discovery through both native compiler executables."""
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]


class DirectoryTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix='sev-directory-tests-')
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.binaries = {
            'rust': Path(os.environ.get('SEVERIAN_NATIVE_RUST_COMPILER', ROOT / 'package.pkg/release/sev')),
            'source': Path(os.environ.get('SEVERIAN_NATIVE_SOURCE_COMPILER', ROOT / 'sev_compiler/package.pkg/host/dev/bin/sev_compiler')),
        }
        blocked = self.root / '.blocked-python'
        blocked.mkdir()
        for name in ('python', 'python3'):
            executable = blocked / name
            executable.write_text('#!/bin/sh\necho unexpected-python >&2\nexit 99\n')
            executable.chmod(0o755)
        self.environment = {**os.environ, 'SEVERIAN_SYSROOT': str(ROOT),
                            'PATH': str(blocked) + os.pathsep + os.environ['PATH']}

    def invoke(self, compiler, *arguments, cwd=None):
        result = subprocess.run([str(self.binaries[compiler]), 'test', *map(str, arguments)],
                                cwd=cwd or self.root, env=self.environment,
                                capture_output=True, text=True, timeout=60)
        self.assertNotIn('unexpected-python', result.stderr)
        return result

    def write(self, name, contents):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(contents)
        return path

    def test_current_directory_recurses_and_skips_generated_files(self):
        self.write('first.sev', 'test "first":\n    assert(true)\n')
        self.write('nested/first.sev', 'test "nested":\n    assert(true)\n')
        self.write('package.pkg/stale.sev', 'this is not valid Sev!\n')
        self.write('notes.txt', 'not a source file')
        for compiler in self.binaries:
            with self.subTest(compiler=compiler):
                result = self.invoke(compiler)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                self.assertIn('first', result.stdout)
                self.assertIn('nested', result.stdout)
                self.assertNotIn('stale.sev', result.stdout + result.stderr)

    def test_explicit_directory_preserves_profile_and_ignores_parent_package(self):
        self.write('package.toml', '[package]\nname = "parent"\n')
        directory = self.write('loose files/one.sev', 'test "one":\n    assert(true)\n').parent
        for compiler in self.binaries:
            with self.subTest(compiler=compiler):
                report = self.root / f'{compiler}-profile'
                result = self.invoke(compiler, directory, '--profile', '--profile-output', report)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                self.assertEqual(json.loads((report / 'report.json').read_text())['exit_code'], 0)

    def test_failure_does_not_skip_later_files(self):
        self.write('a-invalid.sev', 'test "invalid":\n    assert(unknown_batch_symbol)\n')
        self.write('b-failing.sev', 'test "failing":\n    assert(false)\n')
        self.write('z-passing.sev', 'test "last":\n    assert(true)\n')
        for compiler in self.binaries:
            with self.subTest(compiler=compiler):
                result = self.invoke(compiler)
                self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
                self.assertIn('unknown_batch_symbol', result.stdout + result.stderr)
                self.assertIn('failing', result.stdout)
                self.assertTrue('z-passing' in result.stdout or 'last' in result.stdout, result.stdout)

    def test_empty_directory_fails_clearly(self):
        for compiler in self.binaries:
            with self.subTest(compiler=compiler):
                result = self.invoke(compiler)
                self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
                self.assertIn('no Severian test sources', result.stderr)

    def test_source_discovery_skips_hidden_entries_and_symlink_cycles(self):
        self.write('valid.sev', 'test "valid":\n    assert(true)\n')
        self.write('.hidden/invalid.sev', 'not valid source!\n')
        (self.root / 'cycle').symlink_to(self.root, target_is_directory=True)
        result = self.invoke('source')
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn('1 passed; 0 failed; 1 total', result.stdout)

    def test_package_directory_keeps_package_tests(self):
        self.write('package.toml', '[package]\nname = "batch_package"\n\n[[bin]]\n'
                   'name = "batch_package"\npath = "src/main.sev"\n')
        self.write('src/main.sev', 'test "package test":\n    assert(true)\n')
        self.write('unrelated.sev', 'not valid source!\n')
        for compiler in self.binaries:
            with self.subTest(compiler=compiler):
                result = self.invoke(compiler)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                self.assertNotIn('unrelated.sev', result.stdout + result.stderr)


if __name__ == '__main__':
    unittest.main(verbosity=2)
