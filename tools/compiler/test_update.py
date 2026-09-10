#!/usr/bin/env python3
"""Exercise update safety against local Git repositories and real launchers."""
import importlib.util
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location('compiler_update', Path(__file__).with_name('update.py'))
update = importlib.util.module_from_spec(spec)
spec.loader.exec_module(update)


class CompilerUpdate(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix='sev-update-test-')
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)

    def git(self, cwd, *args):
        return subprocess.run(['git', '-C', str(cwd), *args], check=True,
                              text=True, capture_output=True).stdout.strip()

    def test_install_preserves_old_commands_and_dispatches_update(self):
        directory = self.root / 'bin'
        directory.mkdir()
        old = directory / 'sev'
        old.write_text('old compiler')
        update.install(directory)
        self.assertEqual((directory / 'sev.before-source-default').read_text(), 'old compiler')
        for name in ('sev', 'sev_rust'):
            command = directory / name
            self.assertEqual(command.resolve(), update.ROOT / 'bin' / name)
            result = subprocess.run([str(command), 'update', '--help'], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn('--local', result.stdout)
        update.install(directory)
        self.assertEqual((directory / 'sev.before-source-default').read_text(), 'old compiler')

    def test_update_fast_forwards_upstream_and_refuses_dirty_checkout(self):
        origin = self.root / 'origin'
        origin.mkdir()
        self.git(origin, 'init', '-b', 'main')
        self.git(origin, 'config', 'user.email', 'test@example.invalid')
        self.git(origin, 'config', 'user.name', 'Compiler test')
        (origin / 'source').write_text('first')
        self.git(origin, 'add', 'source')
        self.git(origin, 'commit', '-m', 'first')
        checkout = self.root / 'checkout'
        self.git(self.root, 'clone', str(origin), str(checkout))
        (origin / 'source').write_text('latest')
        self.git(origin, 'commit', '-am', 'latest')
        real_run = update.run
        def in_checkout(*args, **kwargs):
            return real_run(*args, cwd=checkout, **kwargs)
        with patch.object(update, 'run', in_checkout):
            update.update_checkout()
            self.assertEqual((checkout / 'source').read_text(), 'latest')
            (checkout / 'source').write_text('local work')
            with self.assertRaisesRegex(RuntimeError, 'local changes'):
                update.update_checkout()
            self.assertEqual((checkout / 'source').read_text(), 'local work')

    def test_failed_source_build_restores_working_binary_and_does_not_install(self):
        compiler = self.root / 'sev_compiler/package.pkg/host/dev/bin/sev_compiler'
        compiler.parent.mkdir(parents=True)
        compiler.write_bytes(b'working compiler')
        def fail_build(*args, **kwargs):
            if len(args) > 1 and args[1] == 'build' and args[0] != 'cargo':
                compiler.write_bytes(b'incomplete build')
                raise RuntimeError('build failed')
        with patch.object(update, 'ROOT', self.root), patch.object(update, 'run', fail_build), \
             patch.object(update, 'install') as install, patch('sys.argv', ['update.py', '--local']):
            with self.assertRaisesRegex(RuntimeError, 'build failed'):
                update.main()
            install.assert_not_called()
        self.assertEqual(compiler.read_bytes(), b'working compiler')


if __name__ == '__main__':
    unittest.main(verbosity=2)
