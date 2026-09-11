import os
import signal
from pathlib import Path
import sys
import tempfile
import unittest

from resource_guard import run


class ResourceGuard(unittest.TestCase):
    def test_child_signal_is_preserved(self):
        result = run([sys.executable, '-c', 'import os; os.abort()'])
        self.assertEqual(result.returncode, -signal.SIGABRT)

    def test_nested_guard_retains_tighter_inherited_limit(self):
        source = ('from resource_guard import run\n'
                  'result = run(["/bin/true"], memory_bytes=128 * 1024 * 1024)\n'
                  'assert result.resources["memory_bytes"] == 64 * 1024 * 1024\n'
                  'raise SystemExit(result.returncode)\n')
        result = run([sys.executable, '-c', source], cwd=Path(__file__).parent,
                     memory_bytes=64 * 1024 * 1024)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_native_metrics_and_exit_status(self):
        result = run(['/bin/sh', '-c', 'echo hello; exit 7'])
        self.assertEqual(result.returncode, 7)
        self.assertEqual(result.stdout, 'hello\n')
        self.assertGreater(result.resources['peak_rss_kib'], 0)
        self.assertIn('user_seconds', result.resources)

    def test_address_space_limit_is_inherited(self):
        result = run([sys.executable, '-c',
                      'import resource; print(resource.getrlimit(resource.RLIMIT_AS)[0]); '
                      'data = bytearray(128 * 1024 * 1024)'], memory_bytes=64 * 1024 * 1024)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn(str(64 * 1024 * 1024), result.stdout)
        self.assertIn('MemoryError', result.stderr)

    def test_timeout_cleans_up_descendant_process_groups(self):
        with tempfile.TemporaryDirectory() as temporary:
            pid_file = Path(temporary) / 'child.pid'
            source = ('import os, time, pathlib\n'
                      'pid = os.fork()\n'
                      'if pid == 0:\n'
                      ' os.setpgid(0, 0)\n'
                      f' pathlib.Path({str(pid_file)!r}).write_text(str(os.getpid()))\n'
                      ' time.sleep(30)\n'
                      'else:\n'
                      ' time.sleep(30)\n')
            result = run([sys.executable, '-c', source], timeout=0.3)
            self.assertEqual(result.returncode, 124)
            self.assertEqual(result.resources['limit'], 'timeout')
            self.assertLess(result.resources['seconds'], 3)
            self.assertTrue(pid_file.exists())
            child = Path('/proc') / pid_file.read_text() / 'stat'
            if child.exists():
                self.assertEqual(child.read_text().rsplit(')', 1)[1].split()[0], 'Z')

    def test_combined_child_rss_is_bounded(self):
        child = 'import time; data = bytearray(64 * 1024 * 1024); time.sleep(30)'
        source = (f'import subprocess, sys, time\n'
                  f'children = [subprocess.Popen([sys.executable, "-c", {child!r}]) for _ in range(2)]\n'
                  'time.sleep(30)\n')
        result = run([sys.executable, '-c', source], timeout=3,
                     memory_bytes=128 * 1024 * 1024)
        self.assertEqual(result.returncode, 125)
        self.assertEqual(result.resources['limit'], 'memory')
        self.assertGreater(result.resources['sampled_tree_peak_rss_bytes'], 128 * 1024 * 1024)
        self.assertLess(result.resources['seconds'], 3)

    def test_invalid_budgets_fail_before_start(self):
        for limits in ({'timeout': 0}, {'timeout': float('nan')}, {'memory_bytes': -1}):
            with self.subTest(limits=limits), self.assertRaises(ValueError):
                run(['/bin/true'], **limits)


if __name__ == '__main__':
    unittest.main()
