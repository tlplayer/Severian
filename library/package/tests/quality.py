#!/usr/bin/env python3
"""Exercise package lint policy and JSON5 through the source CLI."""
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[3]
COMPILER = Path(os.environ.get('SEVERIAN_SOURCE_COMPILER', ROOT / 'sev_compiler/package.pkg/host/dev/bin/sev_compiler'))


class QualityPipeline(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix='sev-quality-')
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.manifest = {
            'package': {'name': 'quality', 'version': '0.1.0'},
            'bin': [{'name': 'quality', 'path': 'main.sev'}],
            'diagnostics': {'message-format': 'json'},
            'lint': {'parameters': 1, 'rules': {'L0003': 'error'}},
        }
        self.write_manifest()
        (self.root / 'main.sev').write_text('def helper(a: int, b: int):\n    print(a, b)\ndef main():\n    helper(1, 2)\n')

    def write_manifest(self):
        (self.root / 'package.json').write_text('// package policy\n' + json.dumps(self.manifest))

    def run_sev(self, *arguments, ok=True):
        result = subprocess.run([str(COMPILER), *arguments, '--sysroot', str(ROOT)],
                                cwd=self.root, text=True, capture_output=True, timeout=300)
        if ok:
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        else:
            self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
        return result

    @staticmethod
    def diagnostics(result):
        return [json.loads(line) for line in result.stderr.splitlines() if line.startswith('{')]

    def test_error_policy_is_deterministic_and_prevents_codegen(self):
        first = self.run_sev('build', ok=False)
        second = self.run_sev('build', ok=False)
        self.assertEqual(self.diagnostics(first), self.diagnostics(second))
        finding = next(d for d in self.diagnostics(first) if d['rule'] == 'L0003')
        self.assertEqual((finding['file'], finding['line'], finding['column']), ('main.sev', 1, 1))
        self.assertEqual((finding['severity'], finding['measured'], finding['threshold']), ('error', 2, 1))
        self.assertTrue(finding['remediation'])
        self.assertNotIn('compiling ', first.stderr)
        self.assertFalse(list(self.root.glob('package.pkg/bin/**/*quality')))
        lock = json.loads((self.root / 'package.lock').read_text())
        self.assertEqual(lock['version'], 1)

    def test_warning_build_cache_hit_and_tightened_policy(self):
        self.manifest['lint']['rules']['L0003'] = 'warning'
        self.write_manifest()
        first = self.run_sev('build')
        second = self.run_sev('build')
        self.assertEqual(self.diagnostics(first), self.diagnostics(second))
        self.assertNotIn('compiling ', second.stderr)
        executable = Path(first.stdout.strip().splitlines()[-1])
        result = subprocess.run([str(executable)], capture_output=True, text=True, check=True)
        self.assertEqual(result.stdout, '1 2\n')
        self.manifest['lint']['rules']['L0003'] = 'error'
        self.write_manifest()
        self.assertIn('package lint policy failed', self.run_sev('build', ok=False).stderr)

    def test_disabled_policy_force_flag_and_suppression(self):
        self.manifest['lint']['enabled'] = False
        self.write_manifest()
        self.assertEqual(self.diagnostics(self.run_sev('build')), [])
        self.assertTrue(self.diagnostics(self.run_sev('build', '--lint', ok=False)))
        self.manifest['lint']['enabled'] = True
        self.write_manifest()
        source = self.root / 'main.sev'
        source.write_text('# sev-lint-next-line: allow L0003\n' + source.read_text())
        self.assertFalse(any(d['rule'] == 'L0003' for d in self.diagnostics(self.run_sev('build'))))

    def test_new_options_and_json5_discovery(self):
        self.run_sev('new', 'created')
        text = (self.root / 'created/package.json').read_text()
        generated = json.loads('\n'.join(line for line in text.splitlines() if not line.lstrip().startswith('//')))
        self.assertTrue(generated['lint']['enabled'])
        self.assertEqual(generated['lint']['rules']['L0001'], 'warning')
        self.assertIn('off, info, hint, warning, error', text)
        json.loads((self.root / 'created/package.lock').read_text())
        options = self.run_sev('options').stdout
        self.assertIn('L0003', options)
        self.assertEqual((self.root / 'package.json').read_text(), '// package policy\n' + json.dumps(self.manifest))

    def test_dead_cycles_unused_symbols_and_exclusions(self):
        self.manifest['lint'] = {'exclude': ['generated/']}
        self.write_manifest()
        (self.root / 'main.sev').write_text('def main():\n    print("ok")\ndef dead():\n    other()\ndef other():\n    dead()\n')
        (self.root / 'generated').mkdir()
        (self.root / 'generated/invalid.sev').write_text('"unterminated')
        findings = self.diagnostics(self.run_sev('build'))
        self.assertEqual([d['line'] for d in findings if d['rule'] == 'L0005'], [3, 5])

    def test_unknown_rule_and_invalid_threshold_fail_even_when_disabled(self):
        self.manifest['lint'] = {'enabled': False, 'rules': {'L9999': 'off'}}
        self.write_manifest()
        self.assertIn('unknown lint rule', self.run_sev('build', ok=False).stderr)
        self.manifest['lint'] = {'parameters': -1}
        self.write_manifest()
        self.assertIn('nonnegative integer', self.run_sev('build', ok=False).stderr)


if __name__ == '__main__':
    unittest.main()
