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
        self.manifest.setdefault('lint', {}).setdefault('rules', {}).update({'L0011': 'off', 'L0012': 'off'})
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
        cache = self.root / 'package.pkg/cache/quality/analysis.json'
        self.assertFalse(json.loads(cache.read_text())['cache_hit'])
        second = self.run_sev('build', ok=False)
        self.assertTrue(json.loads(cache.read_text())['cache_hit'])
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

    def test_test_quality_golden_diagnostics_stop_before_compilation(self):
        self.manifest['lint'] = {}
        self.write_manifest()
        (self.root / 'main.sev').write_text(
            'def answer() -> int:\n    return 42\n'
            'def main():\n    print("ready")\n'
            'test "placeholder":\n    assert(answer() == 42)\n')
        first = self.run_sev('test', ok=False)
        second = self.run_sev('test', ok=False)
        expected = [
            ('L0005', 1, 1, 1, 4, 'code has no reachable use in this package',
             'Check external uses before removing or exporting the callable.'),
            ('L0010', 1, 1, 1, 4, 'callable only returns a constant',
             'Implement the operation or use a named value; explicitly suppress intentional constant APIs.'),
            ('L0009', 5, 1, 5, 5, 'test has no production behavior check',
             'Assert an observable production result; replace placeholders and checks of constant stubs.'),
        ]
        self.assertEqual(self.diagnostics(first), [
            dict(rule=rule, file='main.sev', line=line, column=column,
                 end_line=end_line, end_column=end_column, severity='error',
                 message=message, measured=1, threshold=0, remediation=remediation)
            for rule, line, column, end_line, end_column, message, remediation in expected
        ])
        self.assertEqual(self.diagnostics(first), self.diagnostics(second))
        self.assertNotIn('compiling ', first.stderr)

    def test_real_production_behavior_passes_and_executes_assertions(self):
        self.manifest['lint'] = {}
        self.write_manifest()
        source = self.root / 'main.sev'
        text = ('def doubled(value: int) -> int:\n    return value * 2\n'
                'def main():\n    print(doubled(3))\n'
                'test "doubled":\n    assert(doubled(4) == 8)\n')
        source.write_text(text)
        self.assertEqual(self.diagnostics(self.run_sev('test')), [])
        # A real implementation mutation must be caught by the same test.
        source.write_text(text.replace('value * 2', 'value * 3'))
        result = self.run_sev('test', ok=False)
        self.assertNotIn('package lint policy failed', result.stderr)

    def test_test_only_cycle_does_not_make_production_reachable(self):
        self.manifest['lint'] = {}
        self.write_manifest()
        (self.root / 'main.sev').write_text(
            'def left(value: int) -> int:\n    return right(value)\n'
            'def right(value: int) -> int:\n    return left(value)\n'
            'def main():\n    print("ready")\n'
            'test "cycle":\n    assert(left(1) == 1)\n')
        findings = self.diagnostics(self.run_sev('test', ok=False))
        self.assertEqual([d['line'] for d in findings if d['rule'] == 'L0005'], [1, 3])

    def test_coverage_golden_path_measures_both_outcomes(self):
        self.manifest['lint'] = {}
        self.manifest['lib'] = {'path': 'main.sev'}
        self.manifest.pop('bin')
        self.write_manifest()
        (self.root / 'main.sev').write_text(
            'def magnitude(value: int) -> int:\n'
            '    if value < 0:\n        return -value\n    return value\n'
            'test "both signs":\n'
            '    assert(magnitude(-3) == 3)\n    assert(magnitude(2) == 2)\n')
        result = self.run_sev('test', '--coverage')
        report = Path(next(line.removeprefix('quality report: ') for line in result.stderr.splitlines() if line.startswith('quality report: ')))
        summary = json.loads(report.read_text())
        self.assertEqual(summary['tests'], 1)
        self.assertEqual(summary['failures'], [])
        self.assertEqual(summary['metrics']['function'], {'covered': 1, 'total': 1, 'threshold': 91})
        for kind in ('line', 'branch', 'condition'):
            metric = summary['metrics'][kind]
            self.assertGreater(metric['total'], 0, kind)
            self.assertEqual(metric['covered'], metric['total'], kind)
        source = self.root / 'main.sev'
        source.write_text(source.read_text().replace('    assert(magnitude(-3) == 3)\n', ''))
        self.assertIn('branch coverage below', self.run_sev('test', '--coverage', ok=False).stderr)

    def test_editor_snapshot_resolves_real_call_and_parameter(self):
        self.manifest['lint'] = {}
        self.write_manifest()
        (self.root / 'main.sev').write_text(
            '# Doubles the supplied value.\n'
            'def doubled(value: int) -> int:\n    result = value * 2\n    return result\n'
            'def main():\n    print(doubled(3))\n')
        output = self.root / 'editor.json'
        self.run_sev('check', '--emit', 'editor', '-o', str(output))
        snapshot = json.loads(output.read_text())
        declarations = [d for d in snapshot['definitions'] if d['source']['path'] == str(self.root / 'main.sev')]
        doubled = next(d for d in declarations if d['name'] == 'doubled')
        self.assertIn('Doubles the supplied value.', doubled['documentation'])
        self.assertIn('value: int', doubled['type'])
        self.assertTrue(any(r['symbol'] == doubled['id'] for r in snapshot['references']))
        result = next(d for d in declarations if d['name'] == 'result')
        self.assertEqual(result['type'], 'int')
        self.assertTrue(any(r['symbol'] == result['id'] for r in snapshot['references']))

    def test_native_debugger_reads_arguments_locals_and_watch_expression(self):
        self.manifest['lint'] = {}
        self.write_manifest()
        source = self.root / 'main.sev'
        source.write_text('def doubled(value: int) -> int:\n    result = value * 2\n    return result\ndef main():\n    print(doubled(3))\n')
        executable = self.run_sev('build').stdout.strip().splitlines()[-1]
        result = subprocess.run(['gdb', '-q', '-batch', executable,
                                 '-ex', f'break {source}:3', '-ex', 'run',
                                 '-ex', 'print value', '-ex', 'print result',
                                 '-ex', 'print result + value', '-ex', 'print result__ownership', '-ex', 'backtrace',
                                 '-ex', 'continue'], capture_output=True, text=True, timeout=60)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn('$1 = 3', result.stdout)
        self.assertIn('$2 = 6', result.stdout)
        self.assertIn('$3 = 9', result.stdout)
        self.assertIn('$4 = 0', result.stdout)
        self.assertIn('doubled', result.stdout)
        self.assertIn('exited normally', result.stdout)

    def test_native_debugger_observes_borrow_and_move_transitions(self):
        self.manifest['lint'] = {}
        self.write_manifest()
        source = self.root / 'main.sev'
        source.write_text('def shifted(value: int) -> int:\n    shared = view value\n'
                          '    temporary = value + 1\n    transferred = move temporary\n'
                          '    return transferred + shared\ndef main():\n    print(shifted(3))\n')
        executable = self.run_sev('build').stdout.strip().splitlines()[-1]
        result = subprocess.run(['gdb', '-q', '-batch', executable,
                                 '-ex', f'break {source}:5', '-ex', 'run',
                                 '-ex', 'print shared__ownership', '-ex', 'print temporary__ownership',
                                 '-ex', 'print transferred__ownership', '-ex', 'print transferred + shared',
                                 '-ex', 'continue'], capture_output=True, text=True, timeout=60)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        for expected in ('$1 = 1', '$2 = 4', '$3 = 0', '$4 = 7', 'exited normally'):
            self.assertIn(expected, result.stdout)

    def test_churn_uses_the_configured_revision_window(self):
        self.manifest['lint'] = {'churn-commits': 1}
        self.manifest['quality'] = {'revision-window': 2}
        self.write_manifest()
        subprocess.run(['git', 'init', '-q', str(self.root)], check=True)
        for value in (1, 2):
            (self.root / 'main.sev').write_text(f'def main():\n    print({value})\n')
            subprocess.run(['git', '-C', str(self.root), 'add', 'main.sev'], check=True)
            subprocess.run(['git', '-C', str(self.root), '-c', 'user.name=Quality Test',
                            '-c', 'user.email=quality@example.invalid', 'commit', '-qm', str(value)], check=True)
        result = self.run_sev('lint')
        findings = [json.loads(line) for line in result.stdout.splitlines()]
        churn = next(item for item in findings if item['rule'] == 'L0014')
        self.assertEqual((churn['measured'], churn['threshold']), (2, 1))
        self.assertEqual(result.stdout, self.run_sev('lint').stdout)

    def test_generated_library_has_documented_behavior_and_executable_tests(self):
        destination = self.root / 'example'
        self.run_sev('new', str(destination), '--lib')
        self.assertEqual(self.diagnostics(self.run_sev('test', str(destination))), [])
        output = self.root / 'api.md'
        self.run_sev('check', str(destination), '--emit', 'docs', '-o', str(output))
        documentation = output.read_text()
        self.assertIn('def doubled(value: int) -> int', documentation)
        self.assertIn('Number to double.', documentation)

    def test_runtime_records_real_allocations_release_and_retention(self):
        driver = self.root / 'runtime.c'
        driver.write_text('''#include <stdint.h>
#include <stddef.h>
void __sev_quality_test_begin(uint64_t);
void __sev_quality_test_end(uint64_t);
void __sev_quality_hit(uint64_t);
void *__sev_memory_allocate(size_t);
void __sev_memory_release(void *);
int main(void) {
    __sev_quality_test_begin(7);
    void *released = __sev_memory_allocate(16);
    __sev_quality_hit(123);
    __sev_memory_release(released);
    __sev_quality_test_end(7);
    __sev_quality_test_begin(8);
    void *retained = __sev_memory_allocate(32);
    __sev_quality_hit(456);
    __sev_quality_test_end(8);
    __sev_memory_release(retained);
    return 0;
}
''')
        binary = self.root / 'runtime'
        subprocess.run(['clang-21', '-std=c11', '-Wall', '-Wextra', '-Werror',
                        '-DSEV_QUALITY_TRACK_ALLOCATIONS', str(driver),
                        str(ROOT / 'sev_compiler/runtime/native/quality.c'),
                        str(ROOT / 'library/core/memory/native/memory.c'), '-o', str(binary)], check=True)
        records = self.root / 'records'
        subprocess.run([str(binary)], env={**os.environ, 'SEV_COVERAGE_FILE': str(records)}, check=True)
        self.assertEqual(records.read_text().splitlines(), [
            'B:7', 'H:123', 'A:1', 'M:16', 'L:0', 'E:7',
            'B:8', 'H:456', 'A:1', 'M:32', 'L:32', 'E:8'])

    def test_relative_declaration_cycle_keeps_both_implementations(self):
        self.manifest['lint'] = {}
        self.write_manifest()
        (self.root / 'main.sev').write_text('import * from "even.sev"\ndef main():\n    print(even(4))\n')
        (self.root / 'even.sev').write_text('import * from "odd.sev"\ndef even(value: int) -> bool:\n    if value == 0:\n        return true\n    return odd(value - 1)\n')
        (self.root / 'odd.sev').write_text('import * from "even.sev"\ndef odd(value: int) -> bool:\n    if value == 0:\n        return false\n    return even(value - 1)\n')
        executable = self.run_sev('build').stdout.strip().splitlines()[-1]
        result = subprocess.run([executable], capture_output=True, text=True, check=True)
        self.assertEqual(result.stdout, 'true\n')


if __name__ == '__main__':
    unittest.main()
