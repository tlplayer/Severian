"""Exercise declared recipes through either compiler's package build command."""
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
COMPILER = Path(os.environ.get('SEVERIAN_TEST_COMPILER', ROOT / 'package.pkg/release/sev'))


class BuildGenerators(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix='sev-generators-')
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        (self.root / 'src').mkdir()
        (self.root / 'build').mkdir()
        self.caller = self.root / 'caller'
        self.caller.mkdir()
        (self.root / 'package.toml').write_text('''[package]
name = "generated-example"
version = "0.1.0"
[[bin]]
name = "example"
path = "src/main.sev"
[build]
generators = ["build/generate.py"]
''')
        (self.root / 'src/main.sev').write_text('''import * from "../package.pkg/build/value.sev"
def main():
    print(answer())
''')
        (self.root / 'value.txt').write_text('42')
        self.recipe = self.root / 'build/generate.py'
        self.recipe.write_text('''from pathlib import Path
root = Path(__file__).resolve().parents[1]
output = root / 'package.pkg/build/value.sev'
selected = root / 'package.pkg/value.txt'
if not selected.is_file():
    selected = root / 'value.txt'
text = 'def answer() -> int:\\n    return ' + selected.read_text() + '\\n'
output.parent.mkdir(parents=True, exist_ok=True)
if not output.is_file() or output.read_text() != text:
    output.write_text(text)
''')
        self.generated = self.root / 'package.pkg/build/value.sev'
        self.binary = self.root / 'package.pkg/bin/example'
        self.env = {**os.environ, 'SEVERIAN_SYSROOT': str(ROOT),
                    'SEVERIAN_HOME': str(ROOT),
                    'SEVERIAN_REGISTRY': os.environ.get('SEVERIAN_TEST_REGISTRY', str(self.root / 'registry'))}

    def invoke(self, command='build'):
        args = [str(COMPILER), command, str(self.root), '--bin', 'example']
        if command == 'build':
            args += ['-o', str(self.binary)]
        return subprocess.run(args, cwd=self.caller, env=self.env, text=True,
                              capture_output=True, timeout=240)

    def build(self, expected):
        result = self.invoke()
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(subprocess.check_output([self.binary], text=True).strip(), expected)

    def test_clean_build_warm_reuse_repair_and_input_changes(self):
        self.build('42')
        stamp = self.binary.stat().st_mtime_ns
        self.build('42')
        self.assertEqual(self.binary.stat().st_mtime_ns, stamp)
        self.generated.unlink()
        self.build('42')
        self.assertEqual(self.binary.stat().st_mtime_ns, stamp)
        (self.root / 'value.txt').write_text('43')
        self.build('43')
        self.recipe.write_text(self.recipe.read_text().replace("read_text() +", "read_text().replace('43', '44') +"))
        self.build('44')
        # The package directory scan excludes package.pkg. The generated
        # module must nevertheless be fingerprinted as a compiler input.
        (self.root / 'package.pkg/value.txt').write_text('45')
        self.build('45')

    def test_check_generates_before_source_discovery(self):
        result = self.invoke('check')
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertTrue(self.generated.is_file())

    def test_recipe_failure_does_not_reuse_previous_binary(self):
        self.build('42')
        before = self.binary.read_bytes()
        self.recipe.write_text('raise SystemExit(7)\n')
        result = self.invoke()
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(self.binary.read_bytes(), before)


if __name__ == '__main__':
    unittest.main()
