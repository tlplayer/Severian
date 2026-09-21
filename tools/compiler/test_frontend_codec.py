"""The frontend codec is reproducible, disposable package build output."""
from pathlib import Path
import re
import runpy
import shutil
import subprocess
import sys
import tempfile
import unittest
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
RECIPE = ROOT / 'sev_compiler/build/frontend_codec.py'
codec = SimpleNamespace(**runpy.run_path(str(RECIPE)))


class FrontendCodec(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix='sev-codec-')
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        sources = {RECIPE}
        for entry in ['universal/src/lib.sev', 'frontend/lexer/src/lib.sev']:
            sources.update(codec.exported_sources(ROOT / 'sev_compiler' / entry))
        sources.update(ROOT / 'sev_compiler' / entry for entry in [
            'universal/type/source/source.sev', 'frontend/semantic/src/callable.sev',
            'frontend/semantic/src/definitions.sev'])
        for source in sources:
            destination = self.root / source.relative_to(ROOT)
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, destination)
        self.recipe = self.root / RECIPE.relative_to(ROOT)
        self.entry = self.root / 'sev_compiler/boundaries/driver/package.pkg/build/frontend_archive.sev'

    def generate(self):
        subprocess.run([sys.executable, self.recipe], cwd='/', check=True)
        return self.entry.parent / re.search(r'"([^"]+)"', self.entry.read_text())[1]

    def test_clean_warm_repair_and_schema(self):
        output = self.generate()
        self.assertEqual(output.parent.name, 'staging')
        self.assertRegex(output.parent.parent.name, r'^[0-9a-f]{64}$')
        files = [self.entry, output, *output.with_suffix('').glob('*.sev')]
        before = {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in files}
        self.assertEqual(self.generate(), output)
        self.assertEqual(before, {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in files})
        files[-1].write_bytes(b'\xffcorrupt output')
        files[-2].unlink()
        self.generate()
        self.assertEqual({p: data[0] for p, data in before.items()},
                         {p: p.read_bytes() for p in files})
        shutil.rmtree(self.entry.parent)
        self.assertEqual(self.generate(), output)
        # Every generated relative import resolves, including semantic models.
        for path in files:
            for locator in re.findall(r'^import \* from "([^"]+)"', path.read_text(), re.M):
                target = (self.root / 'sev_compiler/frontend/semantic' / locator[len('package:semantic/'):]
                          if locator.startswith('package:semantic/') else path.parent / locator)
                self.assertTrue(target.is_file(), (path, locator))

    def test_model_and_recipe_changes_select_new_builds(self):
        original = self.generate()
        source = self.root / 'sev_compiler/frontend/source/source.sev'
        source.write_text(source.read_text().replace('class SourceId:',
                                                    'class SourceId:\n    generation: u32 = 0'))
        changed = self.generate()
        self.assertNotEqual(original, changed)
        schema = lambda path: re.search(r'return "([0-9a-f]{64})"', path.read_text())[1]
        self.assertNotEqual(schema(original), schema(changed))
        self.recipe.write_text(self.recipe.read_text() + '\n# Recipe revision.\n')
        revised = self.generate()
        self.assertNotEqual(changed, revised)
        self.assertEqual(schema(changed), schema(revised))

    def test_concurrent_generators_publish_complete_output(self):
        children = [subprocess.Popen([sys.executable, self.recipe]) for _ in range(4)]
        for child in children:
            self.assertEqual(child.wait(timeout=30), 0)
        output = self.generate()
        self.assertTrue(list(output.with_suffix('').glob('part_*.sev')))
        self.assertFalse((self.root / 'sev_compiler/boundaries').exists())


if __name__ == '__main__':
    unittest.main()
