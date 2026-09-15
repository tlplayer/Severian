#!/usr/bin/env python3
import json
from pathlib import Path
import tempfile
import unittest

from prelude_packages import stage, IMPORT


class PreludePackages(unittest.TestCase):
    def test_releases_are_closed_and_dependencies_precede_consumers(self):
        with tempfile.TemporaryDirectory() as temporary:
            packages = stage(Path(temporary))
            published = set()
            for root in packages:
                manifest = json.loads((root/'package.json').read_text())
                dependencies = manifest['dependencies']
                for relative in manifest.get('xxi', {}).get('c', {}).get('sources', []):
                    self.assertTrue((root/relative).is_file())
                for alias, dependency in dependencies.items():
                    self.assertIn(alias, published)
                    self.assertEqual(dependency['package'], 'sev-prelude-'+alias)
                for source in root.rglob('*.sev'):
                    for match in IMPORT.finditer(source.read_text()):
                        locator = match[2]
                        if locator.startswith('package:'):
                            alias, _, relative = locator[8:].partition('/')
                            self.assertIn(alias, dependencies)
                            child = Path(temporary)/alias/(relative or 'lib.sev')
                        else:
                            child = (source.parent/locator).resolve()
                            self.assertTrue(child.is_relative_to(root))
                        self.assertTrue(child.is_file(), child)
                published.add(root.name)
            self.assertEqual(packages[-1].name, 'prelude')
            runtime = Path(temporary)/'runtime'
            self.assertTrue((runtime/'library/core/storage/native/statistics.c').is_file())
            self.assertTrue((runtime/'library/core/memory/native/memory.h').is_file())
            self.assertIn('import print from ', (runtime/'lib.sev').read_text())
            self.assertEqual((Path(temporary)/'math/lib.sev').read_text(),
                             'import round from "library/core/math/src/lib.sev"\n')


if __name__ == '__main__': unittest.main()
