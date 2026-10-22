"""Initialize source packages without building artifacts or replacing user files."""
from pathlib import Path
import re

from py_compiler.packaging.package import encode


def init_package(location='.', *, name=None, library=False):
    root = Path(location).resolve()
    selected = root.name if name is None else name
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*', selected):
        raise ValueError('package name must contain letters, digits, dots, underscores or hyphens; use --name')
    if root.exists():
        if not root.is_dir():
            raise ValueError('init requires a directory: ' + str(root))
        if any(root.iterdir()):
            raise ValueError('init requires an empty directory: ' + str(root))
    source = 'src/lib.sev' if library else 'src/main.sev'
    manifest = {
        'package': {'name': selected, 'version': '0.1.0', 'edition': '2026'},
        'sources': [source],
        'dependencies': {},
        'lint': {'enabled': True},
    }
    target = {'name': selected, 'path': source}
    manifest['lib' if library else 'bin'] = target if library else [target]
    content = ('def doubled(value: int) -> int:\n    return value * 2\n\n'
               'test "doubles positive and negative values":\n'
               '    assert(doubled(3) == 6)\n    assert(doubled(-2) == -4)\n')
    if not library:
        content += '\ndef main() -> int:\n    return doubled(0)\n'
    root.mkdir(parents=True, exist_ok=True)
    (root / 'src').mkdir()
    files = {source: content, 'package.lock': encode({'version': 1, 'packages': []}),
             'package.json': encode(manifest)}
    for relative, text in files.items():
        with (root / relative).open('x', encoding='utf-8') as stream:
            stream.write(text)
    return {'root': str(root), 'created': list(files)}


import unittest


class InitTests(unittest.TestCase):
    def test_empty_directory_produces_a_frontend_valid_package(self):
        import json
        import tempfile
        from py_compiler.frontend.src.lib import compile_source
        from py_compiler.syntax.recognition import Syntax
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / '16-sorting'
            root.mkdir()
            result = init_package(root)
            manifest = json.loads((root / 'package.json').read_text())
            self.assertEqual(manifest['package']['name'], '16-sorting')
            self.assertEqual(manifest['sources'], ['src/main.sev'])
            source = root / manifest['sources'][0]
            compiled = compile_source('src/main.sev', source.read_text(), Syntax(include_tests=True))
            self.assertFalse(compiled.diagnostics, compiled.diagnostics)
            self.assertTrue(any(body.test for body in compiled.program.bodies))
            self.assertEqual(set(result['created']), {'package.json', 'package.lock', 'src/main.sev'})
            self.assertFalse((root / 'package.pkg').exists())

    def test_existing_files_and_repeated_init_are_preserved(self):
        import tempfile
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            init_package(root, name='existing')
            before = {p.relative_to(root): p.read_bytes() for p in root.rglob('*') if p.is_file()}
            with self.assertRaisesRegex(ValueError, 'empty directory'):
                init_package(root)
            self.assertEqual(before, {p.relative_to(root): p.read_bytes() for p in root.rglob('*') if p.is_file()})
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / '.keep').write_text('keep')
            with self.assertRaisesRegex(ValueError, 'empty directory'):
                init_package(root)
            self.assertEqual((root / '.keep').read_text(), 'keep')
            self.assertFalse((root / 'package.json').exists())

    def test_library_and_invalid_names(self):
        import json
        import tempfile
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / 'nested/library'
            with self.assertRaisesRegex(ValueError, 'package name'):
                init_package(root, name='../bad')
            self.assertFalse(root.exists())
            init_package(root, name='sorting', library=True)
            manifest = json.loads((root / 'package.json').read_text())
            self.assertEqual(manifest['lib']['path'], 'src/lib.sev')
            self.assertNotIn('def main', (root / 'src/lib.sev').read_text())
