#!/usr/bin/env python3
"""Native acceptance: init, build, test, publish, add, import, relocate, reuse."""
import hashlib
import os
from pathlib import Path
import statistics
import subprocess
import tempfile
import time
import unittest

ROOT = Path(__file__).resolve().parents[3]
COMPILER = Path(os.environ.get('SEVERIAN_SOURCE_COMPILER', ROOT / 'bin/sev')).resolve()


class HelloWorldFlow(unittest.TestCase):
    def test_init_preserves_files_and_publication_requires_source(self):
        with tempfile.TemporaryDirectory(prefix='sev-package-contract-') as temporary:
            root = Path(temporary)
            env = {**os.environ, 'SEVERIAN_SYSROOT': str(ROOT),
                   'SEVERIAN_HOME': str(root / 'home')}
            env.pop('SEVERIAN_REGISTRY', None)

            def invoke(*args, cwd=root):
                return subprocess.run([str(COMPILER), *map(str, args)], cwd=cwd,
                                      env=env, text=True, capture_output=True, timeout=15)

            occupied = root / 'occupied'
            occupied.mkdir()
            marker = occupied / 'keep.txt'
            marker.write_text('authored content')
            rejected = invoke('init', cwd=occupied)
            self.assertNotEqual(rejected.returncode, 0)
            self.assertIn('empty directory', rejected.stderr)
            self.assertEqual(marker.read_text(), 'authored content')
            self.assertFalse((occupied / 'package.toml').exists())

            producer = root / 'source_required'
            created = invoke('init', producer)
            self.assertEqual(created.returncode, 0, created.stderr)
            manifest = producer / 'package.toml'
            with manifest.open('a') as stream:
                stream.write('\n[publish]\ninclude-source = false\n')
            original = manifest.read_bytes()
            rejected = invoke('publish', '--local', cwd=producer)
            self.assertNotEqual(rejected.returncode, 0)
            self.assertIn('must include source', rejected.stderr)
            self.assertNotIn('compiling ', rejected.stderr)
            self.assertEqual(manifest.read_bytes(), original)

    def test_published_function_survives_relocation_and_reuses_builds(self):
        with tempfile.TemporaryDirectory(prefix='sev-hello-world-') as temporary:
            root = Path(temporary)
            env = {**os.environ, 'SEVERIAN_SYSROOT': str(ROOT),
                   'SEVERIAN_HOME': str(root / 'home')}
            env.pop('SEVERIAN_REGISTRY', None)
            invocation = 0

            def invoke(*args, cwd=root, warm=False):
                nonlocal invocation
                invocation += 1
                timing = root / f'timing-{invocation}'
                start = time.monotonic()
                result = subprocess.run([str(COMPILER), *map(str, args)], cwd=cwd,
                                        env={**env, 'SEVERIAN_TIMINGS': str(timing)},
                                        text=True, capture_output=True, timeout=120)
                elapsed = time.monotonic() - start
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                print(f'{cwd.name}: sev {" ".join(map(str, args))} ({elapsed:.3f}s)', flush=True)
                if warm:
                    self.assertNotIn('compiling ', result.stderr)
                    self.assertFalse(list(root.glob(timing.name + '*')),
                                     'reused build entered the compiler pipeline')
                return elapsed, result

            # 1. Initialize a package with an importable library target.
            producer = root / 'hello_world'
            invoke('init', producer, '--lib')
            source = producer / 'src/lib.sev'
            source.write_text('def hello() -> string:\n    return "hello world"\n\n'
                              'test "hello returns its greeting":\n'
                              '    assert(hello() == "hello world")\n')
            # 2. Build and execute producer tests.
            invoke('build', cwd=producer)
            invoke('test', cwd=producer)
            invoke('build', cwd=producer, warm=True)
            invoke('test', cwd=producer, warm=True)
            # 3. Publish source and completed artifacts to the local registry.
            invoke('publish', '--local', cwd=producer)
            registry = root / 'home/packages/registry/hello_world/0.1.0'
            self.assertTrue((registry / 'source/src/lib.sev').is_file())
            self.assertTrue((registry / 'source/package.lock').is_file())
            publication = {str(p.relative_to(registry)): hashlib.sha256(p.read_bytes()).hexdigest()
                           for p in registry.rglob('*') if p.is_file()}
            for directory in ('build', 'cache', 'debug'):
                self.assertFalse((registry / directory).exists())
            # No consumer can accidentally find the original producer path.
            moved_producer = root / 'moved-producer'
            producer.rename(moved_producer)
            invoke('build', cwd=moved_producer, warm=True)
            # 4. Initialize a different package, including init in an empty cwd.
            consumer = root / 'consumer'
            consumer.mkdir()
            invoke('init', cwd=consumer)
            # 5. Add the published package without starting a compiler.
            invoke('add', 'hello_world', cwd=consumer, warm=True)
            # 6–7. Build an actual import and test its result.
            (consumer / 'src/main.sev').write_text(
                'import hello_world\n\ndef main():\n    print(hello_world.hello())\n\n'
                'test "published hello is callable":\n'
                '    assert(hello_world.hello() == "hello world")\n')
            cold, _ = invoke('build', cwd=consumer)
            invoke('test', cwd=consumer)
            _, run = invoke('run', cwd=consumer, warm=True)
            self.assertEqual(run.stdout.strip(), 'hello world')
            # 8. Warm builds must be faster and relocation must preserve reuse.
            warm = [invoke('build', cwd=consumer, warm=True)[0] for _ in range(3)]
            self.assertLess(statistics.median(warm), cold, (cold, warm))
            nested = root / 'unrelated/deep'
            nested.mkdir(parents=True)
            moved_consumer = nested / 'renamed-consumer'
            consumer.rename(moved_consumer)
            invoke('build', cwd=moved_consumer, warm=True)
            invoke('test', cwd=moved_consumer, warm=True)
            _, run = invoke('run', cwd=moved_consumer, warm=True)
            self.assertEqual(run.stdout.strip(), 'hello world')
            # Reuse must still invalidate for real edits after relocation.
            consumer_source = moved_consumer / 'src/main.sev'
            consumer_source.write_text(consumer_source.read_text().replace(
                'print(hello_world.hello())', 'print(hello_world.hello() + " again")'))
            _, run = invoke('run', cwd=moved_consumer)
            self.assertEqual(run.stdout.strip(), 'hello world again')
            invoke('build', cwd=moved_consumer, warm=True)
            self.assertEqual(publication,
                             {str(p.relative_to(registry)): hashlib.sha256(p.read_bytes()).hexdigest()
                              for p in registry.rglob('*') if p.is_file()})
            print(f'cold build {cold:.3f}s; warm median {statistics.median(warm):.3f}s', flush=True)


if __name__ == '__main__':
    unittest.main(verbosity=2)
