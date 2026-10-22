"""C test coverage from LLVM source locations, with crash-safe hit records."""
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
import re
import struct

from py_compiler.packaging.package import encode, write_atomic


RUNTIME = r'''
#include <stdint.h>
#include <stdlib.h>
#include <unistd.h>
#include <fcntl.h>
#include <pthread.h>
#include <stdatomic.h>
#include <sys/mman.h>
#ifndef SEV_COVERAGE_SLOTS
#define SEV_COVERAGE_SLOTS 65536
#endif
struct counter { _Atomic uint64_t identity, hits; };
static struct counter *counters;
static pthread_once_t once = PTHREAD_ONCE_INIT;
static void initialize(void) {
    const char *path = getenv("SEV_COVERAGE_FILE");
    if (!path) return;
    int output = open(path, O_RDWR | O_CREAT | O_TRUNC | O_CLOEXEC, 0600);
    size_t size = 16 + SEV_COVERAGE_SLOTS * sizeof(struct counter);
    if (output < 0 || ftruncate(output, size)) _exit(91);
    char *mapping = mmap(0, size, PROT_READ | PROT_WRITE, MAP_SHARED, output, 0);
    close(output);
    if (mapping == MAP_FAILED) _exit(92);
    const char magic[8] = {'S','E','V','C','O','V','2',0};
    for (int i = 0; i < 8; ++i) mapping[i] = magic[i];
    *(uint64_t *)(mapping + 8) = SEV_COVERAGE_SLOTS;
    counters = (struct counter *)(mapping + 16);
}
void __sev_coverage_hit(uint64_t value) {
    pthread_once(&once, initialize);
    if (!counters) return;
    // Slot zero is reserved for identity zero; all u64 identities are valid.
    if (!value) { atomic_fetch_add_explicit(&counters[0].hits, 1, memory_order_relaxed); return; }
    size_t slot = 1 + value % (SEV_COVERAGE_SLOTS - 1);
    for (size_t count = 1; count < SEV_COVERAGE_SLOTS; ++count) {
        uint64_t expected = 0;
        if (atomic_compare_exchange_strong_explicit(&counters[slot].identity, &expected,
                value, memory_order_relaxed, memory_order_relaxed) || expected == value) {
            atomic_fetch_add_explicit(&counters[slot].hits, 1, memory_order_relaxed);
            return;
        }
        if (++slot == SEV_COVERAGE_SLOTS) slot = 1;
    }
    _exit(93);
}
'''


def artifact_slot(build_id=None):
    if build_id is None:
        return 'dev'
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*', build_id):
        raise ValueError('invalid published coverage build identity')
    return build_id


def instrument(text, root, points, excluded=(), *, existing_sources_only=False):
    """Instrument local source lines, function entries, and conditional edges."""
    root = Path(root).resolve()
    metadata = dict(re.findall(r'^!(\d+) = (.+)$', text, re.MULTILINE))
    def quoted(field, entry):
        match = re.search(field + r': "((?:\\.|[^"])*)"', entry)
        return re.sub(r'\\([0-9A-Fa-f]{2})', lambda m: chr(int(m[1], 16)), match[1]) if match else ''
    def location(index, seen=()):
        if index in seen:
            return None
        entry = metadata.get(index, '')
        file = re.search(r'file: !(\d+)', entry)
        if file:
            entry_file = metadata.get(file[1], '')
            path = Path(quoted('directory', entry_file)) / quoted('filename', entry_file)
            path = path.resolve()
            if not path.is_relative_to(root) or path.suffix != '.sev' or (existing_sources_only and not path.is_file()):
                return None
            line = re.search(r'\bline: (\d+)', entry)
            column = re.search(r'\bcolumn: (\d+)', entry)
            return (path.relative_to(root).as_posix(), int(line[1]) if line else 0,
                    int(column[1]) if column else 0)
        scope = re.search(r'scope: !(\d+)', entry)
        parent = location(scope[1], (*seen, index)) if scope else None
        if parent:
            line = re.search(r'\bline: (\d+)', entry)
            column = re.search(r'\bcolumn: (\d+)', entry)
            return (parent[0], int(line[1]) if line else parent[1], int(column[1]) if column else 0)
        return None
    def point(kind, loc, function, extra=''):
        payload = {'kind': kind, 'file': loc[0], 'line': loc[1], 'column': loc[2],
                   'function': function, 'outcome': extra}
        identity = int(sha256(encode(payload).encode()).hexdigest()[:16], 16)
        points[identity] = payload
        return identity
    def call(identity):
        return f'  call void @__sev_coverage_hit(i64 {identity})'
    lines, function, entry_loc, entered, previous, serial = [], '', None, False, None, 0
    for line in text.splitlines():
        definition = re.match(r'^define .*?@("[^"]+"|[^ (]+)\(', line)
        if definition:
            function = definition[1].strip('"')
            debug = re.search(r'!dbg !(\d+)', line)
            entry_loc = location(debug[1]) if debug and function not in excluded else None
            entered, previous = False, None
        debug = re.search(r'!dbg !(\d+)', line)
        loc = location(debug[1]) if debug and entry_loc else None
        # PHI instructions must remain first in their block.
        instruction = line.startswith('  ') and not re.search(r' = phi ', line)
        if instruction and entry_loc and not entered:
            lines.append(call(point('function', entry_loc, function)))
            entered = True
        if instruction and loc and loc[1] and loc != previous:
            lines.append(call(point('line', (loc[0], loc[1], 0), function)))
            previous = loc
        branch = re.match(r'  br i1 ([^,]+), label ', line)
        if branch and loc:
            yes = point('branch', loc, function, f'{serial}:true')
            no = point('branch', loc, function, f'{serial}:false')
            lines += [f'  %sev_cov_{serial} = select i1 {branch[1]}, i64 {yes}, i64 {no}',
                      f'  call void @__sev_coverage_hit(i64 %sev_cov_{serial})']
            serial += 1
        lines.append(line)
        if line == '}':
            entry_loc = None
        if re.match(r'^[A-Za-z0-9_.]+:', line):
            previous = None
    return '\n'.join(lines) + '\ndeclare void @__sev_coverage_hit(i64)\n'


def read_hits(path):
    data = Path(path).read_bytes() if Path(path).exists() else b''
    if data.startswith(b'SEVCOV2\0'):
        if len(data) < 16:
            return Counter()
        slots, = struct.unpack_from('=Q', data, 8)
        available = min(slots, (len(data) - 16) // 16)
        return Counter({identity: hits for identity, hits in
                        struct.iter_unpack('=QQ', data[16:16 + available * 16]) if hits})
    return Counter(value[0] for value in struct.iter_unpack('=Q', data[:len(data) // 8 * 8]))


class Coverage:
    def __init__(self, root, build_id=None):
        self.root = Path(root).resolve()
        self.output = self.root / 'package.pkg/debug/test/coverage' / artifact_slot(build_id)
        self.points, self.hits, self.tests, self.hashes = {}, Counter(), [], {}

    def add(self, record, path):
        hits = read_hits(path)
        self.hits.update(hits)
        entry = {'label': record['label'], 'passed': record['passed'],
                 'partial': record.pop('_coverage-partial', not record.get('passed', False)),
                 'hits': {str(k): v for k, v in hits.items() if k in self.points}}
        self.tests.append(entry)
        record['coverage'] = entry
        return set(hits) & self.points.keys()

    def save(self):
        files = {}
        for identity, point in self.points.items():
            name = point['file']
            item = files.setdefault(name, {'source': name, 'source-sha256': self.hashes.get(name),
                                          'points': [], 'lines': {}})
            hits = self.hits[identity]
            item['points'].append({**point, 'id': str(identity), 'hits': hits})
            if point['kind'] == 'line':
                key = str(point['line'])
                item['lines'][key] = item['lines'].get(key, 0) + hits
        summary = {'instrumentation': 'llvm-source-probes-v1', 'files': {},
                   'limitations': ['condition expressions are represented by lowered branch outcomes',
                                   'foreign code and dependency packages are excluded']}
        for name, item in files.items():
            item['covered-lines'] = sum(count > 0 for count in item['lines'].values())
            item['total-lines'] = len(item['lines'])
            for kind in ('function', 'branch'):
                values = [p for p in item['points'] if p['kind'] == kind]
                item[kind] = {'covered': sum(p['hits'] > 0 for p in values), 'total': len(values)}
            summary['files'][name] = {k: v for k, v in item.items() if k not in ('points', 'lines')}
            write_atomic(self.output / 'files' / (name + '.json'), encode(item))
        write_atomic(self.output / 'tests.json', encode(self.tests))
        write_atomic(self.output / 'summary.json', encode(summary))
        return str(self.output / 'summary.json')


import unittest


class CoverageTests(unittest.TestCase):
    def test_profile_coverage_keeps_sanitizer_on_both_executions(self):
        import tempfile
        from py_compiler.packaging.testing import test_package
        with tempfile.TemporaryDirectory(prefix='sev-profile-coverage-') as directory:
            root = Path(directory)
            (root / 'src').mkdir()
            (root / 'package.json').write_text(encode({'package': {'name': 'profile-coverage', 'version': '0.1.0'}}))
            (root / 'src/lib.sev').write_text('test with profile "allocation":\n'
                '    values = tensor[int,(4,)]()\n    values[0] = 42\n    assert(values[0] == 42)\n')
            result = test_package(root, coverage=True, sanitize='address')
            self.assertTrue(result['success'], result)
            output = root / 'package.pkg/debug/test/test'
            self.assertTrue(list(output.glob('asan-baseline-*.instrumented.ll')))
            self.assertIn(b'__asan_init', (output / 'test-0000').read_bytes())
            self.assertIn(b'__asan_init', (output / 'test-0000.coverage-exe').read_bytes())
            self.assertTrue((root / 'package.pkg/debug/test/coverage/dev/summary.json').is_file())

    def test_c_failure_is_persisted_as_a_failing_regression(self):
        import tempfile
        from py_compiler.packaging.testing import test_package
        with tempfile.TemporaryDirectory(prefix='sev-fuzz-failure-') as directory:
            root = Path(directory)
            (root / 'src').mkdir()
            (root / 'package.json').write_text(encode({'package': {'name': 'failure', 'version': '0.1.0'}}))
            path = root / 'src/lib.sev'
            path.write_text('test with fuzz "failure":\n'
                            '    n: int {0 <= n and n <= 2} = 0\n'
                            '    assert(n != 2)\n')
            result = test_package(root, fuzz=True)
            self.assertFalse(result['success'])
            self.assertIn('    test "fuzz regression ', path.read_text())
            self.assertNotIn('throws', path.read_text())
            replay = test_package(root)
            self.assertFalse(replay['success'])
            report = json.loads(Path(replay['report']).read_text())
            self.assertTrue(any(t['label'].startswith('fuzz regression ') and not t['passed']
                                for t in report['tests']))

    def test_c_fuzz_covers_branches_and_appends_replay(self):
        import tempfile
        from py_compiler.packaging.testing import test_package
        with tempfile.TemporaryDirectory(prefix='sev-coverage-') as directory:
            root = Path(directory)
            (root / 'src').mkdir()
            (root / 'package.json').write_text(encode({'package': {'name': 'coverage', 'version': '0.1.0'}}))
            path = root / 'src/lib.sev'
            path.write_text('def sign(n: int) -> int:\n'
                            '    if n < 0:\n        return -1\n    return 1\n'
                            'test with fuzz "sign":\n'
                            '    n: int {-2 <= n and n <= 2} = 0\n'
                            '    result = sign(n)\n'
                            '    assert(result * result == 1)\n'
                            'test with profile "profile sign":\n'
                            '    assert(sign(1) == 1)\n')
            result = test_package(root, fuzz=True)
            self.assertTrue(result['success'], result)
            report = json.loads((root / 'package.pkg/debug/test/coverage/dev/files/src/lib.sev.json').read_text())
            self.assertEqual(report['branch']['covered'], report['branch']['total'])
            self.assertGreaterEqual(report['branch']['covered'], 2)
            self.assertIn('    test "fuzz regression ', path.read_text())
            replay = test_package(root, coverage=True)
            self.assertTrue(replay['success'], replay)
            self.assertLess(replay['tests'], result['tests'])

    def test_slots(self):
        self.assertEqual(artifact_slot(), 'dev')
        self.assertEqual(artifact_slot('published-1'), 'published-1')
        with self.assertRaises(ValueError):
            artifact_slot('../escape')

    def test_lines_and_both_branch_outcomes_exist_before_execution(self):
        source = '''define i32 @choose(i1 %flag) !dbg !1 {
  br i1 %flag, label %yes, label %no, !dbg !3
yes:
  ret i32 1, !dbg !4
no:
  ret i32 0, !dbg !5
}
!1 = distinct !DISubprogram(name: "choose", file: !2, line: 1)
!2 = !DIFile(filename: "sample.sev", directory: "/tmp")
!3 = !DILocation(line: 2, column: 3, scope: !1)
!4 = !DILocation(line: 3, column: 3, scope: !1)
!5 = !DILocation(line: 4, column: 3, scope: !1)
'''
        points = {}
        result = instrument(source, '/tmp', points)
        self.assertIn('select i1 %flag', result)
        self.assertEqual(sum(p['kind'] == 'branch' for p in points.values()), 2)
        self.assertEqual({p['line'] for p in points.values() if p['kind'] == 'line'}, {2, 3, 4})

    def test_virtual_dependency_paths_are_not_local_coverage(self):
        import tempfile
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            (root/'main.sev').write_text('def work() -> int:\n    return 1\n')
            template='''define i64 @work() !dbg !1 {
  ret i64 1, !dbg !3
}
!1 = distinct !DISubprogram(name: "work", file: !2, line: 1)
!2 = !DIFile(filename: "FILENAME", directory: "DIRECTORY")
!3 = !DILocation(line: 2, column: 1, scope: !1)
'''.replace('DIRECTORY',str(root))
            points={}
            instrument(template.replace('FILENAME','core.collections/lib.sev'),root,points,existing_sources_only=True)
            self.assertEqual(points,{})
            instrument(template.replace('FILENAME','main.sev'),root,points,existing_sources_only=True)
            self.assertTrue(points)
            self.assertEqual({p['file'] for p in points.values()},{'main.sev'})


class CoverageCounterTests(unittest.TestCase):
    def test_shared_counters_preserve_repeated_hits_zero_and_u64_identity(self):
        import tempfile
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'hits.bin'
            path.write_bytes(b'SEVCOV2\0' + struct.pack('=Q', 4) +
                             b''.join(struct.pack('=QQ', *entry) for entry in
                                      [(0, 7), (2**64-1, 1000000), (42, 0), (0, 0)]))
            self.assertEqual(read_hits(path), Counter({0: 7, 2**64-1: 1000000}))
            path.write_bytes(struct.pack('=QQQ', 42, 42, 9) + b'partial')
            self.assertEqual(read_hits(path), Counter({42: 2, 9: 1}))

    def test_interrupted_counter_initialization_does_not_invent_hits(self):
        import tempfile
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'hits.bin'
            path.write_bytes(b'SEVCOV2\0' + struct.pack('=Q', 8) + struct.pack('=QQ', 42, 0))
            self.assertEqual(read_hits(path), Counter())
