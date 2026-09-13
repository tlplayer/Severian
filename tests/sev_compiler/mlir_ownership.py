#!/usr/bin/env python3
"""Exercise the shared ownership pipeline with real MLIR and physical adapters."""
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
PIPELINE = ROOT / 'sev_compiler/frontend/ownership/mlir.pipeline'


class MlirOwnership(unittest.TestCase):
    def test_aliases_returns_branches_and_loops_release_once(self):
        with tempfile.TemporaryDirectory(prefix='sev-mlir-owner-') as tmp:
            root = Path(tmp)
            source = root / 'subject.mlir'
            source.write_text('''module {
  func.func private @identity(%value: memref<?xi8>) -> memref<?xi8> {
    return %value : memref<?xi8>
  }
  func.func @exercise(%choose: i1) {
    %zero = arith.constant 0 : index
    %one = arith.constant 1 : index
    %count = arith.constant 32 : index
    %expected = arith.constant 42 : i8
    scf.for %i = %zero to %count step %one {
      %first = memref.alloc(%count) : memref<?xi8>
      %second = memref.alloc(%count) : memref<?xi8>
      memref.store %expected, %first[%zero] : memref<?xi8>
      memref.store %expected, %second[%zero] : memref<?xi8>
      %selected = scf.if %choose -> memref<?xi8> {
        scf.yield %first : memref<?xi8>
      } else {
        scf.yield %second : memref<?xi8>
      }
      %alias = func.call @identity(%selected) : (memref<?xi8>) -> memref<?xi8>
      %actual = memref.load %alias[%zero] : memref<?xi8>
      %valid = arith.cmpi eq, %actual, %expected : i8
      cf.assert %valid, "buffer released before its alias"
    }
    return
  }
}
''')
            opt = os.environ.get('SEVERIAN_MLIR_OPT', 'mlir-opt-21')
            owned = subprocess.run([opt, source, '--verify-each', '--pass-pipeline=' + PIPELINE.read_text().strip()],
                                   capture_output=True, text=True, check=True).stdout
            self.assertIn('memref.dealloc', owned)
            self.assertNotIn('__sev_storage_', owned)
            lower = subprocess.run([opt, '--verify-each', '--convert-scf-to-cf', '--expand-strided-metadata',
                                    '--lower-affine', '--convert-arith-to-llvm', '--convert-func-to-llvm',
                                    '--convert-cf-to-llvm', '--finalize-memref-to-llvm=use-generic-functions',
                                    '--reconcile-unrealized-casts'], input=owned, capture_output=True, text=True, check=True).stdout
            llvm = subprocess.run([os.environ.get('SEVERIAN_MLIR_TRANSLATE', 'mlir-translate-21'), '--mlir-to-llvmir'],
                                  input=lower, capture_output=True, text=True, check=True).stdout
            ir = root / 'subject.ll'
            # Instrument MLIR-generated functions, not only the C adapter.
            ir.write_text('\n'.join(' sanitize_address {'.join(line.rsplit(' {', 1)) if line.startswith('define ') else line
                                    for line in llvm.splitlines()) + '\n')
            adapter = root / 'adapter.c'
            adapter.write_text('''#include <stdbool.h>
#include <assert.h>
#include "memory.h"
static unsigned allocations, releases;
void *_mlir_memref_to_llvm_alloc(size_t bytes) { ++allocations; return sev_memory_allocate(bytes); }
void _mlir_memref_to_llvm_free(void *value) { ++releases; sev_memory_release(value); }
extern void exercise(bool choose);
int main(void) {
    exercise(false);
    assert(allocations >= 64 && allocations == releases);
    exercise(true);
    assert(allocations >= 128 && allocations == releases);
}
''')
            binary = root / 'subject'
            linked = subprocess.run([os.environ.get('SEVERIAN_CLANG', 'clang-21'), '-O1', '-g', '-fsanitize=address,undefined',
                            ir, adapter, '-I', ROOT / 'library/core/memory/native', '-o', binary],
                           capture_output=True, text=True)
            self.assertEqual(linked.returncode, 0, linked.stderr)
            subprocess.run([binary], env={**os.environ, 'ASAN_OPTIONS': 'detect_leaks=0'}, check=True,
                           capture_output=True, text=True, timeout=30)


if __name__ == '__main__':
    unittest.main(verbosity=2)
