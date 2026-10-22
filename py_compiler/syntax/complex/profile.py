"""Test-local wall time and allocator measurements from the c executable."""
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
import subprocess

RUNTIME = r'''
#include <stdint.h>
#include <stdlib.h>
#include <malloc.h>
#include <stdatomic.h>
#include <time.h>
static _Atomic uint64_t live, peak, allocations;
static _Thread_local uint64_t started, initial_live, initial_allocations;
void *__real_malloc(size_t); void __real_free(void *);
void *__real_calloc(size_t,size_t); void *__real_realloc(void *,size_t);
static void added(void *p) {
    if (!p) return;
    atomic_fetch_add(&allocations,1);
    uint64_t current=atomic_fetch_add(&live,malloc_usable_size(p))+malloc_usable_size(p);
    uint64_t previous=atomic_load(&peak);
    while (current>previous && !atomic_compare_exchange_weak(&peak,&previous,current)) {}
}
void *__wrap_malloc(size_t size) { void *p=__real_malloc(size); added(p); return p; }
void *__wrap_calloc(size_t count,size_t size) { void *p=__real_calloc(count,size); added(p); return p; }
void __wrap_free(void *p) { if (p) atomic_fetch_sub(&live,malloc_usable_size(p)); __real_free(p); }
void *__wrap_realloc(void *p,size_t size) {
    size_t before=p?malloc_usable_size(p):0; void *q=__real_realloc(p,size);
    if (q || size==0) { atomic_fetch_sub(&live,before); added(q); }
    return q;
}
static uint64_t now(void) { struct timespec t; if(clock_gettime(CLOCK_MONOTONIC,&t)) abort(); return (uint64_t)t.tv_sec*1000000000+t.tv_nsec; }
extern void __sev_gpu_begin(void) __attribute__((weak));
void __sev_profile_begin(void) { if (__sev_gpu_begin) __sev_gpu_begin(); initial_live=atomic_load(&live); atomic_store(&peak,initial_live); initial_allocations=atomic_load(&allocations); started=now(); }
double __sev_profile_time(void) { return (double)(now()-started)/1000000000.0; }
double __sev_profile_memory(void) { uint64_t maximum=atomic_load(&peak); return (double)(maximum>initial_live?maximum-initial_live:0)*8.0; }
int64_t __sev_profile_allocations(void) { return (int64_t)(atomic_load(&allocations)-initial_allocations); }
'''


def linker_options(objects):
    return [f'-Wl,--wrap={name}' for name in ('malloc','free','calloc','realloc')] if any(str(p).endswith('.profile.o') for p in objects) else []


@dataclass(frozen=True)
class ProfileOperation:
    metric: str

    def atom(self, operation):
        from py_compiler.syntax.generic.atom import Atom
        return Atom(self, (), operation.result.type if operation.result else None, (), ('IO',), self)

    @property
    def external_symbol_contracts(self):
        return (('__sev_profile_' + self.metric, (), {'begin':'','save':'','difference':'','input':'i64','allocations':'i64'}.get(self.metric,'f64')),)

    def c_dependencies(self, target, pointer_bits):
        if pointer_bits != 64 or not target.startswith('x86_64-') or 'linux' not in target:
            raise ValueError('c allocator profiling currently requires x86_64 Linux')
        from py_compiler.lir.src.lib import companion
        from py_compiler.mlir.src.lib import verifier_path
        from py_compiler.syntax.complex.campaign import RUNTIME as CAMPAIGN_RUNTIME
        runtime = RUNTIME + CAMPAIGN_RUNTIME
        key=sha256((runtime+target).encode()).hexdigest()
        directory=Path(__file__).resolve().parents[3]/'package.pkg'/'profile'
        directory.mkdir(parents=True,exist_ok=True)
        source, obj=directory/(key+'.c'),directory/(key+'.profile.o')
        if not obj.exists():
            source.write_text(runtime)
            result=subprocess.run([companion('clang',verifier_path()),'-std=c11','-D_POSIX_C_SOURCE=200809L','-fPIC','-O2','-c',str(source),'-o',str(obj)],capture_output=True,text=True,timeout=20)
            if result.returncode:raise ValueError('profile runtime compilation failed: '+result.stderr)
        return (str(obj),)

    def render_operation(self, operation):
        from py_compiler.mlir.src.cfg import name
        type_=operation.result.type.mlir if operation.result else '()'
        prefix=name(operation.result)+' = ' if operation.result else ''
        return [f'{prefix}func.call @__sev_profile_{self.metric}() : () -> {type_}']


def read(node,cfg,env):
    if not getattr(cfg,'profiling',False):return None
    from py_compiler.syntax.complex.gpu import read_metric
    gpu_metric = read_metric(node, cfg)
    if gpu_metric is not None: return gpu_metric
    metric=(node.token.text if node.kind=='name' and node.token.text not in env else
            node.token.text if node.kind=='member' and node.operands[0].kind=='name' and node.operands[0].token.text=='profile' else None)
    if metric not in ('time','memory','allocations'):return None
    from py_compiler.syntax.complex.units import quantity
    type_=cfg.context.type('int') if metric=='allocations' else quantity((0,1,0) if metric=='time' else (1,0,0))
    return cfg.emit('profile-read',type_,(),ProfileOperation(metric),node.token.span)
