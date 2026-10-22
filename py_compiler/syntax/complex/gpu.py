"""Checked scalar maps: MLIR GPU kernels, ROCDL code objects, HIP execution."""
from dataclasses import dataclass, replace
from hashlib import sha256
from pathlib import Path
import os
import re
import shutil
import subprocess
import tempfile
from py_compiler.syntax.generic.atom import Atom


def run(command):
    result = subprocess.run(command, capture_output=True, text=True, timeout=45)
    if result.returncode:
        raise ValueError('ROCm GPU compilation failed: ' + result.stderr)
    return result.stdout


def architecture():
    chip = os.environ.get('SEV_GPU_ARCH', '')
    if not chip:
        tool = shutil.which('rocm_agent_enumerator')
        if tool:
            chips = {line.strip() for line in run([tool]).splitlines()
                     if re.fullmatch(r'gfx[0-9a-f]+', line.strip()) and line.strip() != 'gfx000'}
            if len(chips) == 1:
                chip = chips.pop()
    if not re.fullmatch(r'gfx[0-9a-f]+', chip):
        raise ValueError('with gpu requires an AMD target; set SEV_GPU_ARCH=gfx... (no CPU fallback)')
    return chip





def runtime_dependencies(target, pointer_bits):
    if pointer_bits != 64 or 'linux' not in target or not target.startswith('x86_64-'):
        raise ValueError('ROCm GPU execution requires x86_64 Linux with 64-bit pointers')
    from py_compiler.lir.src.lib import companion
    from py_compiler.mlir.src.lib import verifier_path
    root = Path(os.environ.get('ROCM_PATH', '/opt/rocm'))
    includes = next((p for p in (root / 'include', Path('/usr/include'))
                     if (p / 'hip/hip_runtime_api.h').is_file()), None)
    library = next((p for p in (root / 'lib/libamdhip64.so', Path('/usr/lib/x86_64-linux-gnu/libamdhip64.so')) if p.is_file()), None)
    if includes is None or library is None:
        raise ValueError('with gpu requires ROCm HIP headers and libamdhip64.so; set ROCM_PATH')
    directory = Path(__file__).resolve().parents[3] / 'package.pkg/gpu'
    directory.mkdir(parents=True, exist_ok=True)
    source = Path(__file__).resolve().parents[3] / 'library/system/gpu/c/rocm.cpp'
    key = sha256(source.read_bytes() + (target + str(includes)).encode()).hexdigest()
    obj = directory / (key + '.o')
    if not obj.exists():
        with tempfile.TemporaryDirectory(dir=directory) as temp:
            output = Path(temp) / 'runtime.o'
            run([companion('clang', verifier_path()), '-x', 'c++', '-std=c++17', '-D__HIP_PLATFORM_AMD__',
                 '-I' + str(includes), '-fno-exceptions', '-fno-rtti', '-fPIC', '-O2', '-c', str(source), '-o', str(output)])
            output.replace(obj)
    return str(obj), str(library)


@dataclass(frozen=True)
class Map:
    scalar: str
    tree: tuple
    captures: int
    inputs: int = 1

    @property
    def symbol(self):
        return '__sev_gpu_map_' + sha256(repr(self).encode()).hexdigest()[:24]

    def kernel(self):
        scalar = self.scalar
        args = ', '.join(f'%c{i}: {scalar}' for i in range(self.captures))
        lines = ['module {', f'gpu.module @{self.symbol}_device {{',
                 'gpu.func @map(%a: !llvm.ptr<1>, %b: !llvm.ptr<1>, %n: i64' + (', ' + args if args else '') + ') kernel {',
                 '%block = gpu.block_id x', '%thread = gpu.thread_id x', '%width = gpu.block_dim x',
                 '%base = arith.muli %block, %width : index', '%idx = arith.addi %base, %thread : index',
                 '%i = arith.index_cast %idx : index to i64', '%ok = arith.cmpi ult, %i, %n : i64', 'scf.if %ok {',
                 f'%p = llvm.getelementptr %a[%i] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, {scalar}',
                 f'%x = llvm.load %p : !llvm.ptr<1> -> {scalar}']
        def emit(tree):
            if tree[0] == 'item': return '%x'
            if tree[0] == 'other':
                lines.extend(['%other_i = arith.addi %i, %n : i64',
                    f'%other_p = llvm.getelementptr %a[%other_i] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, {scalar}',
                    f'%y = llvm.load %other_p : !llvm.ptr<1> -> {scalar}'])
                return '%y'
            if tree[0] == 'capture': return '%c' + str(tree[1])
            left, right = emit(tree[1]), emit(tree[2])
            name = '%r' + str(len(lines))
            opcode = {'+':'addf','-':'subf','*':'mulf','/':'divf'}[tree[0]]
            lines.append(f'{name} = arith.{opcode} {left}, {right} : {scalar}')
            return name
        result = emit(self.tree)
        lines += [f'%q = llvm.getelementptr %b[%i] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, {scalar}',
                  f'llvm.store {result}, %q : {scalar}, !llvm.ptr<1>', '}', 'gpu.return', '}', '}', '}']
        return '\n'.join(lines)

    def c_dependencies(self, target, pointer_bits):
        return runtime_dependencies(target, pointer_bits)

    def global_definitions(self):
        scalar = self.scalar
        inputs = ', '.join(f'%c{i}: {scalar}' for i in range(self.captures))
        lines = [f'func.func private @{self.symbol}(%a: !llvm.ptr, %b: !llvm.ptr, %n: i64' +
                 (', ' + inputs if inputs else '') + ') attributes {llvm.linkage = #llvm.linkage<internal>} {',
                 '%zero = arith.constant 0 : i64', '%one = arith.constant 1 : i64', '%ok = arith.cmpi sge, %n, %zero : i64',
                 'cf.assert %ok, "negative GPU extent"',
                 f'%image = llvm.mlir.addressof @{self.symbol}_device_image : !llvm.ptr',
                 f'%count = arith.constant {max(1,self.captures)} : i64',
                 '%args = llvm.alloca %count x !llvm.ptr : (i64) -> !llvm.ptr']
        for i in range(self.captures):
            lines += [f'%c{i}_ptr = llvm.alloca %one x {scalar} : (i64) -> !llvm.ptr',
                      f'llvm.store %c{i}, %c{i}_ptr : {scalar}, !llvm.ptr',
                      f'%c{i}_slot = llvm.getelementptr %args[{i}] : (!llvm.ptr) -> !llvm.ptr, !llvm.ptr',
                      f'llvm.store %c{i}_ptr, %c{i}_slot : !llvm.ptr, !llvm.ptr']
        lines += [f'%width = arith.constant {4 if scalar == "f32" else 8} : i64',
                  f'%captures = arith.constant {self.captures} : i64',
                  f'%inputs = arith.constant {self.inputs} : i64',
                  '%status = llvm.call @__sev_gpu_execute(%image, %a, %b, %n, %width, %args, %captures, %inputs) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64, !llvm.ptr, i64, i64) -> i32',
                  '%success = arith.constant 0 : i32', '%completed = arith.cmpi eq, %status, %success : i32',
                  'cf.assert %completed, "GPU driver failed after resource cleanup"', 'func.return', '}']
        return (self.kernel().removeprefix('module {\n').removesuffix('\n}'),
                f'llvm.mlir.global external @{self.symbol}_device_image() : i8',
                'llvm.func @__sev_gpu_execute(!llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64, !llvm.ptr, i64, i64) -> i32',
                '\n'.join(lines))

    def atom(self, operation):
        return Atom(self, tuple(v.type for v in operation.operands), None,
                    ('view','borrow','copy', *(('copy',)*self.captures)), ('read','mutation','IO'), self)

    def render_operation(self, operation):
        from py_compiler.mlir.src.cfg import name, temporary_suffix
        a,b,n,*captures = operation.operands
        prefix = '%gpu_' + temporary_suffix(operation)
        lines = []
        for label, value in (('a',a),('b',b)):
            lines += [f'{prefix}{label} = memref.extract_aligned_pointer_as_index {name(value)} : {value.type.mlir} -> index',
                      f'{prefix}{label}64 = arith.index_cast {prefix}{label} : index to i64',
                      f'{prefix}{label}ptr = llvm.inttoptr {prefix}{label}64 : i64 to !llvm.ptr']
        arguments = ', '.join([prefix+'aptr',prefix+'bptr',name(n),*(name(v) for v in captures)])
        types = ', '.join(('!llvm.ptr','!llvm.ptr','i64', *((self.scalar,)*self.captures)))
        return [*lines, f'func.call @{self.symbol}({arguments}) : ({types}) -> ()']


def lower_device_modules(text, optimizer):
    """Serialize GPU modules within the program's MLIR pipeline, then embed images.

    The host adapters already exist in this module. Only GPU binary storage is
    materialized here; no expression, host function, or C source is synthesized.
    """
    chip = architecture()
    toolkit = os.environ.get('ROCM_PATH', '')
    if not toolkit:
        llvm_root = Path(optimizer).resolve().parent.parent
        candidates = [Path('/opt/rocm'), Path('/usr'), *sorted((llvm_root / 'lib/clang').glob('*'), reverse=True)]
        toolkit = str(next((root for root in candidates if (root / 'amdgcn/bitcode/ocml.bc').is_file()), Path('/opt/rocm')))
    if not re.fullmatch(r'[A-Za-z0-9_./-]+', toolkit):
        raise ValueError('ROCM_PATH must be a plain filesystem path for the MLIR pass pipeline')
    # Distribution packages split device bitcode from LLVM tools. Present the
    # ROCm toolkit layout MLIR expects without modifying the installation.
    from py_compiler.lir.src.lib import companion
    with tempfile.TemporaryDirectory(prefix='sev-rocm-toolkit-') as temporary:
        root = Path(temporary)
        (root / 'amdgcn').symlink_to(Path(toolkit) / 'amdgcn', target_is_directory=True)
        tool_root = Path(companion('ld.lld', optimizer)).resolve().parent
        (root / 'llvm').mkdir()
        (root / 'llvm/bin').symlink_to(tool_root, target_is_directory=True)
        (root / 'bin').symlink_to(tool_root, target_is_directory=True)
        toolkit = str(root)
        pipeline = ('builtin.module(rocdl-attach-target{chip=' + chip + '},'
                    'gpu.module(convert-scf-to-cf,convert-gpu-to-rocdl{chipset=' + chip +
                    ' runtime=HIP},reconcile-unrealized-casts),gpu-module-to-binary{toolkit=' + toolkit + '})')
        tool_environment = dict(os.environ)
        tool_environment['PATH'] = str(Path(optimizer).resolve().parent) + os.pathsep + tool_environment.get('PATH', '')
        result = subprocess.run([optimizer, '--pass-pipeline=' + pipeline], input=text,
                                text=True, capture_output=True, timeout=60, env=tool_environment)
    if result.returncode:
        raise ValueError('ROCm MLIR lowering failed: ' + result.stderr)
    return embed_device_images(result.stdout)


def embed_device_images(text):
    binaries = re.findall(r'^\s*gpu.binary @([A-Za-z_0-9]+)[^\n]*', text, re.MULTILINE)
    if not binaries:
        raise ValueError('ROCDL did not produce an executable GPU binary')
    for symbol in binaries:
        pattern = r'^\s*gpu.binary @' + re.escape(symbol) + r'[^\n]*'
        line = re.search(pattern, text, re.MULTILINE).group(0)
        images = []
        for match in re.finditer(r'"((?:\\[0-9a-fA-F]{2}|\\.|[^"\\])*)"', line):
            encoded = match.group(1)
            data = bytearray(); i = 0
            while i < len(encoded):
                if encoded[i] == '\\':
                    if re.fullmatch(r'[0-9a-fA-F]{2}', encoded[i+1:i+3]):
                        data.append(int(encoded[i+1:i+3],16)); i += 3
                    else:
                        escapes={'\\':92, '"':34, 'n':10, 'r':13, 't':9}
                        if encoded[i+1] not in escapes: raise ValueError('unsupported GPU image byte escape')
                        data.append(escapes[encoded[i+1]]); i += 2
                else:
                    data.extend(encoded[i].encode()); i += 1
            if data.startswith(b'\x7fELF'): images.append(bytes(data))
        if len(images) != 1:
            raise ValueError('GPU binary must contain exactly one executable ELF image')
        data = images[0]
        encoded = ''.join('\\' + format(byte, '02X') for byte in data)
        placeholders = re.findall(r'llvm\.mlir\.global external @' + re.escape(symbol) + r'_image\(\)(?: \{[^\n}]*\})? : i8', text)
        if len(placeholders) != 1:
            raise ValueError('GPU image must have exactly one host declaration')
        placeholder = placeholders[0]
        text = text.replace(placeholder, f'llvm.mlir.global private constant @{symbol}_image("{encoded}") : !llvm.array<{len(data)} x i8>')
        text = re.sub(pattern, '', text, flags=re.MULTILINE)
    return text


def elementwise(node, left, cfg, env):
    from py_compiler.syntax.complex.tensor import TensorType
    right = cfg.expr(node.operands[1], env)
    if (left.type.rank != 1 or right.type.family != 'tensor' or right.type.rank != 1
            or left.type.element != right.type.element or left.type.element.mlir not in ('f32','f64')
            or node.token.text not in ('+','*') or left.type.strided or right.type.strided):
        raise ValueError('GPU elementwise arithmetic requires contiguous rank-one matching floating tensors')
    result = cfg.emit('gpu-elementwise', TensorType(left.type.element), (left,right),
                     Elementwise(node.token.text,left.type.element.mlir), node.token.span)
    cfg.owned_results.add(result.identity)
    cfg.body.placements = [*getattr(cfg.body,'placements',()), {'requested':'gpu','selected':'rocm'}]
    return result


@dataclass(frozen=True)
class Elementwise:
    operator: str
    scalar: str

    @property
    def map(self): return Map(self.scalar,(self.operator,('item',),('other',)),0,2)

    def c_dependencies(self,target,pointer_bits): return self.map.c_dependencies(target,pointer_bits)
    def global_definitions(self): return self.map.global_definitions()

    def atom(self,operation):
        return Atom(self,tuple(v.type for v in operation.operands),operation.result.type,
                    ('view','view'),('read','allocation','IO'),self)

    def render_operation(self,operation):
        from py_compiler.mlir.src.cfg import name,temporary_suffix
        left,right=operation.operands; result=operation.result
        a,b=name(left),name(right); p='%gpu_tensor_'+temporary_suffix(operation)
        scalar=self.scalar; buffer=f'memref<?x{scalar}>'
        lines=[f'{p}z = arith.constant 0 : index', f'{p}one = arith.constant 1 : index',
               f'{p}n = memref.dim {a}, {p}z : {left.type.mlir}',
               f'{p}m = memref.dim {b}, {p}z : {right.type.mlir}',
               f'{p}equal = arith.cmpi eq, {p}n, {p}m : index',
               f'cf.assert {p}equal, "GPU tensor shapes do not match"',
               f'{p}max = arith.constant {((1<<63)-1)//(2*(4 if scalar=="f32" else 8))} : index',
               f'{p}fits = arith.cmpi ule, {p}n, {p}max : index',
               f'cf.assert {p}fits, "GPU tensor extent overflows"',
               f'{p}extent = arith.addi {p}n, {p}n : index',
               f'{p}packed = memref.alloc({p}extent) : {buffer}',
               f'{name(result)} = memref.alloc({p}n) : {result.type.mlir}',
               f'scf.for {p}i = {p}z to {p}n step {p}one {{',
               f'{p}x = memref.load {a}[{p}i] : {left.type.mlir}',
               f'{p}y = memref.load {b}[{p}i] : {right.type.mlir}',
               f'{p}j = arith.addi {p}i, {p}n : index',
               f'memref.store {p}x, {p}packed[{p}i] : {buffer}',
               f'memref.store {p}y, {p}packed[{p}j] : {buffer}', '}']
        for label,value,type_ in (('input',p+'packed',buffer),('output',name(result),result.type.mlir)):
            lines += [f'{p}{label}_idx = memref.extract_aligned_pointer_as_index {value} : {type_} -> index',
                      f'{p}{label}_int = arith.index_cast {p}{label}_idx : index to i64',
                      f'{p}{label} = llvm.inttoptr {p}{label}_int : i64 to !llvm.ptr']
        lines += [f'{p}count = arith.index_cast {p}n : index to i64',
                  f'func.call @{self.map.symbol}({p}input, {p}output, {p}count) : (!llvm.ptr, !llvm.ptr, i64) -> ()',
                  f'memref.dealloc {p}packed : {buffer}']
        return lines


@dataclass(frozen=True)
class Buffer:
    kind: str

    def atom(self, operation):
        modes = ('copy',) if self.kind == 'allocate' else ('move',) if self.kind == 'free' else ('borrow','copy','copy')
        effects = ('allocation',) if self.kind == 'allocate' else ('destruction',) if self.kind == 'free' else ('mutation',)
        return Atom(self, tuple(v.type for v in operation.operands), operation.result.type if operation.result else None, modes, effects, self)

    def render_operation(self, operation):
        from py_compiler.mlir.src.cfg import name, temporary_suffix
        values = operation.operands
        if self.kind == 'free': return [f'memref.dealloc {name(values[0])} : {values[0].type.mlir}']
        p = '%gpu_index_' + temporary_suffix(operation)
        value = values[0] if self.kind == 'allocate' else values[1]
        lines = [f'{p} = arith.index_cast {name(value)} : {value.type.mlir} to index']
        if self.kind == 'allocate':
            lines += [f'{name(operation.result)} = memref.alloc({p}) : {operation.result.type.mlir}']
        else:
            lines += [f'memref.store {name(values[2])}, {name(values[0])}[{p}] : {values[0].type.mlir}']
        return lines


def lower(node, cfg, env):
    from py_compiler.syntax.grammar.expression import Expression, expression
    from py_compiler.syntax.complex.tensor import ArgumentValue
    from py_compiler.syntax.generic.place import expression_type
    from py_compiler.syntax.generic.storage import root_binding
    from py_compiler.syntax.function.calls import literal_value
    from py_compiler.syntax.function.length import Length
    from py_compiler.syntax.generic.owned import select
    from py_compiler.syntax.complex.array import ArrayType
    from py_compiler.hir.thir.src.model import Edge, Terminator
    token, source, _ = node.header
    if source.kind != 'name':
        raise ValueError('GPU input requires a live list binding; expressions with effects are unsupported')
    if isinstance(token, tuple) or len(node.children) != 1 or node.children[0].children:
        raise ValueError('GPU loop requires one unconditional output.append(scalar expression)')
    call = expression(node.children[0].tokens)
    if (call.kind != 'call' or len(call.operands) != 2 or call.operands[0].kind != 'member'
            or call.operands[0].token.text != 'append' or call.operands[0].operands[0].kind != 'name'):
        raise ValueError('GPU loop requires one unconditional output.append(scalar expression)')
    destination = call.operands[0].operands[0]
    source_type, destination_type = expression_type(source,cfg,env), expression_type(destination,cfg,env)
    def list_element(type_):
        declaration = getattr(type_,'declaration',None)
        if getattr(declaration,'template_name','').rsplit('.',1)[-1] != 'list':
            raise ValueError('GPU map requires list input and output')
        return type_.generic_arguments[0]
    element = list_element(source_type)
    if element != list_element(destination_type) or element.mlir not in ('f32','f64'):
        raise ValueError('GPU map requires matching f32 or f64 list elements')
    source_owner, output_owner = root_binding(source,cfg,env), root_binding(destination,cfg,env)
    if source_owner is None or output_owner is None or cfg.flow.resolve(source_owner.identity) == cfg.flow.resolve(output_owner.identity):
        raise ValueError('GPU input and output must be distinct live lists')
    captures = []
    def tree(term):
        if term.kind == 'name' and term.token.text == token.text: return ('item',)
        if term.kind == 'binary' and term.token.text in ('+','-','*','/'):
            return (term.token.text, tree(term.operands[0]), tree(term.operands[1]))
        if term.kind not in ('name','literal') or (term.kind == 'name' and term.token.text in (source.token.text,destination.token.text)):
            raise ValueError('GPU expression supports scalar captures and +, -, *, /; calls and other effects are unsupported')
        value = cfg.expr(term,env,element)
        if value.type != element: raise ValueError('GPU capture must match the element floating type')
        captures.append(value)
        return ('capture',len(captures)-1)
    kernel = Map(element.mlir,tree(call.operands[1]),len(captures))
    count = Length().call(None,(source,),cfg,env,node.span)
    buffer_type = ArrayType(element)
    input_buffer = cfg.emit('gpu-buffer',buffer_type,(count,),Buffer('allocate'),node.span)
    output_buffer = cfg.emit('gpu-buffer',buffer_type,(count,),Buffer('allocate'),node.span)
    def capture(value): return Expression('capture',token,(ArgumentValue(value),))
    def loop(action):
        zero = literal_value(0,count.type,cfg,node.span)
        one = literal_value(1,count.type,cfg,node.span)
        header,body,done = cfg.block((count.type,)),cfg.block(),cfg.block()
        cfg.current.terminator = Terminator('jump',(Edge(header.identity,(zero,)),))
        cfg.current = header
        i = header.parameters[0]
        condition = select(count.type,'F.operator','<').expand(cfg,i,count,node.span)
        cfg.current.terminator = Terminator('conditional',(Edge(body.identity),Edge(done.identity)),condition)
        cfg.current = body
        action(i)
        next_ = select(count.type,'F.operator','+').expand(cfg,i,one,node.span)
        cfg.current.terminator = Terminator('jump',(Edge(header.identity,(next_,)),))
        cfg.current = done
    def snapshot(i):
        from py_compiler.syntax.primitive.numeric.conversion import ScalarConversion
        index_type = cfg.context.type('usize')
        position = cfg.emit('convert', index_type, (i,), ScalarConversion(i.type,index_type),node.span)
        value = cfg.expr(Expression('index',token,(source,capture(position))),env,element)
        cfg.effect('gpu-store',(input_buffer,i,value),Buffer('store'),node.span)
    loop(snapshot)
    cfg.effect('gpu-map',(input_buffer,output_buffer,count,*captures),kernel,node.span)
    def publish(i):
        value = cfg.expr(Expression('index',token,(capture(output_buffer),capture(i))),env,element)
        cfg.expr(replace(call,operands=(call.operands[0],capture(value))),env)
    loop(publish)
    for value in (input_buffer,output_buffer): cfg.effect('gpu-free',(value,),Buffer('free'),node.span)
    cfg.body.placements = [*getattr(cfg.body,'placements',()), {'requested':'gpu','selected':'rocm'}]
    return 1


@dataclass(frozen=True)
class Metric:
    metric: str

    @property
    def external_symbol_contracts(self):
        return (('__sev_gpu_' + self.metric, (), 'i64' if self.metric in ('kernels','allocations') else 'f64'),)

    def c_dependencies(self, target, pointer_bits):
        return runtime_dependencies(target,pointer_bits)

    def atom(self, operation):
        return Atom(self,(),operation.result.type,(),('IO',),self)

    def render_operation(self, operation):
        from py_compiler.mlir.src.cfg import name
        return [f'{name(operation.result)} = func.call @__sev_gpu_{self.metric}() : () -> {operation.result.type.mlir}']


def read_metric(node,cfg):
    parts=[]
    term=node
    while term.kind=='member':
        parts.append(term.token.text)
        term=term.operands[0]
    if term.kind!='name' or term.token.text!='profile': return None
    path=tuple(reversed(parts))
    metric={('gpu','kernels'):'kernels',('gpu','time'):'time',
            ('gpu','memory','peak'):'peak',('gpu','memory','allocations'):'allocations'}.get(path)
    if metric is None: return None
    from py_compiler.syntax.complex.units import quantity
    type_=cfg.context.type('int') if metric in ('kernels','allocations') else quantity((0,1,0) if metric=='time' else (1,0,0))
    return cfg.emit('gpu-profile',type_,(),Metric(metric),node.token.span)


import unittest


class GpuTests(unittest.TestCase):
    def test_kernel_has_indexed_multiply_and_tail_guard(self):
        text = Map('f64',('*',('item',),('capture',0)),1).kernel()
        for expected in ('gpu.func @map', 'gpu.block_id', 'arith.cmpi ult', 'arith.mulf %x, %c0', 'llvm.store'):
            self.assertIn(expected,text)
        self.assertNotIn('append',text)

    def test_missing_target_is_an_error(self):
        from unittest.mock import patch
        with patch.dict(os.environ,{'SEV_GPU_ARCH':''}), patch.object(shutil,'which',return_value=None):
            with self.assertRaisesRegex(ValueError,'no CPU fallback'): architecture()

    def test_bad_target_cannot_be_pass_injection(self):
        from unittest.mock import patch
        with patch.dict(os.environ,{'SEV_GPU_ARCH':'gfx1100} evil'}):
            with self.assertRaisesRegex(ValueError,'AMD target'): architecture()


class RocmExecutionTests(unittest.TestCase):
    """Explicit hardware tests: python3 -m unittest ...gpu.RocmExecutionTests."""

    def test_f32_and_f64_empty_tail_and_device_metrics(self):
        import ctypes
        from py_compiler.lir.src.lib import companion
        from py_compiler.mlir.src.lib import verifier_path
        for scalar, ctype in (('f32',ctypes.c_float),('f64',ctypes.c_double)):
            with self.subTest(scalar=scalar), tempfile.TemporaryDirectory() as directory:
                operation=Map(scalar,('*',('item',),('capture',0)),1)
                dependencies=operation.c_dependencies('x86_64-unknown-linux-gnu',64)
                from py_compiler.lir.src.lib import Lir, compile_object
                from unittest.mock import patch
                text='module {\n'+'\n'.join(operation.global_definitions())+'\n}'
                text=text.replace('func.func private @'+operation.symbol,'func.func @'+operation.symbol).replace(' attributes {llvm.linkage = #llvm.linkage<internal>}', '')
                with patch('py_compiler.lir.src.lib.render',return_value=text):
                    artifact=compile_object(Lir(None,'x86_64-unknown-linux-gnu',64,dependencies))
                obj=Path(directory)/'map.o'; obj.write_bytes(artifact.object_bytes)
                library=Path(directory)/'map.so'
                run([companion('clang',verifier_path()),'-shared',str(obj),*dependencies,'-o',str(library)])
                c=ctypes.CDLL(str(library))
                function=getattr(c,operation.symbol)
                function.argtypes=[ctypes.c_void_p,ctypes.c_void_p,ctypes.c_int64,ctype]
                function.restype=None
                # getattr avoids Python's class-private name mangling.
                begin=getattr(c,'__sev_gpu_begin')
                kernels=getattr(c,'__sev_gpu_kernels'); kernels.restype=ctypes.c_int64
                allocations=getattr(c,'__sev_gpu_allocations'); allocations.restype=ctypes.c_int64
                peak=getattr(c,'__sev_gpu_peak'); peak.restype=ctypes.c_double
                for count in (0,1,3,257):
                    begin()
                    source=(ctype*count)(*range(count))
                    result=(ctype*count)()
                    function(source,result,count,ctype(4))
                    self.assertEqual(list(result),[float(i*4) for i in range(count)])
                    self.assertEqual(kernels(),int(count>0))
                    self.assertEqual(allocations(),2 if count else 0)
                    self.assertEqual(peak(),count*ctypes.sizeof(ctype)*2*8)


class GpuDiagnosticTests(unittest.TestCase):
    def test_effectful_body_reports_error_instead_of_cpu_fallback(self):
        from py_compiler.frontend.src.lib import compile_source
        from py_compiler.syntax.recognition import Syntax
        result = compile_source('gpu-effect.sev', '''def invalid(values: array[f64]):
    for value in values with gpu:
        print(value)
''', Syntax(include_tests=False))
        self.assertTrue(result.diagnostics)
        self.assertIn('GPU loop requires', str(result.diagnostics))


class GpuBoundaryTests(unittest.TestCase):
    def test_tensor_placements_share_arithmetic_and_render_without_c_compilation(self):
        from unittest.mock import patch
        from py_compiler.frontend.src.lib import compile_source
        from py_compiler.syntax.recognition import Syntax
        from py_compiler.mlir.src.lib import lower, render, verify_c
        for placement in ('gpu','simd'):
            with self.subTest(placement=placement):
                source = ('def multiply(a: tensor[f64], b: tensor[f64]) -> tensor[f64]:\n'
                          f'    with {placement}:\n        return a * b\n')
                checked=compile_source('placement.sev',source,Syntax(include_tests=False))
                self.assertFalse(checked.diagnostics,str(checked.diagnostics))
                with patch.object(subprocess,'run',side_effect=AssertionError('render spawned a compiler')):
                    text=render(lower(checked.program))
                verify_c(text)
                if placement=='gpu':
                    for fragment in ('gpu.module @','arith.mulf','llvm.call @__sev_gpu_execute','GPU tensor shapes do not match','memref.dealloc'):
                        self.assertIn(fragment,text)
                else:
                    self.assertIn('vector<4xf64>',text)
                    self.assertNotIn('__sev_gpu_execute',text)

    def test_binary_is_embedded_in_the_same_mlir_module(self):
        text = r'''module {
llvm.mlir.global external @map_device_image() : i8
gpu.binary @map_device [#gpu.object<#rocdl.target<chip = "gfx1100">, "\7FELF\00\FF">]
}'''
        embedded=embed_device_images(text)
        self.assertIn('private constant @map_device_image',embedded)
        self.assertIn('!llvm.array<6 x i8>',embedded)
        self.assertNotIn('gpu.binary',embedded)
        from py_compiler.mlir.src.lib import verify_c
        verify_c(embedded)
        with self.assertRaisesRegex(ValueError,'executable ELF'):
            embed_device_images(text.replace(r'\7FELF', 'oops'))

    def test_mlir_printer_escapes_and_global_attributes_are_preserved(self):
        text = r'''module {
llvm.mlir.global external @map_device_image() {addr_space = 0 : i32} : i8
gpu.binary @map_device [#gpu.object<#rocdl.target<chip = "gfx1100">, bin = "\7FELF\\\00\FF">]
}'''
        embedded=embed_device_images(text)
        self.assertIn('!llvm.array<7 x i8>',embedded)
        self.assertIn(r'\7F\45\4C\46\5C\00\FF',embedded)

    def test_gpu_rejects_unsupported_shapes_without_fallback(self):
        from py_compiler.frontend.src.lib import compile_source
        from py_compiler.syntax.recognition import Syntax
        source='def add(a: tensor[i32], b: tensor[i32]) -> tensor[i32]:\n    with gpu:\n        return a + b\n'
        checked=compile_source('unsupported-gpu.sev',source,Syntax(include_tests=False))
        self.assertIn('matching floating tensors',str(checked.diagnostics))
