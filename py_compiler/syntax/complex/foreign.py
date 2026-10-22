"""Explicit C/Rust/Python boundaries, with owned UTF-8 conversion at the ABI."""
import ast
import json
import re
import subprocess
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path


def register(node, entry, context):
    annotations = [a for a in node.decorators if a.name in ('c', 'rust', 'python')]
    if not annotations:
        return
    if len(annotations) != 1 or node.children or node.tokens[-1].text == ':':
        raise ValueError('foreign annotation requires one bodyless function')
    annotation = annotations[0]
    args = annotation.arguments
    if args and (len(args) != 3 or args[:2] != ('symbol', '=')):
        raise ValueError('foreign annotation accepts symbol="external_name"')
    symbol = ast.literal_eval(args[2]) if args else entry.name.rsplit('.', 1)[-1]
    if not isinstance(symbol, str) or not re.fullmatch(r'[A-Za-z_][A-Za-z_0-9]*', symbol):
        raise ValueError('foreign symbol must be a plain identifier')
    source = context.source.debug_path or context.source.path
    path = Path(source).resolve()
    root = next((parent for parent in path.parents if (parent / 'package.json').is_file()), None)
    spec = json.loads((root / 'package.json').read_text()).get('xxi', {}).get(annotation.name, {}) if root else {}
    sources = tuple(str((root / name).resolve()) for name in spec.get('sources', ()))
    if root and any(not Path(name).is_relative_to(root) for name in sources):
        raise ValueError('foreign source escapes its package')
    if annotation.name in ('rust', 'python') and not sources:
        raise ValueError('foreign provider requires manifest xxi sources')
    inputs = tuple(context.type(t) for _, t in entry.parameters)
    result = context.type(entry.result or 'absent')
    operation = ForeignOperation(annotation.name, symbol, inputs, result, sources, context.syntax.target, context.syntax.pointer_bits)
    operation.validate(inputs, result)
    node.mlir_operation = operation
    entry.foreign_boundary = True


def c_objects(language, sources, target):
    from py_compiler.lir.src.lib import companion
    from py_compiler.mlir.src.lib import verifier_path
    objects = []
    for name in sources:
        source = Path(name)
        content = source.read_bytes()
        key = sha256(content + target.encode() + language.encode()).hexdigest()
        destination = source.parent / 'package.pkg' / 'foreign' / (key + '.o')
        destination.parent.mkdir(parents=True, exist_ok=True)
        if not destination.exists():
            temporary = destination.with_suffix('.tmp.o')
            if language == 'c':
                command = [companion('clang', verifier_path()), '-target', target, '-fPIC', '-O2', '-c', str(source), '-o', str(temporary)]
            else:
                command = ['rustc', '--crate-type=lib', '--emit=obj', '-C', 'panic=abort', '-C', 'relocation-model=pic', '-O', '--target', target, str(source), '-o', str(temporary)]
            result = subprocess.run(command, capture_output=True, text=True, timeout=20)
            if result.returncode:
                raise ValueError('foreign compilation failed: ' + result.stderr)
            temporary.replace(destination)
        objects.append(str(destination))
    return tuple(objects)


@dataclass(frozen=True)
class ForeignOperation:
    language: str
    foreign_symbol: str
    inputs: tuple
    result: object
    sources: tuple
    target: str = "x86_64-unknown-linux-gnu"
    pointer_bits: int = 64

    @property
    def symbol(self):
        text = repr((self.language, self.foreign_symbol, tuple(t.name for t in self.inputs), self.result.name, self.sources, self.target, self.pointer_bits))
        return '__sev_foreign_' + sha256(text.encode()).hexdigest()[:24]

    def validate(self, inputs, result):
        if self.language == 'c':
            from py_compiler.syntax.complex.abi import CAbi
            abi = CAbi(self.target, self.pointer_bits)
            for type_ in inputs: abi.validate_type(type_)
            abi.validate_type(result, result=True)
            return
        for type_ in (*inputs, result):
            raw_pointer = self.language in ('c', 'rust') and type_.family == 'pointer' and type_.mlir == '!llvm.ptr'
            if type_.family not in ('integer', 'float', 'bool', 'string', 'absent') and not raw_pointer:
                raise ValueError('foreign boundary requires scalar or string types, got ' + type_.name)
        if self.result.family == 'string' and self.language != 'python':
            raise ValueError('foreign string results require an explicit ownership adapter')

    def validate_target(self, target, pointer_bits):
        if self.language == 'c':
            from py_compiler.syntax.complex.abi import CAbi
            CAbi(target, pointer_bits)
            if (target, pointer_bits) != (self.target, self.pointer_bits):
                raise ValueError('C ABI target changed after checking; recompile the declaration')
            return
        if pointer_bits != 64 or 'linux' not in target or not target.startswith('x86_64-'):
            raise ValueError('foreign UTF-8 adapters currently require x86_64 Linux')

    def c_dependencies(self, target, pointer_bits):
        self.validate_target(target, pointer_bits)
        if self.language == 'python':
            return self.python_bridge(target)
        return c_objects(self.language, self.sources, target)

    def python_bridge(self, target):
        import sysconfig
        from py_compiler.lir.src.lib import companion
        from py_compiler.mlir.src.lib import verifier_path
        if len(self.sources) != 1:
            raise ValueError('Python provider requires exactly one module source')
        def ctype(t):
            if t.family == 'string': return 'char *'
            if t.family == 'float': return 'float' if t.bits == 32 else 'double'
            if t.family == 'bool': return '_Bool'
            if t.family == 'absent': return 'void'
            return ('int' if getattr(t, 'signed', True) else 'uint') + str(t.bits) + '_t'
        parameters = ', '.join(ctype(t) + ' a' + str(i) for i, t in enumerate(self.inputs)) or 'void'
        conversions = []
        for i, type_ in enumerate(self.inputs):
            name = 'a' + str(i)
            make = ('PyUnicode_FromString(' + name + ')' if type_.family == 'string' else
                    'PyFloat_FromDouble(' + name + ')' if type_.family == 'float' else
                    ('PyLong_FromLongLong(' if getattr(type_, 'signed', True) else 'PyLong_FromUnsignedLongLong(') + name + ')')
            conversions.append(f'PyObject *arg{i} = {make}; if (!arg{i}) goto failed; PyTuple_SET_ITEM(args, {i}, arg{i});')
        if self.result.family == 'string':
            extraction = 'Py_ssize_t length; const char *text = PyUnicode_AsUTF8AndSize(value, &length); if (!text) goto failed; char *out = malloc((size_t)length + 1); if (!out) abort(); memcpy(out, text, length); out[length] = 0;'
        elif self.result.family == 'absent':
            extraction = ''
        else:
            conversion = 'PyFloat_AsDouble(value)' if self.result.family == 'float' else ('PyLong_AsLongLong(value)' if getattr(self.result, 'signed', True) else 'PyLong_AsUnsignedLongLong(value)')
            extraction = ctype(self.result) + ' out = ' + conversion + '; if (PyErr_Occurred()) goto failed;'
        code = '#include <Python.h>\n#include <stdint.h>\n#include <stdlib.h>\n#include <string.h>\n#include <pthread.h>\n'
        code += 'static pthread_once_t once = PTHREAD_ONCE_INIT; static void init(void) { if (!Py_IsInitialized()) { Py_Initialize(); PyEval_SaveThread(); } }\n'
        code += ctype(self.result) + ' ' + self.symbol + '_python(' + parameters + ') {\n'
        code += 'pthread_once(&once, init); PyGILState_STATE gil = PyGILState_Ensure();\n'
        code += 'PyObject *globals = PyDict_New(), *args = NULL, *value = NULL; if (!globals) goto failed;\n'
        code += 'if (PyDict_SetItemString(globals, "__builtins__", PyEval_GetBuiltins()) < 0) goto failed;\n'
        code += 'FILE *file = fopen(' + json.dumps(self.sources[0]) + ', "r"); if (!file) { PyErr_SetFromErrno(PyExc_OSError); goto failed; }\n'
        code += 'PyObject *loaded = PyRun_FileEx(file, ' + json.dumps(self.sources[0]) + ', Py_file_input, globals, globals, 1); if (!loaded) goto failed; Py_DECREF(loaded);\n'
        code += 'PyObject *callable = PyDict_GetItemString(globals, ' + json.dumps(self.foreign_symbol) + '); if (!callable || !PyCallable_Check(callable)) { PyErr_SetString(PyExc_TypeError, "foreign symbol is not callable"); goto failed; }\n'
        code += 'args = PyTuple_New(' + str(len(self.inputs)) + '); if (!args) goto failed;\n' + '\n'.join(conversions)
        code += '\nvalue = PyObject_CallObject(callable, args); if (!value) goto failed;\n' + extraction
        code += '\nPy_DECREF(value); Py_DECREF(args); Py_DECREF(globals); PyGILState_Release(gil); return' + (' out' if self.result.mlir else '') + ';\n'
        code += 'failed: PyErr_Print(); Py_XDECREF(value); Py_XDECREF(args); Py_XDECREF(globals); PyGILState_Release(gil); abort(); }\n'
        key = sha256((code + target).encode() + Path(self.sources[0]).read_bytes()).hexdigest()
        directory = Path(self.sources[0]).parent / 'package.pkg' / 'foreign'
        directory.mkdir(parents=True, exist_ok=True)
        source, obj = directory / (key + '.c'), directory / (key + '.o')
        if not obj.exists():
            source.write_text(code)
            command = [companion('clang', verifier_path()), '-fPIC', '-O2', '-I' + sysconfig.get_path('include'), '-c', str(source), '-o', str(obj)]
            completed = subprocess.run(command, capture_output=True, text=True, timeout=20)
            if completed.returncode: raise ValueError('Python bridge compilation failed: ' + completed.stderr)
        library = Path(sysconfig.get_config_var('LIBDIR')) / sysconfig.get_config_var('LDLIBRARY')
        if not library.is_file(): raise ValueError('Python embedding library unavailable: ' + str(library))
        return str(obj), str(library)

    def render_operation(self, operation):
        from py_compiler.mlir.src.cfg import name
        operands = ', '.join(name(value) for value in operation.operands)
        types = ', '.join(t.mlir for t in self.inputs)
        result = self.result.mlir or '()'
        prefix = name(operation.result) + ' = ' if operation.result else ''
        return [f'{prefix}func.call @{self.symbol}({operands}) : ({types}) -> {result}']

    def global_definitions(self):
        from py_compiler.syntax.generic.storage import descriptor_type
        descriptor = descriptor_type(64)
        target = self.symbol + '_python' if self.language == 'python' else self.foreign_symbol
        def abi(t): return '!llvm.ptr' if t.family == 'string' else t.mlir
        from py_compiler.syntax.complex.abi import CAbi
        contract = CAbi(self.target, self.pointer_bits) if self.language == 'c' else None
        def attrs(t): return contract.attributes(t) if contract else '{}'
        def declaration_type(t):
            return abi(t) + (' ' + attrs(t) if attrs(t) != '{}' else '')
        call_attrs = ' {arg_attrs = [' + ', '.join(attrs(t) for t in self.inputs) + ']'
        if self.result.mlir: call_attrs += ', res_attrs = [' + attrs(self.result) + ']'
        call_attrs += '}'
        inputs = ', '.join(f'%a{i}: {t.mlir}' for i, t in enumerate(self.inputs))
        suffix = ' -> ' + self.result.mlir if self.result.mlir else ''
        lines = [f'func.func private @{self.symbol}({inputs}){suffix} attributes {{llvm.linkage = #llvm.linkage<internal>}} {{', '%zero = arith.constant 0 : i64', '%one = arith.constant 1 : i64']
        arguments, buffers = [], []
        for i, type_ in enumerate(self.inputs):
            if type_.family != 'string':
                arguments.append('%a' + str(i)); continue
            p = '%s' + str(i)
            lines += [f'{p}_descriptor = builtin.unrealized_conversion_cast %a{i} : {type_.mlir} to {descriptor}',
                      f'{p}_base = llvm.extractvalue {p}_descriptor[1] : {descriptor}',
                      f'{p}_offset = llvm.extractvalue {p}_descriptor[2] : {descriptor}',
                      f'{p}_length = llvm.extractvalue {p}_descriptor[3, 0] : {descriptor}',
                      f'{p}_data = llvm.getelementptr {p}_base[{p}_offset] : (!llvm.ptr, i64) -> !llvm.ptr, i8',
                      f'{p}_size = arith.addi {p}_length, %one : i64',
                      f'{p}_buffer = llvm.call @malloc({p}_size) : (i64) -> !llvm.ptr',
                      f'{p}_null = llvm.mlir.zero : !llvm.ptr',
                      f'{p}_ok = llvm.icmp "ne" {p}_buffer, {p}_null : !llvm.ptr',
                      f'cf.assert {p}_ok, "foreign string allocation failed"',
                      f'"llvm.intr.memcpy"({p}_buffer, {p}_data, {p}_length) <{{isVolatile = false}}> : (!llvm.ptr, !llvm.ptr, i64) -> ()',
                      f'{p}_end = llvm.getelementptr {p}_buffer[{p}_length] : (!llvm.ptr, i64) -> !llvm.ptr, i8',
                      f'{p}_nul = arith.constant 0 : i8', f'llvm.store {p}_nul, {p}_end : i8, !llvm.ptr']
            arguments.append(p + '_buffer'); buffers.append(p + '_buffer')
        prefix = '%foreign = ' if self.result.mlir else ''
        lines.append(f'{prefix}llvm.call @{target}({", ".join(arguments)}){call_attrs} : ({", ".join(map(abi, self.inputs))}) -> {abi(self.result) or "()"}')
        lines += [f'llvm.call @free({buffer}) : (!llvm.ptr) -> ()' for buffer in buffers]
        if self.result.family == 'string':
            lines += ['%length = llvm.call @strlen(%foreign) : (!llvm.ptr) -> i64',
                      f'%d0 = llvm.mlir.undef : {descriptor}',
                      f'%d1 = llvm.insertvalue %foreign, %d0[0] : {descriptor}',
                      f'%d2 = llvm.insertvalue %foreign, %d1[1] : {descriptor}',
                      f'%d3 = llvm.insertvalue %zero, %d2[2] : {descriptor}',
                      f'%d4 = llvm.insertvalue %length, %d3[3, 0] : {descriptor}',
                      f'%d5 = llvm.insertvalue %one, %d4[4, 0] : {descriptor}',
                      f'%result = builtin.unrealized_conversion_cast %d5 : {descriptor} to {self.result.mlir}',
                      f'func.return %result : {self.result.mlir}']
        else:
            lines.append('func.return' + (' %foreign : ' + self.result.mlir if self.result.mlir else ''))
        lines.append('}')
        definitions = [f'llvm.func @{target}({", ".join(map(declaration_type, self.inputs))})' + (' -> (' + declaration_type(self.result) + ')' if self.result.mlir else ''), '\n'.join(lines)]
        if any(t.family == 'string' for t in (*self.inputs, self.result)):
            definitions += ['llvm.func @malloc(i64) -> !llvm.ptr', 'llvm.func @free(!llvm.ptr)']
        if self.result.family == 'string': definitions.append('llvm.func @strlen(!llvm.ptr) -> i64')
        return tuple(definitions)


import unittest


class ForeignLinkageTests(unittest.TestCase):
    def test_same_foreign_adapter_links_from_independent_objects(self):
        import tempfile
        from py_compiler.frontend.src.lib import compile_source
        from py_compiler.syntax.recognition import Syntax
        from py_compiler.lir.src.lib import lower, compile_object, companion
        from py_compiler.mlir.src.lib import verifier_path
        with tempfile.TemporaryDirectory(prefix='sev-foreign-linkage-') as directory:
            objects = []
            for name in ('left', 'right'):
                source = ('@c(symbol="labs")\ndef absolute(value: i64) -> i64\n'
                          f'def {name}(value: i64) -> i64:\n'
                          '    unsafe:\n        return absolute(value)\n')
                result = compile_source(name + '.sev', source, Syntax())
                self.assertFalse(result.diagnostics, result.diagnostics)
                artifact = compile_object(lower(result.program, 'x86_64-unknown-linux-gnu', 64))
                self.assertIn('define internal', artifact.llvm_ir)
                path = Path(directory) / (name + '.o')
                path.write_bytes(artifact.object_bytes)
                objects.append(str(path))
            linked = subprocess.run(
                [companion('clang', verifier_path()), '-r', *objects,
                 '-o', str(Path(directory) / 'combined.o')],
                capture_output=True, text=True, timeout=10)
            self.assertEqual(linked.returncode, 0, linked.stderr)


class CInteropTests(unittest.TestCase):
    def test_call_and_extension_survive_mlir_for_each_target(self):
        from py_compiler.frontend.src.lib import compile_source
        from py_compiler.syntax.recognition import Syntax
        from py_compiler.mlir.src.lib import lower, render, verify_c
        for target,extension in (('x86_64-unknown-linux-gnu','llvm.signext'),
                                 ('aarch64-unknown-linux-gnu',''),
                                 ('x86_64-pc-windows-msvc',''),
                                 ('aarch64-apple-darwin','llvm.signext')):
            with self.subTest(target=target):
                source='@c(symbol="external_i8")\ndef external(value: i8) -> i8\ndef use(value: i8) -> i8:\n    unsafe:\n        return external(value)\n'
                result=compile_source('abi-call.sev',source,Syntax(target=target))
                self.assertFalse(result.diagnostics,str(result.diagnostics))
                text=render(lower(result.program))
                self.assertIn('llvm.call @external_i8',text)
                self.assertEqual('llvm.signext' in text,bool(extension))
                verify_c(text)

    def test_call_requires_unsafe(self):
        from py_compiler.frontend.src.lib import compile_source
        from py_compiler.syntax.recognition import Syntax
        result=compile_source('abi-unsafe.sev','@c\ndef external(value: i32) -> i32\ndef use() -> i32:\n    return external(1)\n',Syntax())
        self.assertIn('unsafe',str(result.diagnostics))

    def test_c_rendering_does_not_compile_an_adapter(self):
        from unittest.mock import patch
        from py_compiler.syntax.recognition import Syntax
        types=Syntax().types
        operation=ForeignOperation('c','scalar',(types['i32'],),types['i32'],())
        with patch.object(subprocess,'run',side_effect=AssertionError('rendering must not invoke a compiler')):
            self.assertIn('llvm.call @scalar','\n'.join(operation.global_definitions()))
