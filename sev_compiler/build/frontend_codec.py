#!/usr/bin/env python3
"""Generate the versioned prelude frontend archive codec from concrete IR models.

The archive preserves syntax, diagnostics, and generic bodies; it is not a C ABI.
Transient lowering fields are intentionally absent and retain their defaults.
"""
from pathlib import Path
import re
import json
import hashlib
import fcntl
import os
import tempfile

ROOT = Path(__file__).resolve().parents[2]
SKIP = {
    'model_hir_module_module.Module': {'initializer_cfg', 'cfg_bodies', 'values', 'globals', 'binding_values', 'quality_files', 'quality_exclusions', 'quality_regions', 'debug_files'},
    'model_syntax_function_function.FunctionDeclaration': {'binary_requirements', 'capability_requirements', 'cfg'},
    'model_syntax_block_block.Block': {'operations', 'lowered_operations'},
}

def split(text, separator=','):
    result, start, depth = [], 0, 0
    for i, c in enumerate(text):
        if c in '[(': depth += 1
        if c in '])': depth -= 1
        if c == separator and depth == 0:
            result.append(text[start:i].strip()); start = i + 1
    result.append(text[start:].strip())
    return result


def exported_sources(entry):
    seen = set()
    def visit(path):
        path = path.resolve()
        if path in seen: return
        seen.add(path)
        for locator in re.findall(r'^import \* from "([^"\n]+)"$', path.read_text(), re.M):
            if locator.startswith('package:'):
                alias, _, member = locator[len('package:'):].partition('/')
                owner = next((directory for directory in path.parents if (directory/'package.json').is_file()), None)
                if owner is None:
                    raise ValueError(f'package import outside a package: {path}: {locator}')
                text = re.sub(r'^\s*//.*$', '', (owner/'package.json').read_text(), flags=re.M)
                manifest = json.loads(text)
                dependency = manifest.get('dependencies', {}).get(alias)
                if not isinstance(dependency, dict) or 'path' not in dependency:
                    raise ValueError(f'codec needs a resolved local dependency: {path}: {alias}')
                dependency_root = (owner/dependency['path']).resolve()
                if not member:
                    dependency_text = re.sub(r'^\s*//.*$', '', (dependency_root/'package.json').read_text(), flags=re.M)
                    member = json.loads(dependency_text)['lib']['path']
                target = (dependency_root/member).resolve()
                if not target.is_relative_to(dependency_root):
                    raise ValueError(f'package import escapes dependency: {locator}')
                visit(target)
            else:
                visit(path.parent / locator)
    visit(entry)
    return seen


def generate(root=ROOT, mir=False, interface=False, bodies=False):
    owners = {}
    model_sources = [
        'sev_compiler/frontend/modules/graph/model.sev',
        'sev_compiler/frontend/modules/graph/queries.sev',
        'sev_compiler/frontend/modules/graph/verify.sev',
        'sev_compiler/hir/module/hierarchy.sev',
        'sev_compiler/hir/module/module.sev',
        'sev_compiler/mir/cfg/availability.sev',
        'sev_compiler/mir/cfg/block.sev',
        'sev_compiler/mir/cfg/branch.sev',
        'sev_compiler/mir/cfg/cfg.sev',
        'sev_compiler/mir/cfg/control_flow.sev',
        'sev_compiler/mir/cfg/dominance.sev',
        'sev_compiler/mir/cfg/effects.sev',
        'sev_compiler/mir/cfg/lazy.sev',
        'sev_compiler/mir/value.sev',
        'sev_compiler/syntax/block/block.sev',
        'sev_compiler/syntax/constraint/dispatch.sev',
        'sev_compiler/syntax/constraint/graph.sev',
        'sev_compiler/syntax/constraint/model.sev',
        'sev_compiler/syntax/constraint/proof.sev',
        'sev_compiler/syntax/constraint/refinement.sev',
        'sev_compiler/syntax/diagnostics/diagnostics/src/lib.sev',
        'sev_compiler/syntax/function/function.sev',
        'sev_compiler/syntax/function/lowering.sev',
        'sev_compiler/syntax/function/syntax.sev',
        'sev_compiler/syntax/function/trait.sev',
        'sev_compiler/syntax/grammar/contracts.sev',
        'sev_compiler/syntax/grammar/grammar.sev',
        'sev_compiler/syntax/lexeme.sev',
        'sev_compiler/syntax/literal/bool.sev',
        'sev_compiler/syntax/literal/lowered.sev',
        'sev_compiler/syntax/literal/value.sev',
        'sev_compiler/syntax/lowered.sev',
        'sev_compiler/syntax/primitive/array/array.sev',
        'sev_compiler/syntax/primitive/array/storage.sev',
        'sev_compiler/syntax/primitive/bool/bool.sev',
        'sev_compiler/syntax/primitive/bytes/bytes.sev',
        'sev_compiler/syntax/primitive/catalog.sev',
        'sev_compiler/syntax/primitive/char/char.sev',
        'sev_compiler/syntax/primitive/char/encoding.sev',
        'sev_compiler/syntax/primitive/char/units.sev',
        'sev_compiler/syntax/primitive/char/utf8.sev',
        'sev_compiler/syntax/primitive/contract.sev',
        'sev_compiler/syntax/primitive/declarations.sev',
        'sev_compiler/syntax/primitive/enum/enum.sev',
        'sev_compiler/syntax/primitive/error/error.sev',
        'sev_compiler/syntax/primitive/float/float.sev',
        'sev_compiler/syntax/primitive/int/int.sev',
        'sev_compiler/syntax/primitive/iterator/iterator.sev',
        'sev_compiler/syntax/primitive/literal.sev',
        'sev_compiler/syntax/primitive/mlir.sev',
        'sev_compiler/syntax/primitive/number.sev',
        'sev_compiler/syntax/primitive/numeric/power.sev',
        'sev_compiler/syntax/primitive/pointer/pointer.sev',
        'sev_compiler/syntax/primitive/string/core.sev',
        'sev_compiler/syntax/primitive/string/format.sev',
        'sev_compiler/syntax/primitive/string/intrinsics.sev',
        'sev_compiler/syntax/primitive/string/methods.sev',
        'sev_compiler/syntax/primitive/string/storage.sev',
        'sev_compiler/syntax/primitive/string/string.sev',
        'sev_compiler/syntax/primitive/tuple/tuple.sev',
        'sev_compiler/syntax/primitive/units/units.sev',
        'sev_compiler/syntax/sentence/sentence.sev',
        'sev_compiler/syntax/source.sev',
        'sev_compiler/syntax/symbol/symbol.sev',
        'sev_compiler/syntax/token.sev',
        'sev_compiler/syntax/type/annotation.sev',
        'sev_compiler/syntax/type/integer_spelling.sev',
        'sev_compiler/syntax/type/mlir.sev',
        'sev_compiler/syntax/type/scalar_conversion.sev',
        'sev_compiler/syntax/type/slice_bounds.sev',
        'sev_compiler/syntax/type/storage.sev',
        'sev_compiler/syntax/type/system.sev',
        'sev_compiler/syntax/type/tagged.sev',
        'sev_compiler/syntax/type/type.sev',
    ]
    for relative in model_sources:
        path = root / relative
        namespace = "model_" + re.sub(r"\W+", "_", str(Path(relative).relative_to("sev_compiler").with_suffix("")))
        for name in re.findall(r"^(?:class|enum) (\w+)", path.read_text(), re.M):
            owners[name] = {"path": relative, "namespace": namespace}
    paths_by_namespace = {}
    for owner in owners.values():
        paths_by_namespace[owner['namespace']] = [root / owner['path']]
    models = {}
    inputs = {}
    paths_by_namespace.update({
        'lexer': exported_sources(root/'sev_compiler/frontend/lexer/lexer/src/lib.sev'),
        'source': [root/'sev_compiler/frontend/source/source.sev'],
        'definitions': [root/'sev_compiler/frontend/semantic/semantic/src/definitions.sev'],
        'semantic': [root/'sev_compiler/frontend/semantic/semantic/src/callable.sev'],
    })
    for ns, paths in paths_by_namespace.items():
        for path in sorted(paths):
            contents = path.read_text()
            inputs[str(path.relative_to(root))] = contents
            name = None
            for line in contents.splitlines():
                match = re.match(r'^(class|enum) (\w+)(?::.*)?$', line)
                if match:
                    name = ns + '.' + match[2]
                    models[name] = (match[1], [], ns)
                    continue
                if line and not line.startswith((' ', '#')): name = None
                if name and re.match(r'^    \w', line):
                    if models[name][0] == 'class':
                        field = re.match(r'    (\w+): (.+?)(?: = .*)?$', line)
                        if field and field[1] not in ({'model_syntax_function_function.FunctionDeclaration': {'binary_requirements', 'capability_requirements'}, 'model_hir_module_module.Module': {'binding_values'}} if mir else SKIP).get(name, set()):
                            if name == 'source.SourceFile' and field[1] in {'characters', 'byte_offsets'}: continue
                            models[name][1].append((field[1], field[2]))
                    else:
                        variant = re.fullmatch(r'    (\w+)(?:\((.*)\))?', line)
                        if variant:
                            fields = [tuple(v.split(': ', 1)) for v in split(variant[2])] if variant[2] else []
                            models[name][1].append((variant[1], fields))

    def resolve(type, ns):
        type = type.strip()
        if ' | ' in type: return ' | '.join(resolve(t, ns) for t in type.split(' | '))
        if '[' in type:
            base, rest = type.split('[', 1)
            return base + '[' + ', '.join(resolve(t, ns) for t in split(rest[:-1])) + ']'
        if type in ('int', 'bool', 'string', 'float', 'None', 'absent') or re.fullmatch('[uif][0-9]+', type): return type
        if '.' in type:
            if type in models:
                return type
            member = type.rsplit('.', 1)[1]
            if member in owners:
                return owners[member]['namespace'] + '.' + member
            raise ValueError(f'unresolved archive model {type} from {ns}')
        if ns + '.' + type in models: return ns + '.' + type
        if type in ('SourceId', 'Span', 'SourceFile'): return 'source.' + type
        if type in owners: return owners[type]['namespace'] + '.' + type
        if 'lexer.' + type in models: return 'lexer.' + type
        raise ValueError((ns, type))

    types, functions, optional_classes = set(), [], []
    def key(type): return re.sub(r'\W+', '_', type).strip('_')
    def ensure(type):
        if type in types: return
        types.add(type)
        name = key(type)
        encode = [f'def put_{name}(output: list[string], value: {type}):']
        result_type = 'ArchiveOption_' + name if ' | ' in type else type
        decode = [f'def get_{name}(input: ArchiveInput) -> {result_type} | Error:']
        def put(type, value):
            ensure(type)
            return f'put_{key(type)}(output, {value})'
        def get(type):
            ensure(type)
            return f'get_{key(type)}(input)'
        def field(type, index, indent='    '):
            call = get(type)
            if ' | ' in type:
                return [f'{indent}optional_{index} = {call}', f'{indent}field_{index} ?= optional_{index}.value']
            return [f'{indent}field_{index} = {call}']
        if type == 'string':
            encode += ['    output.append(package.toml_quote(value))']
            decode += ['    return package.unquote(take(input))']
        elif type == 'bool':
            encode += ['    output.append("1" if value else "0")']
            decode += ['    value = take(input)', '    if value not in ["0", "1"]:', '        throw Error("invalid frontend archive boolean")', '    return value == "1"']
        elif type in ('int', 'float') or re.fullmatch('[uif][0-9]+', type):
            encode += ['    output.append(string(value))']
            decode += [f'    return {type}(take(input))']
        elif ' | ' in type:
            child, empty = type.split(' | ')
            optional_classes.append(f'class {result_type}:\n    value: {type}')
            assert empty in ('None', 'absent'), type
            encode += ['    supplied ?= value', f'    if supplied != {empty}:', '        output.append("1")', '        ' + put(child, 'supplied'), '    else:', '        output.append("0")']
            decode += ['    tag = take(input)', '    if tag == "0":', f'        return {result_type}({empty})', '    if tag != "1":', '        throw Error("invalid frontend archive optional tag")', f'    return {result_type}(' + get(child) + ')']
        elif type.startswith('list['):
            child = type[5:-1]
            encode += ['    output.append(string(len(value)))', '    for element in borrow value:', '        ' + put(child, 'element')]
            decode += [f'    result: {type} = []', '    count = archive_count(input)', '    for index in range(count):', '        result.append(' + get(child) + ')', '    return result']
        elif type.startswith('tuple['):
            children = split(type[6:-1])
            encode += ['    ' + put(t, f'value[{i}]') for i,t in enumerate(children)]
            for i,t in enumerate(children): decode += field(t, i)
            decode += ['    return (' + ', '.join(f'field_{i}' for i in range(len(children))) + ')']
        else:
            kind, fields, ns = models[type]
            if kind == 'class':
                fields = [(n, resolve(t, ns)) for n,t in fields]
                encode += ['    ' + put(t, 'value.'+n) for n,t in fields]
                for i,(n,t) in enumerate(fields): decode += field(t, i)
                args = ', '.join(('' if type == 'source.SourceFile' else n+'=')+f'field_{i}' for i,(n,t) in enumerate(fields))
                if interface and type == 'source.SourceId':
                    decode += ['    if len(input.source_ids) > 0:',
                               '        if field_0 >= len(input.source_ids):',
                               '            throw Error("semantic interface has a dangling source identity")',
                               '        return source.SourceId(index=u32(input.source_ids[field_0]))']
                decode += [f'    return {type}({args})']
            else:
                encode += ['    match value:']
                decode += ['    tag = archive_count(input)']
                for i,(variant, fields) in enumerate(fields):
                    fields = [(n,resolve(t,ns)) for n,t in fields]
                    encode += [f'        case {variant}:', f'            output.append("{i}")']
                    encode += ['            '+put(t,n) for n,t in fields]
                    decode += [f'    if tag == {i}:']
                    for j,(n,t) in enumerate(fields): decode += field(t, j, '        ')
                    decode += [f'        return {type}.{variant}' + ('('+', '.join(f'field_{j}' for j in range(len(fields)))+')' if fields else '')]
                decode += ['    throw Error("invalid frontend archive variant: '+type+'")']
        functions.append('\n'.join(encode)+'\n\n\n'+'\n'.join(decode))

    contracts = ('model_hir_module_module.Module', 'lexer.SyntaxRegistry', 'list[source.SourceFile]', 'list[string]') if interface else (('model_hir_module_module.Module',) if mir else ('model_hir_module_module.Module', 'lexer.SyntaxRegistry', 'list[source.SourceFile]', 'list[semantic.PreludeAnalysis]'))
    if bodies:
        contracts = ('model_syntax_block_block.Block',)
    for contract in contracts: ensure(contract)
    body = '\n\n\n'.join(optional_classes + functions)
    schema = hashlib.sha256(body.encode()).hexdigest()
    header = '''# Generated by sev_compiler/build/frontend_codec.py during the package build.
import syntax
import lexer
import source
import package
import * from "package:semantic/semantic/src/callable.sev" as semantic
import * from "package:semantic/semantic/src/definitions.sev" as definitions


class ArchiveInput:
    lines: list[string]
    cursor: int = 0
    source_ids: list[int] = []


def take(input: ArchiveInput) -> string | Error:
    if input.cursor >= len(input.lines):
        throw Error("truncated frontend archive")
    value = input.lines[input.cursor]
    input.cursor += 1
    return value


def archive_count(input: ArchiveInput) -> int | Error:
    value = int(take(input))
    if value < 0 or value > len(input.lines):
        throw Error("invalid frontend archive count")
    return value


'''
    if bodies:
        header += r'''def archive_quote(value: string) -> string:
    return value.replace("\\", "\\\\").replace("\n", "\\n").replace("\r", "\\r")


def archive_unquote(value: string) -> string | Error:
    output: list[string] = []
    escaped := false
    for character in value.characters():
        part = string(character)
        if escaped:
            if part not in ["n", "r", "\\"]:
                throw Error("invalid body archive escape")
            output.append("\n" if part == "n" else ("\r" if part == "r" else "\\"))
            escaped = false
        elif part == "\\":
            escaped = true
        else:
            output.append(part)
    if escaped:
        throw Error("truncated body archive escape")
    return output.join("")


'''
    identity = hashlib.sha256(Path(__file__).read_bytes())
    for name, contents in sorted(inputs.items()):
        identity.update(name.encode() + b'\0' + contents.encode() + b'\0')
    identity.update(b'mir' if mir else b'frontend')
    build = root / ('library/package/compile/interface/package.pkg/build' if interface else 'sev_compiler/driver/package.pkg/build')
    if bodies:
        build = root / 'sev_compiler/frontend/semantic/package.pkg/build'
    output = build / identity.hexdigest() / 'staging/frontend_archive.sev'
    directory = output.with_suffix('')
    semantic = root / 'sev_compiler/frontend/semantic/semantic/src'
    header = header.replace('../../../../frontend/semantic/src',
                            os.path.relpath(semantic, output.parent))
    def model_imports(directory):
        lines = []
        for namespace, paths in sorted(paths_by_namespace.items()):
            if namespace in ('source', 'lexer', 'semantic', 'definitions'):
                continue
            if not any(namespace + '.' in type for type in types):
                continue
            path = next(iter(paths))
            lines.append('import * from "' + os.path.relpath(path, directory) + '" as ' + namespace)
        return '\n'.join(lines) + '\n'
    header = model_imports(output.parent) + header
    chunks, chunk, lines = [], [], 0
    for function in functions:
        count = len(function.splitlines()) + 3
        if chunk and lines + count > 650:
            chunks.append(chunk)
            chunk, lines = [], 0
        chunk.append(function)
        lines += count
    if chunk: chunks.append(chunk)
    imports = []
    outputs = {}
    for index, chunk in enumerate(chunks):
        name = f'part_{index:02}.sev'
        outputs[directory/name] = (model_imports(directory) + '# Generated by sev_compiler/build/frontend_codec.py.\n'
                                   'import syntax\nimport lexer\nimport source\nimport package\n'
                                   f'import * from "package:semantic/semantic/src/callable.sev" as semantic\n'
                                   f'import * from "package:semantic/semantic/src/definitions.sev" as definitions\n'
                                   'import * from "../frontend_archive.sev"\n\n\n' +
                                   '\n\n\n'.join(chunk) + '\n')
        imports.append(f'import * from "frontend_archive/{name}"')
    outputs[output] = (header+f'def schema() -> string:\n    return "{schema}"\n\n\n'+
                      '\n\n\n'.join(optional_classes)+'\n\n'+ '\n'.join(imports)+'\n')
    # Publish the entry last: readers only discover complete generations. Keep
    # unchanged mtimes and repair missing/corrupt outputs before cache checks.
    outputs[build/'frontend_archive.sev'] = (
        '# Generated frontend codec entry.\n'
        f'import * from "{output.relative_to(build).as_posix()}"\n')
    if mir:
        outputs = {Path(str(p).replace('frontend_archive', 'mir_archive')): text.replace('frontend_archive', 'mir_archive') for p, text in outputs.items()}
    if interface:
        outputs = {Path(str(p).replace('frontend_archive', 'semantic_archive')): text.replace('frontend_archive', 'semantic_archive') for p, text in outputs.items()}
    if bodies:
        outputs = {Path(str(p).replace('frontend_archive', 'body_archive')): '\n'.join(line for line in text.replace('frontend_archive', 'body_archive').split('\n') if not line.startswith('import * from \"package:semantic/') and line != 'import package') for p, text in outputs.items()}
    if bodies:
        outputs = {p: text.replace('package.toml_quote', 'archive_quote').replace('package.unquote', 'archive_unquote') for p, text in outputs.items()}
    build.mkdir(parents=True, exist_ok=True)
    with (build/'frontend_archive.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        for destination, contents in outputs.items():
            if destination.is_file() and destination.read_bytes() == contents.encode():
                continue
            destination.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(mode='w', dir=destination.parent,
                                             delete=False) as temporary:
                temporary.write(contents)
            os.replace(temporary.name, destination)
    return output

if __name__ == '__main__':
    generate()
    generate(mir=True)
    generate(interface=True)
    generate(bodies=True)
