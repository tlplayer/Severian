#!/usr/bin/env python3
"""Generate the versioned prelude frontend archive codec from concrete IR models.

The archive preserves syntax, diagnostics, and generic bodies; it is not a C ABI.
Transient lowering fields are intentionally absent and retain their defaults.
"""
from pathlib import Path
import re
import hashlib
import fcntl
import os
import tempfile

ROOT = Path(__file__).resolve().parents[2]
SKIP = {
    'universal.Module': {'initializer_cfg', 'cfg_bodies', 'values', 'globals', 'binding_values', 'quality_files', 'quality_exclusions', 'quality_regions', 'debug_files'},
    'universal.FunctionDeclaration': {'binary_requirements', 'capability_requirements', 'cfg'},
    'universal.Block': {'operations', 'lowered_operations'},
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
            visit(path.parent / locator)
    visit(entry)
    return seen


def generate(root=ROOT):
    models = {}
    inputs = {}
    for ns, paths in [('universal', exported_sources(root/'sev_compiler/universal/src/lib.sev')),
                      ('lexer', exported_sources(root/'sev_compiler/frontend/lexer/src/lib.sev')),
                      ('source', [root/'sev_compiler/source/src/lib.sev']),
                      ('definitions', [root/'sev_compiler/frontend/semantic/src/definitions.sev']),
                      ('semantic', [root/'sev_compiler/frontend/semantic/src/callable.sev'])]:
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
                        if field and field[1] not in SKIP.get(name, set()):
                            if name == 'source.SourceFile' and field[1] == 'characters': continue
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
        if '.' in type: return type
        if ns + '.' + type in models: return ns + '.' + type
        if 'universal.' + type in models: return 'universal.' + type
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

    for contract in ('universal.Module', 'lexer.SyntaxRegistry', 'list[source.SourceFile]', 'list[semantic.PreludeAnalysis]'): ensure(contract)
    body = '\n\n\n'.join(optional_classes + functions)
    schema = hashlib.sha256(body.encode()).hexdigest()
    header = '''# Generated by sev_compiler/build/frontend_codec.py during the package build.
import universal
import lexer
import source
import package
import * from "../../../../frontend/semantic/src/callable.sev" as semantic
import * from "../../../../frontend/semantic/src/definitions.sev" as definitions


class ArchiveInput:
    lines: list[string]
    cursor: int = 0


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
    identity = hashlib.sha256(Path(__file__).read_bytes())
    for name, contents in sorted(inputs.items()):
        identity.update(name.encode() + b'\0' + contents.encode() + b'\0')
    build = root / 'sev_compiler/package.pkg/build'
    output = build / identity.hexdigest() / 'staging/frontend_archive.sev'
    directory = output.with_suffix('')
    semantic = root / 'sev_compiler/frontend/semantic/src'
    header = header.replace('../../../../frontend/semantic/src',
                            os.path.relpath(semantic, output.parent))
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
        outputs[directory/name] = ('# Generated by sev_compiler/build/frontend_codec.py.\n'
                                   'import universal\nimport lexer\nimport source\nimport package\n'
                                   f'import * from "{os.path.relpath(semantic, directory)}/callable.sev" as semantic\n'
                                   f'import * from "{os.path.relpath(semantic, directory)}/definitions.sev" as definitions\n'
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

if __name__ == '__main__': generate()
