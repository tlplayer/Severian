"""Label calls from unambiguous source signatures before dependency checking."""
from pathlib import Path
import json
from py_compiler.frontend.source.source import SourceFile
from py_compiler.frontend.lint.rules import SourceEdit, apply_edits, policy
from py_compiler.frontend.lint.import_names import statements, import_contract, module_level


_source_rows = {}


def source_rows(path, syntax):
    path = Path(path)
    if not path.is_file() or path.suffix != '.sev':
        return None, ()
    text = path.read_text()
    previous = _source_rows.get(path)
    if previous is not None and previous[0].text == text:
        return previous
    source = SourceFile(str(path), text, str(path))
    result = source, tuple(statements(source, syntax))
    _source_rows[path] = result
    return result


def package_entry(source, alias):
    origin = Path(source.debug_path or source.path).resolve()
    for directory in origin.parents:
        manifest = directory / 'package.json'
        if not manifest.is_file():
            continue
        data = json.loads(manifest.read_text())
        dependency = data.get('dependencies', {}).get(alias)
        if not isinstance(dependency, dict) or 'path' not in dependency:
            continue
        root = (directory / dependency['path']).resolve()
        if root.is_file():
            return root
        package = root / 'package.json'
        if package.is_file():
            spec = json.loads(package.read_text())
            entry = spec.get('lib', {}).get('path', 'src/lib.sev')
            return root / entry
        return root / 'src/lib.sev'
    return None


def parameters(items):
    from py_compiler.frontend.lint.layout import groups, split_at
    pair = next(((opening, closing) for opening, closing in groups(items) if items[opening].text == '('), None)
    if pair is None:
        return None
    names, required = [], 0
    for group in split_at(items[pair[0] + 1:pair[1]], {','}):
        if not group:
            continue
        if group[0].kind != 'IDENTIFIER' or (len(group) > 1 and group[1].text == '...'):
            return None
        names.append(group[0].text)
        if not any(token.text == '=' for token in group):
            required += 1
    return tuple(names), required


def declared_contract(source, rows, name):
    from py_compiler.frontend.lint.names import canonical_name
    from py_compiler.syntax.grammar.continuation import indentation
    candidates = []
    for index, items in enumerate(rows):
        if not module_level(source, items) or len(items) < 2:
            continue
        if items[0].text == 'def' and items[1].text == name:
            candidates.append(parameters(items))
        elif items[0].text == 'class' and canonical_name(items[1].text, 'type') == canonical_name(name, 'type'):
            fields, required, constructors = [], 0, []
            for child in rows[index + 1:]:
                depth = indentation(source, child[0])
                if depth == 0:
                    break
                if depth != policy()['indentation']:
                    continue
                if len(child) > 1 and child[0].text == 'def' and child[1].text == items[1].text:
                    contract = parameters(child)
                    if contract is not None and contract[0] and contract[0][0] == 'self':
                        contract = contract[0][1:], max(0, contract[1] - 1)
                    constructors.append(contract)
                elif len(child) > 1 and child[0].kind == 'IDENTIFIER' and child[1].text == ':':
                    fields.append(child[0].text)
                    required += not any(token.text in ('=', ':=') for token in child)
            candidates.extend(constructors or [(tuple(fields), required)])
    if candidates and all(candidate is not None for candidate in candidates):
        orders = {candidate[0] for candidate in candidates}
        if len(orders) == 1:
            return next(iter(orders)), min(candidate[1] for candidate in candidates)
    return None


def exported_contract(path, name, syntax, active=None):
    active = set() if active is None else active
    key = (Path(path), name)
    if key in active:
        return None
    active.add(key)
    source, rows = source_rows(path, syntax)
    if source is None:
        return None
    direct = declared_contract(source, rows, name)
    if direct is not None:
        return direct
    for items in rows:
        if not module_level(source, items):
            continue
        contract = import_contract(source, items)
        if contract:
            target, names, _ = contract
            from py_compiler.frontend.lint.names import canonical_name
            exported = next((token.text for token in names if token.text == name or canonical_name(token.text, 'type') == canonical_name(name, 'type')), None)
            if exported:
                result = exported_contract(target, exported, syntax, active)
                if result is not None:
                    return result
        elif any(token.text == 'as' for token in items):
            position = next(i for i, token in enumerate(items) if token.text == 'as')
            if position + 1 < len(items) and items[position + 1].text == name:
                return exported_contract(path, items[0].text, syntax, active)
    return None


def label_declared_arguments(source, syntax):
    from py_compiler.frontend.lint.layout import groups, split_at
    rows = tuple(statements(source, syntax))
    imports, namespaces, definitions, shadowed = {}, {}, set(), set()
    for items in rows:
        if items[0].text in ('def', 'operator', 'grammar'):
            pair = next(((a, b) for a, b in groups(items) if items[a].text == '('), None)
            if pair:
                definitions.add(items[pair[0]].span.start)
                for part in split_at(items[pair[0] + 1:pair[1]], {','}):
                    if part and part[0].kind == 'IDENTIFIER':
                        shadowed.add(part[0].text)
        if len(items) > 1 and items[0].kind == 'IDENTIFIER' and items[1].text in (':', '=', ':='):
            shadowed.add(items[0].text)
        if not module_level(source, items):
            continue
        contract = import_contract(source, items)
        if contract:
            path, names, namespace = contract
            for name in names:
                imports[name.text] = path
            if namespace:
                namespaces[namespace.text] = path
        elif items[0].text == 'import' and len(items) == 2 and items[1].kind == 'IDENTIFIER':
            path = package_entry(source, items[1].text)
            if path:
                namespaces[items[1].text] = path
        elif items[0].text == 'from' and len(items) > 3 and items[1].kind == 'IDENTIFIER':
            path = package_entry(source, items[1].text)
            if path:
                for token in items[3:]:
                    if token.kind == 'IDENTIFIER':
                        imports[token.text] = path
    edits, local_contracts, imported_contracts = [], {}, {}
    for items in rows:
        if items[0].text in ('grammar', 'import', 'from'):
            continue
        for opening, closing in groups(items):
            if opening == 0 or items[opening].text != '(' or items[opening].span.start in definitions:
                continue
            callee = items[opening - 1]
            if callee.kind != 'IDENTIFIER' or callee.text in shadowed:
                continue
            arguments = [part for part in split_at(items[opening + 1:closing], {','}) if part]
            named = any(len(part) > 1 and part[1].text == '=' for part in arguments)
            if len(arguments) <= policy()['positional_arguments'] and not named:
                continue
            if any(part[0].text == '...' for part in arguments):
                continue
            if opening > 2 and items[opening - 2].text == '.':
                path = namespaces.get(items[opening - 3].text)
                key = (path, callee.text)
                if key not in imported_contracts:
                    imported_contracts[key] = exported_contract(path, callee.text, syntax) if path else None
                contract = imported_contracts[key]
            else:
                if callee.text not in local_contracts:
                    local_contracts[callee.text] = declared_contract(source, rows, callee.text)
                contract = local_contracts[callee.text]
                if contract is None and callee.text in imports:
                    key = (imports[callee.text], callee.text)
                    if key not in imported_contracts:
                        imported_contracts[key] = exported_contract(*key, syntax)
                    contract = imported_contracts[key]
            if contract is None:
                continue  # Typed call resolution owns ambiguous/opaque callables.
            names, required = contract
            if not required <= len(arguments) <= len(names):
                continue
            positional = 0
            for part in arguments:
                if len(part) > 1 and part[1].text == '=':
                    from py_compiler.frontend.lint.names import canonical_name
                    matches = [name for name in names if canonical_name(name, 'value') == canonical_name(part[0].text, 'value')]
                    if len(matches) == 1 and matches[0] != part[0].text:
                        edits.append(SourceEdit(part[0].span, matches[0], 'G.call.arguments'))
                    continue
                name = names[positional]
                positional += 1
                edits.append(SourceEdit(source.span(part[0].span.start, part[0].span.start), name + ' = ', 'G.call.arguments'))
    return apply_edits(source, edits)


import unittest


class DeclaredArgumentTests(unittest.TestCase):
    def test_source_window_constructor_is_labeled_without_a_package_build(self):
        from py_compiler.syntax.recognition import Syntax
        text = 'class SourceWindow:\n    source: string\n    start: int\n    end: int\n    cursor: int\ndef capture():\n    return SourceWindow("input", 0, 10, 0)\n'
        result = label_declared_arguments(SourceFile('window.sev', text), Syntax())
        self.assertIn('SourceWindow(source = "input", start = 0, end = 10, cursor = 0)', result)

    def test_variadic_and_ambiguous_overloads_keep_typed_resolution(self):
        from py_compiler.syntax.recognition import Syntax
        text = 'def collect(values...):\n    pass\ndef choose(a: int,b: int,c: int,d: int):\n    pass\ndef choose(w: bool,x: bool,y: bool,z: bool):\n    pass\ncollect(1,2,3,4)\nchoose(1,2,3,4)\n'
        self.assertEqual(label_declared_arguments(SourceFile('calls.sev', text), Syntax()), text)
