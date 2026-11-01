"""Resolve source type spellings through declarations and re-exported imports."""
from pathlib import Path
import ast
from py_compiler.frontend.source.source import SourceFile, Diagnostic
from py_compiler.frontend.lint.rules import SourceEdit, apply_edits


def statements(source, syntax):
    from py_compiler.frontend.lint.layout import scan
    from py_compiler.syntax.grammar.continuation import continues_statement, next_term
    current, depth = [], 0
    tokens = scan(source, syntax)
    for index, token in enumerate(tokens):
        if token.kind in ('INDENT', 'TRIVIA'):
            continue
        if token.kind == 'NEWLINE':
            if depth == 0 and current and not continues_statement(source, current, next_term(tokens, index)):
                yield tuple(current)
                current = []
            continue
        current.append(token)
        depth += (token.text in ('(', '[', '{')) - (token.text in (')', ']', '}'))
    if current:
        yield tuple(current)


def module_level(source, items):
    start = source.text.rfind('\n', 0, items[0].span.start) + 1
    return not source.text[start:items[0].span.start]


def separate_import_statements(source, syntax):
    """Reserved import/from keywords identify adjacent joined declarations."""
    edits = []
    for items in statements(source, syntax):
        if items[0].text not in ('from', 'import'):
            continue
        start, depth = 0, 0
        line_start = source.text.rfind('\n', 0, items[0].span.start) + 1
        prefix = source.text[line_start:items[0].span.start]
        for index, token in enumerate(items):
            if depth == 0 and index > start:
                new_from = token.text == 'from' and index + 2 < len(items) and items[index + 2].text == 'import'
                new_import = token.text == 'import' and not (items[start].text == 'from' and index == start + 2)
                if new_from or new_import:
                    gap = source.span(items[index - 1].span.end, token.span.start)
                    edits.append(SourceEdit(gap, '\n' + prefix, 'G.import.boundary'))
                    start = index
            depth += (token.text in ('(', '[', '{')) - (token.text in (')', ']', '}'))
    return SourceFile(source.path, apply_edits(source, edits), source.debug_path)


def import_contract(source, items):
    from py_compiler.syntax.block.imports_layout import normalize_import_tokens
    items = normalize_import_tokens(items)
    if items[0].text == 'from' and len(items) > 3 and items[1].kind == 'STRING':
        locator, names, namespace = items[1], items[3:], None
    elif items[0].text == 'import' and len(items) > 1:
        position = next((i for i, token in enumerate(items) if token.text == 'from'), None)
        if position is not None and position + 1 < len(items) and items[position + 1].kind == 'STRING':
            locator, names, namespace = items[position + 1], items[1:position], None
        elif items[1].kind == 'STRING':
            locator, names = items[1], ()
            namespace = items[3] if len(items) == 4 and items[2].text == 'as' else None
        else:
            return None
    else:
        return None
    base = Path(source.debug_path or source.path).resolve().parent
    path = (base / ast.literal_eval(locator.text)).resolve()
    return path, tuple(token for token in names if token.kind == 'IDENTIFIER'), namespace


def direct_types(source, rows):
    from py_compiler.frontend.lint.names import canonical_name
    exports, declarations = {}, {}
    for items in rows:
        if module_level(source, items) and items[0].text in ('class', 'trait', 'enum', 'union') and len(items) > 1:
            token = items[1]
            canonical = canonical_name(token.text, 'type')
            for previous, target in exports.items():
                if previous != token.text and target == canonical:
                    raise Diagnostic('lint', f'name {token.text!r} would collide with {previous!r}', source, token.span)
            exports[token.text] = canonical
            declarations[token.span.start] = canonical
    return exports, declarations


def exported_type(name, exports):
    from py_compiler.frontend.lint.names import canonical_name
    if name in exports:
        return exports[name]
    matches = {target for spelling, target in exports.items()
               if canonical_name(name, 'type') == canonical_name(spelling, 'type')}
    return next(iter(matches)) if len(matches) == 1 else None


def source_type_exports(path, syntax, cache, active):
    path = Path(path)
    if not path.is_file() or path.suffix != '.sev':
        return {}
    if path in cache:
        return cache[path]
    from py_compiler.frontend.lint.call_names import source_rows
    source, rows = source_rows(path, syntax)
    exports, _ = direct_types(source, rows)
    cache[path] = exports
    if path in active:
        return exports
    active.add(path)
    for items in rows:
        if not module_level(source, items):
            continue
        contract = import_contract(source, items)
        if contract:
            target, names, _ = contract
            public = source_type_exports(target, syntax, cache, active)
            for name in names:
                canonical = exported_type(name.text, public)
                if canonical:
                    exports[name.text] = canonical
        elif len(items) == 3 and items[1].text == 'as':
            canonical = exported_type(items[0].text, exports)
            if canonical:
                if items[2].text == canonical and items[0].text != canonical:
                    exports[items[0].text] = items[0].text
                    exports[items[2].text] = canonical
                else:
                    exports[items[2].text] = items[2].text
    active.remove(path)
    return exports


def reconcile_type_names(source, syntax, packages=None):
    """Fix imports and their references together, including barrel modules."""
    from py_compiler.frontend.lint.layout import scan, groups
    rows = list(statements(source, syntax))
    bindings, declarations = direct_types(source, rows)
    cache, active, namespaces, imports, protected, edits = {}, set(), {}, [], set(), []
    alias_targets = {}
    for items in rows:
        if not module_level(source, items):
            continue
        contract = import_contract(source, items)
        if contract is None and items[0].text in ('from', 'import') and len(items) > 1 and items[1].kind == 'IDENTIFIER':
            alias = items[1].text
            if packages is None or alias not in packages:
                from py_compiler.frontend.lint.call_names import package_entry
                path = package_entry(source, alias)
                if path:
                    contract = (path, tuple(token for token in items[3:] if token.kind == 'IDENTIFIER'), None) if items[0].text == 'from' else (path, (), items[1])
        if contract:
            path, names, namespace = contract
            public = source_type_exports(path, syntax, cache, active)
            if namespace:
                namespaces[namespace.text] = public
                protected.add(namespace.span.start)
            for name in names:
                canonical = exported_type(name.text, public)
                if canonical:
                    bindings[name.text] = canonical
                    declarations[name.span.start] = canonical
                    imports.append((items, name.text, canonical))
        elif len(items) == 3 and items[1].text == 'as':
            left, _, right = items
            protected.add(right.span.start)
            target = exported_type(left.text, bindings)
            if target and right.text == target and left.text != target:
                edits.append(SourceEdit(right.span, left.text, 'source.alias.identity'))
                alias_targets[right.text] = target
            else:
                alias_targets[right.text] = right.text
    bindings.update(alias_targets)
    # An explicit compatibility alias retains its declared spelling.
    alias_names = set(alias_targets)
    shadows = set()
    for items in rows:
        if len(items) > 1 and items[0].kind == 'IDENTIFIER' and items[1].text in (':', '=', ':='):
            shadows.add(items[0].text)
        if items[0].text in ('def', 'operator', 'grammar'):
            for opening, closing in groups(items):
                if items[opening].text == '(':
                    for index in range(opening + 1, closing - 1):
                        if items[index].kind == 'IDENTIFIER' and items[index + 1].text == ':':
                            shadows.add(items[index].text)
                    break
    compatibility = {}
    for items, old, canonical in imports:
        if old in shadows and old != canonical and old not in alias_names:
            compatibility.setdefault(items[-1].span.end, []).append(canonical + ' as ' + old)
    for offset, aliases in compatibility.items():
        edits.append(SourceEdit(source.span(offset, offset), '\n' + '\n'.join(aliases), 'source.import.compatibility'))
    tokens = [token for token in scan(source, syntax) if token.kind not in ('TRIVIA', 'INDENT', 'NEWLINE')]
    for index, token in enumerate(tokens):
        if token.kind != 'IDENTIFIER' or token.span.start in protected:
            continue
        target = declarations.get(token.span.start)
        if target is None and index and tokens[index - 1].text == '.':
            public = namespaces.get(tokens[index - 2].text, {}) if index > 1 else {}
            target = exported_type(token.text, public)
        elif target is None and token.text not in shadows:
            target = bindings.get(token.text)
        if target and target != token.text:
            edits.append(SourceEdit(token.span, target, 'source.import.identity'))
    return SourceFile(source.path, apply_edits(source, edits), source.debug_path)


import unittest


class SourceTypeNameTests(unittest.TestCase):
    def test_adjacent_import_statements_keep_their_boundaries(self):
        from py_compiler.syntax.recognition import Syntax
        source = SourceFile('imports.sev', 'import os from math import approximate\nfrom util import Range from io import capture_stdout\n')
        result = separate_import_statements(source, Syntax())
        self.assertEqual(result.text, 'import os\nfrom math import approximate\nfrom util import Range\nfrom io import capture_stdout\n')
    def test_reexports_and_namespace_members_follow_the_declared_type(self):
        import tempfile
        from py_compiler.syntax.recognition import Syntax
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'types.sev').write_text('class MLIRTypeBinding:\n    value: int\n')
            (root / 'barrel.sev').write_text('import MlirTypeBinding from "types.sev"\n')
            text = 'import MlirTypeBinding from "barrel.sev"\nimport "types.sev" as types\nvalue: MlirTypeBinding\nother: types.MlirTypeBinding\n'
            source = SourceFile(str(root / 'app.sev'), text)
            result = reconcile_type_names(source, Syntax())
            self.assertIn('import MLIRTypeBinding from', result.text)
            self.assertIn('value: MLIRTypeBinding', result.text)
            self.assertIn('other: types.MLIRTypeBinding', result.text)
