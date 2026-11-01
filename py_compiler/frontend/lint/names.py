"""Canonical names are edits of scoped declarations and their resolved references."""
from dataclasses import dataclass, field
from functools import lru_cache
import re
from py_compiler.frontend.lint.rules import SourceEdit, apply_edits, policy
from py_compiler.frontend.source.source import Diagnostic


@lru_cache(maxsize=4096)
def compound_acronyms(word, acronyms):
    lower = word.lower()
    if lower in acronyms or not (word.isupper() or lower.endswith('id')):
        return (word,)
    def split(text):
        if not text:
            return ()
        for acronym in sorted(acronyms, key=len, reverse=True):
            if text.startswith(acronym):
                tail = split(text[len(acronym):])
                if tail is not None:
                    return (acronym, *tail)
        return None
    result = split(lower)
    return result if result is not None and len(result) > 1 else (word,)


def canonical_name(name, role):
    settings = policy()
    if role in ('generic', 'alias') or name in settings['language_types']:
        return name
    prefix = name[:len(name) - len(name.lstrip('_'))]
    suffix = '_' if name.endswith('_') and name.strip('_') else ''
    body = name.strip('_')
    words = re.findall(r'[A-Z]+(?=[A-Z][a-z]|[0-9]|$)|[A-Z]?[a-z]+[0-9]*|[0-9]+', body)
    if not words or ''.join(words).lower() != body.replace('_', '').lower():
        words = body.split('_')
    words = [part for word in words for part in compound_acronyms(word, tuple(settings['acronyms']))]
    words = [word.lower() if word.lower() in settings['allowed_abbreviations'] else
             settings['abbreviations'].get(word.lower(), word.lower()) for word in words]
    if len(words) == 1 and words[0] in settings['acronyms']:
        return prefix + words[0] + suffix
    if role == 'constant':
        result = '_'.join(words).upper()
    elif role in ('type', 'variant'):
        result = ''.join(word.upper() if word in settings['acronyms'] else word[:1].upper() + word[1:]
                         for word in words)
    else:
        result = '_'.join(words)
    return prefix + result + suffix


@dataclass
class NameBinding:
    name: str
    canonical: str
    token: object
    role: str
    type_name: str = ''
    members: object = None
    parameters: dict = field(default_factory=dict)


@dataclass
class NameScope:
    parent: object = None
    owner: object = None
    bindings: dict = field(default_factory=dict)

    def lookup(self, name):
        scope = self
        while scope is not None:
            if name in scope.bindings:
                return scope.bindings[name]
            scope = scope.parent
        scope = self
        while scope is not None:
            matches = [binding for binding in scope.bindings.values()
                       if canonical_name(name, binding.role) == binding.canonical]
            if len(matches) == 1:
                return matches[0]
            if len(matches) > 1:
                return None
            scope = scope.parent
        return None


def fix_names(source, syntax, packages=None):
    from py_compiler.frontend.lint.import_names import reconcile_type_names
    source = reconcile_type_names(source, syntax, packages)
    from py_compiler.frontend.lexer.lexer import lex
    from py_compiler.frontend.parser.blocks import parse_blocks
    tokens = lex(source, syntax)
    root, errors = parse_blocks(source, tokens, syntax)
    if errors:
        return source.text  # The parser reports invalid source; no guessed rename.
    module = NameScope()
    scopes, definitions, classes, edits = {}, {}, {}, []
    imports = {}
    loaded_sources = set()
    pending_aliases = []
    def imported_scope(alias):
        if alias in imports:
            return imports[alias]
        members = NameScope()
        imports[alias] = members
        package = packages.get(alias) if packages is not None else None
        if package is None:
            return members
        def publish(name, role, type_name=''):
            binding = NameBinding(name, name, None, role, type_name)
            members.bindings[name] = binding
            return binding
        for contract in package.get('globals', ()):
            publish(contract['name'], 'constant' if contract['constant'] else 'value', contract['type'])
        for contract in package.get('types', ()):
            binding = publish(contract['name'], 'type')
            binding.members = NameScope(owner=binding)
            for name, annotation in contract.get('fields', ()):
                binding.members.bindings[name] = NameBinding(name, name, None, 'value', annotation)
        for contract in package.get('exports', ()):
            name = contract['name']
            if '.' not in name:
                binding = publish(name, 'function')
                for parameter in contract.get('parameters', ()):
                    binding.parameters[parameter['name']] = NameBinding(parameter['name'], parameter['name'], None, 'value', parameter['type'])
        for contract in package.get('templates', ()):
            if contract.get('name') and '.' not in contract['name']:
                publish(contract['name'], 'type' if contract.get('kind') in ('record', 'trait', 'enum') else 'function')
        return members
    def find_member(members, name):
        if members is None:
            return None
        if name in members.bindings:
            return members.bindings[name]
        candidates = [binding for binding in members.bindings.values()
                      if canonical_name(name, binding.role) == binding.canonical]
        if len(candidates) > 1:
            raise Diagnostic('lint', 'ambiguous imported/member spelling ' + name, source, source.span(0, min(1, len(source.text))))
        if candidates:
            binding = candidates[0]
            return NameBinding(name, binding.canonical, None, binding.role, binding.type_name,
                               binding.members, binding.parameters)
        return None
    def define(scope, token, role='value', type_name=''):
        if token.kind != 'IDENTIFIER':
            return None
        canonical = canonical_name(token.text, role)
        previous = scope.bindings.get(token.text)
        if previous:
            if previous.role != role and previous.canonical != canonical:
                raise Diagnostic('lint', f'{token.text!r} has conflicting naming roles; use distinct declaration names',
                                 source, token.span)
            definitions[(token.span.source, token.span.start)] = previous
            return previous
        for existing in scope.bindings.values():
            if existing.canonical == canonical and existing.name != token.text:
                raise Diagnostic('lint', f'name {token.text!r} would collide with {existing.name!r}', source, token.span)
        binding = NameBinding(token.text, canonical, token, role, type_name)
        scope.bindings[token.text] = binding
        definitions[(token.span.source, token.span.start)] = binding
        return binding
    def split(items):
        from py_compiler.frontend.lint.layout import split_at
        return split_at(list(items), {','})
    def walk(nodes, scope):
        for node in nodes:
            items = node.tokens
            if not items:
                continue
            spelling = items[0].text
            child_scope = scope
            declared_variant = definitions.get((items[0].span.source, items[0].span.start))
            if declared_variant is not None and declared_variant.role == 'variant':
                scopes[node.identity] = scope
                continue
            if spelling in ('class', 'trait', 'enum', 'union') and len(items) > 1:
                binding = define(scope, items[1], 'type')
                child_scope = NameScope(scope, binding)
                if binding:
                    binding.members = child_scope
                    classes.setdefault(binding.name, []).append(binding)
                    if spelling == 'enum':
                        for variant in node.children:
                            if variant.tokens and variant.tokens[0].kind == 'IDENTIFIER':
                                label = define(child_scope, variant.tokens[0], 'variant')
                                if label and label.name not in scope.bindings:
                                    scope.bindings[label.name] = label
            elif spelling in ('def', 'macro', 'operator', 'grammar') and len(items) > 1:
                constructor = scope.owner is not None and scope.owner.name == items[1].text
                binding = define(scope, items[1], 'type' if constructor else 'function')
                child_scope = NameScope(scope, scope.owner)
                opening = next((i for i, token in enumerate(items) if token.text == '('), None)
                if opening is not None:
                    depth, closing = 1, opening + 1
                    while closing < len(items) and depth:
                        depth += (items[closing].text == '(') - (items[closing].text == ')')
                        closing += 1
                    for group in split(items[opening + 1:closing - 1]):
                        if group and group[0].kind == 'IDENTIFIER':
                            annotation = ''
                            if len(group) > 2 and group[1].text == ':':
                                annotation = next((token.text for token in group[2:]
                                                   if token.kind == 'IDENTIFIER' and token.text not in ('view', 'borrow', 'move', 'copy')), '')
                            parameter = define(child_scope, group[0], type_name=annotation)
                            if binding and parameter:
                                binding.parameters[parameter.name] = parameter
                    arrow = next((i for i, token in enumerate(items[closing:], closing) if token.text == '->'), None)
                    if binding and arrow is not None and len(items) > arrow + 1:
                        binding.type_name = items[arrow + 1].text
            elif node.provider is None and any(token.text == 'as' for token in items) and not any(token.text in ('=', ':=') for token in items):
                position = next(i for i, token in enumerate(items) if token.text == 'as')
                if position + 1 < len(items) and items[position + 1].kind == 'IDENTIFIER':
                    name = items[position + 1]
                    binding = NameBinding(name.text, name.text, name, 'alias')
                    previous = scope.bindings.get(name.text)
                    if previous is not None:
                        raise Diagnostic('lint', 'alias must introduce a distinct name: ' + name.text, source, name.span)
                    scope.bindings[name.text] = binding
                    definitions[(name.span.source, name.span.start)] = binding
                    pending_aliases.append((binding, scope, items[0].text))
            elif getattr(node.provider, 'scope_kind', None) == 'test':
                child_scope = NameScope(scope, scope.owner)
            elif spelling in ('import', 'from'):
                # Import spellings identify external modules; they are not local
                # declarations to rename without updating the import contract.
                for token in items[1:]:
                    if token.kind == 'IDENTIFIER':
                        binding = NameBinding(token.text, token.text, token, 'import')
                        if spelling == 'import' or token in items[3:]:
                            scope.bindings.setdefault(token.text, binding)
                        definitions[(token.span.source, token.span.start)] = binding
                if spelling == 'import' and len(items) == 2:
                    scope.bindings[items[1].text].members = imported_scope(items[1].text)
                elif spelling == 'from' and len(items) > 3 and items[1].kind == 'IDENTIFIER':
                    members = imported_scope(items[1].text)
                    for token in items[3:]:
                        if token.kind == 'IDENTIFIER':
                            binding = find_member(members, token.text)
                            if binding:
                                scope.bindings[token.text] = binding
                                definitions[(token.span.source, token.span.start)] = binding
                elif spelling == 'from' and len(items) > 3 and items[1].kind == 'STRING':
                    import ast
                    from pathlib import Path
                    from py_compiler.frontend.source.source import SourceFile
                    from py_compiler.frontend.lint.layout import format_source
                    origin = items[0].lexeme.source
                    base = Path(origin.debug_path or origin.path).resolve().parent
                    path = (base / ast.literal_eval(items[1].text)).resolve()
                    if path.is_file() and path.suffix == '.sev' and path.is_relative_to(base) and path not in loaded_sources:
                        loaded_sources.add(path)
                        imported = SourceFile(str(path), path.read_text(), str(path))
                        imported = SourceFile(str(path), format_source(imported, syntax), str(path))
                        imported_root, failures = parse_blocks(imported, lex(imported, syntax), syntax)
                        if not failures:
                            walk(imported_root.children, scope)
            else:
                if spelling == 'for':
                    end = next((i for i, token in enumerate(items) if token.text == 'in'), 1)
                    for token in items[1:end]:
                        if token.kind == 'IDENTIFIER':
                            define(scope, token)
                elif spelling == 'catch' and len(items) > 1:
                    define(scope, items[1])
                elif items[0].kind == 'IDENTIFIER' and len(items) > 1 and items[1].text in (':', '=', ':='):
                    annotation = items[2].text if items[1].text == ':' and len(items) > 2 else ''
                    if annotation == 'dynamic' and len(items) > 3:
                        annotation = items[3].text
                    assignment = next((i for i, token in enumerate(items) if token.text in ('=', ':=')), None)
                    if assignment is not None and not annotation and len(items) > assignment + 2 and items[assignment + 2].text in ('(', '['):
                        annotation = items[assignment + 1].text
                    role = 'constant' if scope is module and assignment is not None and items[assignment].text == ':=' else 'value'
                    existing = scope.lookup(items[0].text)
                    if (scope is not module and assignment is not None and items[1].text == '='
                            and existing is not None and existing.name == items[0].text
                            and items[0].text not in scope.bindings):
                        definitions[(items[0].span.source, items[0].span.start)] = existing
                    else:
                        define(scope, items[0], role, annotation)
                depth, assignment = 0, None
                for position, token in enumerate(items):
                    if depth == 0 and token.text in ('=', ':='):
                        assignment = position
                        break
                    depth += (token.text in ('(', '[', '{')) - (token.text in (')', ']', '}'))
                if assignment is not None:
                    target = items[:assignment]
                    if any(token.text == ',' for token in target) and not any(token.text in ('.', '[', 'with') for token in target):
                        for token in target:
                            if token.kind == 'IDENTIFIER':
                                define(scope, token)
                with_index = next((i for i, token in enumerate(items) if token.text == 'with'), None)
                if with_index is not None:
                    for group in split(items[with_index + 1:-1]):
                        if len(group) > 1 and group[1].text in ('=', ':=', ':'):
                            define(scope, group[0])
            # Generic parameter names have concrete type contracts and are not
            # ordinary variable spellings; preserve T/V/Container/etc.
            if spelling in ('class', 'trait', 'def', 'enum', 'macro') and len(items) > 2 and items[2].text == '[':
                depth = 1
                expect = True
                for token in items[3:]:
                    if token.text == ']' and depth == 1:
                        break
                    if depth == 1 and expect and token.kind == 'IDENTIFIER':
                        define(child_scope, token, 'generic')
                        expect = False
                    if depth == 1 and token.text == ',':
                        expect = True
                    depth += (token.text == '[') - (token.text == ']')
            scopes[node.identity] = child_scope
            walk(node.children, child_scope)
    walk(root.children, module)
    for alias, scope, target in pending_aliases:
        binding = scope.lookup(target)
        if binding is not None and binding is not alias:
            alias.members, alias.type_name, alias.parameters = binding.members, binding.type_name, binding.parameters
    def member_scope(binding, scope):
        if binding is None:
            return None
        if binding.members:
            return binding.members
        target = scope.lookup(binding.type_name)
        if target and target.members:
            return target.members
        candidates = classes.get(binding.type_name, ())
        return candidates[0].members if len(candidates) == 1 else None
    def check_capture(binding, scope, token, comprehension=()):
        candidates = [candidate for candidate in comprehension if candidate.canonical == binding.canonical]
        current = scope
        while not candidates and current is not None:
            candidates = [candidate for candidate in current.bindings.values()
                          if candidate.canonical == binding.canonical]
            current = current.parent
        if any(candidate.name != binding.name for candidate in candidates):
            raise Diagnostic('lint', f'renaming {binding.name!r} to {binding.canonical!r} would capture another symbol',
                             source, token.span)
    def visit(nodes):
        for node in nodes:
            scope = scopes[node.identity]
            items = node.tokens
            type_tokens = set()
            def mark_annotation(group):
                colon = next((i for i, token in enumerate(group) if token.text == ':'), None)
                if colon is not None:
                    end = next((i for i, token in enumerate(group[colon + 1:], colon + 1) if token.text in ('=', ':=')), len(group))
                    type_tokens.update(token.span.start for token in group[colon + 1:end])
            if items and items[0].text in ('def', 'macro'):
                from py_compiler.frontend.lint.layout import groups, split_at
                pair = next(((opening, closing) for opening, closing in groups(items) if items[opening].text == '('), None)
                if pair:
                    for group in split_at(items[pair[0] + 1:pair[1]], {','}):
                        mark_annotation(group)
                    arrow = next((i for i, token in enumerate(items[pair[1] + 1:], pair[1] + 1) if token.text == '->'), None)
                    if arrow is not None:
                        end = next((i for i, token in enumerate(items[arrow + 1:], arrow + 1) if token.text == 'with'), len(items))
                        type_tokens.update(token.span.start for token in items[arrow + 1:end])
            elif len(items) > 1 and items[1].text == ':':
                mark_annotation(items)
            def lookup_type(name):
                current = scope
                while current is not None:
                    binding = current.bindings.get(name)
                    if binding and binding.role in ('type', 'generic', 'import', 'alias'):
                        return binding
                    candidates = [binding for binding in current.bindings.values()
                                  if binding.role == 'type' and canonical_name(name, 'type') == binding.canonical]
                    if len(candidates) == 1:
                        return candidates[0]
                    current = current.parent
                return None
            resolved, delimiters, comprehension_bindings = {}, [], {}
            for index, token in enumerate(items):
                if token.text == 'for' and index > 0:
                    end = next((i for i in range(index + 1, len(items)) if items[i].text == 'in'), index + 1)
                    for target in items[index + 1:end]:
                        if target.kind == 'IDENTIFIER':
                            comprehension_bindings[target.text] = NameBinding(
                                target.text, canonical_name(target.text, 'value'), target, 'value')
            for index, token in enumerate(items):
                word = token.text
                if token.kind == 'FSTRING':
                    import ast
                    try:
                        from py_compiler.syntax.symbol.forms import interpolation_parser_text
                        parsed = ast.parse(interpolation_parser_text(token.text), mode='eval')
                    except SyntaxError:
                        continue  # Interpolation supplies its own diagnostic.
                    line_starts, cursor = [], 0
                    for line in token.text.splitlines(keepends=True):
                        line_starts.append(cursor)
                        cursor += len(line)
                    def offset(line, byte_column):
                        byte_column -= int(token.text.startswith('"""') and line == 1)
                        start = line_starts[line - 1]
                        raw = token.text[start:].split('\n', 1)[0]
                        return start + len(raw.encode('utf-8')[:byte_column].decode('utf-8'))
                    def resolve_term(term):
                        if isinstance(term, ast.Name):
                            return scope.owner if term.id == 'self' else scope.lookup(term.id)
                        if isinstance(term, ast.Attribute):
                            if isinstance(term.value, ast.Name) and term.value.id == 'module':
                                return find_member(module, term.attr)
                            return find_member(member_scope(resolve_term(term.value), scope), term.attr)
                        if isinstance(term, ast.Call):
                            callee = resolve_term(term.func)
                            return scope.lookup(callee.type_name) if callee else None
                        return None
                    for term in ast.walk(parsed):
                        if isinstance(term, ast.Name):
                            binding = scope.lookup(term.id)
                            start = offset(term.lineno, term.col_offset)
                            end = offset(term.end_lineno, term.end_col_offset)
                            old = term.id
                            if binding:
                                check_capture(binding, scope, token)
                        elif isinstance(term, ast.Attribute):
                            binding = resolve_term(term)
                            end = offset(term.end_lineno, term.end_col_offset)
                            start = end - len(term.attr)
                            old = term.attr
                        else:
                            continue
                        if binding and binding.canonical != old:
                            edits.append(SourceEdit(source.span(token.span.start + start, token.span.start + end),
                                                    binding.canonical, 'source.names'))
                    continue
                if word in ('(', '[', '{'):
                    callee = resolved.get(index - 1) if word == '(' else None
                    delimiters.append(callee)
                elif word in (')', ']', '}') and delimiters:
                    delimiters.pop()
                if token.kind != 'IDENTIFIER':
                    continue
                binding = definitions.get((token.span.source, token.span.start))
                if binding is None and index and items[index - 1].text == '.':
                    receiver = items[index - 2].text if index > 1 else ''
                    owner = module if receiver == 'module' else scope.owner.members if receiver == 'self' and scope.owner else member_scope(resolved.get(index - 2), scope)
                    binding = find_member(owner, word)
                elif binding is None and index + 1 < len(items) and items[index + 1].text == '=' and delimiters and delimiters[-1]:
                    callable_ = delimiters[-1]
                    binding = callable_.parameters.get(word)
                    if binding is None and callable_.members:
                        binding = callable_.members.bindings.get(word)
                elif binding is None:
                    binding = lookup_type(word) if token.span.start in type_tokens else comprehension_bindings.get(word) or scope.lookup(word)
                if binding:
                    resolved[index] = binding
                    if token.span.start not in type_tokens and not (index and items[index - 1].text == '.') and not (
                            index + 1 < len(items) and items[index + 1].text == '=' and delimiters and delimiters[-1]):
                        check_capture(binding, scope, token, comprehension_bindings.values())
                    if binding.canonical != word:
                        edits.append(SourceEdit(token.span, binding.canonical, 'source.names'))
                elif word == 'self' and scope.owner:
                    resolved[index] = scope.owner
            visit(node.children)
    visit(root.children)
    return apply_edits(source, edits)


import unittest


class NamingTests(unittest.TestCase):
    def test_acronyms_and_abbreviations_have_distinct_maps(self):
        self.assertEqual(canonical_name('MlirBuilder', 'type'), 'MLIRBuilder')
        self.assertEqual(canonical_name('HTTP', 'value'), 'http')
        self.assertEqual(canonical_name('frmt', 'function'), 'format')
        self.assertEqual(canonical_name('min', 'function'), 'min')
        self.assertEqual(canonical_name('Container', 'generic'), 'Container')
        self.assertEqual(canonical_name('DefinitionId', 'type'), 'DefinitionID')
        self.assertEqual(canonical_name('DefinitionID', 'type'), 'DefinitionID')
        self.assertEqual(canonical_name('Identity', 'type'), 'Identity')
        for name in ('AbiId', 'ABIID', 'Abiid'):
            self.assertEqual(canonical_name(name, 'type'), 'ABIID')
        for name in ('HirId', 'HIRID', 'Hirid'):
            self.assertEqual(canonical_name(name, 'type'), 'HIRID')

    def test_compatibility_alias_preserves_both_names_and_identity(self):
        from py_compiler.frontend.source.source import SourceFile
        from py_compiler.syntax.recognition import Syntax
        text = 'class DefinitionId:\n    value: int\nDefinitionId as DefinitionID\ndef work(value: DefinitionID) -> DefinitionId:\n    return value\n'
        result = fix_names(SourceFile('aliases.sev', text), Syntax())
        self.assertIn('class DefinitionID:', result)
        self.assertIn('DefinitionID as DefinitionId', result)
        self.assertNotIn('DefinitionID as DefinitionID', result)
        self.assertEqual(fix_names(SourceFile('aliases.sev', result), Syntax()), result)

    def test_scoped_renames_preserve_strings_members_and_shadowing(self):
        from py_compiler.frontend.source.source import SourceFile
        from py_compiler.syntax.recognition import Syntax
        original = 'def frmt(someValue: int) -> int:\n    return someValue\ndef work() -> int:\n    otherValue = 7\n    return frmt(otherValue)\n'
        result = fix_names(SourceFile('names.sev', original), Syntax())
        self.assertIn('def format(some_value: int)', result)
        self.assertIn('return format(other_value)', result)

    def test_colliding_fixes_are_errors(self):
        from py_compiler.frontend.source.source import SourceFile
        from py_compiler.syntax.recognition import Syntax
        with self.assertRaisesRegex(Diagnostic, 'collide'):
            fix_names(SourceFile('names.sev', 'someValue = 1\nsome_value = 2\n'), Syntax())

    def test_members_strings_and_generic_contract_names_keep_their_roles(self):
        from py_compiler.frontend.source.source import SourceFile
        from py_compiler.syntax.recognition import Syntax
        original = '''class MlirBuilder[T]:
    someValue: T
    def getValue() -> T:
        return someValue
def work() -> int:
    builderValue = MlirBuilder[int](7)
    text = "someValue frmt"
    return builderValue.someValue
'''
        result = fix_names(SourceFile('members.sev', original), Syntax())
        self.assertIn('class MLIRBuilder[T]', result)
        self.assertIn('def get_value()', result)
        self.assertIn('"someValue frmt"', result)
        self.assertIn('builder_value.some_value', result)

    def test_constants_and_mutable_globals_follow_declaration_contracts(self):
        from py_compiler.frontend.source.source import SourceFile
        from py_compiler.syntax.recognition import Syntax
        result = fix_names(SourceFile('globals.sev', 'maxItems := 7\nCURRENT_VALUE = 1\nvalue = maxItems + CURRENT_VALUE\n'), Syntax())
        self.assertIn('MAX_ITEMS := 7', result)
        self.assertIn('current_value = 1', result)
        self.assertIn('MAX_ITEMS + current_value', result)

    def test_interpolation_renames_only_expression_references(self):
        from py_compiler.frontend.source.source import SourceFile
        from py_compiler.syntax.recognition import Syntax
        text = 'someValue = 7\ntext = f"someValue λ {someValue} / {someValue + 1}"\n'
        result = fix_names(SourceFile('interpolation.sev', text), Syntax())
        self.assertIn('some_value = 7', result)
        self.assertIn('f"someValue λ {some_value} / {some_value + 1}"', result)

    def test_fix_cannot_capture_a_different_symbol_in_a_child_scope(self):
        from py_compiler.frontend.source.source import SourceFile
        from py_compiler.syntax.recognition import Syntax
        text = 'def fooBar() -> int:\n    return 7\ndef work(foo_bar: int) -> int:\n    return fooBar()\n'
        with self.assertRaisesRegex(Diagnostic, 'capture another symbol'):
            fix_names(SourceFile('capture.sev', text), Syntax())

    def test_comprehension_cannot_capture_a_renamed_outer_binding(self):
        from py_compiler.frontend.source.source import SourceFile
        from py_compiler.syntax.recognition import Syntax
        text = 'someValue = 7\nvalues = [someValue for some_value in [1, 2]]\n'
        with self.assertRaisesRegex(Diagnostic, 'capture another symbol'):
            fix_names(SourceFile('capture.sev', text), Syntax())

    def test_type_annotations_do_not_resolve_through_runtime_variable_shadowing(self):
        from py_compiler.frontend.source.source import SourceFile
        from py_compiler.syntax.recognition import Syntax
        text = 'class UserType:\n    value: int\ndef work(UserType: int) -> UserType:\n    unimplemented\n'
        result = fix_names(SourceFile('types.sev', text), Syntax())
        self.assertIn('def work(user_type: int) -> UserType:', result)

    def test_tests_have_distinct_local_scopes_and_global_assignments_keep_identity(self):
        from py_compiler.frontend.source.source import SourceFile
        from py_compiler.syntax.recognition import Syntax
        text = ('maxRetries := 3\nmutableGlobal = 0\n'
                'test:\n    localValue := maxRetries\n    mutableGlobal = 1\n'
                'test:\n    localValue = 2\n    assert(localValue == 2)\n')
        result = fix_names(SourceFile('tests.sev', text), Syntax())
        self.assertIn('MAX_RETRIES := 3', result)
        self.assertIn('local_value := MAX_RETRIES', result)
        self.assertIn('mutable_global = 1', result)
        self.assertIn('local_value = 2', result)

    def test_enum_variant_reference_is_not_a_similarly_spelled_local(self):
        from py_compiler.frontend.source.source import SourceFile
        from py_compiler.syntax.recognition import Syntax
        text = 'enum Status:\n    Connecting -> Received | Failed\n    Received\n    Failed\ntest:\n    received = Connecting\n    received = Received\n    failed = Status.Failed\n    assert(failed == Failed)\n'
        result = fix_names(SourceFile('enum.sev', text), Syntax())
        self.assertIn('received = Received', result)
        self.assertIn('assert(failed == Failed)', result)
