"""Group consecutive test operations using parsed statement boundaries."""
from py_compiler.frontend.lint.rules import SourceEdit, apply_edits, policy


def expand_declared_enum_defaults(source, syntax):
    """Explicit module enum parameter contracts do not require a package build."""
    if 'enum ' not in source.text or 'case _:' not in source.text:
        return source.text
    from py_compiler.frontend.lexer.lexer import lex
    from py_compiler.frontend.parser.blocks import parse_blocks
    from py_compiler.frontend.lint.layout import groups, split_at
    root, errors = parse_blocks(source, lex(source, syntax), syntax)
    if errors:
        return source.text
    enums = {node.tokens[1].text: tuple(child.tokens[0].text for child in node.children)
             for node in root.children if node.tokens and node.tokens[0].text == 'enum'}
    if not enums:
        return source.text
    edits = []
    variants = {}
    for enum, names in enums.items():
        for name in names:
            variants.setdefault(name, []).append(enum)
    results = {}
    for function in root.children:
        if function.tokens[0].text != 'def':
            continue
        terms = function.tokens
        arrow = next((i for i, term in enumerate(terms) if term.text == '->'), None)
        if arrow is not None and arrow + 2 < len(terms) and terms[arrow + 1].text in enums and terms[arrow + 2].text in (':', 'with'):
            results[terms[1].text] = terms[arrow + 1].text
    function_names = [node.tokens[1].text for node in root.children if node.tokens[0].text == 'def']
    results = {name: enum for name, enum in results.items() if function_names.count(name) == 1}

    def subject_type(terms, bindings):
        if not terms:
            return None
        from py_compiler.syntax.grammar.expression import expression
        try:
            term = expression(terms)
        except ValueError:
            return None
        called = term.kind == 'call'
        callee = term.operands[0] if called else term
        if callee.kind not in ('name', 'member'):
            return None
        if callee.kind == 'member':
            if callee.operands[0].kind != 'name':
                return None
            owner = callee.operands[0].token.text
            return owner if owner not in bindings and owner in enums and callee.token.text in enums[owner] else None
        name = callee.token.text
        if not called and name in bindings:
            return bindings[name]
        if name in bindings:
            return None
        owners = variants.get(name, ())
        return owners[0] if len(owners) == 1 else results.get(name) if called else None

    def visit(nodes, parameters):
        for node in nodes:
            if node.tokens[0].text in ('def', 'class', 'trait', 'enum', 'reject'):
                continue
            terms = node.tokens
            if len(terms) > 1 and terms[0].kind == 'IDENTIFIER' and terms[1].text in (':', '=', ':='):
                if terms[1].text == ':':
                    parameters[terms[0].text] = terms[2].text if len(terms) > 2 and terms[2].text in enums else None
                elif terms[1].text == ':=' or terms[0].text not in parameters:
                    parameters[terms[0].text] = subject_type(terms[2:], parameters)
            if node.kind == 'match':
                enum = subject_type(node.header, parameters)
                if enum:
                    seen, wildcard, valid = set(), None, True
                    for position, case in enumerate(node.children):
                        if len(case.header) == 1 and case.header[0].text == '_':
                            wildcard = case.header[0] if position == len(node.children) - 1 else None
                            valid = valid and position == len(node.children) - 1
                            continue
                        header = case.header
                        enclosing = groups(header)
                        if enclosing and enclosing[0] == (0, len(header) - 1) and header[0].text == '(':
                            header = header[1:-1]
                        for pattern in split_at(header, {'|'}):
                            if pattern and pattern[0].text == '|':
                                pattern = pattern[1:]
                            if not pattern:
                                valid = False
                                continue
                            leaf = pattern[0].text
                            if len(pattern) > 2 and pattern[1].text == '.':
                                valid = valid and leaf == enum
                                leaf = pattern[2].text
                            valid = valid and leaf in enums[enum] and leaf not in seen
                            seen.add(leaf)
                    remaining = [enum + '.' + variant for variant in enums[enum] if variant not in seen]
                    if valid and wildcard is not None and remaining:
                        edits.append(SourceEdit(wildcard.span, ' | '.join(remaining), 'G.match.exhaustive'))
            children_bindings = dict(parameters)
            if terms[0].text in ('for', 'catch'):
                for token in terms[1:]:
                    if token.text == 'in':
                        break
                    if token.kind == 'IDENTIFIER':
                        children_bindings[token.text] = None
            if node.kind == 'case':
                for opening, closing in groups(node.header):
                    if node.header[opening].text == '(':
                        for token in node.header[opening + 1:closing]:
                            if token.kind == 'IDENTIFIER':
                                children_bindings[token.text] = None
            visit(node.children, children_bindings)

    for function in root.children:
        items = function.tokens
        if not items or items[0].text != 'def' or (len(items) > 2 and items[2].text == '['):
            continue
        signature = next(((opening, closing) for opening, closing in groups(items) if items[opening].text == '('), None)
        if signature is None:
            continue
        opening, closing = signature
        parameters = {}
        for part in split_at(items[opening + 1:closing], {','}):
            if part and part[0].kind == 'IDENTIFIER':
                parameters[part[0].text] = None
            if len(part) >= 3 and part[1].text == ':' and part[2].text in enums and (len(part) == 3 or part[3].text == '='):
                parameters[part[0].text] = part[2].text
        # Explicit contracts remain authoritative on reassignment. A local
        # declaration shadowing the parameter must be resolved by typed lint.
        def shadowed(nodes):
            for node in nodes:
                terms = node.tokens
                if len(terms) > 1 and terms[0].text in parameters and terms[1].text in (':', ':='):
                    return True
                if terms[0].text in ('for', 'catch', 'class', 'enum', 'def'):
                    return True
                if shadowed(node.children):
                    return True
            return False
        if not shadowed(function.children):
            visit(function.children, parameters)
    visit(root.children, {})
    return apply_edits(source, edits)


def group_operations(source, syntax):
    if all(source.text.count(name) < policy()['operation_block'] for name in ('assert', 'mock', 'when', 'throws')):
        return source.text
    from py_compiler.frontend.lexer.lexer import lex
    from py_compiler.frontend.parser.blocks import parse_blocks
    from py_compiler.frontend.lint.layout import split_at
    root, errors = parse_blocks(source, lex(source, syntax), syntax)
    if errors:
        return source.text
    edits = []

    def end(node):
        return max([node.span.end, *(end(child) for child in node.children)])

    def operation(node):
        items = node.tokens
        if not items or (len(items) == 2 and items[-1].text == ':'):
            return None
        name = items[0].text
        if node.scope is None or not node.scope.contains_kind('test'):
            return None
        if name == 'assert':
            if (len(items) > 3 and items[1].text == '(' and items[-1].text == ')'
                    and len(split_at(items[2:-1], {','})) == 1):
                return name
        elif name in ('mock', 'when', 'throws') and node.provider is not None:
            return name
        return None

    def visit(parent):
        index = 0
        while index < len(parent.children):
            first = parent.children[index]
            name = operation(first)
            if len(parent.tokens) == 2 and parent.tokens[-1].text == ':' and parent.tokens[0].text == name:
                name = None
            stop = index + 1
            while name and stop < len(parent.children):
                previous, following = parent.children[stop - 1:stop + 1]
                gap = source.text[end(previous):following.span.start]
                if operation(following) != name or gap.strip():
                    break
                stop += 1
            if name and stop - index >= policy()['operation_block']:
                if name == 'when':
                    while stop < len(parent.children) and getattr(parent.children[stop].provider, 'continuation', False):
                        stop += 1
                run = parent.children[index:stop]
                line_start = source.text.rfind('\n', 0, first.span.start) + 1
                prefix = source.text[line_start:first.span.start]
                lines = [name + ':']
                for node in run:
                    if name == 'assert':
                        text = source.text[node.tokens[2].span.start:node.tokens[-2].span.end]
                        if '\n' in text:
                            text = '(' + text + ')'
                    else:
                        text = source.text[node.span.start:end(node)]
                    for position, line in enumerate(text.splitlines()):
                        relative = line if position == 0 else line[len(prefix):]
                        lines.append(prefix + '    ' + relative)
                edits.append(SourceEdit(source.span(first.span.start, end(run[-1])), '\n'.join(lines),
                                        'G.test.operation_block'))
            else:
                for node in parent.children[index:stop]:
                    visit(node)
            index = stop

    visit(root)
    return apply_edits(source, edits)


import unittest


class OperationGroupingTests(unittest.TestCase):
    def test_production_asserts_do_not_acquire_a_test_only_block(self):
        from py_compiler.frontend.source.source import SourceFile
        from py_compiler.syntax.recognition import Syntax
        text = 'def work():\n    assert(true)\n    assert(true)\n    assert(true)\n'
        self.assertEqual(group_operations(SourceFile('production.sev', text), Syntax()), text)
    def test_three_assertions_group_but_two_and_comment_boundaries_do_not(self):
        from py_compiler.frontend.source.source import SourceFile
        from py_compiler.syntax.recognition import Syntax
        text = 'test:\n    assert(true)\n    assert(false)\n    assert(true)\n'
        result = group_operations(SourceFile('group.sev', text), Syntax())
        self.assertEqual(result, 'test:\n    assert:\n        true\n        false\n        true\n')
        for text in ('test:\n    assert(true)\n    assert(true)\n',
                     'test:\n    assert(true)\n    # boundary\n    assert(true)\n    assert(true)\n'):
            self.assertEqual(group_operations(SourceFile('group.sev', text), Syntax()), text)

    def test_multiline_condition_keeps_its_group_when_moved_under_assert(self):
        from py_compiler.frontend.lint.lint import prepare_source
        from py_compiler.frontend.source.source import SourceFile
        from py_compiler.syntax.recognition import Syntax
        text = 'test:\n    assert(true)\n    assert(\n        first == second\n        or third == fourth\n    )\n    assert(true)\n'
        result = prepare_source(SourceFile('multiline.sev', text), Syntax())
        self.assertIn('assert:\n', result.text)
        self.assertIn('first == second or third == fourth', result.text)
        self.assertEqual(prepare_source(result, Syntax()).text, result.text)

    def test_explicit_enum_parameter_expands_without_unrelated_dependency_context(self):
        from py_compiler.frontend.source.source import SourceFile
        from py_compiler.syntax.recognition import Syntax
        text = 'enum Shape:\n    Circle(radius: float)\n    Rectangle(width: float)\n    Point\ndef check(shape: Shape) -> bool:\n    return match shape:\n        case Rectangle:\n            true\n        case _:\n            false\ndef unrelated():\n    missing_dependency()\n'
        result = expand_declared_enum_defaults(SourceFile('enum.sev', text), Syntax())
        self.assertIn('case Shape.Circle | Shape.Point:', result)
        shadowed = text.replace('    return match shape:', '    shape: int = 1\n    return match shape:')
        self.assertEqual(expand_declared_enum_defaults(SourceFile('enum.sev', shadowed), Syntax()), shadowed)
