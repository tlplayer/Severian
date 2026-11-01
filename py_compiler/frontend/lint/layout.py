"""Canonical layout over active Y matches; literal contents are never text-replaced."""
import json
import textwrap
from py_compiler.frontend.source.source import SourceFile, Diagnostic
from py_compiler.frontend.lexer.lexer import Token, Lexeme
from py_compiler.syntax.generic.grammar import SourceWindow
from py_compiler.frontend.lint.rules import policy


def scan(source, syntax):
    cursor, result = 0, []
    while cursor < len(source.text):
        try:
            match = syntax.registry.recognize('Y', SourceWindow(source, cursor, len(source.text)))
            kind, end = match.provider.construct(match)
        except ValueError as failure:
            raise Diagnostic('lexer', str(failure), source, source.span(cursor, cursor + 1)) from failure
        result.append(Token(kind, Lexeme(source.text[cursor:end], source.span(cursor, end), source),
                            match.provider, syntax))
        cursor = end
    return result


def quoted(text):
    return json.dumps(text, ensure_ascii=False).replace('\\b', '\\u0008').replace('\\f', '\\u000c')


def literal_text(token):
    from py_compiler.syntax.symbol.forms import decode_quoted
    if token.kind == 'STRING':
        return decode_quoted(token.text)
    import ast
    from py_compiler.syntax.symbol.forms import interpolation_parser_text
    value = ast.parse(interpolation_parser_text(token.text), mode='eval').body
    if isinstance(value, ast.JoinedStr) and all(isinstance(part, ast.Constant) and isinstance(part.value, str) for part in value.values):
        return ''.join(part.value for part in value.values)
    raise ValueError('test labels require a constant string')


def formatted_string_block(value, width):
    """Soft source wraps use continuations; literal newlines remain newlines."""
    lines, current = ['"""\\'], ''
    for character in value:
        if character == '\n':
            lines.append(current)
            current = ''
            continue
        encoded = quoted(character)[1:-1].replace('{', '{{').replace('}', '}}')
        if len(current + encoded) > width - 1:
            lines.append(current + '\\')
            current = ''
        current += encoded
    if current:
        lines.append(current + '\\')
    lines.append('"""')
    return lines


def concatenated_string(tokens, width, indentation):
    """Render a string + chain as one implicit formatted block in source order."""
    parts = split_at(tokens, {'+'})
    if len(parts) < 3:
        return None
    parts = [part[1:] if part and part[0].text == '+' else part for part in parts]
    if not all(parts) or not any(part[0].kind in ('STRING', 'FSTRING') for part in parts[:2]):
        return None
    segments = []
    for part in parts:
        if all(token.kind == 'STRING' for token in part):
            segments.append(('text', ''.join(literal_text(token) for token in part)))
        elif len(part) == 1 and part[0].kind == 'FSTRING':
            import ast
            from py_compiler.syntax.symbol.forms import interpolation_parser_text
            text = interpolation_parser_text(part[0].text)
            parsed = ast.parse(text, mode='eval').body
            for value in parsed.values:
                if isinstance(value, ast.Constant):
                    segments.append(('text', value.value))
                elif value.conversion == -1 and value.format_spec is None:
                    segments.append(('value', ast.get_source_segment(text, value.value)))
                else:
                    return None  # The interpolation provider owns this error.
        else:
            pairs = groups(part)
            if len(part) > 3 and part[0].text == 'string' and pairs and pairs[0] == (1, len(part) - 1) and len(split_at(part[2:-1], {','})) == 1:
                part = part[2:-1]
            segments.append(('value', flat(part)))
    encoded = []
    for kind, value in segments:
        if kind == 'text':
            encoded.append(quoted(value)[1:-1].replace('{', '{{').replace('}', '}}'))
        else:
            encoded.append('{' + value + '}')
    inline = '"""' + ''.join(encoded) + '"""'
    if len(inline) + indentation <= width and all(kind != 'text' or '\n' not in value for kind, value in segments):
        return inline
    lines, current = ['"""\\'], ''
    for kind, value in segments:
        units = [('{'+value+'}')] if kind == 'value' else [quoted(character)[1:-1].replace('{', '{{').replace('}', '}}') if character != '\n' else '\n' for character in value]
        for unit in units:
            if unit == '\n':
                lines.append(current)
                current = ''
                continue
            if len(current + unit) > width - 1:
                lines.append(current + '\\')
                current = ''
            if len(unit) > width - 1:
                raise ValueError('formatted value exceeds the line width: ' + value)
            current += unit
    if current:
        lines.append(current + '\\')
    lines.append('"""')
    return '\n'.join(lines)


def normalize_string_arguments(tokens, width, indentation):
    """Apply the same string-chain rule inside ordinary call arguments."""
    pairs = {opening: closing for opening, closing in groups(tokens)}
    output, index = [], 0
    while index < len(tokens):
        closing = pairs.get(index)
        if closing is None:
            output.append(tokens[index])
            index += 1
            continue
        inside = normalize_string_arguments(tokens[index + 1:closing], width, indentation)
        call = tokens[index].text == '(' and index > 0 and tokens[index - 1].kind == 'IDENTIFIER'
        if call:
            parts = split_at(inside, {','})
            rewritten = []
            for position, part in enumerate(parts):
                if position:
                    rewritten.append(synthetic(tokens[index], ','))
                start = 2 if len(part) > 1 and part[1].text == '=' else 0
                value = concatenated_string(part[start:], width, indentation) if part[start:] else None
                rewritten.extend([*part[:start], synthetic(part[start], value, 'FSTRING')] if value is not None else part)
            inside = rewritten
        output.extend([tokens[index], *inside, tokens[closing]])
        index = closing + 1
    return output


def synthetic(token, text, kind='SYMBOL'):
    return token.with_text(text, kind=kind)


def flat(tokens):
    result, stack, previous = '', [], None
    slice_openings = {opening for opening, closing in groups(tokens)
                      if tokens[opening].text == '(' and tokens[closing].text == ']'}
    unary = False
    for index, token in enumerate(tokens):
        word = token.text
        before = previous.text if previous else ''
        is_unary = token.kind != 'INFIX_OPERATOR' and word in ('-', '+', '~', '&') and (not before or before in
                    ('(', '[', '{', ',', ':', '=', ':=', 'return', 'yield', 'and', 'or', 'not',
                     '+', '-', '*', '**', '/', '//', '%', '|', '&', '^', '==', '!=', '<', '>', '<=', '>=', '..',
                     'else', 'if', '->', '?=', 'throw', 'borrow', 'view', 'move', 'copy'))
        space = bool(result)
        if word in (')', ']', '}', ',', '.', ':', '...') or before in ('(', '[', '{', '.', '...', '@'):
            space = False
        elif word in ('(', '['):
            space = before in ('import', ',', 'if', 'elif', 'while', 'return', 'yield', 'not', 'and', 'or', 'in',
                               'is', '=', ':=', '+', '-', '*', '/', '//', '==', '!=', '<', '>', '<=', '>=', '->')
        elif word == '{':
            space = True
        elif before == ':':
            space = not (stack and stack[-1] == '[')
        elif word == '..':
            space = before in ('in', 'not', 'return', 'yield', '=', ':=', 'and', 'or')
        elif before == '..' or unary:
            space = False
        if word == '**' or before == '**':
            space = False
        if word == '(' and len(tokens) > 1 and tokens[0].text == 'operator':
            space = False
        if space:
            result += ' '
        result += word
        if word in ('(', '[', '{'):
            generic = word == '[' and index == 2 and tokens[0].text in ('def', 'class', 'trait', 'enum', 'macro', 'grammar')
            stack.append('generic[' if generic else '[' if index in slice_openings else word)
        elif word in (')', ']', '}') and stack:
            stack.pop()
        previous, unary = token, is_unary
    return result


def groups(tokens):
    stack, pairs = [], []
    for index, token in enumerate(tokens):
        if token.text in ('(', '[', '{'):
            stack.append(index)
        elif token.text in (')', ']', '}') and stack:
            opening = stack.pop()
            pairs.append((opening, index))
    return sorted(pairs)


def split_at(tokens, spellings):
    depth, result, start = 0, [], 0
    for index, token in enumerate(tokens):
        if depth == 0 and token.text in spellings:
            result.append(tokens[start:index])
            start = index + (token.text == ',')
        depth += (token.text in ('(', '[', '{')) - (token.text in (')', ']', '}'))
    result.append(tokens[start:])
    return result


def split_literal(token, width):
    from py_compiler.syntax.symbol.forms import decode_quoted
    value = decode_quoted(token.text)
    chunks, current = [], ''
    for character in value:
        if len(quoted(current + character)) > width and current:
            chunks.append(quoted(current))
            current = ''
        if len(quoted(current + character)) > width:
            return None
        current += character
    if current or not chunks:
        chunks.append(quoted(current))
    return [synthetic(token, chunk, 'STRING') for chunk in chunks]


def simplify_parentheses(tokens):
    tokens = list(tokens)
    while True:
        removed = False
        for opening, closing in reversed(groups(tokens)):
            if tokens[opening].text != '(':
                continue
            before = tokens[opening - 1] if opening else None
            call = before and (before.kind == 'IDENTIFIER' or before.text in (')', ']'))
            if call:
                continue
            inside = tokens[opening + 1:closing]
            if inside and all(token.kind == 'STRING' for token in inside):
                inside = [synthetic(inside[0], quoted(''.join(literal_text(token) for token in inside)), 'STRING')]
            nested = groups(inside)
            negated_comparison = False
            if before is not None and before.text == 'not' and (closing + 1 == len(tokens) or tokens[closing + 1].text in (')', ']', '}', ',', ':', 'with', 'and', 'or')):
                from py_compiler.syntax.grammar.expression import expression, Expression
                def meaning(term):
                    if isinstance(term, Expression):
                        return term.kind, term.token.text, term.interval, tuple(meaning(value) for value in term.operands)
                    return term
                try:
                    negated_comparison = meaning(expression([before, *tokens[opening:closing + 1]])) == meaning(expression([before, *inside]))
                except (ValueError, SyntaxError):
                    pass
            if ((len(inside) == 1 and (inside[0].kind in ('IDENTIFIER', 'NUMBER', 'STRING', 'FSTRING', 'CHAR')
                                     or inside[0].text in ('true', 'false', 'None', 'absent'))) or
                    negated_comparison or
                    (nested and nested[0] == (0, len(inside) - 1)
                     and inside[0].text == '(')):
                tokens = tokens[:opening] + inside + tokens[closing + 1:]
                removed = True
                break
        if not removed:
            return tokens


def wrap(tokens, indentation, source, *, force=False, depth=0, columns=None):
    settings = policy()
    width, unit = columns if columns is not None else settings['columns'], settings['indentation']
    prefix = ' ' * indentation
    tokens = simplify_parentheses(tokens)
    merged = []
    for token in tokens:
        if merged and merged[-1].kind == token.kind == 'STRING':
            merged[-1] = synthetic(merged[-1], quoted(literal_text(merged[-1]) + literal_text(token)), 'STRING')
        else:
            merged.append(token)
    tokens = merged
    # A singleton tuple comma specifies arity, not a vertical layout.
    compact = list(tokens)
    optional_commas = set()
    for opening, closing in groups(tokens):
        if closing <= opening + 1 or compact[closing - 1].text != ',':
            continue
        call = compact[opening].text == '(' and opening > 0 and compact[opening - 1].kind == 'IDENTIFIER'
        if call or compact[opening].text in ('[', '{'):
            optional_commas.add(closing - 1)
    compact = [token for index, token in enumerate(tokens) if index not in optional_commas]
    if '\n' not in flat(compact) and len(prefix + flat(compact)) <= width:
        tokens = compact
    rendered = flat(tokens)
    multiline_string = any(token.kind == 'FSTRING' and '\n' in token.text for token in tokens)
    force = force or (multiline_string and any(opening > 0 and tokens[opening].text == '(' and
                                               tokens[opening - 1].kind == 'IDENTIFIER'
                                               for opening, closing in groups(tokens)))
    if multiline_string and not force:
        lines = (prefix + rendered).splitlines()
        if all(len(line) <= width for line in lines):
            return lines
    # A trailing comma is a request for a vertical member list, even when short.
    if not force and '\n' not in rendered and len(prefix + rendered) <= width:
        return [prefix + rendered]
    if depth > 64:
        raise Diagnostic('lint', 'layout nesting exceeds 64 levels', source, tokens[0].span)
    for index, token in enumerate(tokens):
        if token.kind == 'FSTRING' and '\n' not in token.text:
            block = concatenated_string([token, synthetic(token, '+'), synthetic(token, '""', 'STRING'), synthetic(token, '+'), synthetic(token, '""', 'STRING')], width, indentation + len(flat(tokens[:index])) + 1)
            if block is not None and '\n' in block:
                replacement = [*tokens[:index], synthetic(token, block, 'FSTRING'), *tokens[index + 1:]]
                return wrap(replacement, indentation, source, force=force, depth=depth + 1, columns=width)
    if tokens[0].text == 'print':
        pairs = groups(tokens)
        if pairs and pairs[0] == (1, len(tokens) - 1):
            pieces = tokens[2:-1]
            if pieces and pieces[-1].text == ',':
                pieces = pieces[:-1]
            if pieces and all(piece.kind == 'STRING' for piece in pieces):
                block = formatted_string_block(''.join(literal_text(piece) for piece in pieces), width)
                return [prefix + 'print(', ' ' * (indentation + unit) + block[0],
                        *block[1:-1], block[-1] + ',', prefix + ')']
    if tokens[-1].text == ',' and all(token.kind == 'STRING' for token in tokens[:-1]):
        from py_compiler.syntax.symbol.forms import decode_quoted
        value = ''.join(decode_quoted(token.text) for token in tokens[:-1])
        chunks = split_literal(synthetic(tokens[0], quoted(value), 'STRING'), width - indentation - 1)
        if chunks is not None:
            lines = [prefix + chunk.text for chunk in chunks]
            lines[-1] += ','
            return lines
    if all(token.kind == 'STRING' for token in tokens):
        lines = []
        for token in tokens:
            chunks = split_literal(token, width - indentation)
            if chunks is None:
                raise Diagnostic('lint', 'string scalar cannot fit the declared line width', source, token.span)
            lines.extend(prefix + chunk.text for chunk in chunks)
        return lines
    if tokens[-1].text == ',' and len(tokens) > 1:
        lines = wrap(tokens[:-1], indentation, source, force=force, depth=depth + 1, columns=width - 1)
        lines[-1] += ','
        return lines
    # Split comma-delimited members before considering their nested literals.
    for opening, closing in groups(tokens):
        inside = tokens[opening + 1:closing]
        parts = split_at(inside, {','})
        call = opening > 0 and tokens[opening].text == '(' and tokens[opening - 1].kind == 'IDENTIFIER'
        container = tokens[opening].text in ('[', '{') and (opening == 0 or tokens[opening - 1].text in ('=', ':=', ',', '(', ':', 'return', 'yield'))
        if (len(parts) < 2 and not call and not container) or not any(parts):
            continue
        head, tail = prefix + flat(tokens[:opening + 1]), prefix + flat(tokens[closing:])
        if len(head) > width or len(tail) > width:
            continue
        lines = [head]
        for part in parts:
            if part:
                lines.extend(wrap([*part, synthetic(part[-1], ',')], indentation + unit,
                                  source, depth=depth + 1, columns=width))
        return [*lines, tail]
    # Reuse an existing literal group as a whole. Wrapping just its first
    # piece would manufacture ("piece") "tail" and grow parentheses each pass.
    for opening, closing in groups(tokens):
        pieces = tokens[opening + 1:closing]
        if not pieces or not all(piece.kind == 'STRING' for piece in pieces):
            continue
        head, tail = prefix + flat(tokens[:opening + 1]), prefix + flat(tokens[closing:])
        if len(head) <= width and len(tail) <= width:
            from py_compiler.syntax.symbol.forms import decode_quoted
            combined = synthetic(pieces[0], quoted(''.join(decode_quoted(piece.text) for piece in pieces)), 'STRING')
            chunks = split_literal(combined, width - indentation - unit)
            if chunks is not None:
                lines = [' ' * (indentation + unit) + chunk.text for chunk in chunks]
                if opening > 0 and tokens[opening - 1].kind == 'IDENTIFIER':
                    lines[-1] += ','
                return [head, *lines, tail]
    # A literal can be split only through syntax-owned adjacent string pieces.
    for index, token in enumerate(tokens):
        if token.kind == 'STRING' and tokens[0].text not in ('import', 'from'):
            chunks = split_literal(token, width - indentation - unit)
            if chunks is None:
                break
            head = prefix + flat(tokens[:index])
            tail = ' ' * (indentation + unit) + flat(tokens[index + 1:]) if tokens[index + 1:] else ''
            if len(head) <= width and len(tail) <= width:
                if not head.strip():
                    return [prefix + chunks[0].text, *[' ' * (indentation + unit) + chunk.text for chunk in chunks[1:]], *([tail] if tail else [])]
                return [head, *[' ' * (indentation + unit) + chunk.text for chunk in chunks], *([tail] if tail else [])]
    # Prefer the outermost group that has comma-separated members. Each member
    # is formatted recursively, so nested calls and type arguments stay intact.
    for opening, closing in groups(tokens):
        inside = tokens[opening + 1:closing]
        parts = split_at(inside, {','})
        if len(parts) < 2:
            continue
        head = prefix + flat(tokens[:opening + 1])
        tail = prefix + flat(tokens[closing:])
        if len(head) > width or len(tail) > width:
            continue
        lines = [head]
        for part in parts:
            if part:
                lines.extend(wrap([*part, synthetic(part[-1], ',')], indentation + unit,
                                  source, depth=depth + 1, columns=width))
        lines.append(tail)
        return lines
    # Parenthesized boolean expressions can wrap without changing precedence.
    for opening, closing in groups(tokens):
        inside = tokens[opening + 1:closing]
        operators = {'and', 'or'} if len(split_at(inside, {'and', 'or'})) > 1 else {'if', 'else', '+', '-', '*', '/', '//', '%', '==', '!=', '<', '>', '<=', '>=', 'in', 'is'}
        parts = split_at(inside, operators)
        parts = [part for part in parts if part]
        if len(parts) < 2:
            continue
        head, tail = prefix + flat(tokens[:opening + 1]), prefix + flat(tokens[closing:])
        if len(head) <= width and len(tail) <= width:
            lines = [head]
            for part in parts:
                lines.extend(wrap(part, indentation + unit, source, depth=depth + 1, columns=width))
            return [*lines, tail]
    # Break a single nested call/group even when it has no comma.
    for opening, closing in groups(tokens):
        if closing <= opening + 1:
            continue
        head, tail = prefix + flat(tokens[:opening + 1]), prefix + flat(tokens[closing:])
        if len(head) <= width and len(tail) <= width:
            inside = tokens[opening + 1:closing]
            if tokens[opening].text == '(' and opening > 0 and tokens[opening - 1].kind == 'IDENTIFIER':
                inside = [*inside, synthetic(inside[-1], ',')]
            return [head, *wrap(inside, indentation + unit, source,
                                depth=depth + 1, columns=width), tail]
    # Assignment/return expressions can acquire an enclosing pair and break at
    # operators, including the language's conditional-expression if/else form.
    nesting, beginning = 0, 0
    assignments = {'=', ':=', '+=', '-=', '*=', '/=', '//=', '%=', '|=', '&=', '^='}
    for index, token in enumerate(tokens):
        if nesting == 0 and token.text in assignments:
            beginning = index + 1
            break
        nesting += (token.text in ('(', '[', '{')) - (token.text in (')', ']', '}'))
    if tokens[0].text in ('return', 'yield', 'throw'):
        beginning = 1
    expression = tokens[beginning:]
    parts = [part for part in split_at(expression, {'and', 'or', 'if', 'else', '+', '-', '*', '/', '//', '%',
                                                    '|', '&', '^', '==', '!=', '<', '>', '<=', '>=', 'in', 'is'}) if part]
    if len(parts) > 1 and tokens[-1].text != ':':
        head = prefix + flat(tokens[:beginning])
        if len(head) <= width:
            lines = [head] if beginning else []
            for index, part in enumerate(parts):
                if part[0].text in ('+', '-', '&'):
                    part = [synthetic(part[0], part[0].text, 'INFIX_OPERATOR'), *part[1:]]
                lines.extend(wrap(part, indentation + (unit if beginning or index else 0), source, depth=depth + 1, columns=width))
            return lines
    if beginning and expression:
        head = prefix + flat(tokens[:beginning])
        if len(head) <= width and len(' ' * (indentation + unit) + flat(expression)) <= width:
            return [head, *wrap(expression, indentation + unit, source, depth=depth + 1, columns=width)]
    raise Diagnostic('lint', f'line exceeds {width} columns and has no meaning-preserving automatic split',
                     source, tokens[0].span)


def format_statement(tokens, indentation, source):
    settings = policy()
    tokens = simplify_parentheses(tokens)
    if tokens[0].text not in ('grammar', 'import', 'from'):
        try:
            tokens = normalize_string_arguments(tokens, settings['columns'], indentation)
        except (ValueError, SyntaxError) as failure:
            raise Diagnostic('lint', str(failure), source, tokens[0].span) from failure
    nesting, beginning = 0, None
    for index, token in enumerate(tokens):
        if nesting == 0 and token.text in ('=', ':=', '+=', '-=', '*=', '/=', '//=', '%=', '|=', '&=', '^='):
            beginning = index + 1
            break
        nesting += (token.text in ('(', '[', '{')) - (token.text in (')', ']', '}'))
    if tokens[0].text in ('return', 'yield', 'throw'):
        beginning = 1
    if beginning is not None:
        value = tokens[beginning:]
        while value and value[0].text == '(' and value[1:-1] and groups(value) and groups(value)[0] == (0, len(value) - 1) and len(split_at(value[1:-1], {','})) == 1:
            value = value[1:-1]
        tokens = [*tokens[:beginning], *value]
        if value:
            try:
                block = concatenated_string(value, settings['columns'], indentation + len(flat(tokens[:beginning])) + 1)
            except (ValueError, SyntaxError) as failure:
                raise Diagnostic('lint', str(failure), source, value[0].span) from failure
            if block is not None:
                tokens = [*tokens[:beginning], synthetic(value[0], block, 'FSTRING')]
    if tokens[0].text in ('if', 'elif', 'while', 'when') and tokens[-1].text == ':':
        condition = tokens[1:-1]
        while condition and condition[0].text == '(' and condition[1:-1] and groups(condition) and groups(condition)[0] == (0, len(condition) - 1) and len(split_at(condition[1:-1], {','})) == 1:
            condition = condition[1:-1]
        tokens = [tokens[0], *condition, tokens[-1]]
    tokens = [synthetic(token, token.text[1:], 'FSTRING') if token.kind == 'FSTRING' and token.text.startswith('f"""') else token
              for token in tokens]
    tokens = [synthetic(token, '\n'.join(formatted_string_block(literal_text(token), settings['columns'])), 'FSTRING')
              if token.kind == 'STRING' and token.text.startswith(('"""', "'''")) else token
              for token in tokens]
    prefix, unit = ' ' * indentation, settings['indentation']
    pairs = groups(tokens)
    if tokens[0].text in ('import', 'from'):
        from py_compiler.syntax.block.imports_layout import normalize_import_tokens
        tokens = list(normalize_import_tokens(tokens))
        if len(prefix + flat(tokens)) > settings['columns']:
            if tokens[0].text == 'from':
                start, end = 3, len(tokens)
            elif any(token.text == 'from' for token in tokens):
                start, end = 1, next(i for i, token in enumerate(tokens) if token.text == 'from')
            else:
                start, end = 1, 2
            parts = [part for part in split_at(tokens[start:end], {','}) if part]
            if tokens[0].text == 'import' and tokens[1].kind == 'STRING':
                return [prefix + flat(tokens[:2]), ' ' * (indentation + unit) + flat(tokens[2:])]
            if tokens[0].text == 'from':
                return [prefix + flat(tokens[:start]), *[' ' * (indentation + unit) + flat(part) + (',' if index < len(parts) - 1 else '') for index, part in enumerate(parts)]]
            lines = [prefix + flat([tokens[0], *parts[0]]) + (',' if len(parts) > 1 else '')]
            lines.extend(' ' * (indentation + unit) + flat(part) + (',' if index < len(parts) - 1 else '') for index, part in enumerate(parts[1:], 1))
            suffix = flat(tokens[end:])
            if len(lines[-1] + ' ' + suffix) <= settings['columns']:
                lines[-1] += ' ' + suffix
            else:
                lines.append(' ' * (indentation + unit) + suffix)
            return lines
        return [prefix + flat(tokens)]
    if tokens[0].text == 'case' and tokens[-1].text == ':':
        patterns = tokens[1:-1]
        enclosing = groups(patterns)
        if enclosing and enclosing[0] == (0, len(patterns) - 1):
            patterns = patterns[1:-1]
        parts = split_at(patterns, {'|'})
        if len(parts) > 1 and len(prefix + 'case ' + flat(patterns) + ':') > settings['columns']:
            lines = [prefix + 'case ' + flat(parts[0])]
            lines.extend(' ' * (indentation + unit) + flat(part) for part in parts[1:])
            lines[-1] += ':'
            return lines
        tokens = [tokens[0], *patterns, tokens[-1]]
    # Test labels are labels, not parenthesized expressions.
    if tokens[0].text == 'test' and not any(token.text == 'skip' for token in tokens):
        label_start = next((i for i, token in enumerate(tokens) if token.kind in ('STRING', 'FSTRING')), None)
        if label_start is not None:
            start = label_start - (label_start > 0 and tokens[label_start - 1].text == '(')
            end = label_start
            while end < len(tokens) and tokens[end].kind in ('STRING', 'FSTRING'):
                end += 1
            try:
                label = ''.join(literal_text(token) for token in tokens[label_start:end])
            except (ValueError, SyntaxError) as error:
                raise Diagnostic('lint', 'test labels require a valid constant string', source, tokens[label_start].span) from error
            after = end + (end < len(tokens) and tokens[end].text == ')')
            canonical = [*tokens[:start], synthetic(tokens[label_start], quoted(label), 'STRING'), *tokens[after:]]
            if '\n' not in label and len(prefix + flat(canonical)) <= settings['columns']:
                return [prefix + flat(canonical)]
            block = formatted_string_block(label, settings['columns'])
            suffix = tokens[after:]
            if suffix and suffix[0].text == 'with':
                tail = format_statement([synthetic(tokens[0], block[-1]), *suffix], indentation, source)
                tail[0] = tail[0].lstrip()
                return [prefix + flat(tokens[:start]), prefix + block[0], *block[1:-1], *tail]
            return [prefix + flat(tokens[:start]), prefix + block[0], *block[1:-1], block[-1] + flat(suffix)]
    # Constraint braces always own a dedicated block when attached to with or
    # to a field default. Type refinements without defaults remain inline.
    for opening, closing in pairs:
        if tokens[opening].text != '{':
            continue
        head = tokens[:opening]
        if not head or not (head[-1].text == 'with' or
                            (len(head) > 2 and head[1].text == ':' and any(t.text in ('=', ':=') for t in head))):
            continue
        lines = wrap(head, indentation, source)
        lines.append(prefix + '{')
        from importlib import import_module
        clauses = import_module('py_compiler.syntax.block.with').clause_groups(tokens[opening:closing + 1], tuple_bindings=True)
        for part in clauses:
            if part:
                # Error actions are calls too: every vertical argument ends ,.
                lines.extend(wrap([*part, synthetic(part[-1], ',')], indentation + unit, source))
        return [*lines, prefix + flat(tokens[closing:])]
    # Fluent member calls continue directly from the receiver's line.
    depth, dots = 0, []
    beginning, nesting = 0, 0
    for index, token in enumerate(tokens):
        if nesting == 0 and token.text in ('=', ':='):
            beginning = index + 1
            break
        nesting += (token.text in ('(', '[', '{')) - (token.text in (')', ']', '}'))
    expression_tokens = tokens[beginning:]
    if groups(expression_tokens) and groups(expression_tokens)[0] == (0, len(expression_tokens) - 1) and expression_tokens[0].text == '(':
        expression_tokens = expression_tokens[1:-1]
    for index, token in enumerate(expression_tokens):
        if depth == 0 and token.text == '.' and index + 2 < len(expression_tokens) and expression_tokens[index + 2].text == '(':
            dots.append(index)
        depth += (token.text in ('(', '[', '{')) - (token.text in (')', ']', '}'))
    if len(dots) >= 2:
        lines = wrap([*tokens[:beginning], *expression_tokens[:dots[0]]], indentation, source)
        for start, end in zip(dots, [*dots[1:], len(expression_tokens)]):
            lines.extend(wrap(expression_tokens[start:end], indentation + unit, source))
        return lines
    provider = getattr(tokens[0].grammar, 'owner', tokens[0].grammar)
    if getattr(provider, 'lint_condition', False) and tokens[-1].text == ':':
        condition = tokens[1:-1]
        depth, suffix = 0, []
        for index, token in enumerate(condition):
            if token.text == 'with' and depth == 0:
                suffix, condition = condition[index:], condition[:index]
                break
            depth += (token.text in ('(', '[', '{')) - (token.text in (')', ']', '}'))
        condition_count = sum(token.text in ('and', 'or') for token in condition) + 1
        needs_wrap = (condition_count > settings['conditions'] or
                      len(' ' * indentation + flat(tokens)) > settings['columns'])
        if needs_wrap:
            # Remove one enclosing pair; reintroduce a canonical outer pair.
            pairs = groups(condition)
            if pairs and pairs[0] == (0, len(condition) - 1):
                condition = condition[1:-1]
            opening = synthetic(tokens[0], '(')
            closing = synthetic(tokens[-1], ')')
            parts = split_at(condition, {'and', 'or'})
            lines = wrap([tokens[0], *parts[0]], indentation, source)
            for part in parts[1:]:
                lines.extend(wrap(part, indentation + settings['indentation'], source))
            if not suffix and len(lines[-1]) + 1 > settings['columns']:
                if len(parts) > 1:
                    previous = wrap(parts[-1], indentation + settings['indentation'], source)
                    lines = lines[:-len(previous)] + wrap(parts[-1], indentation + settings['indentation'], source, columns=settings['columns'] - 1)
                else:
                    lines = wrap([tokens[0], *parts[0]], indentation, source, columns=settings['columns'] - 1)
            lines[-1] += (' ' + flat(suffix) if suffix else '') + ':'
            return lines
    if tokens[-1].text == ':' and any(token.text == 'with' for token in tokens):
        position = next(i for i, token in enumerate(tokens) if token.text == 'with')
        guard = tokens[position + 1:-1]
        if guard and len(prefix + flat(tokens)) > settings['columns']:
            lines = wrap(tokens[:position + 1], indentation, source)
            for part in split_at(guard, {'and', 'or'}):
                if part:
                    lines.extend(wrap(part, indentation + unit, source))
            lines[-1] += ':'
            return lines
    return wrap(tokens, indentation, source)


def comment_lines(comments, indentation):
    width = policy()['columns'] - indentation
    prefix = ' ' * indentation
    contents = []
    for token in comments:
        if token.text.startswith('#!'):
            contents.append(token.text)
        elif token.text.startswith('#'):
            body = token.text[1:]
            body = body[1:] if body.startswith(' ') else body
            contents.append(body.rstrip())
        else:
            body = token.text[3:-3]
            contents.extend(textwrap.dedent(body).strip('\n').splitlines())
    if len(contents) == 1 or any("'''" in line for line in contents) or any(line.startswith('#!') for line in contents):
        result = []
        for line in contents:
            if line.startswith('#!'):
                result.append(prefix + line)
                continue
            leading = line[:len(line) - len(line.lstrip())]
            for part in textwrap.wrap(line.lstrip(), max(1, width - 2 - len(leading)),
                                      break_long_words=True, break_on_hyphens=False) or ['']:
                result.append(prefix + '# ' + leading + part)
        return result
    lines = [prefix + "'''"]
    for line in contents:
        if not line.strip():
            lines.append('')
            continue
        leading = line[:len(line) - len(line.lstrip())]
        lines.extend(prefix + leading + part for part in textwrap.wrap(line.lstrip(), max(1, width - len(leading)),
                                                            break_long_words=True, break_on_hyphens=False) or [''])
    return [*lines, prefix + "'''"]


def format_source(source, syntax):
    """Normalize layout using lexical matches, preserving statement boundaries."""
    tokens = scan(source, syntax)
    output, current, comments, pending_comments = [], [], [], []
    delimiters, previous_end, previous_header = [], 0, False
    indentation_stack = [0]
    current_indent = 0
    def finish():
        nonlocal current, comments, pending_comments, previous_end, previous_header
        if not current:
            pending_comments.extend(comments)
            comments = []
            return
        original = current_indent
        if original > indentation_stack[-1]:
            if not previous_header:
                raise Diagnostic('lint', 'unexpected indentation without an owning block', source, current[0].span)
            indentation_stack.append(original)
        elif original < indentation_stack[-1]:
            while original < indentation_stack[-1]:
                indentation_stack.pop()
            if original != indentation_stack[-1]:
                raise Diagnostic('lint', 'indentation does not identify an enclosing block', source, current[0].span)
        indent = (len(indentation_stack) - 1) * policy()['indentation']
        gap = source.text[previous_end:current[0].span.start]
        if output and gap.count('\n') > 1 and output[-1] != '':
            output.append('')
        if pending_comments or comments:
            output.extend(comment_lines([*pending_comments, *comments], indent))
        output.extend(format_statement(current, indent, source))
        previous_header = current[-1].text == ':'
        owner = getattr(current[0].grammar, 'owner', current[0].grammar)
        if callable(getattr(owner, 'has_body', None)):
            from types import SimpleNamespace
            previous_header = owner.has_body(SimpleNamespace(tokens=current))
        previous_end = current[-1].span.end
        current, comments, pending_comments = [], [], []
    for token_index, token in enumerate(tokens):
        if token.kind == 'INDENT':
            if not current:
                current_indent = len(token.text.expandtabs(policy()['indentation']))
            continue
        if token.kind == 'TRIVIA':
            if token.text.startswith('#') or token.text.startswith(("'''", '"""')):
                comments.append(token)
            continue
        if token.kind == 'NEWLINE':
            from py_compiler.syntax.grammar.continuation import continues_statement, next_term
            continuation = continues_statement(source, current, next_term(tokens, token_index))
            if not delimiters and not continuation:
                finish()
                current_indent = 0
            continue
        if not current:
            start = max(source.text.rfind('\n', 0, token.span.start), source.text.rfind('\r', 0, token.span.start)) + 1
            current_indent = len(source.text[start:token.span.start].expandtabs(policy()['indentation']))
        current.append(token)
        if token.text in ('(', '[', '{'):
            delimiters.append(token.text)
        elif token.text in (')', ']', '}') and delimiters:
            delimiters.pop()
    finish()
    if pending_comments:
        output.extend(comment_lines(pending_comments, 0))
    overflow = next((line for line in output if len(line) > policy()['columns']), None)
    if overflow is not None:
        raise Diagnostic('lint', 'layout cannot satisfy the declared line width: ' + overflow, source,
                         source.span(0, min(1, len(source.text))))
    return '\n'.join(output).rstrip() + ('\n' if output else '')


import unittest


class LayoutTests(unittest.TestCase):
    def formatted(self, text):
        from py_compiler.syntax.recognition import Syntax
        return format_source(SourceFile('layout.sev', text), Syntax())

    def test_indent_spacing_and_inline_comment_are_fixed(self):
        self.assertEqual(self.formatted('def work( a:int,b:int ) -> int:\n  return a+b # sum\n'),
                         'def work(a: int, b: int) -> int:\n    # sum\n    return a + b\n')

    def test_conditions_use_inferred_continuation_and_keep_precedence(self):
        result = self.formatted('if a and b or c and d:\n    work()\n')
        self.assertEqual(result, 'if a\n    and b\n    or c\n    and d:\n    work()\n')
        self.assertEqual(self.formatted(result), result)

    def test_literal_content_and_intervals_survive_layout(self):
        result = self.formatted('text="a # b, c"\nfor i in [0..3):\n  print(text)\n')
        self.assertIn('text = "a # b, c"', result)
        self.assertIn('for i in [0..3):', result)

    def test_long_strings_and_comments_are_bounded_and_idempotent(self):
        result = self.formatted('# first\n# second\nvalue = "' + 'word ' * 40 + '"\n')
        self.assertTrue(all(len(line) <= 80 for line in result.splitlines()))
        self.assertTrue(result.startswith("'''\nfirst\nsecond\n'''\n"))
        self.assertEqual(self.formatted(result), result)

    def test_unbreakable_identifier_is_a_lint_diagnostic(self):
        with self.assertRaisesRegex(Diagnostic, 'no meaning-preserving automatic split'):
            self.formatted('value = ' + 'a' * 90 + '\n')

    def test_multiline_calls_keep_trailing_commas_and_fit_width(self):
        result = self.formatted('result = work(' + ', '.join('parameter_' + str(i) + '=123456' for i in range(7)) + ')\n')
        self.assertTrue(all(len(line) <= 80 for line in result.splitlines()))
        self.assertIn('parameter_6 = 123456,\n)', result)
        self.assertEqual(self.formatted(result), result)

    def test_inherited_class_owns_its_body_without_an_extra_trailing_colon(self):
        text = 'class TypeReference: TypeTerm\n    identity: int\n    def value() -> int:\n        return identity\n'
        self.assertEqual(self.formatted(text), text)

    def test_long_literal_group_inside_assert_does_not_grow_nested_groups(self):
        text = 'assert(stdout == "' + 'hello\\n' * 40 + '")\n'
        first = self.formatted(text)
        self.assertEqual(self.formatted(first), first)
        self.assertTrue(all(len(line) <= 80 for line in first.splitlines()))

    def test_header_can_wrap_around_a_literal_that_fits_by_itself(self):
        text = 'test with compiler "' + 'word ' * 12 + '":\n    unimplemented\n'
        first = self.formatted(text)
        self.assertTrue(all(len(line) <= 80 for line in first.splitlines()))
        self.assertEqual(self.formatted(first), first)

    def test_existing_parenthesized_arithmetic_group_is_reused(self):
        text = 'result += (' + ' + '.join('component_' + str(i) for i in range(12)) + ')\n'
        first = self.formatted(text)
        self.assertEqual(self.formatted(first), first)
        self.assertEqual(first.count('('), 0)

    def test_operator_spacing_and_redundant_groups(self):
        text = 'address = & owner[0]\nassert(not ([1,2,3] is (1,2,3)))\nvalue = ((other))\nempty = ((()))\nvalue = (2 ** 2+1 ** 2) ** .5\ndef affine[T:Numeric](x:T)->T:\n    return x\n'
        result = self.formatted(text)
        for expected in ('&owner[0]', 'is (1, 2, 3)', 'not ', 'value = other', 'empty = ()', '(2**2 + 1**2)**.5', '[T: Numeric]'):
            self.assertIn(expected, result)
        self.assertEqual(self.formatted(result), result)

    def test_constraints_builders_and_named_lists(self):
        text = ('class User:\n    age: int = 0{age < 0 -> Error("invalid"),}\n'
                '    x: int{0 <= x <= 100}\n'
                'user = (User().set("name", "Alice").set("age", 36))\n'
                'foo(a=1,b=2,c=3)\n'
                'def work() with {true,}:\n    return\n')
        result = self.formatted(text)
        for expected in ('age: int = 0\n    {\n', 'x: int {0 <= x <= 100}',
                         'user = User()\n    .set("name", "Alice")\n    .set("age", 36)',
                         'foo(a = 1, b = 2, c = 3)', 'with\n{\n    true,\n}:'):
            self.assertIn(expected, result)
        self.assertEqual(self.formatted(result), result)

    def test_comment_code_indentation_is_retained(self):
        text = '# Example:\n#\n# trait Unit[T]:\n#     scale: f64\n#\n# voltage := 950mV\nvalue = 1\n'
        result = self.formatted(text)
        self.assertIn('trait Unit[T]:\n    scale: f64\n\nvoltage := 950mV', result)
        self.assertEqual(self.formatted(result), result)

    def test_long_print_uses_one_formatted_block_and_preserves_every_byte(self):
        from py_compiler.frontend.lexer.lexer import lex
        from py_compiler.syntax.recognition import Syntax
        value = 'prefix {literal} \\ "quoted"\n' * 12
        result = self.formatted('print(' + quoted(value) + ')\n')
        self.assertIn('print(\n    """', result)
        literal = next(token for token in lex(SourceFile('print.sev', result), Syntax()) if token.kind == 'FSTRING')
        self.assertEqual(literal_text(literal), value)
        self.assertEqual(self.formatted(result), result)
        self.assertTrue(all(len(line) <= 80 for line in result.splitlines()))

    def test_nested_container_opens_before_its_vertical_constructor(self):
        text = 'token_forms: list[TokenForm] = [TokenForm("DYNAMIC", "dynamic", recognize=recognize_dynamic)]\n'
        result = self.formatted(text)
        self.assertIn('= [\n    TokenForm(', result)
        self.assertIn('recognize = recognize_dynamic),\n', result)
        self.assertEqual(self.formatted(result), result)

    def test_literal_groups_singleton_shapes_and_not_are_compact(self):
        text = ('class NumericType: Y\n    token_forms: list[TokenForm] = [TokenForm(("NUMERIC"), "numeric", recognize=recognize_numeric)]\n'
                'values = tensor[\n    u8,\n    (\n        3,\n    ),\n](tuple[u8,3](65,66,67))\n'
                'if not(span.start == lexeme.span.start):\n    fail()\n'
                'if not(a and b):\n    fail()\n')
        result = self.formatted(text)
        self.assertNotIn('("NUMERIC")', result)
        self.assertIn('values = tensor[u8, (3,)](tuple[u8, 3](65, 66, 67))', result)
        self.assertIn('if not span.start == lexeme.span.start:', result)
        self.assertIn('if not (a and b):', result)
        self.assertEqual(self.formatted(result), result)

    def test_imports_cases_defaults_and_conditional_arguments_fit(self):
        imports = 'import ' + ', '.join('PublicName' + str(i) for i in range(12)) + ' from "types.sev"\n'
        text = imports + ('match value:\n    case Union | Function | DimensionConstant | DimensionRuntime | ShapeSpread | Another:\n        pass\n'
                          'class Enum:\n    representation: PrimitiveRepresentation = PrimitiveRepresentation.TaggedAggregate\n'
                          'steps.append(package.Step(stdout_fixture=stdout_fixture if package.exists(stdout_fixture) else "", memory_bytes=memory))\n')
        result = self.formatted(text)
        self.assertIn('import PublicName0,\n', result)
        self.assertIn('case Union\n', result)
        self.assertIn('| Another:', result)
        self.assertNotIn('case (', result)
        self.assertTrue(all(len(line) <= 80 for line in result.splitlines()))
        self.assertEqual(self.formatted(result), result)

    def test_nested_calls_collapse_when_the_whole_line_fits(self):
        text = '    print(\n        describe(\n            "width",\n            scale(\n                value = 12.0,\n                factor = 2.0,\n            ),\n        ),\n    )\n'
        result = self.formatted('def work():\n' + text)
        self.assertEqual(result, 'def work():\n    print(describe("width", scale(value = 12.0, factor = 2.0)))\n')

    def test_string_chains_become_implicit_formatted_blocks(self):
        text = 'def describe(line: int, column: int) -> string:\n    result = ""\n    result += (path + ":" + string(line) + ":" + string(column) + "\\n")\n    return result\n'
        result = self.formatted(text)
        self.assertIn('result += """', result)
        self.assertIn('{path}:{line}:{column}', result)
        self.assertNotIn('f"""', result)
        self.assertNotIn('+ string', result)
        self.assertEqual(self.formatted(result), result)

    def test_open_slice_bounds_and_final_condition_colon_fit(self):
        text = 'values = names(0: -1]\n'
        self.assertEqual(self.formatted(text), 'values = names(0:-1]\n')
        condition = 'def work():\n    if a and b or c and ' + 'x' * 24 + ' < ' + 'y' * 41 + ':\n        pass\n'
        result = self.formatted(condition)
        self.assertTrue(all(len(line) <= 80 for line in result.splitlines()))
        self.assertEqual(self.formatted(result), result)

    def test_with_tuple_assignment_keeps_its_target_on_one_line(self):
        text = 'while queue with {i, j = queue.pop(),}:\n    use(i, j)\n'
        result = self.formatted(text)
        self.assertIn('    i, j = queue.pop(),\n', result)
        self.assertEqual(self.formatted(result), result)
