"""Test-local call replacements, specialized through ordinary callable lowering."""
from py_compiler.syntax.generic.block import BlockProvider


class Mock(BlockProvider):
    spelling = 'mock'
    required_scope = 'test'

    def parse_header(self, items):
        if len(items) < 4 or items[1].text != '(' or items[-1].text != ')':
            raise ValueError('mock requires parenthesized call-to-result cases')
        return tuple(items[2:-1])

    def has_body(self, node):
        return False

    def lower(self, node, cfg, env, nodes, index):
        from py_compiler.syntax.block.test import groups
        from py_compiler.syntax.grammar.expression import expression
        cases, fallback = {}, None
        for group in groups(node.header):
            if group[0].text == 'else':
                if fallback is not None:
                    raise ValueError('mock permits one fallback')
                throwing = len(group) > 1 and group[1].text == 'throw'
                fallback = (throwing, expression(group[2:] if throwing else group[1:]))
                continue
            depth, arrow = 0, None
            for position, token in enumerate(group):
                if token.text in ('(', '['):
                    depth += 1
                elif token.text in (')', ']'):
                    depth -= 1
                elif token.text == '->' and depth == 0:
                    arrow = position
                    break
            if arrow is None:
                raise ValueError('mock case requires call -> result')
            call = expression(group[:arrow])
            if call.kind != 'call' or call.operands[0].kind != 'name':
                raise ValueError('mock case requires a named callable')
            entries = cfg.context.lookup_functions(call.operands[0].token.text, cfg.declaration_scope)
            entries = [entry for entry in entries if len(entry.parameters) == len(call.operands) - 1]
            if len(entries) != 1:
                raise ValueError('mock requires one unambiguous callable signature')
            entry = entries[0]
            cases.setdefault(entry.symbol, []).append((call.operands[1:], expression(group[arrow + 1:])))
        if not cases:
            raise ValueError('mock requires at least one call case')
        cfg.mock_cases = dict(getattr(cfg, 'mock_cases', {}))
        for symbol, replacements in cases.items():
            cfg.mock_cases[symbol] = (tuple(replacements), fallback, node.tokens[0])
        cfg.mock_scope = node.identity
        return 1


def replacement(entry, values, cfg, span):
    cases, fallback, token = cfg.mock_cases[entry.symbol]
    from py_compiler.syntax.function.calls import coerce, literal_value
    from py_compiler.syntax.grammar.expression import Expression
    from py_compiler.syntax.generic.owned import select
    from py_compiler.syntax.block.union import Union
    from py_compiler.syntax.complex.error import is_error, unwrap_result, release_temporary
    from py_compiler.hir.thir.src.model import Edge, Terminator
    target = cfg.context.type(entry.result or 'absent')
    if not target.mlir:
        raise ValueError('mock replacement requires a value result')
    result_type = Union().resolve((target, cfg.context.type('Error')), cfg.context)
    done = cfg.block((result_type,))
    for arguments, answer in cases:
        condition = literal_value(True, cfg.context.type('bool'), cfg, span)
        for actual, pattern in zip(values, arguments):
            expected = cfg.expr(pattern, {}, actual.type)
            matches = select(actual.type, 'F.operator', '==').expand(cfg, actual, expected, span)
            condition = select(condition.type, 'F.operator', '&').expand(cfg, condition, matches, span)
        matched, following = cfg.block(), cfg.block()
        cfg.current.terminator = Terminator('conditional', (Edge(matched.identity), Edge(following.identity)), condition)
        cfg.current = matched
        result = coerce(cfg.expr(answer, {}, target), result_type, cfg, span)
        cfg.current.terminator = Terminator('jump', (Edge(done.identity, (result,)),))
        cfg.current = following
    if fallback is None:
        message = Expression('literal', token.with_text('"unmatched mock call"', kind='STRING'))
        error = Expression('call', token, (Expression('name', token.with_text('Error')), message))
        value = cfg.expr(error, {})
    else:
        throwing, term = fallback
        value = cfg.expr(term, {}, None if throwing else target)
        if throwing and not is_error(value.type):
            raise ValueError('mock fallback throw requires an Error value')
        if throwing and value.type not in result_type.variants:
            raise ValueError('mock fallback must use Error declared by the replacement result')
    result = coerce(value, result_type, cfg, span)
    release_temporary(value, cfg, span)
    cfg.current.terminator = Terminator('jump', (Edge(done.identity, (result,)),))
    cfg.current = done
    output = done.parameters[0]
    cfg.owned_results.add(output.identity)
    return unwrap_result(output, cfg, span)


def specialize(entry, cfg):
    from dataclasses import replace
    scope = getattr(cfg, 'mock_scope', cfg.body.identity)
    key = ('mock', entry.symbol, scope)
    specialized = cfg.context.realizations.get(key)
    if specialized is None:
        specialized = replace(entry, body=None, compiling=False,
                              symbol_namespace=entry.symbol_namespace + ':mock:' + scope)
        specialized.mock_cases = cfg.mock_cases
        specialized.mock_scope = scope
        cfg.context.realizations[key] = specialized
        cfg.context.compilers[id(specialized)] = cfg.context.compilers[id(entry)]
    return specialized
