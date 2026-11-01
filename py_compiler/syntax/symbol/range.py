from py_compiler.syntax.symbol.symbol import Symbol, TokenForm


class Range(Symbol):
    token_forms = (TokenForm('RANGE', '..'),)

    def grammars(self):
        return (*super().grammars(), RangeOperation(self))


from py_compiler.syntax.generic.owned import OwnedGrammar


class RangeOperation(OwnedGrammar):
    spelling = '..'
    precedence = 3.5
    associativity = 'left'

    def __init__(self, owner):
        super().__init__(owner, 'F.operator', ('..',))

    def recognize(self, window):
        from py_compiler.syntax.generic.grammar import Match
        if len(window.tokens) == 1 and window.tokens[0].text == '..':
            return Match(self, window, window.end, {})
        return None

    def lower_expression(self, cfg, node, env):
        return lower_interval(node, cfg, env)


def interval_expression(node, cfg):
    from py_compiler.syntax.grammar.expression import Expression
    from py_compiler.syntax.complex.collection import collection_namespace
    closed = node.interval or (True, True)
    flags = tuple(Expression('literal', node.token.with_text('true' if flag else 'false')) for flag in closed)
    callee = Expression('name', node.token.with_text(collection_namespace(cfg.context) + '_Interval', kind='IDENTIFIER'))
    return Expression('call', node.token, (callee, *node.operands, *flags))


def lower_interval(node, cfg, env):
    return cfg.expr(interval_expression(node, cfg), env)


def collection_item_types(type_, context):
    from py_compiler.syntax.complex.collection import indexed_arguments
    primitive = indexed_arguments(type_, context)
    if primitive is not None:
        return primitive
    entries = context.functions.get(type_.name + '.get', ())
    if entries and entries[0].result:
        entry = entries[0]
        return context.type(entry.parameters[0][1]), context.type(entry.result)
    return None


def endpoint_type(type_, context):
    pair = collection_item_types(type_, context)
    return context.type('tuple[' + ','.join(t.name for t in pair) + ']') if pair else None


def domain_axes(type_, context):
    """The full coordinate domain of nested lists and ranked storage."""
    if type_.family == 'tensor':
        return [('tensor', axis) for axis in range(type_.rank)]
    result = []
    while True:
        pair = collection_item_types(type_, context)
        if pair is None:
            raise ValueError(type_.name + ' has no collection index domain')
        template = getattr(getattr(type_, 'declaration', None), 'template_name', '').rsplit('.', 1)[-1]
        if pair[0].family != 'integer' or template in ('map', 'dict', 'set', 'count'):
            if result:
                raise ValueError('nested noninteger keys require explicit key traversal')
            return [('keys', pair[0])]
        result.append(('list', None))
        child = pair[1]
        template = getattr(getattr(child, 'declaration', None), 'template_name', '').rsplit('.', 1)[-1]
        if template != 'list':
            return result
        type_ = child


def endpoint_interval(node):
    """..C.last uses the endpoint item's K, not the index domain of its V."""
    from py_compiler.syntax.grammar.expression import Expression
    term = node.operands[0]
    if term.kind != 'member' or term.token.text not in ('first', 'last'):
        return None
    zero = Expression('literal', node.token.with_text('0', kind='NUMBER'))
    key = Expression('index', node.token, (term, zero))
    return Expression('interval', node.token, (zero, key), interval=(True, True))


def special_type(node, cfg, env):
    from py_compiler.syntax.complex.collection import collection_namespace
    from py_compiler.syntax.generic.place import expression_type
    prefix = collection_namespace(cfg.context)
    if node.kind != 'index_domain' or endpoint_interval(node) is not None:
        return cfg.context.type(prefix + '_Interval')
    axes = domain_axes(expression_type(node.operands[0], cfg, env), cfg.context)
    item = axes[0][1].name if axes[0][0] == 'keys' else 'int' if len(axes) == 1 else 'tuple[' + ','.join(['int'] * len(axes)) + ']'
    return cfg.context.type(prefix + 'list[' + item + ']')


def lowered_statements(text, node, cfg):
    from py_compiler.frontend.source.source import SourceFile
    from py_compiler.frontend.lexer.lexer import lex
    from py_compiler.frontend.parser.blocks import parse_blocks
    source = SourceFile(cfg.source.path + ':index-domain', text + '\n')
    root, errors = parse_blocks(source, lex(source, cfg.syntax), cfg.syntax)
    if errors:
        raise ValueError(str(errors[0]))
    def attach(children):
        for child in children:
            child.scope = cfg.block_scope
            child.parent = cfg.body.identity
            attach(child.children)
    attach(root.children)
    return root.children


def domain_iterator(node, cfg, env):
    """A synchronous producer keeps direct index loops allocation-free."""
    from types import SimpleNamespace
    from py_compiler.frontend.parser.blocks import Node
    from py_compiler.syntax.generic.place import expression_type
    source_type = expression_type(node.operands[0], cfg, env)
    axes = domain_axes(source_type, cfg.context)
    if axes[0][0] == 'keys':
        return None
    item = 'int' if len(axes) == 1 else 'tuple[' + ','.join(['int'] * len(axes)) + ']'
    lines, indices, subject = [], [], 'self'
    for depth, (kind, axis) in enumerate(axes):
        index = '__coordinate_' + str(depth)
        extent = subject + '.dim(' + str(axis) + ')' if kind == 'tensor' else 'len(' + subject + ')'
        lines.append('    ' * depth + 'while ' + index + ' < ' + extent + ' with ' + index + ' = 0:')
        indices.append(index)
        if kind == 'list': subject += '[' + index + ']'
    value = indices[0] if len(indices) == 1 else '(' + ','.join(indices) + ')'
    lines.append('    ' * len(indices) + 'yield ' + value)
    for depth in reversed(range(len(indices))):
        lines.append('    ' * (depth + 1) + indices[depth] + ' += 1')
    identity = cfg.body.identity + ':domain-producer:' + str(node.token.span.start) + ':' + str(cfg.next_value)
    producer = Node(identity, 'grammar', node.token.span, (node.token,), cfg.body.identity,
                    children=lowered_statements('\n'.join(lines), node, cfg), syntax=cfg.syntax, scope=cfg.block_scope)
    return SimpleNamespace(node=producer, arrangement=('indices', '(', ('self', source_type), ')'),
                           result=cfg.context.type('iterator[' + item + ']'))


def lower_domain(node, cfg, env):
    from py_compiler.syntax.grammar.expression import Expression
    from py_compiler.syntax.complex.collection import lower_literal
    from py_compiler.syntax.generic.storage import root_binding
    from py_compiler.mir.ownership.flow import Binding
    endpoint = endpoint_interval(node)
    if endpoint is not None:
        return lower_interval(endpoint, cfg, env)
    source = node.operands[0]
    value = cfg.expr(source, env)
    axes = domain_axes(value.type, cfg.context)
    if axes[0][0] == 'keys':
        from py_compiler.syntax.complex.tensor import ArgumentValue
        captured = Expression('capture', source.token, (ArgumentValue(value, root_binding(source, cfg, env)),))
        return cfg.expr(Expression('call', node.token, (Expression('member', node.token.with_text('keys'), (captured,)),)), env)
    target = special_type(node, cfg, env)
    result = lower_literal(Expression('list', node.token), cfg, env, target)
    suffix = str(cfg.next_value)
    source_name, result_name = '__domain_source_' + suffix, '__domain_result_' + suffix
    local = dict(env)
    owner = root_binding(source, cfg, env)
    source_binding = Binding(cfg.body.identity + ':' + source_name, source_name, value.type, False, 'view' if owner else 'own', node.token.span)
    result_binding = Binding(cfg.body.identity + ':' + result_name, result_name, target, False, 'own', node.token.span)
    for binding, captured in ((source_binding, value), (result_binding, result)):
        local[binding.name] = binding
        cfg.values[binding.identity] = captured
        cfg.body.bindings.append(binding)
        cfg.record_lifetime('declare', binding.identity, binding.name)
    if owner:
        cfg.flow.loan(source_binding, owner)
    lines, indices, subject = [], [], source_name
    for depth, (kind, axis) in enumerate(axes):
        index = '__domain_index_' + suffix + '_' + str(depth)
        extent = subject + '.dim(' + str(axis) + ')' if kind == 'tensor' else 'len(' + subject + ')'
        lines.append('    ' * depth + 'while ' + index + ' < ' + extent + ' with ' + index + ' = 0:')
        indices.append(index)
        if kind == 'list':
            subject += '[' + index + ']'
    item = indices[0] if len(indices) == 1 else '(' + ','.join(indices) + ')'
    lines.append('    ' * len(indices) + result_name + '.append(' + item + ')')
    for depth in reversed(range(len(indices))):
        lines.append('    ' * (depth + 1) + indices[depth] + ' += 1')
    cfg.scope(lowered_statements('\n'.join(lines), node, cfg), local, cleanup=False)
    result = cfg.read_binding(result_binding, node.token.span)
    cfg.flow.move(result_binding)
    cfg.cleanup_memory((source_binding,), node.token.span)
    cfg.flow.end_scope(cfg.live_binding_ids(env) | {binding.identity for scope in cfg.active_scopes for binding in scope.values()})
    cfg.owned_results.add(result.identity)
    cfg.value_sources[result.identity] = set()
    return result


def lower_special(node, cfg, env, expected=None):
    if node.kind == 'interval':
        return lower_interval(node, cfg, env)
    if node.kind == 'index_domain':
        return lower_domain(node, cfg, env)
    from py_compiler.syntax.grammar.expression import Expression
    constructor, initializer = node.operands
    value = cfg.expr(Expression('call', node.token, (constructor,)), env, expected)
    if value.type.family != 'tensor':
        raise ValueError('fill initializer requires a tensor')
    fill = cfg.expr(initializer, env, value.type.element)
    cfg.effect('tensor-fill', (value, fill), TensorFill(), node.token.span)
    return value


def domain_membership(node, cfg, env):
    from py_compiler.syntax.grammar.expression import Expression
    from py_compiler.syntax.complex.tensor import ArgumentValue
    from py_compiler.syntax.generic.place import expression_type
    left, domain = node.operands
    if domain.kind == 'index_domain':
        interval = endpoint_interval(domain)
        if interval is not None:
            domain = interval
        else:
            axes = domain_axes(expression_type(domain.operands[0], cfg, env), cfg.context)
            if axes[0][0] == 'keys':
                return None
            if len(axes) > 1:
                return coordinate_membership(left, domain.operands[0], axes, node, cfg, env)
            if left.kind == 'tuple':
                # Several independent cursors share one one-dimensional domain.
                # Capture their values and the collection length exactly once.
                positions = [cfg.expr(item, env, cfg.context.type('int')) for item in left.operands]
                length = Expression('call', node.token, (Expression('name', node.token.with_text('len')), domain.operands[0]))
                extent = cfg.expr(length, env)
                bound = Expression('capture', node.token, (ArgumentValue(extent),))
                zero = Expression('literal', node.token.with_text('0', kind='NUMBER'))
                condition = Expression('literal', node.token.with_text('true'))
                for position in positions:
                    captured = Expression('capture', node.token, (ArgumentValue(position),))
                    lower = Expression('binary', node.token.with_text('>='), (captured, zero))
                    upper = Expression('binary', node.token.with_text('<'), (captured, bound))
                    inside = Expression('binary', node.token.with_text('and'), (lower, upper))
                    condition = Expression('binary', node.token.with_text('and'), (condition, inside))
                return cfg.expr(condition, env)
            position = cfg.expr(left, env, cfg.context.type('int'))
            captured = Expression('capture', left.token, (ArgumentValue(position),))
            zero = Expression('literal', node.token.with_text('0', kind='NUMBER'))
            length = Expression('call', node.token, (Expression('name', node.token.with_text('len')), domain.operands[0]))
            lower = Expression('binary', node.token.with_text('>='), (captured, zero))
            upper = Expression('binary', node.token.with_text('<'), (captured, length))
            return cfg.expr(Expression('binary', node.token.with_text('and'), (lower, upper)), env)
    if domain.kind != 'interval' and not (domain.kind == 'binary' and domain.token.text == '..'):
        return None
    value = cfg.expr(left, env, cfg.context.type('int'))
    captured = Expression('capture', left.token, (ArgumentValue(value),))
    closed = domain.interval or (True, True)
    lower = Expression('binary', node.token.with_text('>=' if closed[0] else '>'), (captured, domain.operands[0]))
    upper = Expression('binary', node.token.with_text('<=' if closed[1] else '<'), (captured, domain.operands[1]))
    return cfg.expr(Expression('binary', node.token.with_text('and'), (lower, upper)), env)


def coordinate_membership(left, source, axes, node, cfg, env):
    """Check each axis before accessing the next nested collection."""
    from py_compiler.syntax.grammar.expression import Expression
    from py_compiler.syntax.complex.tensor import ArgumentValue
    from py_compiler.syntax.generic.storage import root_binding, is_unbound_temporary
    from py_compiler.mir.ownership.flow import Binding
    temporaries = []
    def capture(term):
        value = cfg.expr(term, env)
        owner = root_binding(term, cfg, env) or cfg.call_result_owners.get(value.identity)
        if is_unbound_temporary(value, cfg):
            owner = None
        if owner is None:
            name = '<coordinate operand ' + str(value.identity) + '>'
            owner = Binding(cfg.body.identity + ':' + name, name, value.type, False, 'own', node.token.span)
            env[name] = owner
            cfg.local_names.add(name)
            cfg.values[owner.identity] = value
            cfg.body.bindings.append(owner)
            cfg.record_lifetime('declare', owner.identity, name)
            cfg.call_result_owners[value.identity] = owner
            cfg.call_result_modes[value.identity] = 'view'
            temporaries.append(owner)
        return Expression('capture', term.token, (ArgumentValue(value, owner),))
    if left.kind == 'tuple':
        if len(left.operands) != len(axes):
            raise ValueError('coordinate count must match collection dimensions')
        positions = left.operands
    else:
        from py_compiler.syntax.generic.place import expression_type
        from py_compiler.syntax.complex.tuple import ProductType, TupleValue
        coordinate_type = expression_type(left, cfg, env)
        count = len(coordinate_type.generic_arguments) if isinstance(coordinate_type, ProductType) else coordinate_type.count if isinstance(coordinate_type, TupleValue) else None
        if count != len(axes):
            raise ValueError('coordinate count must match collection dimensions')
        coordinate = capture(left)
        positions = tuple(Expression('index', node.token, (coordinate,
                          Expression('literal', node.token.with_text(str(index), kind='NUMBER'))))
                          for index in range(len(axes)))
    indices = [Expression('capture', position.token, (ArgumentValue(cfg.expr(position, env, cfg.context.type('int'))),))
               for position in positions]
    subject = capture(source)
    zero = Expression('literal', node.token.with_text('0', kind='NUMBER'))
    condition = Expression('literal', node.token.with_text('true'))
    for position, (kind, axis) in zip(indices, axes):
        if kind == 'tensor':
            dimension = Expression('literal', node.token.with_text(str(axis), kind='NUMBER'))
            extent = Expression('call', node.token, (Expression('member', node.token.with_text('dim'), (subject,)), dimension))
        else:
            extent = Expression('call', node.token, (Expression('name', node.token.with_text('len')), subject))
        lower = Expression('binary', node.token.with_text('>='), (position, zero))
        upper = Expression('binary', node.token.with_text('<'), (position, extent))
        inside = Expression('binary', node.token.with_text('and'), (lower, upper))
        condition = Expression('binary', node.token.with_text('and'), (condition, inside))
        if kind == 'list':
            subject = Expression('index', node.token, (subject, position))
    result = cfg.expr(condition, env)
    cfg.cleanup_memory(temporaries, node.token.span)
    for owner in temporaries:
        env.pop(owner.name, None)
        cfg.local_names.discard(owner.name)
    return result


class TensorFill:
    def atom(self, operation):
        from py_compiler.syntax.generic.atom import Atom
        return Atom(self, tuple(v.type for v in operation.operands), None, ('borrow', 'copy'), ('mutation',), self)

    def render_operation(self, operation):
        from py_compiler.mlir.src.cfg import name, temporary_suffix
        receiver, value = operation.operands
        prefix = '%fill_' + temporary_suffix(operation)
        lines = [prefix + '_zero = arith.constant 0 : index', prefix + '_one = arith.constant 1 : index']
        indices = []
        for axis in range(receiver.type.rank):
            lines += [f'{prefix}_axis{axis} = arith.constant {axis} : index',
                      f'{prefix}_dim{axis} = memref.dim {name(receiver)}, {prefix}_axis{axis} : {receiver.type.mlir}',
                      f'scf.for {prefix}_i{axis} = {prefix}_zero to {prefix}_dim{axis} step {prefix}_one {{']
            indices.append(prefix + '_i' + str(axis))
        lines.append(f'memref.store {name(value)}, {name(receiver)}[{", ".join(indices)}] : {receiver.type.mlir}')
        lines.extend('}' for _ in indices)
        return lines


import unittest


class IndexDomainSyntaxTests(unittest.TestCase):
    def test_endpoint_flags_and_membership_precedence(self):
        from py_compiler.frontend.source.source import SourceFile
        from py_compiler.frontend.lexer.lexer import lex
        from py_compiler.frontend.parser.blocks import parse_blocks
        from py_compiler.syntax.grammar.expression import expression
        from py_compiler.syntax.recognition import Syntax
        syntax = Syntax()
        for spelling, flags in (('[0..3]', (True, True)), ('[0..3)', (True, False)),
                                ('(0..3]', (False, True)), ('(0..3)', (False, False))):
            source = SourceFile('interval.sev', 'value in ' + spelling)
            term = expression([t for t in lex(source, syntax) if t.kind not in ('NEWLINE', 'EOF')])
            self.assertEqual(term.token.text, 'in')
            self.assertEqual(term.operands[1].interval, flags)
            loop = SourceFile('interval-loop.sev', 'while value in ' + spelling + ' with value = 0:\n    value += 1\n')
            self.assertFalse(parse_blocks(loop, lex(loop, syntax), syntax)[1])

    def test_domain_endpoint_and_coordinate_projection(self):
        from py_compiler.frontend.source.source import SourceFile
        from py_compiler.frontend.lexer.lexer import lex
        from py_compiler.syntax.grammar.expression import expression
        from py_compiler.syntax.recognition import Syntax
        syntax = Syntax()
        source = SourceFile('endpoint.sev', '..grid.last')
        term = expression([t for t in lex(source, syntax) if t.kind not in ('NEWLINE', 'EOF')])
        self.assertEqual(term.kind, 'index_domain')
        self.assertEqual(term.operands[0].token.text, 'last')
        interval = endpoint_interval(term)
        self.assertEqual(interval.interval, (True, True))
        self.assertEqual(interval.operands[1].operands[1].token.text, '0')
