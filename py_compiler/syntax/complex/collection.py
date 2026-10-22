"""Collections own their storage operations and default view policy."""


class Collection:
    parameter_ownership = 'view'

    def require_context(self, cfg):
        pass

    @property
    def binding_ownership(self):
        return 'view'

    def element_place(self, receiver, index, cfg, span):
        raise ValueError(f'{self.name} does not declare an indexed storage grammar')


from py_compiler.syntax.generic.owned import OwnedGrammar, Capture


class CollectionCondition(OwnedGrammar):
    def __init__(self, owner):
        super().__init__(owner, 'F.condition', ('bool', '(', Capture('self', owner), ')'),
                         (Capture('self', owner),))

    def recognize(self, window):
        return None

    def expand(self, value, node, cfg, env):
        from py_compiler.syntax.function.length import Length
        from py_compiler.syntax.function.calls import literal_value
        from py_compiler.syntax.grammar.expression import Expression
        from py_compiler.syntax.complex.tensor import ArgumentValue
        from py_compiler.syntax.generic.owned import select
        captured = Expression('capture', node.token, (ArgumentValue(value),))
        length = Length().call(None, (captured,), cfg, env, node.token.span)
        zero = literal_value(0, length.type, cfg, node.token.span)
        return select(length.type, 'F.operator', '!=').expand(cfg, length, zero, node.token.span)


def indexed_write_preserves_shape(type_, context):
    if type_.family in ('array', 'tensor', 'tuple', 'pointer'):
        return True
    entries = context.functions.get(type_.name + '.set', ())
    return bool(entries) and all(preserves_shape(entry, context) for entry in entries)


def preserves_shape(entry, context, parameter=-1, active=()):
    """Prove element mutation without replacing container fields or storage.

    The proof follows receiver projections and callees; unfamiliar operations
    remain exclusive. Reading an element ends the container projection.
    """
    key = entry.symbol, parameter
    if key in active or entry.compiling:
        return False
    context.compile(entry)
    body = entry.body
    if body is None or getattr(entry, 'c_symbol', None) is not None:
        return False
    position = 0 if parameter == -1 else parameter + int(entry.receiver is not None)
    if position >= len(body.blocks[0].parameters):
        return False
    tracked = {body.blocks[0].parameters[position].identity}
    from py_compiler.syntax.generic.storage import FieldPlace, FieldState
    from py_compiler.syntax.complex.object import Invoke
    changed = True
    while changed:
        before = len(tracked)
        for block in body.blocks:
            for operation in block.operations:
                if not operation.result or operation.kind in ('memory-copy', 'allocate', 'allocate-array',
                        'record-element', 'memory-load', 'tensor-load', 'union-member'):
                    continue
                inputs = operation.operands
                if isinstance(operation.payload, Invoke):
                    callee = operation.payload.dependency
                    sources = getattr(getattr(callee, 'body', None), 'result_sources', ())
                    inputs = tuple(inputs[0 if source == -1 else source + int(callee.receiver is not None)] for source in sources)
                if any(value.identity in tracked for value in inputs):
                    tracked.add(operation.result.identity)
            if block.terminator:
                for edge in block.terminator.edges:
                    for value, target in zip(edge.arguments, body.blocks[edge.target].parameters):
                        if value.identity in tracked:
                            tracked.add(target.identity)
        changed = len(tracked) != before
    for block in body.blocks:
        for operation in block.operations:
            inputs = [i for i, value in enumerate(operation.operands) if value.identity in tracked]
            if not inputs:
                continue
            payload = operation.payload
            if isinstance(payload, FieldPlace) and operation.kind == 'write' and 0 in inputs:
                return False
            if isinstance(payload, FieldState) and payload.initialized is not None and 0 in inputs:
                return False
            if isinstance(payload, Invoke):
                callee = payload.dependency
                if callee is None:
                    return False
                modes = payload.ownership
                for index in inputs:
                    if modes[index] in ('borrow', 'move'):
                        argument = -1 if callee.receiver is not None and index == 0 else index - int(callee.receiver is not None)
                        if modes[index] == 'move' or not preserves_shape(callee, context, argument, (*active, key)):
                            return False
            elif operation.contract and any(effect in ('mutation', 'destruction', 'cleanup', 'call') for effect in operation.contract.effects):
                if operation.kind not in ('memory-store', 'element-store', 'element-clear', 'element-initialized', 'record-element', 'write'):
                    return False
    return True


def lower_literal(node, cfg, env, expected=None):
    """Literal syntax constructs the imported library provider, never storage."""
    if expected is not None and expected.family == 'bool':
        if not node.operands:
            from py_compiler.syntax.function.calls import literal_value
            return literal_value(False, expected, cfg, node.token.span)
        from py_compiler.syntax.block.grammar import boolean_grammar
        return boolean_grammar(lower_literal(node, cfg, env), node, cfg, env)
    from py_compiler.syntax.function.calls import invoke, select
    from py_compiler.syntax.grammar.expression import Expression
    from py_compiler.syntax.complex.tensor import ArgumentValue
    name = node.kind
    if name == 'map' and getattr(getattr(expected,'declaration',None),'template_name','').rsplit('.',1)[-1] == 'dict':
        name = 'dict'
    width = 2 if name in ('map','dict') else 1
    target_name = getattr(getattr(expected, 'declaration', None), 'template_name', '')
    template = cfg.context.record_templates.get(target_name if target_name.rsplit('.', 1)[-1] == name
                                                else cfg.context.declaration_name(name, cfg.declaration_scope))
    if template is None:
        raise ValueError(f'{name} literal requires importing {name} from the collections library')
    types = list(getattr(expected, 'generic_arguments', ())) if expected else []
    if expected and getattr(expected.declaration, 'template_name', '') != template.name:
        raise ValueError(f'{name} literal does not satisfy {expected.name}')
    values = []
    for index, term in enumerate(node.operands):
        position = index % width
        value = cfg.expr(term, env, types[position] if position < len(types) else None)
        if position >= len(types):
            types.append(value.type)
        if value.type != types[position]:
            raise ValueError(f'{name} literal elements have incompatible types')
        values.append(value)
    if len(types) != width:
        raise ValueError(f'empty {name} literal requires a type annotation')
    target = template.realize(tuple(t.name for t in types), cfg.context)
    result = cfg.context.provider_by_name[target.name].instantiate(target.declaration, (), cfg, env, node.token.span)
    method = 'set' if name in ('map','dict') else 'add' if name == 'set' else 'append'
    entries = cfg.context.functions[target.name + '.' + method]
    for index in range(0, len(values), width):
        from py_compiler.syntax.generic.storage import root_binding
        terms = tuple(Expression('capture', node.token, (ArgumentValue(v, root_binding(t,cfg,env)),)) for t,v in zip(node.operands[index:index+width], values[index:index+width]))
        entry, operands = select(entries, terms, cfg, env)
        invoke(entry, operands, cfg, node.token.span, result)
    cfg.owned_results.add(result.identity)
    cfg.value_sources[result.identity] = set()
    return result


def static_method(type_, method, arguments, cfg, env, span):
    from py_compiler.syntax.function.calls import select, invoke
    from py_compiler.syntax.complex.object import Allocate
    entry, values = select(cfg.context.functions.get(type_.name + '.' + method, ()), arguments, cfg, env)
    body = cfg.context.compile(entry)
    receiver = body.blocks[0].parameters[0]
    if any(receiver in op.operands for b in body.blocks for op in b.operations) or any(b.terminator and b.terminator.value == receiver for b in body.blocks):
        raise ValueError('type namespace call requires a method independent of its receiver')
    layout = cfg.context.provider_by_name[type_.name].layout(type_.declaration, cfg.context)
    unused = cfg.emit('allocate', type_, (), Allocate(layout), span)
    return invoke(entry, values, cfg, span, unused)


def lower_comprehension(node, cfg, env, expected=None):
    """Use the same library constructor, append, and for-consumer as statements."""
    from importlib import import_module
    from py_compiler.frontend.parser.blocks import Node
    from py_compiler.syntax.grammar.expression import Expression
    from py_compiler.syntax.complex.tensor import ArgumentValue
    from py_compiler.syntax.generic.place import expression_type
    from py_compiler.syntax.generic.storage import root_binding
    from py_compiler.mir.ownership.flow import Binding
    from py_compiler.syntax.function.calls import select, invoke
    from py_compiler.hir.thir.src.model import Edge, Terminator
    element, target, source, predicate = node.operands
    value = cfg.expr(source, env)
    collection_iterator(value.type, cfg.context)
    if value.type.family == 'iterator':
        item_type = value.type.element
    else:
        matches = [g for g in cfg.context.sentence_grammars
                   if g[1] == ('iterator', '(', ('self', value.type), ')')
                   and g[3] is not None and g[3].family == 'iterator']
        if len(matches) != 1:
            raise ValueError('comprehension requires one typed iterator grammar')
        item_type = matches[0][3].element
    identity = cfg.body.identity + ':comprehension:' + str(cfg.next_value)
    probe = {**env, target.token.text: Binding(identity + ':item', target.token.text, item_type, False, 'view', node.token.span)}
    result_type = expected
    if result_type is None:
        output = expression_type(element, cfg, probe)
        template = cfg.context.record_templates.get(cfg.context.declaration_name('list', cfg.declaration_scope))
        if template is None:
            raise ValueError('comprehension requires importing list from the collections library')
        result_type = template.realize((output.name,), cfg.context)
    result = lower_literal(Expression('list', node.token, ()), cfg, env, result_type)
    name = '<comprehension ' + str(cfg.next_value) + '>'
    binding = Binding(identity, name, result_type, False, 'own', node.token.span)
    cfg.body.bindings.append(binding)
    cfg.values[binding.identity] = result
    cfg.record_lifetime('declare', binding.identity, binding.name)
    env[name] = binding

    class Append:
        def lower(self, statement, builder, loop_env, nodes, index):
            done = None
            if predicate is not None:
                condition = builder.expr(predicate, loop_env, builder.context.type('bool'))
                yes, done = builder.block(), builder.block()
                builder.current.terminator = Terminator('conditional', (Edge(yes.identity), Edge(done.identity)), condition)
                builder.current = yes
            entry, arguments = select(builder.context.functions[result_type.name + '.append'], (element,), builder, loop_env)
            invoke(entry, arguments, builder, node.token.span, builder.read_binding(binding, node.token.span))
            if done is not None:
                builder.current.terminator = Terminator('jump', (Edge(done.identity),))
                builder.current = done
            return 1

    owner = root_binding(source, cfg, env)
    captured = Expression('capture', source.token, (ArgumentValue(value, owner),))
    body = Node(identity + ':append', 'sentence', node.token.span, (node.token,), identity,
                provider=Append(), syntax=cfg.syntax, scope=cfg.block_scope)
    loop = Node(identity + ':loop', 'for', node.token.span, (node.token,), cfg.body.identity,
                children=[body], header=(target.token, captured, ()),
                provider=import_module('py_compiler.syntax.block.for').For(), syntax=cfg.syntax, scope=cfg.block_scope)
    loop.provider.lower(loop, cfg, env, [loop], 0)
    result = cfg.read_binding(binding, node.token.span)
    cfg.flow.move(binding)
    env.pop(name)
    cfg.owned_results.add(result.identity)
    cfg.value_sources[result.identity] = set()
    return result


def primitive_contract(actual, contract, context):
    """Structural primitive contracts check the whole signature, not trait names."""
    if indexed_arguments(actual, context) is not None and collection_contract(actual, contract, context):
        return True
    if (actual.family not in ('integer', 'float', 'string', 'unit', 'byte', 'bool', 'char', 'tuple')
            or contract.declaration.fields or contract.declaration.grammar_requirements):
        return False
    numeric = actual.family in ('integer', 'float', 'unit', 'byte')
    supported = {'equal': ((actual,), context.type('bool'))}
    if numeric:
        supported.update(zero=((),actual), add=((actual,),actual))
    if actual.family in ('integer', 'string', 'byte', 'bool', 'char'):
        supported['hash'] = ((),context.type('usize'))
    if actual.family == 'tuple' and actual.element.family in ('integer', 'byte', 'bool', 'char'):
        supported['hash'] = ((), context.type('usize'))
    for method in contract.declaration.methods:
        key = method.name.rsplit('.',1)[-1]
        from py_compiler.syntax.complex.generic import substitute
        resolve = lambda t: context.type(substitute(t, {'Self':actual}))
        if key not in supported or (tuple(resolve(t) for _,t in method.parameters), resolve(method.result or 'absent')) != supported[key]:
            return False
    return bool(contract.declaration.methods)


def indexed_arguments(actual, context):
    """Key/value types supplied by compiler-owned indexed storage."""
    if actual.family == 'string':
        return context.type('int'), context.type('u8')
    if actual.family in ('array', 'tensor', 'tuple') and hasattr(actual, 'element'):
        return context.type('int'), actual.element
    return None


def collection_namespace(context):
    for alias in context.package_imports:
        if context.c_packages[alias]['package']['name'] == 'core.collections':
            return alias + '.'
    if '_indexed_items' in context.functions:
        return ''
    raise ValueError('collection methods require importing the collections library')


def collection_contract(actual, contract, context):
    if contract.declaration.fields:
        return False
    from py_compiler.syntax.block.trait import contract_annotation
    from py_compiler.syntax.complex.generic import substitute
    key, value = indexed_arguments(actual, context)
    try:
        prefix = collection_namespace(context)
    except ValueError:
        return False
    sequence = lambda t: prefix + 'list[' + t + ']'
    pair = 'tuple[' + key.name + ',' + value.name + ']'
    supported = {
        'len': ((), 'i64'), 'empty': ((), 'bool'),
        'get': ((key.name,), value.name),
        'keys': ((), sequence(key.name)), 'values': ((), sequence(value.name)),
        'items': ((), sequence(pair)), 'enumerate': ((), sequence(pair)),
        'sum': ((), value.name), 'iter': ((), 'iterator[' + value.name + ']'),
    }
    canonical = lambda t: contract_annotation(substitute(t, {'Self': actual}), context)
    from py_compiler.syntax.block.trait import grammar_satisfies
    iterator = context.type('iterator[' + value.name + ']')
    grammars = (
        ((('self', actual), '[', ('key', key), ']'), value),
        (('iterator', '(', ('self', actual), ')'), iterator),
        (('bool', '(', ('self', actual), ')'), context.type('bool')),
    )
    for requirement in contract.declaration.grammar_requirements:
        if not any(grammar_satisfies(requirement, arrangement, result, actual, context)
                   for arrangement, result in grammars):
            return False
    for method in contract.declaration.methods:
        name = method.name.rsplit('.', 1)[-1]
        signature = (tuple(canonical(t) for _, t in method.parameters), canonical(method.result or 'absent'))
        if name not in supported or signature != supported[name]:
            return False
        if any(method.parameter_ownership.get(n, 'view') not in ('view', 'copy') for n, _ in method.parameters):
            return False
    return bool(contract.declaration.methods)


def collection_method(receiver, method, arguments, cfg, env, span, owner):
    """Primitive access stays here; collection algorithms live in lib.sev."""
    pair = indexed_arguments(receiver.type, cfg.context)
    if pair is None or method not in ('get', 'len', 'keys', 'values', 'items', 'enumerate', 'sum', 'empty', 'iter'):
        return None
    from py_compiler.syntax.grammar.expression import Expression
    from py_compiler.syntax.complex.tensor import ArgumentValue
    from py_compiler.frontend.lexer.lexer import lex
    from py_compiler.frontend.source.source import SourceFile
    from py_compiler.syntax.function.calls import select, invoke
    token = lex(SourceFile(cfg.source.path + ':collection-method', method), cfg.syntax)[0]
    capture = Expression('capture', token, (ArgumentValue(receiver, owner),))
    if method == 'get':
        if len(arguments) != 1:
            raise ValueError('get requires one container key')
        if receiver.type.family in ('tensor', 'tuple'):
            key = cfg.expr(arguments[0], env, pair[0])
            return cfg.emit('tensor-flat-get', pair[1], (receiver, key), FlatGet(), span)
        value = receiver.type.element_place(receiver, arguments[0], cfg, env, span).read(cfg, span)
        from py_compiler.syntax.generic.storage import owns_storage
        if owns_storage(value.type):
            value = value.type.copy_value(value, cfg, span)
            cfg.owned_results.add(value.identity)
        return value
    if arguments:
        raise ValueError(method + ' takes no arguments')
    if method == 'len':
        from py_compiler.syntax.function.length import Length
        return Length().call(None, (capture,), cfg, env, span)
    prefix = collection_namespace(cfg.context)
    if method == 'iter':
        collection_iterator(receiver.type, cfg.context)
    entries = cfg.context.lookup_functions(prefix + '_indexed_' + method, cfg.declaration_scope)
    explicit = (receiver.type.name,) if method in ('keys', 'empty') else (pair[1].name, receiver.type.name)
    entry, operands = select(entries, (capture,), cfg, env, explicit)
    return invoke(entry, operands, cfg, span)


def collection_iterator(type_, context):
    """Attach the library's indexed traversal grammar to primitive storage."""
    pair = indexed_arguments(type_, context)
    if pair is None:
        return
    arrangement = ('iterator', '(', ('self', type_), ')')
    if any(d[1] == arrangement for d in context.sentence_grammars):
        return
    prefix = collection_namespace(context)
    provider = context.record_templates[prefix + '_Indexed']
    adapter = provider.realize((pair[1].name, type_.name), context)
    original = ('iterator', '(', ('self', adapter), ')')
    matches = [d for d in context.sentence_grammars if d[1] == original]
    if len(matches) != 1:
        raise ValueError('indexed storage requires one library iteration grammar')
    node, _, _, result = matches[0]
    context.sentence_grammars.append((node, arrangement, (), result))


def primitive_method(receiver, method, arguments, cfg, env, span, owner=None):
    from py_compiler.syntax.generic.owned import select
    if receiver.type.family == 'tuple' and method in ('equal', 'hash'):
        if len(arguments) != (1 if method == 'equal' else 0):
            raise ValueError('tuple ' + method + ' has an invalid argument count')
        from py_compiler.syntax.grammar.expression import Expression
        from py_compiler.syntax.complex.tensor import ArgumentValue
        from py_compiler.frontend.lexer.lexer import lex
        from py_compiler.frontend.source.source import SourceFile
        from py_compiler.syntax.function.calls import select as select_call, invoke
        token = lex(SourceFile(cfg.source.path + ':tuple-method', method), cfg.syntax)[0]
        capture = Expression('capture', token, (ArgumentValue(receiver, owner),))
        prefix = collection_namespace(cfg.context)
        entries = cfg.context.lookup_functions(prefix + ('equal' if method == 'equal' else '_tuple_hash'), cfg.declaration_scope)
        if method == 'equal':
            from py_compiler.syntax.generic.storage import root_binding
            other = cfg.expr(arguments[0], env)
            arguments = (Expression('capture', token, (ArgumentValue(other, root_binding(arguments[0], cfg, env)),)),)
            types = receiver.type.name, other.type.name
        else:
            types = (receiver.type.name,)
        entry, operands = select_call(entries, (capture, *arguments), cfg, env, types)
        return invoke(entry, operands, cfg, span)
    numeric = receiver.type.family in ('integer', 'float', 'unit', 'byte')
    if (numeric and method in ('add', 'equal') or receiver.type.family in ('bool', 'char') and method == 'equal') and len(arguments) == 1:
        other = cfg.expr(arguments[0], env, receiver.type)
        return select(receiver.type,'F.operator','+' if method=='add' else '==').expand(cfg,receiver,other,span)
    if receiver.type.family == 'string' and method == 'hash' and not arguments:
        return cfg.emit('string-hash',cfg.context.type('usize'),(receiver,),StringHash(),span)
    if receiver.type.family in ('integer', 'byte', 'bool', 'char') and method == 'hash' and not arguments:
        target = cfg.context.type('usize')
        return cfg.emit('integer-hash',target,(receiver,),IntegerHash(),span)
    return None


def is_sequence(type_):
    if type_.family in ('tuple', 'tensor'):
        return getattr(type_, 'rank', 1) == 1
    template = getattr(getattr(type_, 'declaration', None), 'template_name', '')
    return type_.family == 'record' and template.rsplit('.', 1)[-1] in ('list', 'tuple')


def sequence_equality(node, left, cfg, env):
    from py_compiler.syntax.grammar.expression import Expression
    from py_compiler.syntax.complex.tensor import ArgumentValue
    from py_compiler.syntax.generic.storage import root_binding
    from py_compiler.syntax.function.calls import select as select_call, invoke
    from py_compiler.syntax.generic.owned import select
    right = cfg.expr(node.operands[1], env)
    if not is_sequence(right.type):
        raise ValueError('sequence equality requires another list, tuple, or one-dimensional tensor')
    temporaries = []
    def capture_value(value, term):
        from py_compiler.mir.ownership.flow import Binding
        owner = root_binding(term, cfg, env) or cfg.call_result_owners.get(value.identity)
        if owner is None and callable(getattr(value.type, 'release', None)):
            name = f'<sequence operand {value.identity}>'
            owner = Binding(cfg.body.identity + ':' + name, name, value.type, False, 'own', node.token.span)
            env[name] = owner
            cfg.local_names.add(name)
            cfg.values[owner.identity] = value
            cfg.body.bindings.append(owner)
            cfg.record_lifetime('declare', owner.identity, name)
            temporaries.append(owner)
        return Expression('capture', node.token, (ArgumentValue(value, owner),))

    def finish(result):
        cfg.cleanup_memory(temporaries, node.token.span)
        for owner in temporaries:
            env.pop(owner.name, None)
            cfg.local_names.discard(owner.name)
        return result

    capture = capture_value(left, node.operands[0])
    other = capture_value(right, node.operands[1])
    from py_compiler.syntax.complex.tuple import ProductType
    if isinstance(left.type, ProductType) or isinstance(right.type, ProductType):
        product = left.type if isinstance(left.type, ProductType) else right.type
        count = len(product.generic_arguments)
        remaining = right.type if product is left.type else left.type
        other_count = len(remaining.generic_arguments) if isinstance(remaining, ProductType) else getattr(remaining, 'count', None)
        if other_count is not None and count != other_count:
            from py_compiler.syntax.function.calls import literal_value
            return finish(literal_value(node.token.text == '!=', cfg.context.type('bool'), cfg, node.token.span))
        def length(term):
            name = Expression('name', node.token.with_text('len', kind='IDENTIFIER'))
            return Expression('call', node.token, (name, term))
        comparison = Expression('binary', node.token.with_text('=='), (length(capture), length(other)))
        for index in range(count):
            position = Expression('literal', node.token.with_text(str(index), kind='NUMBER'))
            elements = (Expression('index', node.token, (capture, position)),
                        Expression('index', node.token, (other, position)))
            equal = Expression('binary', node.token.with_text('=='), elements)
            comparison = Expression('binary', node.token.with_text('and'), (comparison, equal))
        result = cfg.expr(comparison, env)
        return finish(result if node.token.text == '==' else select(result.type, 'F.unary', 'not').expand(cfg, result, node.token.span))
    entries = cfg.context.lookup_functions(collection_namespace(cfg.context) + 'equal', cfg.declaration_scope)
    entry, operands = select_call(entries, (capture, other), cfg, env,
                                 (left.type.name, right.type.name))
    result = invoke(entry, operands, cfg, node.token.span)
    return finish(result if node.token.text == '==' else select(result.type, 'F.unary', 'not').expand(cfg, result, node.token.span))


from dataclasses import dataclass


@dataclass(frozen=True)
class IntegerHash:
    """Hash the integer's bits; negative keys are valid, not conversions."""
    def atom(self, operation):
        from py_compiler.syntax.generic.atom import Atom
        return Atom(self, (operation.operands[0].type,), operation.result.type, ('copy',), (), self)

    def render_operation(self, operation):
        from py_compiler.mlir.src.cfg import name, temporary_suffix
        source, result = operation.operands[0], operation.result
        p = '%integer_hash_' + temporary_suffix(operation)
        source_type, target_type = source.type.mlir, result.type.mlir
        value, lines = name(source), []
        width = result.type.bits
        if source_type == 'index':
            source_type = 'i' + str(source.type.bits)
            value = p + '_integer'
            lines.append(f'{value} = arith.index_cast {name(source)} : index to {source_type}')
        if source.type.bits < width:
            opcode = 'extsi' if source.type.signed else 'extui'
            return lines + [f'{name(result)} = arith.{opcode} {value} : {source_type} to {target_type}']
        if source.type.bits == width:
            return lines + [f'{p}_zero = arith.constant 0 : {target_type}',
                            f'{name(result)} = arith.xori {value}, {p}_zero : {target_type}']
        lines.append(f'{p}_low = arith.trunci {value} : {source_type} to {target_type}')
        folded = p + '_low'
        for offset in range(width, source.type.bits, width):
            part = p + '_' + str(offset)
            lines += [f'{part}_shift = arith.constant {offset} : {source_type}',
                      f'{part}_wide = arith.shrui {value}, {part}_shift : {source_type}',
                      f'{part} = arith.trunci {part}_wide : {source_type} to {target_type}',
                      f'{part}_folded = arith.xori {folded}, {part} : {target_type}']
            folded = part + '_folded'
        lines += [f'{p}_zero = arith.constant 0 : {target_type}',
                  f'{name(result)} = arith.xori {folded}, {p}_zero : {target_type}']
        return lines


@dataclass(frozen=True)
class FlatGet:
    """Read a row-major logical offset, respecting a tensor view's strides."""
    def atom(self, operation):
        from py_compiler.syntax.generic.atom import Atom
        return Atom(self, tuple(v.type for v in operation.operands), operation.result.type,
                    ('view', 'copy'), ('read',), self)

    def render_operation(self, operation):
        from py_compiler.mlir.src.cfg import name, temporary_suffix
        source, offset = operation.operands
        p = '%flat_' + temporary_suffix(operation)
        lines = [f'{p}_zero = arith.constant 0 : index',
                 f'{p}_one = arith.constant 1 : index',
                 f'{p}_offset = arith.index_cast {name(offset)} : {offset.type.mlir} to index',
                 f'{p}_nonnegative = arith.cmpi sge, {p}_offset, {p}_zero : index',
                 f'cf.assert {p}_nonnegative, "container index must be nonnegative"']
        count = p + '_one'
        for axis in range(source.type.rank):
            lines += [f'{p}_axis{axis} = arith.constant {axis} : index',
                      f'{p}_dim{axis} = memref.dim {name(source)}, {p}_axis{axis} : {source.type.mlir}',
                      f'{p}_count{axis} = arith.muli {count}, {p}_dim{axis} : index']
            count = p + f'_count{axis}'
        lines += [f'{p}_bounded = arith.cmpi ult, {p}_offset, {count} : index',
                  f'cf.assert {p}_bounded, "container index outside storage"']
        remaining = p + '_offset'
        indices = []
        for axis in reversed(range(source.type.rank)):
            index, quotient = p + f'_index{axis}', p + f'_quotient{axis}'
            lines += [f'{index} = arith.remui {remaining}, {p}_dim{axis} : index',
                      f'{quotient} = arith.divui {remaining}, {p}_dim{axis} : index']
            indices.insert(0, index)
            remaining = quotient
        lines.append(f'{name(operation.result)} = memref.load {name(source)}[{", ".join(indices)}] : {source.type.mlir}')
        return lines


@dataclass(frozen=True)
class StringHash:
    def atom(self, operation):
        from py_compiler.syntax.generic.atom import Atom
        return Atom(self,(operation.operands[0].type,),operation.result.type,('view',),('read',),self)

    def render_operation(self, operation):
        from py_compiler.mlir.src.cfg import name, temporary_suffix
        p = '%hash_' + temporary_suffix(operation)
        value, result = operation.operands[0], operation.result
        type_ = result.type.mlir
        return [f'{p}_zero = arith.constant 0 : index',
                f'{p}_one = arith.constant 1 : index',
                f'{p}_size = memref.dim {name(value)}, {p}_zero : {value.type.mlir}',
                f'{p}_seed = arith.constant 2166136261 : {type_}',
                f'{p}_prime = arith.constant 16777619 : {type_}',
                f'{name(result)} = scf.for {p}_i = {p}_zero to {p}_size step {p}_one iter_args({p}_acc = {p}_seed) -> ({type_}) {{',
                f'  {p}_byte = memref.load {name(value)}[{p}_i] : {value.type.mlir}',
                f'  {p}_wide = arith.extui {p}_byte : i8 to {type_}',
                f'  {p}_xor = arith.xori {p}_acc, {p}_wide : {type_}',
                f'  {p}_next = arith.muli {p}_xor, {p}_prime : {type_}',
                f'  scf.yield {p}_next : {type_}', '}']
