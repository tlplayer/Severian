"""Error owns its nominal type declaration and existing pointer representation."""
from py_compiler.syntax.complex.object import ObjectType, NamedDefinition


class ErrorType(ObjectType):
    def __init__(self):
        from py_compiler.syntax.block.record import TypeDeclaration
        declaration = TypeDeclaration('prelude.Error', 'Error', 'class',
            (('message', 'string'), ('call_stack', 'string')), (), (), None,
            defaults=(None, ''))
        super().__init__('Error', 'record', '!llvm.ptr', declaration=declaration, owns_fields=True)

    def install(self, context):
        from importlib import import_module
        context.by_name[self.name] = self.declaration
        context.provider_by_name[self.name] = import_module('py_compiler.syntax.block.class').Class()
        context.by_name['E'] = self.declaration
        context.provider_by_name['E'] = context.provider_by_name[self.name]

    def raise_failure(self, arguments, cfg, span):
        from py_compiler.syntax.grammar.expression import Expression
        token = arguments[0].token if arguments else None
        if token is None:
            raise ValueError('Error requires a message')
        term = Expression('call', token, (Expression('name', token.with_text('Error')), *arguments))
        throw_expression(term, cfg, {}, span)


def is_error(type_):
    return type_ is not None and (type_.name == 'Error' or
        'Error' in getattr(getattr(type_, 'declaration', None), 'traits', ()))


def error_members(type_):
    return tuple(t for t in (type_.variants if type_ and type_.family == 'union' else (type_,)) if is_error(t))


def throw_expression(term, cfg, env, span):
    # Existing untyped failures keep their diagnostic-only boundary. Typed
    # results carry ordinary class storage through the existing union ABI.
    import ast
    if (term.kind == 'call' and term.operands[0].token.text == 'Error'
            and not error_members(cfg.body.result_type) and not getattr(cfg, 'typed_error_handlers', ())
            and not getattr(getattr(cfg,'hooks',None),'groups',())
            and len(term.operands) == 2 and term.operands[1].kind == 'literal'
            and term.operands[1].token.kind == 'STRING'):
        raise_error(cfg, ast.literal_eval(term.operands[1].token.text), span)
        return
    value = cfg.expr(term, env)
    if not is_error(value.type):
        raise ValueError('throw requires an Error value')
    forward_error(value, cfg, span)



def release_temporary(value, cfg, span):
    if value.identity not in cfg.owned_results:
        return
    live = [b for scope in cfg.active_scopes for b in scope.values() if b.identity not in cfg.flow.moved]
    if any(cfg.values.get(b.identity) is value for b in live):
        return
    from py_compiler.syntax.generic.storage import release_storage
    release_storage(value, cfg, span)

def forward_error(value, cfg, span):
    from py_compiler.syntax.function.calls import coerce
    from py_compiler.syntax.generic.storage import copy_storage, release_storage
    from py_compiler.hir.thir.src.model import Terminator
    if getattr(cfg, 'typed_error_handlers', ()):
        cfg.typed_error_handlers[-1](value, span)
        return
    if getattr(cfg,'hooks',None):
        cfg.hooks.finish(error=value)
    expected = cfg.body.result_type
    if value.type in error_members(expected):
        result = coerce(value, expected, cfg, span) if value.type != expected else copy_storage(value, cfg, span)
        release_temporary(value, cfg, span)
        cfg.run_defers()
        if cfg.phases:
            cfg.phases.leave()
        cfg.cleanup_memory((b for scope in cfg.active_scopes for b in scope.values()), span)
        cfg.current.terminator = Terminator('finish', value=result, span=span)
        return
    # An unhandled error is reported at the outer boundary, never silently
    # converted into a success value.
    if value.type.name == 'Error':
        message = cfg.context.provider_by_name['Error'].field_place(value, 'message', cfg, span).read(cfg, span)
        cfg.effect('panic-message', (message,), PanicMessage(cfg.syntax.pointer_bits), span)
    raise_error(cfg, 'unhandled ' + value.type.name, span)


def unwrap_result(value, cfg, span):
    from py_compiler.syntax.block.union import UnionTag, UnionMember
    from py_compiler.syntax.function.calls import literal_value
    from py_compiler.syntax.generic.owned import select
    from py_compiler.syntax.generic.storage import copy_storage, release_storage
    from py_compiler.hir.thir.src.model import Edge, Terminator
    errors = error_members(value.type)
    if not errors or getattr(cfg, 'preserve_error_result', False):
        return value
    initial_values, initial_flow = dict(cfg.values), cfg.flow.fork()
    if value.type.family != 'union':
        forward_error(value, cfg, span)
        cfg.current = cfg.block()
        cfg.values, cfg.flow = initial_values, initial_flow
        return None
    tag = cfg.emit('union-tag', cfg.context.type('i32'), (value,), UnionTag(), span)
    for error in errors:
        index = value.type.variants.index(error)
        failed, following = cfg.block(), cfg.block()
        matches = select(tag.type, 'F.operator', '==').expand(cfg, tag, literal_value(index, tag.type, cfg, span), span)
        cfg.current.terminator = Terminator('conditional', (Edge(failed.identity), Edge(following.identity)), matches)
        cfg.current = failed
        payload = cfg.emit('union-member', error, (value,), UnionMember(index), span)
        payload = copy_storage(payload, cfg, span)
        release_temporary(value, cfg, span)
        forward_error(payload, cfg, span)
        cfg.current, cfg.values, cfg.flow = following, dict(initial_values), initial_flow.fork()
    successes = [t for t in value.type.variants if not is_error(t)]
    if len(successes) > 1:
        from py_compiler.syntax.block.union import Union, UnionWiden
        target = Union().resolve(successes, cfg.context)
        copied = copy_storage(value, cfg, span)
        result = cfg.emit('union-narrow', target, (copied,), UnionWiden(narrow=True), span)
        release_temporary(value, cfg, span)
        cfg.owned_results.add(result.identity)
        return result
    if not successes:
        return None
    if not successes[0].mlir:
        if value.identity in cfg.owned_results:
            release_storage(value, cfg, span)
        return None
    result = cfg.emit('union-member', successes[0], (value,), UnionMember(value.type.variants.index(successes[0])), span)
    result = copy_storage(result, cfg, span)
    if value.identity in cfg.owned_results:
        release_storage(value, cfg, span)
    cfg.owned_results.add(result.identity)
    return result


class Throw:
    def matches(self, node):
        return node.tokens[0].text == 'throw'

    def lower(self, node, cfg, env, local_names):
        from py_compiler.syntax.grammar.expression import expression
        throw_expression(expression(node.tokens[1:]), cfg, env, node.span)


class Panic:
    def call(self, type_, arguments, cfg, env, span):
        import ast
        from py_compiler.hir.thir.src.model import Terminator
        if type_ is not None or len(arguments) != 1:
            raise ValueError('panic requires one string message')
        argument = arguments[0]
        line, column = cfg.source.position(span.start)
        location = f' at {cfg.source.path}:{line}:{column}'
        if argument.kind == 'literal' and argument.token.kind == 'STRING':
            message = ast.literal_eval(argument.token.text)
        else:
            value = cfg.expr(argument, env, cfg.context.type('string'))
            cfg.effect('panic-message', (value,), PanicMessage(cfg.syntax.pointer_bits), span)
            message = 'panic'
        # Panic is a terminal invariant failure. Recoverable Error handlers do
        # not intercept it or turn it into an ordinary return value.
        cfg.current.terminator = Terminator('panic', message=message + location)
        return None


from dataclasses import dataclass


@dataclass(frozen=True)
class PanicMessage:
    bits: int

    @property
    def external_symbol_contracts(self):
        return (('write', ('i32', '!llvm.ptr', 'i' + str(self.bits)), 'i' + str(self.bits)),)

    def atom(self, operation):
        from py_compiler.syntax.generic.atom import Atom
        return Atom(self, (operation.operands[0].type,), None, ('view',), ('IO',), self)

    def render_operation(self, operation):
        from py_compiler.mlir.src.cfg import name, temporary_suffix
        from py_compiler.syntax.generic.storage import descriptor_type
        value = operation.operands[0]
        prefix = '%panic_text_' + temporary_suffix(operation)
        descriptor = descriptor_type(self.bits)
        width = 'i' + str(self.bits)
        return [f'{prefix}_descriptor = builtin.unrealized_conversion_cast {name(value)} : {value.type.mlir} to {descriptor}',
                f'{prefix}_base = llvm.extractvalue {prefix}_descriptor[1] : {descriptor}',
                f'{prefix}_offset = llvm.extractvalue {prefix}_descriptor[2] : {descriptor}',
                f'{prefix}_length = llvm.extractvalue {prefix}_descriptor[3, 0] : {descriptor}',
                f'{prefix}_address = llvm.getelementptr {prefix}_base[{prefix}_offset] : (!llvm.ptr, {width}) -> !llvm.ptr, i8',
                f'{prefix}_stderr = arith.constant 2 : i32',
                f'{prefix}_written = func.call @write({prefix}_stderr, {prefix}_address, {prefix}_length) : (i32, !llvm.ptr, {width}) -> {width}']


import unittest


class ErrorBoundaryTests(unittest.TestCase):
    def test_literal_fast_path_preserves_error_message_type(self):
        from py_compiler.frontend.src.lib import compile_source
        from py_compiler.syntax.recognition import Syntax
        result = compile_source('invalid-error.sev', 'def fail():\n    throw Error(123)\n', Syntax())
        self.assertTrue(result.diagnostics)
        self.assertIn('string', str(result.diagnostics[0]))

    def test_c_panic_accepts_literal_and_runtime_messages(self):
        import json
        import tempfile
        from pathlib import Path
        from py_compiler.packaging.testing import test_package
        source = '''def fail(message: string):
    panic(message)
test "literal panic":
    throws "invariant violated":
        panic("invariant violated")
test "runtime panic":
    throws "runtime message":
        fail("runtime message")
'''
        with tempfile.TemporaryDirectory(prefix='sev-panic-') as directory:
            root = Path(directory)
            (root / 'src').mkdir()
            (root / 'src/lib.sev').write_text(source)
            (root / 'package.json').write_text(json.dumps({'package': {'name': 'panic', 'version': '0.1.0'}}))
            result = test_package(root)
            self.assertTrue(result['success'], result)

    def test_dynamic_error_payload_is_accepted(self):
        from py_compiler.frontend.src.lib import compile_source
        from py_compiler.syntax.recognition import Syntax
        result = compile_source('error-payload.sev', 'def fail(message: string):\n    throw Error(message)\n', Syntax())
        self.assertFalse(result.diagnostics, result.diagnostics)


def raise_error(cfg, message, span):
    from py_compiler.hir.thir.src.model import Edge, Terminator
    handlers = getattr(cfg, 'error_handlers', ())
    if handlers:
        handler = handlers[-1]
        cfg.run_defers(handler['scope-depth'])
        if handler['message'] and handler['message'] not in message:
            cfg.current.terminator = Terminator('panic', message='unexpected error: ' + message)
        else:
            kept = {b.identity for b in handler['bindings']}
            kept.update(cfg.storage)
            cfg.cleanup_memory((b for scope in cfg.active_scopes for b in scope.values() if b.identity not in kept), span)
            cfg.current.terminator = Terminator('jump', (Edge(handler['block'].identity, tuple(cfg.values[b.identity] for b in handler['bindings'])),))
            handler['flows'].append(cfg.flow.fork())
    else:
        cfg.run_defers()
        cfg.cleanup_memory((b for scope in cfg.active_scopes for b in scope.values()), span)
        line,column=cfg.source.position(span.start)
        cfg.current.terminator=Terminator('panic',message=f'{message} at {cfg.source.path}:{line}:{column}')


from py_compiler.syntax.generic.block import BlockProvider


class Try(BlockProvider):
    spelling = 'try'
    attachment = 'error-chain'

    def parse_header(self, items):
        header = super().parse_header(items)
        if header:
            raise ValueError('try takes an indented body')
        return header

    def lower(self, node, cfg, env, nodes, index):
        from py_compiler.mir.ownership.flow import FlowState, reconcile_owners, Binding
        from py_compiler.hir.thir.src.model import Edge, Terminator
        from py_compiler.syntax.generic.storage import copy_storage, release_storage
        catches = []
        for following in nodes[index + 1:]:
            if not isinstance(following.provider, Catch):
                break
            catches.append(following)
        if not catches:
            raise ValueError('try requires catch')
        outer = tuple(b for b in env.values() if b.identity not in cfg.storage)
        exits = []
        initial_values, initial_flow = dict(cfg.values), cfg.flow.fork()
        scope_depth = len(cfg.active_scopes)
        handlers = getattr(cfg, 'typed_error_handlers', [])
        cfg.typed_error_handlers = handlers
        def handle(value, span):
            selected = None
            for candidate in catches:
                name, annotation = candidate.provider.binding(candidate.header)
                if annotation is None or cfg.context.type(annotation) == value.type:
                    selected = candidate, name
                    break
            handlers.pop()
            try:
                if selected is None:
                    forward_error(value, cfg, span)
                    return
                catch, name = selected
                payload = copy_storage(value, cfg, span)
                release_temporary(value, cfg, span)
                cfg.run_defers(scope_depth)
                cfg.cleanup_memory((b for scope in cfg.active_scopes[scope_depth:] for b in scope.values()
                                    if b.identity not in {item.identity for item in outer}), span)
                local = dict(env)
                binding = Binding(catch.identity + ':' + name, name, payload.type, False, 'own', catch.span)
                local[name] = binding
                cfg.body.bindings.append(binding)
                cfg.values[binding.identity] = payload
                cfg.scope(catch.children, local)
                if cfg.current.terminator is None:
                    exits.append((cfg.current, dict(cfg.values), cfg.flow.fork()))
            finally:
                handlers.append(handle)
        handlers.append(handle)
        try:
            cfg.scope(node.children, dict(env))
            if cfg.current.terminator is None:
                exits.append((cfg.current, dict(cfg.values), cfg.flow.fork()))
        finally:
            handlers.pop()
        if exits:
            reconcile_owners(cfg, exits, outer, node.span)
            outer = tuple(b for b in outer if b.identity not in exits[0][2].moved)
            done = cfg.block(tuple(b.type for b in outer))
            for block, values, _ in exits:
                block.terminator = Terminator('jump', (Edge(done.identity, tuple(values[b.identity] for b in outer)),))
            cfg.current, cfg.values = done, initial_values
            cfg.values.update({b.identity: v for b, v in zip(outer, done.parameters)})
            cfg.flow = FlowState.join([flow for _, _, flow in exits])
        return len(catches) + 1


class Catch(BlockProvider):
    spelling = 'catch'
    attachment = 'error-chain'
    continuation = True

    def binding(self, header):
        if len(header) == 1 and header[0].kind == 'IDENTIFIER':
            return header[0].text, None
        if len(header) >= 3 and header[1].text == ':':
            return header[0].text, ''.join(t.text for t in header[2:])
        if len(header) == 2:
            return header[1].text, header[0].text
        raise ValueError('catch requires a binding and optional Error type')

    def parse_header(self, items):
        header = super().parse_header(items)
        self.binding(header)
        return header


class Recover(BlockProvider):
    spelling = ''

    def recognize(self, window):
        from py_compiler.syntax.generic.grammar import Match
        tokens = window.tokens
        if not tokens or tokens[-1].text != ':' or tokens[0].text in ('if', 'elif', 'else', 'def', 'test'):
            return None
        depth = 0
        for index, token in enumerate(tokens[:-1]):
            depth += (token.text in ('(', '[', '{')) - (token.text in (')', ']', '}'))
            if token.text == 'else' and depth == 0 and index:
                return Match(self, window, window.end, {'header': tokens, 'split': index})
        return None

    def construct(self, match):
        tokens, split = match.captures['header'], match.captures['split']
        if len(tokens[split + 1:-1]) != 1:
            raise ValueError('recovery block requires an error binding')
        return tuple(tokens[:split]), tuple(tokens[split + 1:-1])

    def lower(self, node, cfg, env, nodes, index):
        from dataclasses import replace
        call, binding = node.header
        sentence = replace(node, provider=None, header=(), tokens=call, children=[])
        body = replace(node, provider=Try(), header=(), children=[sentence])
        catch = replace(node, identity=node.identity + ':catch', provider=Catch(), header=binding)
        Try().lower(body, cfg, env, [body, catch], 0)
        return 1
