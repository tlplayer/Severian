"""Owned FIFO channels with c mutex/condition-variable synchronization."""
from dataclasses import dataclass
from hashlib import sha256
from py_compiler.syntax.generic.atom import Atom
from py_compiler.syntax.generic.definition import TypeDefinition


class Channel(TypeDefinition):
    def __init__(self, name='channel'):
        super().__init__(name)

    def specialize(self, element, context):
        from py_compiler.syntax.generic.storage import stored_type
        return ChannelType(element, stored_type(element, context))

    def specialize_arguments(self, arguments, context):
        if len(arguments) != 1:
            raise ValueError('channel type annotations require one element type')
        return self.specialize(context.type(arguments[0]), context)

    def construct_expression(self, argument, arguments, cfg, env, span, expected=None):
        from py_compiler.syntax.grammar.expression import type_spelling, Expression
        from py_compiler.syntax.function.calls import literal_value
        from py_compiler.syntax.generic.storage import stored_type
        types = argument.operands if argument and argument.kind == 'template_arguments' else (argument,)
        if not argument or len(types) not in (1, 2) or len(arguments) > 1:
            raise ValueError('channel requires an element type and optional capacity')
        element = cfg.context.type(type_spelling(types[0]))
        type_ = self.specialize(element, cfg.context)
        capacity = cfg.expr(arguments[0], env, cfg.context.type('int')) if arguments else (
            cfg.expr(types[1], env, cfg.context.type('int')) if len(types) == 2 else
            literal_value(0, cfg.context.type('int'), cfg, span))
        result = cfg.emit('channel-create', type_, (capacity,), ChannelOperation('create', type_), span)
        cfg.owned_results.add(result.identity)
        return result


@dataclass(frozen=True)
class ChannelType:
    element: object
    storage: str
    family: str = 'channel'
    mlir: str = '!llvm.ptr'
    binding_ownership: str = 'view'
    parameter_ownership: str = 'view'

    @property
    def name(self):
        return 'channel[' + self.element.name + ']'

    @property
    def prefix(self):
        return '__sev_channel_' + sha256((self.name + self.storage).encode()).hexdigest()[:20]

    @property
    def frame(self):
        return '!llvm.struct<(ptr, i64, i64, i64, i64, array<40 x i8>, array<48 x i8>, array<48 x i8>)>'

    def require_context(self, cfg):
        pass

    def grammars(self):
        return ()

    def copy_value(self, value, cfg, span):
        raise ValueError('channel storage has one owner; pass a view or move it')

    def ready(self, value, cfg, span):
        return cfg.emit('channel-ready', cfg.context.type('bool'), (value,), ChannelOperation('ready', self), span)

    def receive(self, value, cfg, span):
        result = cfg.emit('channel-receive', self.element, (value,), ChannelOperation('receive', self), span)
        cfg.owned_results.add(result.identity)
        cfg.value_dependencies[result.identity] = set()
        cfg.value_sources[result.identity] = set()
        return result

    def call_method(self, receiver, method, arguments, cfg, env, span, binding):
        from py_compiler.syntax.generic.storage import copy_storage, root_binding
        if method == 'receive' and not arguments:
            return self.receive(receiver, cfg, span)
        if method == 'ready' and not arguments:
            return self.ready(receiver, cfg, span)
        if method != 'send' or len(arguments) != 1:
            raise ValueError('channel supports send(value), receive(), and ready()')
        term = arguments[0]
        moving = term.kind == 'unary' and term.token.text == 'move'
        if moving:
            term = term.operands[0]
        value = cfg.expr(term, env, self.element)
        owner = root_binding(term, cfg, env)
        if moving:
            if owner is None or owner.ownership in ('view', 'borrow'):
                raise ValueError('channel send(move value) requires an owning binding')
            cfg.flow.move(owner)
            cfg.flow.end_retained(owner.identity)
        else:
            value = copy_storage(value, cfg, span)
        if getattr(cfg, 'spawning_async', False):
            from importlib import import_module
            from types import SimpleNamespace
            operation = ChannelOperation('send', self)
            entry = SimpleNamespace(symbol=self.prefix + '_send', result='absent',
                parameters=(('item', self.element.name),), parameter_ownership={'item': 'move'},
                receiver_ownership='view', synthetic=True, definitions=operation.global_definitions)
            return import_module('py_compiler.syntax.operator.async').start(entry, (value,), cfg, span, receiver)
        cfg.effect('channel-send', (receiver, value), ChannelOperation('send', self), span)
        return None

    def release(self, value, cfg, span):
        from py_compiler.hir.thir.src.model import Edge, Terminator
        from py_compiler.syntax.generic.storage import release_storage
        check, body, done = cfg.block(), cfg.block(), cfg.block()
        cfg.current.terminator = Terminator('jump', (Edge(check.identity),))
        cfg.current = check
        ready = self.ready(value, cfg, span)
        cfg.current.terminator = Terminator('conditional', (Edge(body.identity), Edge(done.identity)), ready)
        cfg.current = body
        release_storage(self.receive(value, cfg, span), cfg, span)
        cfg.current.terminator = Terminator('jump', (Edge(check.identity),))
        cfg.current = done
        cfg.effect('channel-free', (value,), ChannelOperation('free', self), span)


@dataclass(frozen=True)
class ChannelOperation:
    kind: str
    owner: ChannelType

    def validate_target(self, target, pointer_bits):
        if pointer_bits != 64 or not target.startswith('x86_64-') or 'linux' not in target:
            raise ValueError('c channel runtime currently requires x86_64 Linux')

    @property
    def external_symbol_contracts(self):
        return (
                ('pthread_mutex_lock', ('!llvm.ptr',), 'i32'),
                ('pthread_mutex_unlock', ('!llvm.ptr',), 'i32'),
                ('pthread_cond_wait', ('!llvm.ptr', '!llvm.ptr'), 'i32'),
                ('pthread_cond_signal', ('!llvm.ptr',), 'i32'),
                ('pthread_mutex_destroy', ('!llvm.ptr',), 'i32'),
                ('pthread_cond_destroy', ('!llvm.ptr',), 'i32'))

    def atom(self, operation):
        from py_compiler.syntax.generic.storage import owns_storage
        modes = (('view', 'move' if owns_storage(self.owner.element) else 'copy')
                 if self.kind == 'send' else ('view',) * len(operation.operands))
        return Atom(self.owner, tuple(v.type for v in operation.operands),
                    operation.result.type if operation.result else None, modes, ('IO', 'mutation'), self)

    def global_definitions(self):
        p, frame, storage = self.owner.prefix, self.owner.frame, self.owner.storage
        from py_compiler.syntax.generic.storage import pack_value
        packed = []
        value = pack_value(packed, '%item', self.owner.element, storage, '%packed')
        fields = '\n'.join(f'%{name} = llvm.getelementptr %queue[0, {i}] : (!llvm.ptr) -> !llvm.ptr, {frame}'
                           for i, name in enumerate(('data', 'capacity', 'head', 'count', 'bounded', 'mutex', 'readable', 'writable')))
        setup = fields + '\n%zero = arith.constant 0 : i64\n%one = arith.constant 1 : i64\n'
        create = f'''func.func private @{p}_create(%requested: i64) -> !llvm.ptr {{
    %zero = arith.constant 0 : i64
    %valid = arith.cmpi sge, %requested, %zero : i64
    cf.assert %valid, "channel capacity must be nonnegative"
    %default = arith.constant 16 : i64
    %fixed = arith.cmpi sgt, %requested, %zero : i64
    %slots = arith.select %fixed, %requested, %default : i64
    %null = llvm.mlir.zero : !llvm.ptr
    %end = llvm.getelementptr %null[1] : (!llvm.ptr) -> !llvm.ptr, {frame}
    %size = llvm.ptrtoint %end : !llvm.ptr to i64
    %queue = llvm.call @malloc(%size) : (i64) -> !llvm.ptr
    %ok = llvm.icmp "ne" %queue, %null : !llvm.ptr
    cf.assert %ok, "channel allocation failed"
    %empty = llvm.mlir.zero : {frame}
    llvm.store %empty, %queue : {frame}, !llvm.ptr
    {fields}
    %last = llvm.getelementptr %null[%slots] : (!llvm.ptr, i64) -> !llvm.ptr, {storage}
    %bytes = llvm.ptrtoint %last : !llvm.ptr to i64
    %buffer = llvm.call @malloc(%bytes) : (i64) -> !llvm.ptr
    %buffer_ok = llvm.icmp "ne" %buffer, %null : !llvm.ptr
    cf.assert %buffer_ok, "channel buffer allocation failed"
    llvm.store %buffer, %data : !llvm.ptr, !llvm.ptr
    llvm.store %slots, %capacity : i64, !llvm.ptr
    llvm.store %fixed, %bounded : i1, !llvm.ptr
    return %queue : !llvm.ptr
}}'''
        send = f'''func.func private @{p}_send(%queue: !llvm.ptr, %item: {self.owner.element.mlir}) {{
    {setup}
    {chr(10).join(packed)}
    %locked = func.call @pthread_mutex_lock(%mutex) : (!llvm.ptr) -> i32
    cf.br ^check
  ^check:
    %length = llvm.load %count : !llvm.ptr -> i64
    %limit = llvm.load %capacity : !llvm.ptr -> i64
    %full = arith.cmpi eq, %length, %limit : i64
    cf.cond_br %full, ^full, ^enqueue
  ^full:
    %fixed = llvm.load %bounded : !llvm.ptr -> i1
    cf.cond_br %fixed, ^wait, ^grow
  ^wait:
    %waited = func.call @pthread_cond_wait(%writable, %mutex) : (!llvm.ptr, !llvm.ptr) -> i32
    cf.br ^check
  ^grow:
    %twice = arith.addi %limit, %limit : i64
    %safe = arith.cmpi sgt, %twice, %limit : i64
    cf.assert %safe, "channel capacity overflow"
    %null = llvm.mlir.zero : !llvm.ptr
    %end = llvm.getelementptr %null[%twice] : (!llvm.ptr, i64) -> !llvm.ptr, {storage}
    %bytes = llvm.ptrtoint %end : !llvm.ptr to i64
    %next = llvm.call @malloc(%bytes) : (i64) -> !llvm.ptr
    %allocated = llvm.icmp "ne" %next, %null : !llvm.ptr
    cf.assert %allocated, "channel growth failed"
    %old = llvm.load %data : !llvm.ptr -> !llvm.ptr
    %start = llvm.load %head : !llvm.ptr -> i64
    cf.br ^copy(%zero : i64)
  ^copy(%index: i64):
    %more = arith.cmpi slt, %index, %length : i64
    cf.cond_br %more, ^element, ^replace
  ^element:
    %sum = arith.addi %start, %index : i64
    %offset = arith.remui %sum, %limit : i64
    %source = llvm.getelementptr %old[%offset] : (!llvm.ptr, i64) -> !llvm.ptr, {storage}
    %destination = llvm.getelementptr %next[%index] : (!llvm.ptr, i64) -> !llvm.ptr, {storage}
    %saved = llvm.load %source : !llvm.ptr -> {storage}
    llvm.store %saved, %destination : {storage}, !llvm.ptr
    %following = arith.addi %index, %one : i64
    cf.br ^copy(%following : i64)
  ^replace:
    llvm.call @free(%old) : (!llvm.ptr) -> ()
    llvm.store %next, %data : !llvm.ptr, !llvm.ptr
    llvm.store %twice, %capacity : i64, !llvm.ptr
    llvm.store %zero, %head : i64, !llvm.ptr
    cf.br ^check
  ^enqueue:
    %buffer = llvm.load %data : !llvm.ptr -> !llvm.ptr
    %first = llvm.load %head : !llvm.ptr -> i64
    %tail_sum = arith.addi %first, %length : i64
    %tail = arith.remui %tail_sum, %limit : i64
    %slot = llvm.getelementptr %buffer[%tail] : (!llvm.ptr, i64) -> !llvm.ptr, {storage}
    llvm.store {value}, %slot : {storage}, !llvm.ptr
    %increased = arith.addi %length, %one : i64
    llvm.store %increased, %count : i64, !llvm.ptr
    %notified = func.call @pthread_cond_signal(%readable) : (!llvm.ptr) -> i32
    %unlocked = func.call @pthread_mutex_unlock(%mutex) : (!llvm.ptr) -> i32
    return
}}'''
        receive = f'''func.func private @{p}_receive(%queue: !llvm.ptr) -> {storage} {{
    {setup}
    %locked = func.call @pthread_mutex_lock(%mutex) : (!llvm.ptr) -> i32
    cf.br ^check
  ^check:
    %length = llvm.load %count : !llvm.ptr -> i64
    %empty = arith.cmpi eq, %length, %zero : i64
    cf.cond_br %empty, ^wait, ^read
  ^wait:
    %waited = func.call @pthread_cond_wait(%readable, %mutex) : (!llvm.ptr, !llvm.ptr) -> i32
    cf.br ^check
  ^read:
    %buffer = llvm.load %data : !llvm.ptr -> !llvm.ptr
    %first = llvm.load %head : !llvm.ptr -> i64
    %limit = llvm.load %capacity : !llvm.ptr -> i64
    %slot = llvm.getelementptr %buffer[%first] : (!llvm.ptr, i64) -> !llvm.ptr, {storage}
    %value = llvm.load %slot : !llvm.ptr -> {storage}
    %sum = arith.addi %first, %one : i64
    %next = arith.remui %sum, %limit : i64
    llvm.store %next, %head : i64, !llvm.ptr
    %decreased = arith.subi %length, %one : i64
    llvm.store %decreased, %count : i64, !llvm.ptr
    %notified = func.call @pthread_cond_signal(%writable) : (!llvm.ptr) -> i32
    %unlocked = func.call @pthread_mutex_unlock(%mutex) : (!llvm.ptr) -> i32
    return %value : {storage}
}}'''
        ready = f'''func.func private @{p}_ready(%queue: !llvm.ptr) -> i1 {{
    {setup}
    %locked = func.call @pthread_mutex_lock(%mutex) : (!llvm.ptr) -> i32
    %length = llvm.load %count : !llvm.ptr -> i64
    %ready = arith.cmpi sgt, %length, %zero : i64
    %unlocked = func.call @pthread_mutex_unlock(%mutex) : (!llvm.ptr) -> i32
    return %ready : i1
}}'''
        free = f'''func.func private @{p}_free(%queue: !llvm.ptr) {{
    {fields}
    %mutex_done = func.call @pthread_mutex_destroy(%mutex) : (!llvm.ptr) -> i32
    %read_done = func.call @pthread_cond_destroy(%readable) : (!llvm.ptr) -> i32
    %write_done = func.call @pthread_cond_destroy(%writable) : (!llvm.ptr) -> i32
    %buffer = llvm.load %data : !llvm.ptr -> !llvm.ptr
    llvm.call @free(%buffer) : (!llvm.ptr) -> ()
    llvm.call @free(%queue) : (!llvm.ptr) -> ()
    return
}}'''
        helpers = tuple(body.replace(' {\n', ' attributes {llvm.linkage = #llvm.linkage<internal>} {\n', 1)
                        for body in (create, send, receive, ready, free))
        return ('llvm.func @malloc(i64) -> !llvm.ptr', 'llvm.func @free(!llvm.ptr)', *helpers)

    def render_operation(self, operation):
        from py_compiler.mlir.src.cfg import name, temporary_suffix
        from py_compiler.syntax.generic.storage import unpack_value
        prefix = '%channel_' + temporary_suffix(operation)
        args = ', '.join(name(v) for v in operation.operands)
        inputs = ', '.join(v.type.mlir for v in operation.operands)
        result = self.owner.storage if self.kind == 'receive' else operation.result.type.mlir if operation.result else ''
        variable = prefix + '_stored' if self.kind == 'receive' else name(operation.result) if operation.result else ''
        lines = [(variable + ' = ' if result else '') + f'func.call @{self.owner.prefix}_{self.kind}({args}) : ({inputs}) -> {result or "()"}']
        if self.kind == 'receive':
            unpack_value(lines, variable, self.owner.element, self.owner.storage, name(operation.result))
        return lines
