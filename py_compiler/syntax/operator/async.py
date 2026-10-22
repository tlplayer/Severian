"""Scoped c tasks use pthreads and own their argument/result storage."""
from dataclasses import dataclass
from hashlib import sha256
from py_compiler.syntax.operator.operation import PrefixOperator
from py_compiler.syntax.generic.atom import Atom


class Async(PrefixOperator):
    def __init__(self):
        super().__init__('async')

    def parse_prefix(self, token, parse, tokens, cursor):
        from py_compiler.syntax.grammar.expression import Expression
        operation = parse(6)
        options = ['local']
        if cursor[0] < len(tokens) and tokens[cursor[0]].text == 'with':
            cursor[0] += 1
            options = []
            while cursor[0] < len(tokens):
                word = tokens[cursor[0]].text
                if word not in ('self', 'local', 'runtime', 'module', 'lock'):
                    raise ValueError('async requires a supported owning scope or lock')
                options.append(word)
                cursor[0] += 1
                if cursor[0] >= len(tokens) or tokens[cursor[0]].text != 'and':
                    break
                cursor[0] += 1
        return Expression('owned', token, (self, operation, tuple(options)))

    def lower_expression(self, node, cfg, env, expected):
        _, operation, options = node.operands
        if operation.kind != 'call':
            raise ValueError('async requires a call')
        if 'runtime' in options and not cfg.unsafe_depth:
            raise ValueError('runtime-owned task requires unsafe scope')
        previous = getattr(cfg, 'async_expression', None), getattr(cfg, 'async_options', ())
        cfg.async_expression, cfg.async_options = operation, options
        try:
            return cfg.expr(operation, env)
        finally:
            cfg.async_expression, cfg.async_options = previous


@dataclass(frozen=True)
class TaskType:
    result: object
    inputs: tuple
    storage: tuple
    modes: tuple
    symbol: str
    result_storage: str
    locked: bool = False
    family: str = 'task'
    mlir: str = '!llvm.ptr'
    binding_ownership: str = 'view'
    parameter_ownership: str = 'view'

    @property
    def name(self):
        return 'task[' + self.symbol + ']'

    @property
    def frame(self):
        return '!llvm.struct<(i64, i1, ' + ', '.join((self.result_storage, *self.storage)) + ')>'

    def require_context(self, cfg):
        pass

    def grammars(self):
        return ()

    def copy_value(self, value, cfg, span):
        raise ValueError('tasks have one owner; move or await the task')

    def join(self, value, cfg, span):
        cfg.effect('task-join', (value,), TaskOperation('join', self), span)

    def read(self, value, cfg, span, position, type_):
        return cfg.emit('task-value', type_, (value,), TaskOperation('read', self, position), span)

    def release(self, value, cfg, span):
        from py_compiler.syntax.generic.storage import owns_storage, release_storage
        self.join(value, cfg, span)
        if self.result.mlir and owns_storage(self.result):
            release_storage(self.read(value, cfg, span, 2, self.result), cfg, span)
        for index, (type_, mode) in enumerate(zip(self.inputs, self.modes)):
            if mode == 'snapshot' and owns_storage(type_):
                release_storage(self.read(value, cfg, span, index + 3, type_), cfg, span)
        cfg.effect('task-free', (value,), TaskOperation('free', self), span)


@dataclass(frozen=True)
class TaskOperation:
    kind: str
    owner: TaskType
    position: int = 0

    def validate_target(self, target, pointer_bits):
        if pointer_bits != 64 or not target.startswith('x86_64-') or 'linux' not in target:
            raise ValueError('c task runtime currently requires x86_64 Linux')

    @property
    def external_symbol_contracts(self):
        return (('pthread_join', ('i64', '!llvm.ptr'), 'i32'),
                ('pthread_create', ('!llvm.ptr',) * 4, 'i32'),
                ('pthread_mutex_lock', ('!llvm.ptr',), 'i32'),
                ('pthread_mutex_unlock', ('!llvm.ptr',), 'i32'))

    def atom(self, operation):
        return Atom(self.owner, tuple(v.type for v in operation.operands),
                    operation.result.type if operation.result else None,
                    ('view',) * len(operation.operands), ('IO', 'mutation'), self)

    def global_definitions(self):
        frame = self.owner.frame
        return ('llvm.func @malloc(i64) -> !llvm.ptr', 'llvm.func @free(!llvm.ptr)', f'''func.func private @{self.owner.symbol}_join(%task: !llvm.ptr) attributes {{llvm.linkage = #llvm.linkage<internal>}} {{
    %flag = llvm.getelementptr %task[0, 1] : (!llvm.ptr) -> !llvm.ptr, {frame}
    %joined = llvm.load %flag : !llvm.ptr -> i1
    cf.cond_br %joined, ^done, ^wait
  ^wait:
    %thread = llvm.load %task : !llvm.ptr -> i64
    %null = llvm.mlir.zero : !llvm.ptr
    %status = func.call @pthread_join(%thread, %null) : (i64, !llvm.ptr) -> i32
    %zero = arith.constant 0 : i32
    %ok = arith.cmpi eq, %status, %zero : i32
    cf.assert %ok, "task join failed"
    %true = arith.constant true
    llvm.store %true, %flag : i1, !llvm.ptr
    cf.br ^done
  ^done:
    return
  }}''',)

    def render_operation(self, operation):
        from py_compiler.mlir.src.cfg import name, temporary_suffix
        from py_compiler.syntax.generic.storage import unpack_value
        value = name(operation.operands[0])
        if self.kind == 'join':
            return [f'func.call @{self.owner.symbol}_join({value}) : (!llvm.ptr) -> ()']
        if self.kind == 'free':
            return [f'llvm.call @free({value}) : (!llvm.ptr) -> ()']
        prefix = '%task_read_' + temporary_suffix(operation)
        storage = self.owner.result_storage if self.position == 2 else self.owner.storage[self.position - 3]
        lines = [f'{prefix}_slot = llvm.getelementptr {value}[0, {self.position}] : (!llvm.ptr) -> !llvm.ptr, {self.owner.frame}',
                 f'{prefix}_stored = llvm.load {prefix}_slot : !llvm.ptr -> {storage}']
        unpack_value(lines, prefix + '_stored', operation.result.type, storage, name(operation.result))
        return lines


@dataclass(frozen=True)
class TaskStart(TaskOperation):
    entry: object = None

    @property
    def external_symbol_contracts(self):
        contracts = super().external_symbol_contracts
        if getattr(self.entry, 'synthetic', False):
            from py_compiler.syntax.complex.channel import ChannelOperation
            contracts += ChannelOperation('send', self.owner.inputs[0]).external_symbol_contracts
        return tuple(dict.fromkeys(contracts))

    @property
    def symbol_contracts(self):
        if getattr(self.entry, 'synthetic', False):
            return ()
        result = self.owner.result_storage if self.owner.result.family == 'record' else self.owner.result.mlir
        return ((self.entry.symbol, tuple(t.mlir for t in self.owner.inputs), result),)

    def global_definitions(self):
        from py_compiler.mlir.src.lib import symbol_ref
        from py_compiler.syntax.generic.storage import unpack_value, pack_value
        owner, frame = self.owner, self.owner.frame
        lines = [f'func.func private @{owner.symbol}_worker(%task: !llvm.ptr) -> !llvm.ptr attributes {{llvm.linkage = #llvm.linkage<internal>}} {{']
        if owner.locked:
            lines += ['%mutex = llvm.mlir.addressof @__sev_task_mutex : !llvm.ptr',
                      '%locked = func.call @pthread_mutex_lock(%mutex) : (!llvm.ptr) -> i32']
        args = []
        for index, (type_, storage, mode) in enumerate(zip(owner.inputs, owner.storage, owner.modes)):
            prefix = f'%argument{index}'
            lines += [f'{prefix}_slot = llvm.getelementptr %task[0, {index + 3}] : (!llvm.ptr) -> !llvm.ptr, {frame}',
                      f'{prefix}_stored = llvm.load {prefix}_slot : !llvm.ptr -> {storage}']
            if mode == 'borrow':
                args.append(prefix + '_stored')
            else:
                unpack_value(lines, prefix + '_stored', type_, storage, prefix)
                args.append(prefix)
        result = owner.result_storage if owner.result.family == 'record' else owner.result.mlir
        lines.append(('%result = ' if result else '') + f'func.call {symbol_ref(self.entry.symbol)}({", ".join(args)}) : ({", ".join(t.mlir for t in owner.inputs)}) -> {result or "()"}')
        if result:
            stored = '%result' if owner.result.family == 'record' else pack_value(lines, '%result', owner.result, owner.result_storage, '%stored_result')
            lines += [f'%result_slot = llvm.getelementptr %task[0, 2] : (!llvm.ptr) -> !llvm.ptr, {frame}',
                      f'llvm.store {stored}, %result_slot : {owner.result_storage}, !llvm.ptr']
        if owner.locked:
            lines.append('%unlocked = func.call @pthread_mutex_unlock(%mutex) : (!llvm.ptr) -> i32')
        lines += ['%null = llvm.mlir.zero : !llvm.ptr', 'return %null : !llvm.ptr', '}']
        definitions = (*super().global_definitions(), '\n'.join(lines))
        if getattr(self.entry, 'synthetic', False):
            definitions += tuple(self.entry.definitions())
        if owner.locked:
            definitions += ('llvm.mlir.global weak @__sev_task_mutex() : !llvm.array<40 x i8> {\n %zero = llvm.mlir.zero : !llvm.array<40 x i8>\n llvm.return %zero : !llvm.array<40 x i8>\n}',)
        return definitions

    def render_operation(self, operation):
        from py_compiler.mlir.src.cfg import name
        from py_compiler.syntax.generic.storage import pack_value
        owner, frame = self.owner, self.owner.frame
        prefix = name(operation.result)
        lines = [f'{prefix}_null = llvm.mlir.zero : !llvm.ptr',
                 f'{prefix}_end = llvm.getelementptr {prefix}_null[1] : (!llvm.ptr) -> !llvm.ptr, {frame}',
                 f'{prefix}_size = llvm.ptrtoint {prefix}_end : !llvm.ptr to i64',
                 f'{prefix} = llvm.call @malloc({prefix}_size) : (i64) -> !llvm.ptr',
                 f'{prefix}_ok = llvm.icmp "ne" {prefix}, {prefix}_null : !llvm.ptr',
                 f'cf.assert {prefix}_ok, "task allocation failed"',
                 f'{prefix}_empty = llvm.mlir.zero : {frame}',
                 f'llvm.store {prefix}_empty, {prefix} : {frame}, !llvm.ptr']
        for index, (value, storage, mode) in enumerate(zip(operation.operands, owner.storage, owner.modes)):
            stored = name(value) if mode == 'borrow' else pack_value(lines, name(value), value.type, storage, prefix + f'_argument{index}')
            lines += [f'{prefix}_slot{index} = llvm.getelementptr {prefix}[0, {index + 3}] : (!llvm.ptr) -> !llvm.ptr, {frame}',
                      f'llvm.store {stored}, {prefix}_slot{index} : {storage}, !llvm.ptr']
        lines += [f'{prefix}_worker = func.constant @{owner.symbol}_worker : (!llvm.ptr) -> !llvm.ptr',
                  f'{prefix}_code = builtin.unrealized_conversion_cast {prefix}_worker : (!llvm.ptr) -> !llvm.ptr to !llvm.ptr',
                  f'{prefix}_status = func.call @pthread_create({prefix}, {prefix}_null, {prefix}_code, {prefix}) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr) -> i32',
                  f'{prefix}_zero = arith.constant 0 : i32',
                  f'{prefix}_started = arith.cmpi eq, {prefix}_status, {prefix}_zero : i32',
                  f'cf.assert {prefix}_started, "task creation failed"']
        return lines


def start(entry, values, cfg, span, receiver=None):
    from py_compiler.syntax.generic.storage import stored_type, copy_storage
    from py_compiler.syntax.function.calls import coerce
    if not getattr(entry, 'synthetic', False):
        cfg.context.compile(entry)
    locked = 'lock' in cfg.async_options
    operands, modes, storage = [], [], []
    dependencies = set()
    if receiver is not None:
        if entry.receiver_ownership != 'view' and not locked:
            raise ValueError('async mutation requires lock')
        operands.append(receiver); modes.append('borrow'); storage.append(receiver.type.mlir)
        dependencies.update(cfg.value_origins.get(receiver.identity, ()))
    for value, (parameter, annotation) in zip(values, entry.parameters):
        value = coerce(value, cfg.context.type(annotation), cfg, span)
        mode = entry.parameter_ownership.get(parameter, getattr(value.type, 'parameter_ownership', 'view'))
        if mode == 'borrow':
            raise ValueError('async mutable parameters require a shared receiver with lock')
        if value.type.family == 'channel' and mode == 'view':
            owners = (cfg.value_origins.get(value.identity, set()) |
                      cfg.value_dependencies.get(value.identity, set()))
            if not owners:
                raise ValueError('async channel parameters require a live channel binding')
            operands.append(value)
            modes.append('borrow')
            storage.append(value.type.mlir)
            dependencies.update(owners)
            continue
        if mode != 'move':
            value = copy_storage(value, cfg, span)
        operands.append(value); modes.append('move' if mode == 'move' else 'snapshot')
        storage.append(stored_type(value.type, cfg.context))
    result = cfg.context.type(entry.result or 'absent')
    symbol = '__sev_task_' + sha256((entry.symbol + str(locked)).encode()).hexdigest()[:24]
    type_ = TaskType(result, tuple(v.type for v in operands), tuple(storage), tuple(modes), symbol,
                     stored_type(result, cfg.context), locked)
    output = cfg.emit('task-start', type_, operands, TaskStart('start', type_, entry=entry), span)
    cfg.owned_results.add(output.identity)
    cfg.value_dependencies[output.identity] = dependencies
    return output
