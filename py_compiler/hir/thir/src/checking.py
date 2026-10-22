"""Check source providers and elaborate typed expressions before MIR construction."""
from hashlib import sha256
from dataclasses import replace
from functools import wraps
from py_compiler.frontend.source.source import Diagnostic
from py_compiler.syntax.grammar.expression import expression, OWNERSHIP
from py_compiler.frontend.parser.contract import Literal
from py_compiler.hir.hir.src.program import Constant, resolve_literal
from py_compiler.hir.thir.src.model import DraftBody as Body, Region as ExecutionBlock, Value, Expression as Operation, Edge, Terminator, OperationContract
from py_compiler.mir.ownership.flow import Binding, FlowState
from py_compiler.syntax.generic.block import DeclarationProvider


def located(method):
    @wraps(method)
    def call(self, node, *args, **kwargs):
        origin = node[0] if isinstance(node, list) else node
        span = origin.token.span if hasattr(origin, 'token') else origin.span
        previous = getattr(self, 'lifetime_span', self.body.source_span)
        self.lifetime_span = span
        try:
            return method(self, node, *args, **kwargs)
        except ValueError as failure:
            origin = node[0] if isinstance(node, list) else node
            span = origin.token.span if hasattr(origin, "token") else origin.span
            raise Diagnostic("THIR", str(failure), self.source, span) from failure
        finally:
            self.lifetime_span = previous
    return call


class Checker:
    def __init__(self, source, syntax, identity, name, result):
        self.source, self.syntax = source, syntax
        self.body = Body(identity, name, source.path, result)
        self.body.source_snapshot = source
        self.body.source_span = source.span(0, 0)
        self.next_value = 0
        self.current = self.block()
        self.flow = FlowState()
        self.values = {}
        self.local_names = set()
        self.namespaces = {}
        self.fallback_scopes = []
        self.storage = {}
        self.storage_factory = None
        self.declared_nodes = set()
        self.context = None
        self.stack_values = set()
        self.receiver = None
        self.receiver_value = None
        self.declaration_scope = ()
        self.uninitialized_fields = set()
        self.constructor = False
        self.phases = None
        self.imported = {}
        self.unsafe_depth = 0
        self.active_scopes = []
        self.deferred_actions = []
        self.loops = []
        self.value_sources = {}
        self.call_result_owners = {}
        self.call_result_modes = {}
        self.call_result_sources = {}
        self.owned_results = set()
        self.value_dependencies = {}
        self.value_origins = {}
        self.parameter_bindings = {}
        self.grammar_captures = {}
        self.active_grammars = set()
        self.grammar_expansions = 0
        self.yield_handler = None
        self.return_handler = None
        self.expression_frames = []

    @property
    def flow(self):
        return self._flow

    @flow.setter
    def flow(self, value):
        self._flow = value
        value.reporter = self.record_lifetime

    def record_lifetime(self, kind, owner, slot='', sources=()):
        from py_compiler.mir.ownership.flow import LifetimeEvent
        self.effect('lifetime', (), LifetimeEvent(kind, owner, slot, frozenset(sources)),
                    getattr(self, 'lifetime_span', self.body.source_span))

    def value(self, type_):
        value = Value(self.next_value, type_)
        self.next_value += 1
        return value

    def block(self, types=()):
        block = ExecutionBlock(len(self.body.blocks), tuple(self.value(t) for t in types))
        self.body.blocks.append(block)
        return block

    def emit(self, kind, type_, operands, payload, span):
        if kind == 'binary' and isinstance(payload, str):
            from py_compiler.syntax.generic.owned import select
            payload = select(operands[0].type, 'F.operator', payload)
            kind = 'scalar'
        result = self.value(type_)
        operation = Operation(kind, result, tuple(operands), payload, span)
        if not callable(getattr(payload, 'atom', None)):
            raise ValueError(f'{kind} has no object-supplied atom implementation')
        operation = replace(operation, contract=OperationContract.check(payload, operation))
        self.current.operations.append(operation)
        if self.expression_frames:
            self.expression_frames[-1][1].append(operation)
        if type_:
            from py_compiler.syntax.generic.storage import contains_borrowed_fields
            dependencies = (set().union(*(self.value_dependencies.get(v.identity, set()) for v in operands))
                            if contains_borrowed_fields(type_, self.context) or type_.family == 'string' else set())
            if type_.family in ('pointer', 'trait'):
                dependencies.update(set().union(*(self.value_origins.get(v.identity, set()) for v in operands)))
            self.value_dependencies[result.identity] = dependencies
            self.value_origins[result.identity] = set().union(*(self.value_origins.get(v.identity, set()) for v in operands)) if kind != 'memory-copy' else set()
        if kind != 'memory-copy' and type_ and type_.binding_ownership != 'copy':
            self.value_sources[result.identity] = set().union(*(self.value_sources.get(v.identity, set()) for v in operands))
        if kind != "memory-copy" and type_ and type_.family in ("record", "trait", "union", "dynamic", "array", "tensor", "tuple", "pointer") and any(v.identity in self.stack_values for v in operands):
            self.stack_values.add(result.identity)
        if kind == 'constant' and type_ and type_.family == 'string':
            self.owned_results.add(result.identity)
        return result

    def effect(self, kind, operands, payload, span):
        operation = Operation(kind, None, tuple(operands), payload, span)
        if not callable(getattr(payload, 'atom', None)):
            raise ValueError(f'{kind} has no object-supplied atom implementation')
        operation = replace(operation, contract=OperationContract.check(payload, operation))
        self.current.operations.append(operation)
        if self.expression_frames:
            self.expression_frames[-1][1].append(operation)

    def capture_expression(self, node):
        from py_compiler.syntax.block.grammar import capture_expression
        return capture_expression(node, self.grammar_captures)

    def lookup(self, node, env):
        if node.kind == 'name' and node.token.text in self.grammar_captures:
            return self.grammar_captures[node.token.text].binding(self)
        if node.kind == 'name' and node.token.text == 'self' and hasattr(self, 'self_binding'):
            return self.self_binding
        if node.kind == "reference":
            provider = self.syntax.scope_providers[node.token.text]
            return provider.bindings(self, env).get(node.operands[0].text)
        for scope in (env, *self.fallback_scopes):
            if node.token.text in scope:
                return scope[node.token.text]
        return None

    def read_binding(self, binding, span):
        self.flow.read(binding)
        value = self._read_binding(binding, span)
        identity = self.flow.resolve(binding.identity)
        from py_compiler.syntax.generic.storage import FieldPlace
        place = self.storage.get(identity)
        origins = self.value_origins.get(place.receiver.identity, {identity}) if isinstance(place, FieldPlace) else {identity}
        if value.type.family == 'pointer' and binding.identity in self.flow.view_sources:
            origins = {self.flow.resolve(source) for source in self.flow.view_sources[binding.identity]}
        self.value_origins[value.identity] = set(origins)
        dependencies = set().union(*(self.flow.sources(origin) for origin in origins)) | self.flow.sources(binding.identity)
        if value.type.family == 'pointer':
            dependencies.update(set().union(*(self.flow.sources(origin) or {origin} for origin in origins)))
        elif value.type.family == 'trait':
            dependencies.update(origins)
        if identity in self.parameter_bindings:
            from py_compiler.syntax.generic.storage import contains_borrowed_fields
            if contains_borrowed_fields(value.type, self.context):
                dependencies.add(identity)
        self.value_dependencies[value.identity] = dependencies
        return value

    def _read_binding(self, binding, span):
        identity = self.flow.resolve(binding.identity)
        locked_tasks = {b.identity for b in self.body.bindings
                        if b.type is not None and b.type.family == 'task' and b.type.locked}
        if (self.flow.borrowers(identity) & locked_tasks
                and not (getattr(self, 'async_expression', None) is not None
                         and 'lock' in getattr(self, 'async_options', ()))):
            raise ValueError('await locked tasks before accessing their shared receiver')
        if identity in self.uninitialized_fields:
            raise ValueError("constructor reads a field before initialization")
        if binding.type.family == 'pointer' and binding.identity in self.values and binding.identity != identity:
            return self.values[binding.identity]
        if identity in self.storage:
            return self.storage[identity].read(self, span)
        value = self.values[identity]
        if binding.type.family in ('record', 'array', 'tensor', 'tuple', 'shared', 'string', 'union', 'dynamic', 'iterator') and binding.identity in self.values and binding.type != value.type:
            # A nested record view borrows the containing binding's lifetime,
            # but retains the address and type of its own field.
            value = self.values[binding.identity]
        if binding.type != value.type and callable(getattr(binding.type, 'from_value', None)):
            return binding.type.from_value(value, self, span)
        return value

    def infer(self, binding, type_, env):
        previous = self.values[binding.identity].type
        if previous is not None:
            if type_ and previous != type_:
                raise ValueError("conflicting parameter type constraints")
            type_ = previous
        if type_ is None:
            raise ValueError(f"type of {binding.name!r} requires an annotation or expected type")
        updated = replace(binding, type=type_, ownership=getattr(type_, 'parameter_ownership', binding.ownership))
        value = self.values[binding.identity]
        value = Value(value.identity, type_)
        self.values[binding.identity] = value
        self.body.blocks[0].parameters = tuple(value if p.identity == value.identity else p for p in self.body.blocks[0].parameters)
        self.body.bindings = [updated if b.identity == binding.identity else b for b in self.body.bindings]
        env[binding.name] = updated
        return updated

    def live_binding_ids(self, env):
        return {b.identity for scope in (env, *self.namespaces.values()) for b in scope.values()}

    @located
    def expr(self, node, env, expected=None):
        from py_compiler.hir.thir.src.model import SourceExpression
        frame = ([], [])
        self.expression_frames.append(frame)
        try:
            value = self.check_expression(node, env, expected)
        finally:
            self.expression_frames.pop()
        binding = self.lookup(node, env) if node.kind in ('name', 'reference') else None
        symbol = binding.identity if binding is not None else next(
            (op.payload.symbol for op in frame[1] if op.kind == 'call' and hasattr(op.payload, 'symbol')), None)
        tokens = node.tokens
        span = (self.source.span(tokens[0].span.start, tokens[-1].span.end)
                if tokens and tokens[0].span.source == self.source.identity else node.token.span)
        expression = SourceExpression(len(self.body.expressions), node.kind,
            value.type if value is not None else self.syntax.types['absent'], span, symbol,
            tuple(frame[0]), tuple(frame[1]), value)
        self.body.expressions.append(expression)
        if self.expression_frames:
            self.expression_frames[-1][0].append(expression.identity)
        return value

    def check_expression(self, node, env, expected=None):
        if expected is not None and expected.family == 'bool':
            return expected.condition(node, self, env)
        require_context = getattr(expected, 'require_context', None)
        if require_context is not None:
            require_context(self)
        previous_capture = getattr(self, 'preserve_error_result', False)
        previous_async = getattr(self, 'spawning_async', False)
        self.spawning_async = node is getattr(self, 'async_expression', None)
        self.preserve_error_result = node is getattr(self, 'captured_error_expression', None)
        try:
            value = self.expression_value(self.capture_expression(node), env, expected)
        finally:
            self.preserve_error_result = previous_capture
            self.spawning_async = previous_async
        if (value is not None and expected is not None and value.type != expected
                and (value.type.family == expected.family == 'integer' or
                     value.type.family in ('tensor', 'tuple') and expected.family in ('tensor', 'tuple'))):
            from py_compiler.syntax.function.calls import coerce
            value = coerce(value, expected, self, node.token.span)
        require_context = getattr(value.type, 'require_context', None) if value is not None else None
        if require_context is not None:
            require_context(self)
        return value

    def expression_value(self, node, env, expected=None):
        token = node.token
        if getattr(self, 'profiling', False):
            from py_compiler.syntax.complex.profile import read
            metric = read(node, self, env)
            if metric is not None:
                return metric
        from py_compiler.syntax.block.grammar import expression_grammar, lower_result
        match = expression_grammar(node, self, env)
        if match is not None:
            return lower_result(match, self)
        if node.kind == 'tuple':
            value = self.context.type('tuple').construct_expression(None, node.operands, self, env, node.token.span, expected)
            if expected is not None and expected.family == 'union':
                from py_compiler.syntax.function.calls import coerce
                from py_compiler.syntax.generic.storage import release_storage
                boxed = coerce(value, expected, self, node.token.span)
                release_storage(value, self, node.token.span)
                return boxed
            return value
        if node.kind == 'interpolation':
            from py_compiler.syntax.complex.interpolation import lower
            return lower(node, self, env)
        if node.kind in ('list', 'map', 'set'):
            from py_compiler.syntax.complex.collection import lower_literal
            return lower_literal(node, self, env, expected)
        if node.kind == 'fallback':
            from py_compiler.syntax.block.union import fallback_expression
            return fallback_expression(node, self, env, expected)
        if node.kind in ('conditional', 'identity'):
            from py_compiler.syntax.operator.logical import conditional_expression, identity_expression
            lower = conditional_expression if node.kind == 'conditional' else identity_expression
            return lower(node, self, env, expected)
        if node.kind == 'capture':
            return node.operands[0].lower(self, expected)
        if node.kind == 'address':
            from py_compiler.syntax.complex.array import ArrayType, MemoryOperation, ElementPlace
            from py_compiler.syntax.function.allocate import resolve_plan
            from py_compiler.syntax.function.calls import literal_value
            source = node.operands[0]
            if source.kind == 'index':
                from py_compiler.syntax.generic.storage import root_binding
                from py_compiler.syntax.primitive.pointer.operations import ElementAddress
                binding = root_binding(source, self, env)
                if binding is None:
                    raise ValueError('address requires a live binding')
                receiver = self.expr(source.operands[0], env)
                if (getattr(receiver.type, 'rank', 1) != 1 or
                        not hasattr(receiver.type, 'element') or
                        receiver.type.element.family not in ('integer', 'float', 'bool', 'char', 'byte')):
                    raise ValueError('indexed address requires rank-one scalar storage')
                place = receiver.type.element_place(receiver, source.operands[1], self, env, token.span)
                position = place.indices[0] if hasattr(place, 'indices') else place.index
                target = self.context.type('pointer[' + place.type.name + ']')
                target.require_context(self)
                return self.emit('element-address', target, (receiver, position), ElementAddress(target.bits), token.span)
            binding = self.lookup(source, env) if source.kind in ('name', 'reference') else None
            if binding is None:
                raise ValueError('address requires a live binding')
            value = self.read_binding(binding, token.span)
            target = self.context.type('pointer[' + binding.type.name + ']')
            if binding.identity in self.storage:
                place = self.storage[binding.identity]
                if not isinstance(place, ElementPlace):
                    raise ValueError('this storage has no bounded address implementation')
                storage = place.receiver
            else:
                one = literal_value(1, self.context.type('index'), self, token.span)
                storage = self.emit('address-storage', ArrayType(binding.type, location='stack'), (one,),
                                    MemoryOperation('allocate', resolve_plan(binding.type, {'location': 'memory.stack'}, self.context)), token.span)
                self.stack_values.add(storage.identity)
                zero = literal_value(0, self.context.type('index'), self, token.span)
                place = ElementPlace(storage, zero, binding.type)
                place.write(self, value, token.span)
                self.storage[binding.identity] = place
            return target.from_value(storage, self, token.span)
        if node.kind == 'comprehension':
            from py_compiler.syntax.complex.collection import lower_comprehension
            return lower_comprehension(node, self, env, expected)
        if node.kind == 'owned':
            return node.operands[0].lower_expression(node, self, env, expected)
        if expected and expected.family == 'dynamic':
            return expected.from_value(self.expr(node, env), self, token.span)
        if expected and expected.family == "union":
            from py_compiler.syntax.function.calls import coerce
            return coerce(self.expr(node, env), expected, self, token.span)
        if node.kind == "index":
            subject, argument = node.operands
            definition = self.context.types.get(subject.token.text) if subject.kind == 'name' else None
            if callable(getattr(definition, 'construct_expression', None)):
                return definition.construct_expression(argument, (), self, env, token.span, expected)
            receiver = self.expr(node.operands[0], env)
            if argument.kind == 'slice':
                slicing = getattr(receiver.type, 'slice_value', None)
                if slicing is None:
                    raise ValueError('type does not supply a slicing operation')
                from py_compiler.syntax.generic.storage import root_binding
                owner = root_binding(subject, self, env) or self.call_result_owners.get(receiver.identity)
                if owner is None and callable(getattr(receiver.type, 'release', None)):
                    # Slices may borrow their source. Keep an unnamed owning
                    # receiver alive until the enclosing lexical scope exits.
                    temporary = f'<slice receiver {receiver.identity}>'
                    owner = Binding(self.body.identity + ':' + temporary, temporary,
                                    receiver.type, False, 'own', token.span)
                    env[temporary] = owner
                    self.local_names.add(temporary)
                    self.values[owner.identity] = receiver
                    self.body.bindings.append(owner)
                    self.record_lifetime('declare', owner.identity, owner.name)
                return slicing(receiver, argument.operands, self, env, token.span, owner,
                               closed=argument.interval or (True, True))
            place = receiver.type.element_place(receiver, node.operands[1], self, env, token.span)
            value = place.read(self, token.span)
            if receiver.type.family == 'array' and value.type.binding_ownership != 'copy':
                self.value_sources[value.identity] = set(self.value_sources.get(receiver.identity, ()))
                self.value_dependencies[value.identity] = set(self.value_dependencies.get(receiver.identity, ()))
            if receiver.identity in self.call_result_owners and value.type.binding_ownership != 'copy':
                self.call_result_owners[value.identity] = self.call_result_owners[receiver.identity]
                self.call_result_modes[value.identity] = self.call_result_modes.get(receiver.identity, 'view')
                if receiver.identity in self.call_result_sources:
                    self.call_result_sources[value.identity] = self.call_result_sources[receiver.identity]
            return value
        if node.kind in ("call", "member"):
            from py_compiler.syntax.function.calls import lower_expression
            value = lower_expression(node, self, env, expected)
            return value
        if node.kind == "construct":
            type_ = self.syntax.types.get(token.text)
            if type_ is None:
                raise ValueError(f"constructor {token.text!r} has no executable provider")
            if expected and type_ != expected:
                raise ValueError(f"constructor {type_.name} does not satisfy {expected.name}")
            return self.expr(node.operands[0], env, type_)
        if node.kind == "literal" or (node.kind == "unary" and token.text in ("-", "+") and node.operands[0].kind == "literal"):
            literal_token = node.operands[0].token if node.kind == "unary" else token
            spelling = (token.text if node.kind == "unary" else "") + literal_token.text
            type_, value = resolve_literal(Literal("", token.span, spelling, literal_token.kind, expected.name if expected else None), self.syntax, self)
            if not type_.mlir:
                return None
            identity = self.body.identity + ":literal:" + str(self.next_value)
            constant = Constant(identity, None, type_, value, token.span, self.source.path, None)
            return self.emit("constant", type_, (), constant, token.span)
        if node.kind in ("name", "reference"):
            binding = self.lookup(node, env)
            if binding is None:
                if node.kind == 'name':
                    entries = self.context.lookup_functions(token.text, self.declaration_scope)
                    if entries:
                        from importlib import import_module
                        return import_module('py_compiler.syntax.function.lambda').named_function(node, entries, self, env, expected)
                    declaration_name = self.context.declaration_name(token.text, self.declaration_scope)
                    variant_enum = getattr(self.context, 'enum_variants', {}).get(declaration_name)
                    if variant_enum is not None:
                        from py_compiler.syntax.function.calls import coerce
                        declaration = self.context.by_name[declaration_name]
                        if declaration.fields:
                            raise ValueError('payload enum variant requires constructor arguments')
                        value = self.context.provider_by_name[declaration_name].instantiate(declaration, (), self, env, token.span)
                        return coerce(coerce(value, variant_enum, self, token.span), expected, self, token.span)
                    declared = self.context.types.get(self.context.declaration_name(token.text, self.declaration_scope))
                    if declared is not None and getattr(declared, 'mlir', None):
                        from py_compiler.syntax.complex.type import type_value
                        return type_value(declared, self, token.span)
                    candidates = [t for name, t in self.context.types.items()
                                  if t.family == 'enum' and token.text in t.variants
                                  and name == self.context.declaration_name(name.rsplit('.', 1)[-1], self.declaration_scope)]
                    candidates = list(dict.fromkeys(candidates))
                    if len(candidates) > 1:
                        raise ValueError(f'ambiguous enum variant {token.text!r}; qualify its enum')
                    if candidates:
                        from py_compiler.syntax.function.calls import literal_value, coerce
                        enum = candidates[0]
                        return coerce(literal_value(enum.variant(token.text), enum, self, token.span), expected, self, token.span)
                raise ValueError(f"unknown binding {token.text!r}")
            if binding.type is None:
                binding = self.infer(binding, expected, env)
            self.flow.read(binding)
            value = self.read_binding(binding, token.span)
            from py_compiler.syntax.operator.logical import refined_value
            value = refined_value(binding, value, self, token.span)
            if expected and binding.type != expected:
                from py_compiler.syntax.function.calls import coerce
                return coerce(value, expected, self, token.span)
            return value
        if node.kind == "unary":
            if token.text in OWNERSHIP:
                raise ValueError("ownership modifiers must apply to the entire binding initializer")
            operand = self.expr(node.operands[0], env, self.context.type('bool') if token.text == 'not' else expected)
            from py_compiler.syntax.generic.owned import select
            return select(operand.type, "F.unary", token.text).expand(self, operand, token.span)
        if node.kind == "chain":
            left = self.expr(node.operands[0], env)
            from py_compiler.mir.ownership.flow import join_expression
            exits = []
            for index in range(1, len(node.operands), 2):
                operator, term = node.operands[index:index + 2]
                from py_compiler.syntax.generic.place import expression_type
                dimensioned = left.type.family == 'unit'
                if not dimensioned:
                    try:
                        dimensioned = expression_type(term, self, env).family == 'unit'
                    except ValueError:
                        pass
                right = self.expr(term, env, None if dimensioned else left.type)
                if dimensioned:
                    from py_compiler.syntax.complex.units import calculate
                    condition = calculate(operator.text, left, right, self, operator.span)
                else:
                    condition = self.emit('binary', self.syntax.types['bool'], (left, right), operator.text, operator.span)
                if index + 2 < len(node.operands):
                    following = self.block()
                    skipped = self.block()
                    exits.append((skipped, dict(self.values), self.flow.fork(), condition))
                    self.current.terminator = Terminator('conditional', (Edge(following.identity), Edge(skipped.identity)), condition)
                    self.current = following
                else:
                    exits.append((self.current, dict(self.values), self.flow.fork(), condition))
                left = right
            return join_expression(self, exits, env, node.token.span)
        if node.kind == "binary":
            from py_compiler.syntax.function.operations import binary
            return binary(self, node, env, expected)
        raise ValueError("unresolved expression")

    @located
    def sentence(self, node, env, local_names):
        if node.identity in self.declared_nodes:
            return
        from py_compiler.syntax.block.grammar import expand_sentence
        if expand_sentence(node, self, env):
            return
        for provider in self.syntax.sentence_providers:
            if provider.matches(node):
                return provider.lower(node, self, env, local_names)
        raise ValueError("sentence has no registered syntax provider")

    def cleanup_memory(self, bindings, span):
        bindings = list(bindings)
        for binding in bindings:
            if binding.ownership in ("view", "borrow"):
                self.flow.end_loan(binding.identity)
        pending = {binding.identity: binding for binding in bindings if binding.identity not in self.flow.moved}
        order = {binding.identity: index for index, binding in enumerate(self.body.bindings)}
        while pending:
            # End contained borrows before destroying their sources. Declaration
            # order alone is insufficient when a later source fills an older array.
            ready = [binding for binding in pending.values()
                     if not (self.flow.borrowers(binding.identity) & pending.keys())]
            if not ready:
                raise ValueError('cyclic stored borrows prevent scope destruction')
            binding = max(ready, key=lambda b: order.get(b.identity, -1))
            pending.pop(binding.identity)
            release = getattr(binding.type, 'release', None)
            if binding.type.family == 'string' and self.values.get(binding.identity) is not None:
                if self.values[binding.identity].identity in self.owned_results:
                    from py_compiler.syntax.generic.storage import release_storage
                    release = release_storage
            if release is not None and binding.ownership in ('own', 'move', 'copy', 'mirror'):
                value = self.read_binding(binding, span)
                self.flow.drop(binding)
                from py_compiler.mir.cleanup import defer
                defer(value, release, self, span)
            if binding.identity not in self.parameter_bindings:
                self.flow.end_retained(binding.identity)

    def run_defers(self, depth=0):
        for scope_depth, actions in reversed(self.deferred_actions):
            if scope_depth >= depth:
                for node, env in reversed(actions):
                    self.sentence(node, env, set())

    def scope(self, nodes, env, declarations=False, local_names=None, cleanup=True):
        local_names = set() if local_names is None else local_names
        previous_scope = self.declaration_scope
        previous_context_scope = self.context.scope if self.context else None
        if nodes and nodes[0].parent:
            parent = nodes[0].parent
            if self.context and parent not in self.declaration_scope and parent != self.body.identity and not parent.endswith(":root"):
                self.declaration_scope = (*self.declaration_scope, parent)
        if self.context:
            self.context.scope = self.declaration_scope
        previous = self.local_names
        self.local_names = local_names
        index = 0
        self.active_scopes.append(env)
        self.deferred_actions.append((len(self.active_scopes) - 1, []))
        try:
            while index < len(nodes):
                node = nodes[index]
                self.remaining_nodes = nodes[index + 1:]
                def referenced_names(terms):
                    return {token.text for term in terms for token in term.tokens} | set().union(*(referenced_names(term.children) for term in terms))
                remaining = referenced_names(nodes[index:])
                remaining.update(token.text for _, actions in self.deferred_actions
                                 for action, _ in actions for token in action.tokens)
                for binding in env.values():
                    if (binding.name in local_names and binding.type and (binding.type.family == 'pointer' or binding.ownership == 'borrow')
                            and binding.name not in remaining):
                        self.flow.end_loan(binding.identity)
                if self.current.terminator:
                    raise ValueError("unreachable sentence after a terminating operation")
                if node.provider is None:
                    self.sentence(node, env, local_names)
                    if self.phases and self.current.terminator is None:
                        self.phases.during()
                    index += 1
                else:
                    if (self.constructor and self.uninitialized_fields and not isinstance(node.provider, DeclarationProvider)
                            and getattr(node.provider, 'spelling', '') != 'unsafe'):
                        raise ValueError("initialize constructor fields before control-flow blocks")
                    index += node.provider.lower(node, self, env, nodes, index)
                    if self.phases and self.current.terminator is None:
                        self.phases.during()
            if nodes and self.current.terminator is None:
                self.run_defers(len(self.active_scopes) - 1)
            if cleanup and nodes and self.current.terminator is None:
                self.cleanup_memory((env[n] for n in local_names if n in env and env[n].identity not in self.context.global_storage), nodes[-1].span)
        finally:
            self.deferred_actions.pop()
            self.active_scopes.pop()
            self.local_names = previous
            self.declaration_scope = previous_scope
            if self.context:
                self.context.scope = previous_context_scope

    def record_retained_parameters(self):
        effects = dict(self.body.retained_parameter_sources)
        for owner in self.flow.retained:
            if owner not in self.parameter_bindings:
                continue
            sources = self.flow.sources(owner)
            if any(source not in self.parameter_bindings for source in sources):
                raise ValueError('stored borrow escapes local storage through a parameter')
            target = self.parameter_bindings[owner]
            effects[target] = tuple(sorted(set(effects.get(target, ())) | {self.parameter_bindings[source] for source in sources}))
        self.body.retained_parameter_sources = tuple(sorted(effects.items()))

    def finish(self):
        self.record_retained_parameters()
        if self.body.result_type is None:
            self.body.result_type = self.syntax.types["absent"]
        if self.current.terminator is None:
            if self.body.result_type.mlir:
                raise ValueError("not every path returns the declared result")
            self.current.terminator = Terminator("finish")
        if self.body.result_type.family == 'record' and self.body.result_ownership == 'own':
            provider = self.context.provider_by_name[self.body.result_type.name]
            self.body.result_layout = provider.layout(self.body.result_type.declaration, self.context)
        return self.body



def check_module(module):
    source, root, context = module.source, module.root, module.context
    syntax = context.syntax
    from py_compiler.mir.ownership.flow import FlowState
    context.initial_flow = FlowState()
    context.resolve_contracts()
    qualifier = '::package=' + context.interface if context.interface else ''
    from hashlib import sha256
    initializer_symbol = sha256((source.path + qualifier).encode()).hexdigest()
    initializer = Checker(source, context.bound_syntax, root.identity + qualifier, "__sev_init_" + initializer_symbol + qualifier, syntax.types["absent"])
    initializer.context = context
    initializer.block_scope = root.scope
    initializer.declared_nodes = context.declared_nodes
    root.provider.configure(initializer, context)
    initializer.scope(root.children, context.global_bindings)
    context.initial_flow = initializer.flow.fork()
    context.resolve()
    return (initializer.finish(), *context.bodies), tuple(context.declarations)


import unittest


class SliceTemporaryTests(unittest.TestCase):
    def test_owning_slice_receiver_gets_scope_cleanup(self):
        from py_compiler.frontend.src.lib import compile_source
        from py_compiler.syntax.recognition import Syntax
        result = compile_source('slice-temporary.sev', '''class Values:
    storage: array[int]
    def Values():
        storage = array[int,4]()
    def __slice(self: view Self, start: int, stop: int, step: int, missing_start: bool,
                missing_stop: bool, start_closed: bool, stop_closed: bool) -> int:
        return 42
def use() -> int:
    value = Values()[::-1]
    return value
''', Syntax(include_tests=False))
        self.assertFalse(result.diagnostics, str(result.diagnostics))
        body = next(body for body in result.program.bodies
                    if any(b.name.startswith('<slice receiver ') for b in body.bindings))
        owners = [b for b in body.bindings if b.name.startswith('<slice receiver ')]
        self.assertEqual(len(owners), 1)
        self.assertTrue(any(op.kind == 'deallocate'
                            for block in body.blocks for op in block.operations))
