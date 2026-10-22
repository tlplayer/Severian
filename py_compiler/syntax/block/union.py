"""Union owns its semantic variants, tagged representation and boxing atom."""
from dataclasses import dataclass, field
from py_compiler.syntax.generic.owned import WordGrammar
from py_compiler.syntax.generic.definition import TypeDefinition
from py_compiler.syntax.generic.block import DeclarationProvider


@dataclass(frozen=True)
class UnionType:
    variants: tuple
    declared_name: str = ''
    storage: tuple = field(default=(), compare=False)
    family: str = 'union'
    binding_ownership: str = 'view'

    def __post_init__(self):
        flattened = []
        for member in self.variants:
            for variant in member.variants if member.family == 'union' else (member,):
                if variant not in flattened:
                    flattened.append(variant)
        if not flattened:
            raise ValueError('union requires at least one member type')
        object.__setattr__(self, 'variants', tuple(flattened))

    @property
    def name(self):
        return self.declared_name or ' | '.join(t.name for t in self.variants)

    def accepts_type(self, actual):
        members = actual.variants if actual.family == 'union' else (actual,)
        return all(member in self.variants for member in members)

    @property
    def mlir(self):
        layouts = self.storage or tuple(t.mlir for t in self.variants)
        return '!llvm.struct<(i32, ' + ', '.join(layouts) + ')>'

    def with_layout(self, context):
        from dataclasses import replace
        from py_compiler.syntax.generic.storage import stored_type
        return replace(self, storage=tuple(stored_type(t, context) for t in self.variants))

    @property
    def release(self):
        from py_compiler.syntax.generic.storage import owns_storage
        return self.destroy if any(owns_storage(t) for t in self.variants) else None

    def copy_value(self, value, cfg, span):
        if not self.release:
            return value
        from py_compiler.syntax.generic.storage import copy_storage
        return self.dispatch(value, cfg, span, lambda member, index:
            cfg.emit('memory-copy', self, (copy_storage(member, cfg, span),), UnionBox(index), span))

    def destroy(self, value, cfg, span):
        from py_compiler.syntax.generic.storage import release_storage
        self.dispatch(value, cfg, span, lambda member, index: release_storage(member, cfg, span), result=False)

    def dispatch(self, value, cfg, span, action, result=True):
        from py_compiler.hir.thir.src.model import Edge, Terminator
        from py_compiler.syntax.function.calls import literal_value
        from py_compiler.syntax.generic.owned import select
        tag = cfg.emit('union-tag', cfg.context.type('i32'), (value,), UnionTag(), span)
        done = cfg.block((self,) if result else ())
        for index, member in enumerate(self.variants):
            branch, following = cfg.block(), cfg.block()
            expected = literal_value(index, tag.type, cfg, span)
            matches = select(tag.type, 'F.operator', '==').expand(cfg, tag, expected, span)
            cfg.current.terminator = Terminator('conditional', (Edge(branch.identity), Edge(following.identity)), matches)
            cfg.current = branch
            if not member.mlir:
                output = cfg.emit('union-none', self, (), UnionNone(index), span) if result else None
            else:
                extracted = cfg.emit('union-member', member, (value,), UnionMember(index), span)
                output = action(extracted, index)
            cfg.current.terminator = Terminator('jump', (Edge(done.identity, (output,) if result else ()),))
            cfg.current = following
        # An invalid tag denotes an empty value after a partial move.
        empty = cfg.emit('union-empty', self, (), UnionEmpty(), span) if result else None
        cfg.current.terminator = Terminator('jump', (Edge(done.identity, (empty,) if result else ()),))
        cfg.current = done
        if result:
            output = done.parameters[0]
            cfg.value_dependencies[output.identity] = set(cfg.value_dependencies.get(value.identity, ()))
            return output
        return None

    def field_place(self, receiver, field, cfg, span, write=False):
        if not write:
            from py_compiler.syntax.complex.error import is_error
            candidates = [(i, member) for i, member in enumerate(self.variants) if is_error(member)
                          and any(name == field for name, _ in member.declaration.fields)]
            if len(candidates) == 1:
                index, member = candidates[0]
                payload = OptionalField(receiver, index, 'value', member).read(cfg, span)
                return cfg.context.provider_by_name[member.name].field_place(payload, field, cfg, span)
        present = [i for i, member in enumerate(self.variants) if member.name not in ('None', 'absent')]
        if write or not present or len(present) == len(self.variants) or field not in ('has_value', 'value'):
            raise ValueError('optional union exposes read-only has_value and value')
        if len(present) > 1:
            result = Union().resolve([self.variants[i] for i in present], cfg.context)
            missing = next(i for i in range(len(self.variants)) if i not in present)
            return OptionalUnionField(receiver, missing, field, cfg.context.type('bool') if field == 'has_value' else result)
        return OptionalField(receiver, present[0], field, cfg.context.type('bool') if field == 'has_value' else self.variants[present[0]])

    def require_context(self, cfg):
        pass

    def grammars(self):
        return (WordGrammar(self, self.name, 'IDENTIFIER'),)


class Union(TypeDefinition):
    def __init__(self):
        super().__init__('union')

    def resolve(self, variants, context):
        return UnionType(tuple(variants)).with_layout(context)


class UnionDeclaration(DeclarationProvider):
    spelling = 'union'

    def grammars(self):
        # The union type definition supplies the shared lexical spelling.
        return (self,)

    def parse_header(self, items):
        header = super().parse_header(items)
        if len(header) != 1 or header[0].kind != 'IDENTIFIER':
            raise ValueError('union requires a name')
        return header

    def declare(self, node, context):
        from py_compiler.syntax.block.record import TypeDeclaration
        name = '.'.join((*context.scope, node.header[0].text))
        spellings = tuple(''.join(t.text for t in child.tokens) for child in node.children)
        # Generic spellings are realized by type resolution; they need not
        # already be dictionary keys before a union mentions them.
        try:
            resolved = tuple(context.type(spelling) for spelling in spellings)
        except ValueError as error:
            if 'unresolved type' not in str(error):
                raise
            resolved = ()
        if not resolved or any(isinstance(member, ForwardUnion) for member in resolved):
            context.types[name] = ForwardUnion(name)
            context.pending_unions = getattr(context, 'pending_unions', []) + [(self,node,context.scope)]
            return
        members = []
        for child, member in zip(node.children, resolved):
            if child.provider or not child.tokens:
                raise ValueError('union block requires member types')
            members.append(member)
        type_ = UnionType(tuple(members), name).with_layout(context)
        context.register(TypeDeclaration(node.identity, name, self.spelling, (),
                         tuple(member.name for member in type_.variants), (), node.span, context.source.path))
        context.types[name] = type_


@dataclass(frozen=True)
class UnionBox:
    variant: int

    def atom(self, operation):
        from py_compiler.syntax.generic.atom import Atom
        result = operation.result.type
        return Atom(self, (result.variants[self.variant],), result, ('view',), (), self)

    def render_operation(self, operation):
        from py_compiler.mlir.src.cfg import name
        value = operation.result
        type_ = value.type.mlir
        from py_compiler.syntax.generic.storage import pack_value
        lines = []
        operand = operation.operands[0]
        stored = pack_value(lines, name(operand), operand.type,
                            value.type.storage[self.variant], f'%union_stored_{value.identity}')
        return lines + [f'%union_{value.identity} = llvm.mlir.zero : {type_}',
                f'%union_tag_{value.identity} = arith.constant {self.variant} : i32',
                f'%union_value_{value.identity} = llvm.insertvalue {stored}, %union_{value.identity}[{self.variant + 1}] : {type_}',
                f'{name(value)} = llvm.insertvalue %union_tag_{value.identity}, %union_value_{value.identity}[0] : {type_}']


@dataclass(frozen=True)
class UnionTag:
    def atom(self, operation):
        from py_compiler.syntax.generic.atom import Atom
        return Atom(self, (operation.operands[0].type,), operation.result.type, ('view',), ('read',), self)

    def render_operation(self, operation):
        from py_compiler.mlir.src.cfg import name
        return [f'{name(operation.result)} = llvm.extractvalue {name(operation.operands[0])}[0] : {operation.operands[0].type.mlir}']


@dataclass(frozen=True)
class UnionMember(UnionTag):
    index: int

    def entry_allocations(self, operation):
        return self.render_operation(operation)[1:3] if operation.result.type.family == 'record' else []

    def render_operation(self, operation):
        from py_compiler.mlir.src.cfg import name
        from py_compiler.syntax.generic.storage import unpack_value
        source, result = operation.operands[0], operation.result
        stored = f'%union_member_{result.identity}'
        lines = [f'{stored} = llvm.extractvalue {name(source)}[{self.index + 1}] : {source.type.mlir}']
        unpack_value(lines, stored, result.type, source.type.storage[self.index], name(result))
        return lines


@dataclass(frozen=True)
class UnionEmpty:
    def atom(self, operation):
        from py_compiler.syntax.generic.atom import Atom
        return Atom(self, (), operation.result.type, (), (), self)

    def render_operation(self, operation):
        from py_compiler.mlir.src.cfg import name
        result = operation.result
        return [f'%empty_{result.identity} = llvm.mlir.zero : {result.type.mlir}',
                f'%empty_tag_{result.identity} = arith.constant -1 : i32',
                f'{name(result)} = llvm.insertvalue %empty_tag_{result.identity}, %empty_{result.identity}[0] : {result.type.mlir}']



@dataclass(frozen=True)
class UnionWiden:
    narrow: bool = False
    def atom(self, operation):
        from py_compiler.syntax.generic.atom import Atom
        return Atom(self, (operation.operands[0].type,), operation.result.type, ('view',), (), self)

    def render_operation(self, operation):
        from py_compiler.mlir.src.cfg import name
        source, result = operation.operands[0], operation.result
        prefix = f'%widen_{result.identity}'
        lines = [f'{prefix}_source_tag = llvm.extractvalue {name(source)}[0] : {source.type.mlir}',
                 f'{prefix}_initial = llvm.mlir.zero : {result.type.mlir}',
                 f'{prefix}_tag_0 = arith.constant -1 : i32']
        aggregate = prefix + '_initial'
        tag = prefix + '_tag_0'
        for index, member in enumerate(source.type.variants):
            if self.narrow and member not in result.type.variants:
                continue
            destination = result.type.variants.index(member)
            base = f'{prefix}_{index}'
            lines += [f'{base}_value = llvm.extractvalue {name(source)}[{index + 1}] : {source.type.mlir}',
                      f'{base}_aggregate = llvm.insertvalue {base}_value, {aggregate}[{destination + 1}] : {result.type.mlir}',
                      f'{base}_source = arith.constant {index} : i32',
                      f'{base}_target = arith.constant {destination} : i32',
                      f'{base}_active = arith.cmpi eq, {prefix}_source_tag, {base}_source : i32',
                      f'{base}_tag = arith.select {base}_active, {base}_target, {tag} : i32']
            aggregate, tag = base + '_aggregate', base + '_tag'
        lines.append(f'{name(result)} = llvm.insertvalue {tag}, {aggregate}[0] : {result.type.mlir}')
        return lines


@dataclass(frozen=True)
class UnionNone(UnionEmpty):
    index: int = 1

    def render_operation(self, operation):
        return [line.replace('constant -1 : i32', f'constant {self.index} : i32') for line in super().render_operation(operation)]


@dataclass(frozen=True)
class OptionalField:
    receiver: object
    index: int
    field: str
    type: object

    def read(self, cfg, span):
        from py_compiler.syntax.function.calls import literal_value
        from py_compiler.syntax.generic.owned import select
        from py_compiler.syntax.complex.object import Assert
        tag = cfg.emit('union-tag', cfg.context.type('i32'), (self.receiver,), UnionTag(), span)
        expected = literal_value(self.index, tag.type, cfg, span)
        present = select(tag.type, 'F.operator', '==').expand(cfg, tag, expected, span)
        if self.field == 'has_value':
            return present
        cfg.effect('assert', (present,), Assert('iterator is exhausted; optional value is absent'), span)
        return cfg.emit('union-member', self.type, (self.receiver,), UnionMember(self.index), span)


@dataclass(frozen=True)
class OptionalUnionField(OptionalField):
    def read(self, cfg, span):
        from py_compiler.syntax.function.calls import literal_value
        from py_compiler.syntax.generic.owned import select
        from py_compiler.syntax.complex.object import Assert
        tag = cfg.emit('union-tag', cfg.context.type('i32'), (self.receiver,), UnionTag(), span)
        missing = literal_value(self.index, tag.type, cfg, span)
        present = select(tag.type, 'F.operator', '!=').expand(cfg, tag, missing, span)
        if self.field == 'has_value':
            return present
        cfg.effect('assert', (present,), Assert('optional value is absent'), span)
        return cfg.emit('union-project', self.type, (self.receiver,), UnionWiden(narrow=True), span)


def fallback_expression(node, cfg, env, expected=None):
    """Evaluate an optional once, and evaluate its fallback only when absent."""
    from py_compiler.hir.thir.src.model import Edge, Terminator
    from py_compiler.mir.ownership.flow import join_expression
    from py_compiler.syntax.generic.storage import root_binding, copy_storage, owns_storage, release_storage
    from py_compiler.syntax.function.calls import coerce
    source, fallback = node.operands
    value = cfg.expr(source, env)
    from py_compiler.syntax.complex.error import is_error
    members = [i for i, t in enumerate(value.type.variants) if t.name not in ('None', 'absent') and not is_error(t)] if value and value.type.family == 'union' else []
    if value is None or len(members) != 1 or not any(t.name in ('None', 'absent') or is_error(t) for t in value.type.variants):
        raise ValueError('else fallback requires an optional value')
    index = members[0]
    target = value.type.variants[index]
    owner = root_binding(source, cfg, env) or cfg.call_result_owners.get(value.identity)
    present = OptionalField(value, index, 'has_value', cfg.context.type('bool')).read(cfg, node.token.span)
    yes, no = cfg.block(), cfg.block()
    cfg.current.terminator = Terminator('conditional', (Edge(yes.identity), Edge(no.identity)), present)
    initial_values, initial_flow = dict(cfg.values), cfg.flow.fork()
    cfg.current = yes
    selected = cfg.emit('union-member', target, (value,), UnionMember(index), node.token.span)
    if owns_storage(target):
        selected = copy_storage(selected, cfg, node.token.span)
    exits = [(cfg.current, dict(cfg.values), cfg.flow.fork(), selected)]
    cfg.current, cfg.values, cfg.flow = no, dict(initial_values), initial_flow.fork()
    fallback_owner = None
    if fallback.kind == 'fallback_throw':
        error = fallback.operands[0]
        if error.kind != 'call' or error.operands[0].kind != 'name':
            raise ValueError('fallback throw requires an Error constructor')
        error_type = cfg.context.types.get(error.operands[0].token.text)
        if not is_error(error_type):
            raise ValueError('fallback throw requires an Error value')
        cfg.cleanup_memory((b for scope in cfg.active_scopes for b in scope.values()), node.token.span)
        error_type.raise_failure(error.operands[1:], cfg, node.token.span)
    else:
        alternative = cfg.expr(fallback, dict(env), target)
        if alternative is None or alternative.type != target:
            raise ValueError('optional fallback must satisfy its present value type')
        fallback_owner = root_binding(fallback, cfg, env) or cfg.call_result_owners.get(alternative.identity)
        if owns_storage(target) and fallback_owner is not None:
            alternative = copy_storage(alternative, cfg, node.token.span)
        exits.append((cfg.current, dict(cfg.values), cfg.flow.fork(), alternative))
    result = join_expression(cfg, exits, env, node.token.span)
    for facts in (cfg.value_sources, cfg.value_dependencies, cfg.value_origins):
        facts[result.identity] = set().union(*(facts.get(output.identity, set()) for _, _, _, output in exits))
    if any(output.identity in cfg.stack_values for _, _, _, output in exits):
        cfg.stack_values.add(result.identity)
    if target.family == 'pointer':
        owners = [binding for binding in (owner, fallback_owner) if binding is not None]
        if owners:
            cfg.call_result_owners[result.identity] = owners[0]
            cfg.call_result_modes[result.identity] = 'view'
            cfg.call_result_sources[result.identity] = frozenset(cfg.flow.resolve(binding.identity) for binding in owners)
    if owns_storage(target):
        cfg.owned_results.add(result.identity)
        cfg.value_sources[result.identity] = set()
    if owner is None and owns_storage(value.type):
        release_storage(value, cfg, node.token.span)
    return coerce(result, expected, cfg, node.token.span) if expected else result


import unittest


class UnionTests(unittest.TestCase):
    def compile(self, source):
        from py_compiler.frontend.src.lib import compile_source
        from py_compiler.syntax.recognition import Syntax
        return compile_source('union.sev', source, Syntax())

    def test_blocks_compose_named_and_inline_unions(self):
        source = ('union Units:\n    u8\n    u32\n'
                  'union Values:\n    Units\n    u8 | bool\n    f64\n'
                  'def choose(value: u32) -> Values:\n    selected: Values = value\n    return selected\n'
                  'def work():\n    choose(u32(955))\n')
        result = self.compile(source)
        self.assertFalse(result.diagnostics, str(result.diagnostics))
        from py_compiler.mlir.src.lib import lower, render
        text = render(lower(result.program))
        self.assertIn('!llvm.struct<(i32, i8, i32, i1, f64)>', text)
        self.assertIn('llvm.insertvalue', text)

    def test_named_union_constraints_and_inference(self):
        source = ('union Number:\n    int\n    float\n'
                  'def add[T: Number](left: T, right: T) -> T:\n    return left + right\n')
        result = self.compile(source + 'def valid():\n    add(1, 2)\n    add(1.0, 2.0)\n')
        self.assertFalse(result.diagnostics, str(result.diagnostics))
        for call in ('add(1, 2.0)', 'add(true, false)'):
            with self.subTest(call=call):
                self.assertTrue(self.compile(source + 'def invalid():\n    ' + call + '\n').diagnostics)

    def test_composed_union_accepts_values_from_a_member_union(self):
        source = ('union Small:\n    u8\n    u32\nunion Large:\n    bool\n    Small\n'
                  'def widen(value: Small) -> Large:\n    return value\n'
                  'def work() -> Large:\n    return widen(u32(955))\n')
        result = self.compile(source)
        self.assertFalse(result.diagnostics, str(result.diagnostics))
        from py_compiler.mlir.src.lib import lower, render
        text = render(lower(result.program))
        self.assertIn('arith.select', text)
        self.assertIn('!llvm.struct<(i32, i1, i8, i32)>', text)

    def test_runtime_memory_union_has_an_owned_descriptor(self):
        result = self.compile('union Text:\n    string\n    char\ndef text() -> Text:\n    return "hello"\n')
        self.assertFalse(result.diagnostics, str(result.diagnostics))
        from py_compiler.mlir.src.lib import lower, render
        text = render(lower(result.program))
        self.assertIn('memref.alloc', text)
        self.assertIn('llvm.insertvalue', text)

    def test_union_can_compose_collection_types_for_constraints(self):
        result = self.compile('union Sequence:\n    string\n    array[int]\ndef length[T: Sequence](value: T) -> int:\n    return 0\ndef work():\n    length("text")\n')
        self.assertFalse(result.diagnostics, str(result.diagnostics))

    def test_unknown_member_and_duplicate_declaration_rejected(self):
        for source in ('union Bad:\n    Missing\n', 'union Value:\n    int\nunion Value:\n    bool\n'):
            self.assertTrue(self.compile(source).diagnostics)


class OptionalFallbackTests(unittest.TestCase):
    def compile(self, source):
        from py_compiler.frontend.src.lib import compile_source
        from py_compiler.syntax.recognition import Syntax
        return compile_source('optional-fallback.sev', source, Syntax())

    def test_scalar_fallback_and_throw_lower_to_conditional_edges(self):
        from py_compiler.mlir.src.lib import lower, render
        for fallback in ('7', 'throw Error("missing")'):
            result = self.compile('def get(flag: bool) -> int | None:\n    if flag:\n        return 42\n    return None\n'
                                  f'def work(flag: bool) -> int:\n    return get(flag) else {fallback}\n')
            self.assertFalse(result.diagnostics, str(result.diagnostics))
            self.assertIn('cf.cond_br', render(lower(result.program)))

    def test_rejects_nonoptional_and_wrong_fallback_type(self):
        for body in ('return 42 else 7', 'value: int | None = None\n    return value else true'):
            result = self.compile('def work() -> int:\n    ' + body + '\n')
            self.assertTrue(result.diagnostics)

    def test_fallback_composes_with_ordinary_conditional_expression(self):
        result = self.compile('def work(value: int | None, flag: bool) -> int:\n    return value else (7 if flag else 9)\n')
        self.assertFalse(result.diagnostics, str(result.diagnostics))


    def test_optional_pointer_cannot_hide_its_local_owner(self):
        result = self.compile('def escape() -> pointer[int]:\n    unsafe:\n'
                              '        owner = tensor[int](1)\n'
                              '        address = &owner[0]\n        choice: pointer[int] | None = address\n'
                              '        selected = choice else address\n        return selected\n')
        self.assertTrue(result.diagnostics)
        self.assertIn('local storage', str(result.diagnostics[0]))

    def test_optional_pointer_preserves_both_parameter_sources(self):
        result = self.compile('def choose(value: pointer[int] | None, fallback: pointer[int]) -> pointer[int]:\n'
                              '    unsafe:\n        selected = value else fallback\n        return selected\n')
        self.assertFalse(result.diagnostics, str(result.diagnostics))
        body = next(b for b in result.program.bodies if b.declaration == 'choose')
        self.assertEqual(body.result_sources, (0, 1))


    def test_direct_optional_pointer_result_preserves_parameter_sources(self):
        result = self.compile('def choose(value: pointer[int] | None, fallback: pointer[int]) -> pointer[int]:\n'
                              '    unsafe:\n        return value else fallback\n')
        self.assertFalse(result.diagnostics, str(result.diagnostics))
        body = next(b for b in result.program.bodies if b.declaration == 'choose')
        self.assertEqual(body.result_sources, (0, 1))


@dataclass(frozen=True)
class ForwardUnion:
    name: str
    family: str = 'union'


def equality(left, node, cfg, env):
    """Compare active variants without converting an unrelated union member."""
    from py_compiler.hir.thir.src.model import Edge, Terminator
    from py_compiler.syntax.function.calls import literal_value
    from py_compiler.syntax.generic.owned import select
    from py_compiler.syntax.generic.storage import root_binding, release_storage, owns_storage
    from py_compiler.syntax.complex.tensor import ArgumentValue
    from py_compiler.syntax.grammar.expression import Expression
    right = cfg.expr(node.operands[1], env)
    boolean = cfg.context.type('bool')
    done = cfg.block((boolean,))
    left_owner, right_owner = (root_binding(term,cfg,env) for term in node.operands)
    left_tag = cfg.emit('union-tag', cfg.context.type('i32'), (left,), UnionTag(), node.token.span)
    right_tag = cfg.emit('union-tag', cfg.context.type('i32'), (right,), UnionTag(), node.token.span) if right.type.family == 'union' else None
    for i, member in enumerate(left.type.variants):
        yes,no = cfg.block(),cfg.block()
        active = select(left_tag.type,'F.operator','==').expand(cfg,left_tag,literal_value(i,left_tag.type,cfg,node.token.span),node.token.span)
        cfg.current.terminator=Terminator('conditional',(Edge(yes.identity),Edge(no.identity)),active)
        cfg.current=yes
        if (right_tag is None and member != right.type) or (right_tag is not None and member not in right.type.variants):
            result=literal_value(False,boolean,cfg,node.token.span)
        else:
            if right_tag is not None:
                matching,mismatch=cfg.block(),cfg.block()
                index=right.type.variants.index(member)
                same=select(right_tag.type,'F.operator','==').expand(cfg,right_tag,literal_value(index,right_tag.type,cfg,node.token.span),node.token.span)
                cfg.current.terminator=Terminator('conditional',(Edge(matching.identity),Edge(mismatch.identity)),same)
                cfg.current=mismatch
                false=literal_value(False,boolean,cfg,node.token.span)
                cfg.current.terminator=Terminator('jump',(Edge(done.identity,(false,)),))
                cfg.current=matching
                other=cfg.emit('union-member',member,(right,),UnionMember(index),node.token.span) if member.mlir else None
            else: other=right
            if not member.mlir:
                result=literal_value(True,boolean,cfg,node.token.span)
            else:
                value=cfg.emit('union-member',member,(left,),UnionMember(i),node.token.span)
                # A projection is borrowed from its union, including temporary unions.
                from py_compiler.mir.ownership.flow import Binding
                def capture(value, owner):
                    if owner is None:
                        owner=Binding('union-comparison:'+str(value.identity),':union-member',value.type,True,'view',node.token.span)
                        cfg.values[owner.identity]=value
                    return Expression('capture',node.token,(ArgumentValue(value,owner),))
                comparison=Expression('binary',node.token.with_text('=='),(capture(value,left_owner),capture(other,right_owner)))
                result=cfg.expr(comparison,env)
        cfg.current.terminator=Terminator('jump',(Edge(done.identity,(result,)),))
        cfg.current=no
    false=literal_value(False,boolean,cfg,node.token.span)
    cfg.current.terminator=Terminator('jump',(Edge(done.identity,(false,)),))
    cfg.current=done
    for value,owner in ((left,left_owner),(right,right_owner)):
        if owner is None and value.identity in cfg.owned_results and owns_storage(value.type):
            release_storage(value,cfg,node.token.span)
    result=done.parameters[0]
    return select(boolean,'F.unary','not').expand(cfg,result,node.token.span) if node.token.text=='!=' else result
