"""Enum owns its declaration block, variant identity, ordering and representation."""
from py_compiler.syntax.block.record import RecordProvider, TypeDeclaration
from dataclasses import dataclass
from py_compiler.syntax.primitive.numeric.types import Integer
from py_compiler.syntax.primitive.numeric.grammar import scalar_grammars
from py_compiler.syntax.generic.owned import WordGrammar


@dataclass(frozen=True)
class EnumType(Integer):
    variants: tuple = ()
    transitions: tuple = ()
    values: tuple = ()
    value_sets: tuple = ()

    def grammars(self):
        return (WordGrammar(self, self.name, "IDENTIFIER"), *scalar_grammars(self, arithmetic=False))

    def variant(self, name):
        if name not in self.variants:
            raise ValueError(f"unknown {self.name} variant {name!r}")
        return self.variants.index(name)

    def field_place(self, receiver, field, cfg, span, write=False):
        if write or field not in ('key', 'value'):
            raise ValueError('enum exposes read-only key and value')
        return EnumField(receiver, field)

    def accept_literal(self, owner, value):
        if owner != self:
            raise ValueError(f"{self.name} requires a named variant")
        return value

    def check_assignment(self, previous, following, cfg):
        if not self.transitions:
            return
        all_states = frozenset(range(len(self.variants)))
        definitions = {op.result.identity: op for block in cfg.body.blocks
                       for op in block.operations if op.result is not None}
        parameters = {value.identity: (block, index) for block in cfg.body.blocks
                      for index, value in enumerate(block.parameters)}

        def states(value, active=frozenset()):
            if value.identity in active:
                return all_states
            operation = definitions.get(value.identity)
            if operation is not None and operation.kind == 'constant':
                return frozenset((operation.payload.value,))
            parameter = parameters.get(value.identity)
            if parameter is None or cfg.loops:
                return all_states
            block, index = parameter
            incoming = [edge.arguments[index] for source in cfg.body.blocks
                        if source.terminator is not None for edge in source.terminator.edges
                        if edge.target == block.identity and len(edge.arguments) > index]
            if not incoming:
                return all_states
            return frozenset().union(*(states(item, active | {value.identity}) for item in incoming))

        allowed = dict(self.transitions)
        for source in states(previous):
            for target in states(following):
                if target not in allowed.get(source, ()):
                    raise ValueError(f'illegal {self.name} transition: '
                                     f'{self.variants[source]} -> {self.variants[target]}')


class Enum(RecordProvider):
    spelling = "enum"

    def declare(self, node, context):
        if any(len(child.tokens)>1 and child.tokens[1].text == '(' for child in node.children):
            return declare_payload_enum(node, context)
        variants, edges, values, value_sets = [], {}, [], []
        for child in node.children:
            if child.provider or not child.tokens or child.tokens[0].kind != "IDENTIFIER":
                raise ValueError("enum provider requires payload-free named variants")
            name = child.tokens[0].text
            if name in variants:
                raise ValueError("duplicate enum variant")
            variants.append(name)
            suffix = child.tokens[1:]
            value, choices = None, ()
            if suffix and suffix[0].text == '{':
                import ast
                close = next((i for i,t in enumerate(suffix) if t.text == '}'), -1)
                if close < 0:
                    raise ValueError('enum value set requires closing brace')
                groups = ''.join(t.text for t in suffix[1:close])
                choices = ast.literal_eval('(' + groups + ',)')
                if not choices or any(not isinstance(v, (int, float, str)) for v in choices):
                    raise ValueError('enum value set requires scalar literals')
                tail = suffix[close+1:]
                if tail:
                    if tail[0].text != '=':
                        raise ValueError('enum default requires = value')
                    value = ast.literal_eval(''.join(t.text for t in tail[1:]))
                    if value not in choices:
                        raise ValueError('enum default must belong to its value set')
                elif len(choices) == 1:
                    value = choices[0]
                else:
                    raise ValueError('enum with multiple values requires a default')
                suffix = ()
            values.append(value)
            value_sets.append(choices)
            if suffix:
                if (suffix[0].text != '->' or len(suffix) < 2 or len(suffix) % 2
                        or any(t.kind != 'IDENTIFIER' for t in suffix[1::2])
                        or any(t.text != '|' for t in suffix[2::2])):
                    raise ValueError('expected enum variant -> destination [| destination]')
                destinations = tuple(t.text for t in suffix[1::2])
                if len(set(destinations)) != len(destinations):
                    raise ValueError('duplicate enum transition destination')
                edges[name] = destinations
        for destinations in edges.values():
            if any(name not in variants for name in destinations):
                raise ValueError('unknown enum transition destination')
        context.register(TypeDeclaration(node.identity, ".".join((*context.scope, node.header[0].text)), self.spelling,
                                         (), tuple(variants), (), node.span, context.source.path))

        name = ".".join((*context.scope, node.header[0].text))
        transitions = tuple((variants.index(source), tuple(variants.index(target) for target in targets))
                            for source, targets in edges.items())
        context.types[name] = EnumType(name, "enum", "i32", 32, False, tuple(variants), transitions, tuple(values), tuple(value_sets))


import unittest


class EnumTransitionTests(unittest.TestCase):
    declaration = '''enum State:
    Start -> Middle | Done
    Middle -> Done
    Done
'''

    def compile(self, source):
        from py_compiler.frontend.src.lib import compile_source
        from py_compiler.syntax.recognition import Syntax
        return compile_source('enum-transitions.sev', self.declaration + source,
                              Syntax(include_tests=True))

    def test_mutable_terminal_reassignment_reports_the_edge(self):
        result = self.compile('def bad():\n    state = State.Done\n    state = State.Start\n')
        self.assertIn('illegal State transition: Done -> Start', str(result.diagnostics))

    def test_join_accepts_an_edge_valid_from_every_predecessor(self):
        result = self.compile('def work(condition: bool) -> State:\n    state = State.Start\n    if condition:\n        state = State.Middle\n    state = State.Done\n    return state\n')
        self.assertFalse(result.diagnostics, str(result.diagnostics))
        for body in result.program.bodies:
            body.verify()

    def test_join_rejects_one_invalid_predecessor(self):
        result = self.compile('def bad(condition: bool):\n    state = State.Start\n    if condition:\n        state = State.Done\n    state = State.Middle\n')
        self.assertIn('illegal State transition: Done -> Middle', str(result.diagnostics))

    def test_loop_carried_state_does_not_keep_first_iteration_facts(self):
        result = self.compile('def bad(count: int):\n    state = State.Start\n    index = 0\n    while index < count:\n        state = State.Middle\n        index += 1\n')
        self.assertIn('illegal State transition:', str(result.diagnostics))

    def test_plain_enums_remain_unrestricted_and_locals_shadow_variants(self):
        result = self.compile('enum Plain:\n    A\n    B\ndef work() -> int:\n    state = Plain.A\n    state = Plain.B\n    Done = 7\n    return Done\n')
        self.assertFalse(result.diagnostics, str(result.diagnostics))

    def test_unknown_and_duplicate_destinations_are_rejected(self):
        for declaration, diagnostic in (
            ('enum Broken:\n    A -> Missing\n', 'unknown enum transition destination'),
            ('enum Broken:\n    A -> B | B\n    B\n', 'duplicate enum transition destination'),
        ):
            with self.subTest(declaration=declaration):
                self.assertIn(diagnostic, str(self.compile(declaration).diagnostics))


@dataclass(frozen=True)
class EnumField:
    receiver: object
    field: str

    def read(self, cfg, span):
        from py_compiler.syntax.function.calls import literal_value, coerce
        from py_compiler.syntax.block.union import UnionType, UnionNone
        from py_compiler.hir.thir.src.model import Edge, Terminator
        from py_compiler.syntax.generic.owned import select
        enum = self.receiver.type
        values = enum.variants if self.field == 'key' else enum.values
        types = tuple(cfg.context.type('absent' if v is None else 'string' if isinstance(v,str) else 'f64' if isinstance(v,float) else 'int') for v in values)
        # A known variant keeps its concrete value type (including absent).
        defining = next((o for b in cfg.body.blocks for o in b.operations if o.result == self.receiver and o.kind == 'constant'), None)
        if defining is not None:
            index = defining.payload.value
            return literal_value(values[index],types[index],cfg,span) if values[index] is not None else cfg.emit('union-none', UnionType((cfg.context.type('absent'),)).with_layout(cfg.context), (), UnionNone(0), span)
        result_type = UnionType(types).with_layout(cfg.context)
        if len(result_type.variants)==1:
            result_type = result_type.variants[0]
        done = cfg.block((result_type,))
        for index,(value,type_) in enumerate(zip(values,types)):
            yes,no = cfg.block(),cfg.block()
            test = select(enum,'F.operator','==').expand(cfg,self.receiver,literal_value(index,enum,cfg,span),span)
            cfg.current.terminator=Terminator('conditional',(Edge(yes.identity),Edge(no.identity)),test)
            cfg.current=yes
            output = (coerce(literal_value(value,type_,cfg,span),result_type,cfg,span) if value is not None else cfg.emit('union-none', result_type, (), UnionNone(result_type.variants.index(type_)), span))
            cfg.current.terminator=Terminator('jump',(Edge(done.identity,(output,)),))
            cfg.current=no
        cfg.current.terminator=Terminator('panic',message='invalid enum variant')
        cfg.current=done
        return done.parameters[0]


from py_compiler.syntax.block.union import UnionType


@dataclass(frozen=True)
class PayloadEnum(UnionType):
    enum_payloads: bool = True


def declare_payload_enum(node, context):
    from py_compiler.syntax.complex.object import ObjectType
    from py_compiler.syntax.generic.storage import owns_storage
    from py_compiler.syntax.complex.generic import split_arguments
    from importlib import import_module
    variants=[]
    for child in node.children:
        if child.provider or not child.tokens or child.tokens[0].kind != 'IDENTIFIER':
            raise ValueError('enum requires named variants')
        name='.'.join((*context.scope,child.tokens[0].text))
        fields=[]
        if len(child.tokens)>1:
            if child.tokens[1].text != '(' or child.tokens[-1].text != ')':
                raise ValueError('payload variant requires named typed parameters')
            for parameter in split_arguments(''.join(t.text for t in child.tokens[2:-1])):
                if ':' not in parameter:
                    raise ValueError('payload field requires a type')
                key,type_=parameter.split(':',1)
                if any(k==key for k,_ in fields):
                    raise ValueError('duplicate payload field')
                fields.append((key,context.type(type_).name))
        if name in context.types:
            raise ValueError('duplicate enum variant ' + name)
        declaration=TypeDeclaration(child.identity,name,'class',tuple(fields),(),(),child.span,defaults=(None,)*len(fields))
        context.by_name[name]=declaration
        context.provider_by_name[name]=import_module('py_compiler.syntax.block.class').Class()
        context.types[name]=ObjectType(name,'record','!llvm.ptr',declaration=declaration,owns_fields=any(owns_storage(context.type(t)) for _,t in fields))
        variants.append(context.types[name])
    name='.'.join((*context.scope,node.header[0].text))
    type_=PayloadEnum(tuple(variants),name).with_layout(context)
    context.types[name]=type_
    context.register(TypeDeclaration(node.identity,name,'enum',(),tuple(v.name for v in variants),(),node.span))
    if not hasattr(context,'enum_variants'):
        context.enum_variants={}
    context.enum_variants.update((v.name,type_) for v in variants)
