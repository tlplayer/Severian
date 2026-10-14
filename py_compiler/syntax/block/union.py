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
        from py_compiler.mir.cfg.cfg import Edge, Terminator
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
            extracted = cfg.emit('union-member', member, (value,), UnionMember(index), span)
            output = action(extracted, index)
            cfg.current.terminator = Terminator('jump', (Edge(done.identity, (output,) if result else ()),))
            cfg.current = following
        # An invalid tag denotes an empty value after a partial move.
        empty = cfg.emit('union-empty', self, (), UnionEmpty(), span) if result else None
        cfg.current.terminator = Terminator('jump', (Edge(done.identity, (empty,) if result else ()),))
        cfg.current = done
        return done.parameters[0] if result else None

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
        members = []
        for child in node.children:
            if child.provider or not child.tokens:
                raise ValueError('union block requires member types')
            members.append(context.type(''.join(token.text for token in child.tokens)))
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
    def atom(self, operation):
        from py_compiler.syntax.generic.atom import Atom
        return Atom(self, (operation.operands[0].type,), operation.result.type, ('view',), (), self)

    def render_operation(self, operation):
        from py_compiler.mlir.src.cfg import name
        source, result = operation.operands[0], operation.result
        prefix = f'%widen_{result.identity}'
        lines = [f'{prefix}_source_tag = llvm.extractvalue {name(source)}[0] : {source.type.mlir}',
                 f'{prefix}_initial = llvm.mlir.zero : {result.type.mlir}',
                 f'{prefix}_tag_0 = arith.constant 0 : i32']
        aggregate = prefix + '_initial'
        tag = prefix + '_tag_0'
        for index, member in enumerate(source.type.variants):
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
