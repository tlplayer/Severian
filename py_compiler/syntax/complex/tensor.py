"""Ranked, owned tensor storage. Shape and strides are carried by MLIR memrefs."""
from dataclasses import dataclass, replace, field
from py_compiler.syntax.generic.atom import Atom
from py_compiler.mlir.src.cfg import name, temporary_suffix


def parse_shape(spelling):
    import ast
    if spelling.startswith('tuple(') and spelling.endswith(')'):
        spelling = '(' + spelling[6:-1] + ',)'
    try:
        shape = ast.literal_eval(spelling)
    except (SyntaxError, ValueError) as failure:
        raise ValueError('tensor shape requires a tuple of nonnegative integer extents') from failure
    if not isinstance(shape, tuple) or not shape:
        raise ValueError('tensor shape must be a nonempty tuple, for example (1, 2, 3)')
    if any(n is not None and (type(n) is not int or n < 0 or n > (1 << 63) - 1) for n in shape):
        raise ValueError('tensor extents must be nonnegative integers within the index range')
    return shape


@dataclass(frozen=True)
class Tensor:
    name: str = 'tensor'
    family: str = 'tensor'

    def grammars(self):
        from py_compiler.syntax.generic.owned import WordGrammar
        return (WordGrammar(self, self.name, 'IDENTIFIER'),)

    def specialize(self, element):
        return TensorType(element)

    def specialize_arguments(self, arguments, context):
        if len(arguments) not in (1, 2, 3) or len(arguments) == 3 and arguments[2] != 'strided':
            raise ValueError('tensor requires T and an optional shape tuple')
        shape = parse_shape(arguments[1]) if len(arguments) >= 2 else (None,)
        return TensorType(context.type(arguments[0]), shape=shape, strided=len(arguments) == 3)

    def construct_expression(self, argument, arguments, cfg, env, span, expected=None):
        from py_compiler.syntax.grammar.expression import type_spelling
        terms = argument.operands if argument and argument.kind == 'template_arguments' else (argument,)
        if not terms[0]:
            raise ValueError('tensor construction requires an element type')
        if len(terms) == 2:
            shape_term = terms[1]
            shape_terms = (shape_term.operands if shape_term.kind == 'tuple' else
                           shape_term.operands[1:] if shape_term.kind == 'call' and shape_term.operands[0].token.text == 'tuple' else None)
            if shape_terms and any(t.kind not in ('literal', 'unary') and t.token.text != 'None' for t in shape_terms):
                if arguments:
                    raise ValueError('runtime shape is supplied in the shape tuple; no constructor extents are needed')
                element = cfg.context.type(type_spelling(terms[0]))
                shape = tuple(int(t.token.text) if t.kind == 'literal' and t.token.kind == 'NUMBER' and t.token.text.isdigit() else None for t in shape_terms)
                result = TensorType(element, shape=shape)
                dimensions = result.dimensions(shape_terms, cfg, env)
                value = cfg.emit('allocate-tensor', result, dimensions, TensorOperation('allocate'), span)
                cfg.owned_results.add(value.identity)
                return value
        result = self.specialize_arguments(tuple(type_spelling(t) for t in terms), cfg.context)
        if result.strided:
            raise ValueError('strided tensor views are created by slicing an existing tensor')
        if len(arguments) == 1:
            from py_compiler.syntax.generic.storage import root_binding
            from py_compiler.syntax.grammar.expression import Expression
            source = cfg.expr(arguments[0], env)
            if isinstance(source.type, TensorType):
                if source.type.element != result.element:
                    raise ValueError('tuple/tensor conversion requires matching element types')
                if len(terms) == 1:
                    result = replace(result, shape=(None,) * source.type.rank)
                if result.count is not None and source.type.count is not None and result.count != source.type.count:
                    raise ValueError('tensor shape element count does not match source storage')
                if None in result.shape and result.rank != source.type.rank:
                    raise ValueError('dynamic tensor conversion must preserve source rank')
                value = cfg.emit('tensor-convert', result, (source,), TensorConvert(), span)
                cfg.owned_results.add(value.identity)
                if source.identity in cfg.owned_results and root_binding(arguments[0], cfg, env) is None:
                    source.type.release(source, cfg, span)
                return value
            capture = ArgumentValue(source, root_binding(arguments[0], cfg, env))
            arguments = (Expression('capture', arguments[0].token, (capture,)),)
        if len(terms) == 1 and len(arguments) > 1:
            result = replace(result, shape=(None,) * len(arguments))
        dimensions = result.allocation_dimensions(arguments, cfg, env, span)
        value = cfg.emit('allocate-tensor', result, dimensions, TensorOperation('allocate'), span)
        cfg.owned_results.add(value.identity)
        return value


@dataclass(frozen=True)
class TensorType:
    element: object
    shape: tuple = (None,)
    location: str = field(default='heap', compare=False)
    strided: bool = False
    family: str = 'tensor'
    binding_ownership: str = 'view'
    parameter_ownership: str = 'view'
    bits: int = 0
    signed: bool = False

    def __post_init__(self):
        scalar = self.element.mlir
        if not (scalar.startswith('i') and scalar[1:].isdigit() or scalar in ('f16', 'bf16', 'f32', 'f64')):
            raise ValueError('tensor elements require a scalar layout')
        parse_shape(repr(self.shape))
        import math
        known = math.prod(n for n in self.shape if n is not None)
        if known > ((1 << 63) - 1) // max(1, (self.element.bits + 7) // 8):
            raise ValueError('tensor static shape exceeds addressable storage')

    @property
    def rank(self):
        return len(self.shape)

    @property
    def count(self):
        import math
        return math.prod(self.shape) if None not in self.shape else None

    @property
    def aggregate(self):
        return False

    @property
    def name(self):
        suffix = '' if self.shape == (None,) and not self.strided else ',' + repr(self.shape).replace(' ', '')
        suffix += ',strided' if self.strided else ''
        return f'tensor[{self.element.name}{suffix}]'

    @property
    def mlir(self):
        layout = ', strided<[' + ','.join('?' for _ in self.shape) + '], offset: ?>' if self.strided else ''
        return 'memref<' + ''.join(('?' if n is None else str(n)) + 'x' for n in self.shape) + self.element.mlir + layout + '>'

    @property
    def generic_arguments(self):
        return self.element, ShapeArgument(self.shape)

    def require_context(self, cfg):
        pass

    def grammars(self):
        from py_compiler.syntax.complex.collection import CollectionCondition
        return (CollectionCondition(self),)

    def allocation_dimensions(self, arguments, cfg, env, span):
        from py_compiler.syntax.function.calls import literal_value
        if len(arguments) != self.shape.count(None):
            raise ValueError('tensor construction requires one extent for each dynamic dimension; static shape is already in the type')
        dynamic = iter(self.dimensions(arguments, cfg, env))
        return tuple(next(dynamic) if n is None else literal_value(n, cfg.context.type('index'), cfg, span) for n in self.shape)

    def accepts_type(self, source):
        return (isinstance(source, TensorType) and source.element == self.element and source.rank == self.rank
                and (self.strided or not source.strided)
                and all(expected is None or expected == actual for expected, actual in zip(self.shape, source.shape)))

    def from_value(self, value, cfg, span):
        if not self.accepts_type(value.type):
            raise ValueError(f'{value.type.name} does not satisfy {self.name}')
        result = cfg.emit('tensor-view', self, (value,), TensorCast(), span)
        cfg.value_sources[result.identity] = set(cfg.value_sources.get(value.identity, ()))
        cfg.value_dependencies[result.identity] = set(cfg.value_dependencies.get(value.identity, ()))
        if value.identity in cfg.stack_values:
            cfg.stack_values.add(result.identity)
        return result

    def release(self, value, cfg, span):
        if value.identity not in cfg.stack_values and self.location != 'stack':
            cfg.effect('deallocate', (value,), TensorOperation('release'), span)

    def dimensions(self, arguments, cfg, env):
        from py_compiler.syntax.generic.owned import select
        return tuple(select(cfg.context.type('index'), 'F.constructor', 'construct').expand(cfg, (a,), env) for a in arguments)

    def element_place(self, receiver, index, cfg, env, span):
        indices = index.operands if index.kind == 'template_arguments' else (index,)
        if len(indices) != self.rank:
            raise ValueError('tensor indexing requires one index per dimension')
        return TensorPlace(receiver, self.dimensions(indices, cfg, env), self.element)

    def length(self, value, cfg, span):
        return cfg.emit('tensor-length', cfg.context.type('int'), (value,), TensorOperation('length'), span)

    def copy_value(self, value, cfg, span):
        result = cfg.emit('memory-copy', replace(self, location='heap', strided=False), (value,), TensorOperation('copy'), span)
        cfg.owned_results.add(result.identity)
        return result

    def slice_value(self, receiver, bounds, cfg, env, span, owner=None, *, closed=(True, True)):
        if self.rank != 1:
            raise ValueError('slice notation requires rank-one storage; use slice(starts, stops, steps) for ranked tensors')
        from py_compiler.syntax.complex.collection import collection_namespace
        from py_compiler.syntax.grammar.expression import Expression
        from py_compiler.syntax.function.calls import literal_value, select, invoke
        from py_compiler.frontend.lexer.lexer import lex
        from py_compiler.frontend.source.source import SourceFile
        token = lex(SourceFile(cfg.source.path + ':slice', 'slice'), cfg.syntax)[0]
        integer = cfg.context.type('int')
        start, stop, step = bounds
        values = [receiver]
        for bound, default in ((start, 0), (stop, 0), (step, 1)):
            values.append(cfg.expr(bound, env, integer) if bound else literal_value(default, integer, cfg, span))
        values.extend(literal_value(bound is None, cfg.context.type('bool'), cfg, span) for bound in (start, stop))
        values.extend(literal_value(flag, cfg.context.type('bool'), cfg, span) for flag in closed)
        arguments = tuple(Expression('capture', token, (ArgumentValue(value, owner if index == 0 else None),))
                          for index, value in enumerate(values))
        entries = cfg.context.lookup_functions(collection_namespace(cfg.context) + '_slice_values', cfg.declaration_scope)
        entry, operands = select(entries, arguments, cfg, env, (self.element.name, self.name))
        return invoke(entry, operands, cfg, span)

    def call_method(self, receiver, method, arguments, cfg, env, span, owner=None):
        if method == 'slice' and len(arguments) == 3:
            if owner is None:
                raise ValueError('tensor slice requires a live source owner')
            terms = [cfg.expr(argument, env) for argument in arguments]
            if any(not isinstance(value.type, TensorType) or value.type.rank != 1 or
                   value.type.count != self.rank or value.type.element.family != 'integer' for value in terms):
                raise ValueError('slice requires fixed integer tuples for start, stop, and step, with one value per axis')
            target = TensorType(self.element, (None,) * self.rank, strided=True)
            result = cfg.emit('tensor-slice', target, (receiver, *terms), TensorSlice(), span)
            cfg.call_result_owners[result.identity] = owner
            cfg.call_result_modes[result.identity] = 'view'
            cfg.value_dependencies[result.identity] = {cfg.flow.resolve(owner.identity)}
            from py_compiler.syntax.generic.storage import root_binding
            for argument, value in zip(arguments, terms):
                if value.identity in cfg.owned_results and root_binding(argument, cfg, env) is None:
                    value.type.release(value, cfg, span)
            return result
        if method in ('flatten', 'materialize') and not arguments:
            target = TensorType(self.element, (self.count,) if self.count is not None else (None,)) if method == 'flatten' else replace(self, location='heap', strided=False)
            result = cfg.emit('tensor-copy', target, (receiver,), TensorConvert(flatten=method == 'flatten'), span)
            cfg.owned_results.add(result.identity)
            return result
        if method in ('shape', 'strides') and not arguments:
            return self.field_place(receiver, method, cfg, span).read(cfg, span)
        if method == 'len' and not arguments:
            return self.length(receiver, cfg, span)
        if method == 'copy' and not arguments:
            return self.copy_value(receiver, cfg, span)
        if method == 'dim' and len(arguments) == 1:
            return cfg.emit('tensor-dimension', cfg.context.type('int'), (receiver, *self.dimensions(arguments, cfg, env)), TensorOperation('dim'), span)
        if (method == 'resize' and len(arguments) == self.rank or
                method == 'append' and self.rank == 1 and len(arguments) == 1):
            from py_compiler.syntax.generic.storage import BindingPlace
            if self.strided:
                raise ValueError('a strided view cannot resize; materialize it first')
            if any(n is not None for n in self.shape):
                raise ValueError('a static shape cannot resize; use dynamic tensor dimensions')
            if owner is None or owner.ownership != 'own':
                raise ValueError('tensor resize requires its owning binding')
            if self.location == 'stack' or receiver.identity in cfg.stack_values:
                raise ValueError('stack tensor cannot resize; copy into heap storage first')
            dimensions = (cfg.expr(arguments[0], env, self.element),) if method == 'append' else self.dimensions(arguments, cfg, env)
            with BindingPlace(owner, rebind=False).capture(cfg, 'borrow'):
                result = cfg.emit('tensor-resize', self, (receiver, *dimensions), TensorOperation(method), span)
            cfg.values[owner.identity] = result
            return None
        raise ValueError('tensor supports dim(axis), resize(extents), rank-1 append(value), and indexed access')

    def field_place(self, receiver, field, cfg, span, write=False):
        if write or field not in ('shape', 'strides', 'values'):
            raise ValueError('tensor exposes read-only shape, strides, and flattened values')
        return TensorMetadata(receiver, field)

    def binary_operation(self, node, left, cfg, env):
        if getattr(cfg, 'placement', None) == 'gpu':
            from py_compiler.syntax.complex.gpu import elementwise
            return elementwise(node, left, cfg, env)
        if getattr(cfg, 'placement', None) == 'simd':
            from py_compiler.syntax.complex.simd import elementwise
            return elementwise(node, left, cfg, env)
        from py_compiler.syntax.grammar.expression import Expression
        from py_compiler.syntax.generic.storage import root_binding
        from py_compiler.syntax.function.calls import select, invoke
        method = {'+': 'add', '*': 'multiply'}.get(node.token.text)
        if method is None:
            raise ValueError('tensor operator requires an implemented tensor math function')
        aliases = [alias for alias in cfg.context.package_imports
                   if cfg.context.c_packages[alias]['package']['name'] == 'math.tensor']
        if not aliases:
            raise ValueError('tensor arithmetic requires an explicit import of math.tensor')
        frozen = Expression('capture', node.token, (ArgumentValue(left, root_binding(node.operands[0], cfg, env)),))
        entries = cfg.context.lookup_functions(aliases[0] + '.' + method, cfg.declaration_scope)
        entry, values = select(entries, (frozen, node.operands[1]), cfg, env)
        return invoke(entry, values, cfg, node.token.span)


@dataclass(frozen=True)
class TensorMetadata:
    receiver: object
    field: str

    def read(self, cfg, span):
        if self.field == 'values':
            target = TensorType(self.receiver.type.element)
            result = cfg.emit('tensor-flatten', target, (self.receiver,), TensorConvert(flatten=True), span)
        else:
            from py_compiler.syntax.complex.tuple import TupleValue
            target = TupleValue(cfg.context.type('int'), shape=(self.receiver.type.rank,))
            result = cfg.emit('tensor-metadata', target, (self.receiver,), self, span)
        cfg.owned_results.add(result.identity)
        return result

    def atom(self, operation):
        return Atom(self, (self.receiver.type,), operation.result.type, ('view',), ('read', 'allocation'), self)

    def render_operation(self, operation):
        result, source = operation.result, self.receiver
        p = name(result)
        lines = [f'{p} = memref.alloc() : {result.type.mlir}', f'{p}_one = arith.constant 1 : index']
        if self.field == 'strides' and source.type.strided:
            names = [p + '_base', p + '_offset', *(p + f'_dim{i}' for i in range(source.type.rank)),
                     *(p + f'_actual_stride{i}' for i in range(source.type.rank))]
            types = [f'memref<{source.type.element.mlir}>', *('index',) * (1 + 2 * source.type.rank)]
            lines.append(f'{", ".join(names)} = "memref.extract_strided_metadata"({name(source)}) : ({source.type.mlir}) -> ({", ".join(types)})')
        stride = p + '_one'
        for axis in reversed(range(source.type.rank)):
            index, size, scalar = p + f'_axis{axis}', p + f'_size{axis}', p + f'_scalar{axis}'
            if source.type.strided and self.field == 'strides':
                stride = p + f'_actual_stride{axis}'
            lines += [f'{index} = arith.constant {axis} : index', f'{size} = memref.dim {name(source)}, {index} : {source.type.mlir}',
                      f'{scalar} = arith.index_cast {size if self.field == "shape" else stride} : index to i64',
                      f'memref.store {scalar}, {p}[{index}] : {result.type.mlir}']
            next_stride = p + f'_stride{axis}'
            lines.append(f'{next_stride} = arith.muli {stride}, {size} : index')
            stride = next_stride
        return lines


@dataclass(frozen=True)
class TensorPlace:
    receiver: object
    indices: tuple
    type: object

    def read(self, cfg, span):
        return cfg.emit('memory-load', self.type, (self.receiver, *self.indices), TensorOperation('load'), span)

    def write(self, cfg, value, span):
        cfg.effect('memory-store', (self.receiver, *self.indices, value), TensorOperation('store'), span)


@dataclass(frozen=True)
class TensorOperation:
    kind: str
    alignment: int = 0

    def atom(self, operation):
        count = len(operation.operands)
        modes = ('move',) if self.kind == 'release' else ('copy',) * count if self.kind == 'allocate' else (('borrow' if self.kind in ('resize', 'append', 'store') else 'view'), *('copy',) * (count - 1))
        effects = ('destruction',) if self.kind == 'release' else ('allocation', 'mutation') if self.kind in ('allocate', 'resize', 'append', 'copy') else ('mutation',) if self.kind == 'store' else ('read',)
        return Atom(self, tuple(v.type for v in operation.operands), operation.result.type if operation.result else None, modes, effects, self)

    def render_operation(self, operation):
        values, result = operation.operands, operation.result
        target = result.type if self.kind == 'allocate' else values[0].type
        rank, mlir = target.rank, target.mlir
        prefix = '%tensor_' + temporary_suffix(operation)
        zero, one = prefix + '_zero', prefix + '_one'
        lines = [f'{zero} = arith.constant 0 : index', f'{one} = arith.constant 1 : index']
        old = name(values[0]) if self.kind != 'allocate' else None
        if self.kind == 'release':
            return [f'memref.dealloc {old} : {mlir}']
        extents = []
        if old:
            for i in range(rank):
                axis, size = prefix + f'_axis{i}', prefix + f'_old{i}'
                lines += [f'{axis} = arith.constant {i} : index', f'{size} = memref.dim {old}, {axis} : {mlir}']
                extents.append(size)
        if self.kind == 'length':
            total = one
            for i, size in enumerate(extents):
                next_ = prefix + f'_total{i}'
                lines.append(f'{next_} = arith.muli {total}, {size} : index')
                total = next_
            lines.append(f'{name(result)} = arith.index_cast {total} : index to {result.type.mlir}')
            return lines
        if self.kind == 'dim':
            axis = name(values[1]); bound = prefix + '_rank'; valid = prefix + '_valid'; size = prefix + '_size'
            lines += [f'{bound} = arith.constant {rank} : index', f'{valid} = arith.cmpi ult, {axis}, {bound} : index', f'cf.assert {valid}, "tensor dimension outside rank"', f'{size} = memref.dim {old}, {axis} : {mlir}', f'{name(result)} = arith.index_cast {size} : index to {result.type.mlir}']
            return lines
        if self.kind in ('load', 'store'):
            indices = [name(v) for v in values[1:rank + 1]]
            for i, (index, size) in enumerate(zip(indices, extents)):
                valid = prefix + f'_valid{i}'
                lines += [f'{valid} = arith.cmpi ult, {index}, {size} : index', f'cf.assert {valid}, "tensor index outside shape"']
            address = old + '[' + ', '.join(indices) + ']'
            lines.append(f'{name(result)} = memref.load {address} : {mlir}' if self.kind == 'load' else f'memref.store {name(values[-1])}, {address} : {mlir}')
            return lines
        sizes = extents if self.kind == 'copy' else [name(v) for v in (values if self.kind == 'allocate' else values[1:])]
        if self.kind == 'append':
            size = prefix + '_grown'
            lines.append(f'{size} = arith.addi {extents[0]}, {one} : index')
            sizes = [size]
        maximum = prefix + '_max'; total = one
        lines.append(f'{maximum} = arith.constant {((1 << 63) - 1) // max(1, (target.element.bits + 7) // 8)} : index')
        for i, size in enumerate(sizes):
            # Avoid division by zero and validate the complete byte product.
            valid, safe, limit, fits, next_ = (prefix + f'_{part}{i}' for part in ('nonnegative', 'safe', 'limit', 'fits', 'product'))
            lines += [f'{valid} = arith.cmpi sge, {size}, {zero} : index', f'cf.assert {valid}, "tensor extent must be nonnegative"', f'{safe} = arith.maxui {size}, {one} : index', f'{limit} = arith.divui {maximum}, {safe} : index', f'{fits} = arith.cmpi ule, {total}, {limit} : index', f'cf.assert {fits}, "tensor allocation byte size overflow"', f'{next_} = arith.muli {total}, {size} : index']
            total = next_
        destination = name(result)
        allocator = 'memref.alloca' if target.location == 'stack' else 'memref.alloc'
        alignment = f' {{alignment = {self.alignment} : i64}}' if self.alignment else ''
        dynamic_sizes = [size for size, extent in zip(sizes, result.type.shape) if extent is None]
        lines.append(f'{destination} = {allocator}({", ".join(dynamic_sizes)}){alignment} : {result.type.mlir}')
        if self.kind == 'copy':
            lines.append(f'memref.copy {old}, {destination} : {mlir} to {result.type.mlir}')
            return lines
        scalar_zero = '0.0' if target.element.mlir in ('f16', 'bf16', 'f32', 'f64') else '0'
        initial = prefix + '_initial'
        lines.append(f'{initial} = arith.constant {scalar_zero} : {target.element.mlir}')
        def loops(bounds, tag, body):
            indices = [prefix + f'_{tag}{i}' for i in range(rank)]
            for i, (index, size) in enumerate(zip(indices, bounds)):
                lines.append('  ' * i + f'scf.for {index} = {zero} to {size} step {one} {{')
            lines.extend('  ' * rank + line for line in body(', '.join(indices)))
            for i in reversed(range(rank)):
                lines.append('  ' * i + '}')
        loops(sizes, 'init', lambda indices: [f'memref.store {initial}, {destination}[{indices}] : {mlir}'])
        if self.kind in ('resize', 'append'):
            common = []
            for i, (previous, size) in enumerate(zip(extents, sizes)):
                bound = prefix + f'_common{i}'
                lines.append(f'{bound} = arith.minui {previous}, {size} : index')
                common.append(bound)
            loops(common, 'copy', lambda indices: [f'{prefix}_item = memref.load {old}[{indices}] : {mlir}', f'memref.store {prefix}_item, {destination}[{indices}] : {mlir}'])
            if self.kind == 'append':
                lines.append(f'memref.store {name(values[1])}, {destination}[{extents[0]}] : {mlir}')
            lines.append(f'memref.dealloc {old} : {mlir}')
        return lines


import unittest


class TensorTests(unittest.TestCase):
    def test_multidimensional_resize_uses_ranked_memrefs(self):
        from py_compiler.frontend.src.lib import compile_source
        from py_compiler.syntax.recognition import Syntax
        from py_compiler.mlir.src.lib import lower, render
        result = compile_source('tensor.sev', 'test "shape":\n    values = tensor[int](2, 3)\n    values[1, 2] = 7\n    values.resize(3, 4)\n    assert(values[1, 2] == 7)\n    assert(values[2, 3] == 0)\n    assert(len(values) == 12)\n    assert(values.dim(1) == 4)\n', Syntax(include_tests=True))
        self.assertFalse(result.diagnostics, str(result.diagnostics))
        output = render(lower(result.program))
        self.assertIn('memref<?x?xi64>', output)
        self.assertIn('arith.minui', output)
        self.assertIn('memref.dealloc', output)

    def test_owning_multidimensional_elements_require_relocation_support(self):
        from py_compiler.frontend.src.lib import compile_source
        from py_compiler.syntax.recognition import Syntax
        result = compile_source('tensor-owner.sev', 'def work():\n    values = tensor[string](2, 3)\n', Syntax())
        self.assertIn('tensor elements require a scalar layout', str(result.diagnostics))

    def test_resize_rejects_a_view(self):
        from py_compiler.frontend.src.lib import compile_source
        from py_compiler.syntax.recognition import Syntax
        result = compile_source('tensor-view.sev', 'def work(values: view tensor[int]):\n    values.resize(4)\n', Syntax())
        self.assertIn('tensor resize requires its owning binding', str(result.diagnostics))

    def test_append_uses_the_same_tensor_layout(self):
        from py_compiler.frontend.src.lib import compile_source
        from py_compiler.syntax.recognition import Syntax
        from py_compiler.mlir.src.lib import lower, render
        result = compile_source('append.sev', 'test "append":\n    values = tensor[int](0)\n    values.append(42)\n    assert(values[0] == 42 and len(values) == 1)\n', Syntax(include_tests=True))
        self.assertFalse(result.diagnostics, str(result.diagnostics))
        output = render(lower(result.program))
        self.assertIn('memref<?xi64>', output)
        self.assertNotIn('sev_vector_', output)


@dataclass(frozen=True)
class ShapeArgument:
    shape: tuple
    family: str = 'shape'

    @property
    def name(self):
        return repr(self.shape).replace(' ', '')


@dataclass(frozen=True)
class ArgumentValue:
    value: object
    owner: object = None

    def type(self, cfg, expected=None):
        return self.value.type

    def binding(self, cfg):
        return self.owner

    def lower(self, cfg, expected=None):
        if self.owner is not None:
            cfg.flow.read(self.owner)
        return self.value


@dataclass(frozen=True)
class TensorConvert:
    flatten: bool = False

    def atom(self, operation):
        return Atom(self, (operation.operands[0].type,), operation.result.type, ('view',), ('allocation', 'read'), self)

    def render_operation(self, operation):
        source, result = operation.operands[0], operation.result
        source_name, destination = name(source), name(result)
        p = '%reshape_' + str(result.identity)
        zero, one = p + '_zero', p + '_one'
        lines = [f'{zero} = arith.constant 0 : index', f'{one} = arith.constant 1 : index']
        source_sizes, sizes = [], []
        total = one
        for i in range(source.type.rank):
            axis, size, product = p + f'_axis{i}', p + f'_source{i}', p + f'_total{i}'
            lines += [f'{axis} = arith.constant {i} : index', f'{size} = memref.dim {source_name}, {axis} : {source.type.mlir}', f'{product} = arith.muli {total}, {size} : index']
            source_sizes.append(size)
            total = product
        target_total = one
        for i, extent in enumerate(result.type.shape):
            size = (total if self.flatten else source_sizes[i]) if extent is None else p + f'_target{i}'
            if extent is not None:
                lines.append(f'{size} = arith.constant {extent} : index')
            product = p + f'_target_total{i}'
            lines.append(f'{product} = arith.muli {target_total}, {size} : index')
            target_total = product
            sizes.append(size)
        lines += [f'{p}_same = arith.cmpi eq, {total}, {target_total} : index', f'cf.assert {p}_same, "tensor shape element count mismatch"']
        dynamic = [size for size, n in zip(sizes, result.type.shape) if n is None]
        lines.append(f'{destination} = memref.alloc({", ".join(dynamic)}) : {result.type.mlir}')
        lines.append(f'scf.for {p}_linear = {zero} to {total} step {one} {{')
        def coordinates(dimensions, label):
            remainder, indices = p + '_linear', [None] * len(dimensions)
            for i in reversed(range(len(dimensions))):
                index, next_ = p + f'_{label}{i}', p + f'_{label}_rest{i}'
                lines.extend([f'  {index} = arith.remui {remainder}, {dimensions[i]} : index', f'  {next_} = arith.divui {remainder}, {dimensions[i]} : index'])
                indices[i], remainder = index, next_
            return ', '.join(indices)
        before = coordinates(source_sizes, 'from')
        after = coordinates(sizes, 'to')
        lines += [f'  {p}_item = memref.load {source_name}[{before}] : {source.type.mlir}', f'  memref.store {p}_item, {destination}[{after}] : {result.type.mlir}', '}']
        return lines


@dataclass(frozen=True)
class TensorCast:
    def atom(self, operation):
        return Atom(self, (operation.operands[0].type,), operation.result.type, ('view',), (), self)

    def render_operation(self, operation):
        source, result = operation.operands[0], operation.result
        return [f'{name(result)} = memref.cast {name(source)} : {source.type.mlir} to {result.type.mlir}']


@dataclass(frozen=True)
class TensorSlice:
    def atom(self, operation):
        return Atom(self, tuple(v.type for v in operation.operands), operation.result.type,
                    ('view',) * 4, ('read', 'panic'), self)

    def render_operation(self, operation):
        source, starts, stops, steps = operation.operands
        result = operation.result
        p = name(result)
        lines = [f'{p}_zero = arith.constant 0 : index', f'{p}_one = arith.constant 1 : index']
        offsets, sizes, strides = [], [], []
        for axis in range(source.type.rank):
            at = p + f'_axis{axis}'
            lines.append(f'{at} = arith.constant {axis} : index')
            loaded = []
            for label, values in (('start', starts), ('stop', stops), ('step', steps)):
                value, index = p + f'_{label}_value{axis}', p + f'_{label}{axis}'
                lines += [f'{value} = memref.load {name(values)}[{at}] : {values.type.mlir}',
                          f'{index} = arith.index_cast {value} : {values.type.element.mlir} to index']
                loaded.append(index)
            start, stop, step = loaded
            bound, valid, ordered, positive, nonnegative = (p + f'_{label}{axis}' for label in ('bound','valid','ordered','positive','nonnegative'))
            lines += [f'{bound} = memref.dim {name(source)}, {at} : {source.type.mlir}',
                      f'{nonnegative} = arith.cmpi sge, {start}, {p}_zero : index',
                      f'cf.assert {nonnegative}, "tensor slice start must be nonnegative"',
                      f'{ordered} = arith.cmpi sge, {stop}, {start} : index',
                      f'cf.assert {ordered}, "tensor slice stop precedes start"',
                      f'{valid} = arith.cmpi ule, {stop}, {bound} : index',
                      f'cf.assert {valid}, "tensor slice exceeds source extent"',
                      f'{positive} = arith.cmpi sgt, {step}, {p}_zero : index',
                      f'cf.assert {positive}, "tensor slice step must be positive"',
                      f'{p}_distance{axis} = arith.subi {stop}, {start} : index',
                      f'{p}_size{axis} = arith.ceildivui {p}_distance{axis}, {step} : index']
            offsets.append(start); sizes.append(p + f'_size{axis}'); strides.append(step)
        lines.append(f'{p} = memref.subview {name(source)}[{", ".join(offsets)}][{", ".join(sizes)}][{", ".join(strides)}] : {source.type.mlir} to {result.type.mlir}')
        return lines
