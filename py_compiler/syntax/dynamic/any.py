"""Heterogeneous values reuse dynamic ownership and retain equality dispatch."""
from dataclasses import dataclass
from hashlib import sha256
from py_compiler.syntax.dynamic.dynamic import DynamicType, DynamicBox, DynamicOperation
from py_compiler.syntax.generic.atom import Atom


@dataclass(frozen=True)
class AnyType(DynamicType):
    name: str = 'any'
    mlir: str = '!llvm.struct<(ptr, ptr, ptr, i64, i64, i64, i64)>'

    def storage_layout(self, context):
        from py_compiler.syntax.function.layout import Layout
        word = context.type('usize').bits // 8
        return Layout((3 * word + 7) // 8 * 8 + 32, max(word, 8))

    def box_operation(self, concrete, symbol, helper, cfg, span):
        comparison = symbol.replace('_drop_', '_equal_')
        helpers = cfg.context.dynamic_helpers
        if comparison not in helpers:
            helpers[comparison] = equality_body(comparison, concrete, self, cfg, span)
        import json
        fingerprint = int(sha256(json.dumps(runtime_type_key(concrete, cfg.context), sort_keys=True).encode()).hexdigest(), 16)
        return AnyBox(concrete, self, symbol, helper, comparison, helpers[comparison], fingerprint)

    def binary_operation(self, node, left, cfg, env):
        if node.token.text not in ('==', '!='):
            raise ValueError('any requires a concrete type before arithmetic')
        from py_compiler.hir.thir.src.model import Edge, Terminator
        from py_compiler.syntax.function.calls import literal_value
        from py_compiler.syntax.generic.owned import select
        from py_compiler.syntax.generic.storage import root_binding
        right_term = node.operands[1]
        right = cfg.expr(right_term, env, self)
        boolean = cfg.context.type('bool')
        same = cfg.emit('any-type-equal', boolean, (left, right), AnyCompare('type'), node.token.span)
        equal, different, done = cfg.block(), cfg.block(), cfg.block((boolean,))
        cfg.current.terminator = Terminator('conditional', (Edge(equal.identity), Edge(different.identity)), same)
        cfg.current = equal
        result = cfg.emit('any-equal', boolean, (left, right), AnyCompare('value'), node.token.span)
        cfg.current.terminator = Terminator('jump', (Edge(done.identity, (result,)),))
        cfg.current = different
        result = literal_value(False, boolean, cfg, node.token.span)
        cfg.current.terminator = Terminator('jump', (Edge(done.identity, (result,)),))
        cfg.current = done
        for value, term in ((left, node.operands[0]), (right, right_term)):
            owner = root_binding(term, cfg, env)
            if value.identity in cfg.owned_results and (owner is None or owner.type != value.type):
                value.type.release(value, cfg, node.token.span)
        result = done.parameters[0]
        return select(boolean, 'F.unary', 'not').expand(cfg, result, node.token.span) if node.token.text == '!=' else result


def runtime_type_key(type_, context):
    """Nominal identities include the defining package, independent of import aliases."""
    from py_compiler.syntax.complex.generic import NumberArgument
    if isinstance(type_, NumberArgument):
        return ('number', type_.value)
    if type_.family == 'record' or getattr(type_, 'declared_name', ''):
        name = type_.name.split('[', 1)[0]
        namespace = context.interface or context.source.path
        for alias, package in context.c_packages.items():
            if name.startswith(alias + '.'):
                namespace, name = package['package']['name'], name[len(alias) + 1:]
                break
        return ('nominal', namespace, name, tuple(runtime_type_key(a, context) for a in getattr(type_, 'generic_arguments', ())))
    if type_.family == 'union':
        return ('union', tuple(runtime_type_key(member, context) for member in type_.variants))
    if hasattr(type_, 'element'):
        return (type_.family, runtime_type_key(type_.element, context), str(getattr(type_, 'shape', '')),
                getattr(type_, 'count', None), type_.mlir)
    return (type_.family, type_.name, type_.mlir)


def equality_body(symbol, concrete, owner, cfg, span):
    from py_compiler.hir.thir.src.checking import Checker
    from py_compiler.hir.thir.src.model import Terminator
    from py_compiler.syntax.complex.array import ArrayType, ElementPlace, RecordElementPlace
    from py_compiler.syntax.complex.shared import Shared, SharedOperation
    from py_compiler.syntax.function.calls import literal_value
    from py_compiler.syntax.complex.tensor import ArgumentValue
    from py_compiler.syntax.grammar.expression import Expression
    from py_compiler.frontend.lexer.lexer import lex
    from py_compiler.frontend.source.source import SourceFile
    from py_compiler.mir.ownership.flow import Binding
    boolean = cfg.context.type('bool')
    builder = Checker(cfg.source, cfg.syntax, symbol, symbol, boolean)
    builder.context = cfg.context
    left, right = builder.value(owner), builder.value(owner)
    builder.current.parameters = (left, right)
    backing = ArrayType(concrete)
    shared = Shared(backing, owner.bits)
    zero = literal_value(0, cfg.context.type('index'), builder, span)
    token = lex(SourceFile(symbol, '=='), cfg.syntax)[0]
    arguments = []
    for index, value in enumerate((left, right)):
        handle = builder.emit('dynamic-handle', shared, (value,), DynamicOperation('handle'), span)
        storage = builder.emit('share-data', backing, (handle,), SharedOperation('data', shared), span)
        place = RecordElementPlace(storage, zero, concrete) if backing.aggregate else ElementPlace(storage, zero, concrete)
        item = place.read(builder, span)
        binding = Binding(symbol + str(index), ':any-item', concrete, True, 'view', span)
        builder.values[binding.identity] = item
        arguments.append(Expression('capture', token, (ArgumentValue(item, binding),)))
    supported = (concrete.family in ('integer', 'float', 'bool', 'char', 'string', 'unit', 'byte', 'union')
                 or isinstance(concrete, AnyType)
                 or concrete.name + '.__operator_61_61' in cfg.context.functions)
    if supported:
        result = builder.expr(Expression('binary', token, tuple(arguments)), {})
    else:
        from py_compiler.syntax.complex.object import Assert
        result = literal_value(False, boolean, builder, span)
        builder.effect('assert', (result,), Assert('boxed type does not supply equality'), span)
    builder.current.terminator = Terminator('finish', value=result)
    body = builder.finish()
    body.source_snapshot = None
    return body.freeze()


@dataclass(frozen=True)
class AnyBox(DynamicBox):
    comparison: str
    comparison_helper: object
    fingerprint: int

    def generated_bodies(self):
        return (*super().generated_bodies(), self.comparison_helper)

    def render_operation(self, operation):
        from py_compiler.mlir.src.cfg import name, temporary_suffix
        lines = super().render_operation(operation)
        p = '%any_' + temporary_suffix(operation)
        output = name(operation.result)
        lines[-1] = lines[-1].replace(output + ' =', p + '_base =', 1)
        type_ = self.owner.mlir
        # A complete type fingerprint travels with the value across modules;
        # function addresses may differ for separately compiled specializations.
        identity = self.fingerprint
        lines += [f'{p}_function = func.constant @{self.comparison} : ({type_}, {type_}) -> i1',
                        f'{p}_pointer = builtin.unrealized_conversion_cast {p}_function : ({type_}, {type_}) -> i1 to !llvm.ptr',
                        f'{p}_compare = llvm.insertvalue {p}_pointer, {p}_base[2] : {type_}']
        previous = p + '_compare'
        for index in range(4):
            tag = (identity >> (64 * index)) & ((1 << 64) - 1)
            result = output if index == 3 else p + '_tagged' + str(index)
            lines += [f'{p}_tag{index} = arith.constant {tag} : i64',
                      f'{result} = llvm.insertvalue {p}_tag{index}, {previous}[{index + 3}] : {type_}']
            previous = result
        return lines


@dataclass(frozen=True)
class AnyCompare:
    kind: str

    def atom(self, operation):
        return Atom(self, tuple(v.type for v in operation.operands), operation.result.type,
                    ('view', 'view'), ('read',), self)

    def render_operation(self, operation):
        from py_compiler.mlir.src.cfg import name, temporary_suffix
        left, right = operation.operands
        p = '%any_equal_' + temporary_suffix(operation)
        if self.kind == 'type':
            lines = []
            previous = None
            for index in range(4):
                result = name(operation.result) if index == 3 else p + '_both' + str(index)
                lines += [f'{p}_left{index} = llvm.extractvalue {name(left)}[{index + 3}] : {left.type.mlir}',
                          f'{p}_right{index} = llvm.extractvalue {name(right)}[{index + 3}] : {right.type.mlir}',
                          f'{p}_same{index} = arith.cmpi eq, {p}_left{index}, {p}_right{index} : i64']
                if previous is not None:
                    lines.append(f'{result} = arith.andi {previous}, {p}_same{index} : i1')
                previous = result if index else p + '_same0'
            return lines
        return [f'{p}_function = llvm.extractvalue {name(left)}[2] : {left.type.mlir}',
                f'{name(operation.result)} = llvm.call {p}_function({name(left)}, {name(right)}) : !llvm.ptr, ({left.type.mlir}, {right.type.mlir}) -> i1']


import unittest


class AnyTests(unittest.TestCase):
    def test_c_package_types_do_not_alias_but_primitives_do(self):
        import json
        import tempfile
        from pathlib import Path
        from py_compiler.packaging.testing import test_package
        with tempfile.TemporaryDirectory(prefix='sev-any-packages-') as directory:
            root = Path(directory)
            for name in ('left', 'right', 'consumer'):
                package = root / name
                (package / 'src').mkdir(parents=True)
                manifest = {'package': {'name': name, 'version': '0.1.0'}}
                source = 'class Value:\n    value: int\ndef box() -> any:\n    return Value(1)\ndef number() -> any:\n    return 42\n'
                if name == 'consumer':
                    manifest['dependencies'] = {alias: {'path': '../' + alias} for alias in ('left', 'right')}
                    source = 'import left\nimport right\ntest "nominal packages remain distinct":\n    assert(left.box() != right.box())\n    assert(left.number() == right.number())\n'
                (package / 'src/lib.sev').write_text(source)
                (package / 'package.json').write_text(json.dumps(manifest))
            result = test_package(root / 'consumer')
            self.assertTrue(result['success'], result)

    source = '''def _equal(left: string, right: string) -> bool:
    if len(left) != len(right):
        return false
    index = 0
    while index < len(left):
        if left[index] != right[index]:
            return false
        index += 1
    return true

test "heterogeneous storage retains types and copied owners":
    values = array[any,3]()
    values[0] = 42
    values[1] = true
    values[2] = 2.5
    saved = copy values
    values[0] = false
    drop(values)
    assert(saved[0] == 42)
    assert(saved[0] != true)
    assert(saved[1] == true)
    assert(saved[2] == 2.5)
    left: any = 42
    right: any = 42
    assert(left == right)
    assert(left != saved[1])

test "boxed string comparison shares ordinary callable declarations":
    text: any = "retained"
    saved = copy text
    drop(text)
    assert(saved == "retained")
    assert(saved != "different")
    slots = array[any,1]()
    slots[0] = saved
    drop(saved)
    assert(slots[0] == "retained")
'''

    def test_c_boxed_equality_and_array_layout(self):
        import json
        import tempfile
        from pathlib import Path
        from py_compiler.packaging.testing import test_package
        with tempfile.TemporaryDirectory(prefix='sev-any-') as directory:
            root = Path(directory)
            (root / 'src').mkdir()
            (root / 'src/lib.sev').write_text(self.source)
            (root / 'package.json').write_text(json.dumps({'package': {'name': 'any-values', 'version': '0.1.0'}}))
            result = test_package(root)
            self.assertTrue(result['success'], result)
