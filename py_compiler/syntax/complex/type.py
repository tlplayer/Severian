"""The language type object wraps a resolved declaration, never a host Python type."""
from dataclasses import dataclass
from py_compiler.syntax.generic.definition import TypeDefinition


@dataclass(frozen=True)
class Type(TypeDefinition):
    name: str = 'type'
    declaration: object = None

    def specialize(self, declaration):
        return TypeValue(declaration)

    def construct_expression(self, argument, arguments, cfg, env, span, expected=None):
        if argument is not None or len(arguments) != 1:
            raise ValueError('type expects one value')
        value = cfg.expr(arguments[0], env)
        return type_value(value.type, cfg, span)

    @classmethod
    def of(cls, value):
        declaration = getattr(value, 'type', None)
        if declaration is None:
            raise ValueError('type(value) requires a resolved Severian value')
        return cls(declaration=declaration)

    def is_type(self, declaration):
        if self.declaration is None:
            raise ValueError('type identity requires a resolved declaration')
        return self.declaration == declaration


@dataclass(frozen=True)
class TypeValue:
    declaration: object
    family: str = 'type'
    mlir: str = 'i64'
    binding_ownership: str = 'copy'
    parameter_ownership: str = 'copy'

    @property
    def name(self):
        return 'type[' + self.declaration.name + ']'

    def grammars(self):
        return ()

    def render_constant(self, value, symbol):
        return [], [f'%value = arith.constant {value} : i64']

    def binary_operation(self, node, left, cfg, env):
        from py_compiler.syntax.function.calls import literal_value
        right = cfg.expr(node.operands[1], env)
        if right is None or right.type.family != 'type':
            raise ValueError('type operations require two type values')
        if node.token.text == '|':
            from py_compiler.syntax.block.union import UnionType
            union = UnionType((self.declaration, right.type.declaration)).with_layout(cfg.context)
            declaration = union.variants[0] if len(union.variants) == 1 else union
            return type_value(declaration, cfg, node.token.span)
        if node.token.text not in ('==', '!='):
            raise ValueError('type values support equality and union composition')
        same = same_type(self.declaration, right.type.declaration)
        return literal_value(same if node.token.text == '==' else not same,
                             cfg.context.type('bool'), cfg, node.token.span)


def same_type(left, right):
    left = left.variants if left.family == 'union' else (left,)
    right = right.variants if right.family == 'union' else (right,)
    return frozenset(left) == frozenset(right)


def type_value(declaration, cfg, span):
    from py_compiler.syntax.function.calls import literal_value
    import hashlib
    tag = int.from_bytes(hashlib.sha256(declaration.name.encode()).digest()[:8], 'little') & ((1 << 63) - 1)
    return literal_value(tag, TypeValue(declaration), cfg, span)


import unittest


class TypeValueTests(unittest.TestCase):
    def test_type_identity_and_union_composition_lower_without_runtime_payloads(self):
        from py_compiler.frontend.src.lib import compile_source
        from py_compiler.syntax.recognition import Syntax
        source = '''class Result:
    value: int
test:
    value = Result(42)
    assert(type(value) == Result)
    assert(type(1) == int)
    assert(type(1) != f64)
    number = int | f64
    reverse = f64 | int
    assert(number == reverse)
    assert(number is reverse)
    assert(number is union)
    assert((int | int) == int)
'''
        result = compile_source('type-values.sev', source, Syntax(include_tests=True))
        self.assertFalse(result.diagnostics, str(result.diagnostics))
        for body in result.program.bodies:
            body.verify()

    def test_type_composition_rejects_runtime_values(self):
        from py_compiler.frontend.src.lib import compile_source
        from py_compiler.syntax.recognition import Syntax
        result = compile_source('invalid-type-union.sev', 'def invalid():\n    value = int | 42\n', Syntax())
        self.assertIn('type operations require two type values', str(result.diagnostics))


class ZipTypes:
    def call(self, type_, arguments, cfg, env, span):
        from py_compiler.syntax.block.record import TypeDeclaration
        from py_compiler.syntax.complex.object import ObjectType
        from py_compiler.syntax.generic.storage import owns_storage
        from py_compiler.syntax.block.function import Function
        from dataclasses import replace
        from importlib import import_module
        values = [cfg.expr(a,env) for a in arguments]
        if type_ is not None or not values or any(v.type.family != 'type' or v.type.declaration.family != 'record' for v in values):
            raise ValueError('class composition requires class type arguments')
        members = tuple(v.type.declaration for v in values)
        name = 'zip[' + ','.join(t.name for t in members) + ']'
        context = cfg.context
        if name in context.types:
            return type_value(context.types[name],cfg,span)
        fields,defaults,methods = [],[],{}
        for member in members:
            for field,default in zip(member.declaration.fields,member.declaration.defaults):
                if any(f[0]==field[0] for f in fields):
                    raise ValueError('class composition has conflicting field ' + field[0])
                fields.append(field);defaults.append(default)
            for key,entries in list(context.functions.items()):
                if not key.startswith(member.name+'.'):
                    continue
                leaf = key[len(member.name)+1:]
                if leaf == member.name.rsplit('.',1)[-1]:
                    continue
                if leaf in methods:
                    raise ValueError('class composition has conflicting method ' + leaf)
                methods[leaf]=entries
        declaration=TypeDeclaration(name,name,'class',tuple(fields),(),(),span,defaults=tuple(defaults))
        context.by_name[name]=declaration
        provider=import_module('py_compiler.syntax.block.class').Class()
        context.provider_by_name[name]=provider
        context.types[name]=ObjectType(name,'record','!llvm.ptr',declaration=declaration,owns_fields=any(owns_storage(context.type(t)) for _,t in fields))
        receiver=import_module('py_compiler.syntax.block.class').Receiver(declaration)
        for entries in methods.values():
            for entry in entries:
                cloned=replace(entry.node,identity=entry.node.identity+':'+name)
                result=Function().register(cloned,context,receiver)
                result.type_arguments['Composition']=context.types[name]
        return type_value(context.types[name],cfg,span)
