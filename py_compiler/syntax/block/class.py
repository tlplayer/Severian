from py_compiler.syntax.block.record import RecordProvider, TypeDeclaration, record_header, register_template
from py_compiler.syntax.complex.object import ObjectType


class Class(RecordProvider):
    spelling = "class"
    scope_kind = 'self'

    def bindings(self, cfg, env):
        return cfg.namespaces.get('self', {})

    def parse_header(self, items):
        header = tuple(items[1:-1] if items[-1].text == ":" else items[1:])
        if not header or header[0].kind != "IDENTIFIER":
            raise ValueError("class requires a name")
        record_header(header)
        return header

    def declare(self, node, context):
        if register_template(self, node, context):
            return
        fields, defaults, members = self.fields(node, context)
        _, _, contracts = record_header(node.header)
        traits = tuple(context.type(name).name if '[' in name else context.declaration_name(name, context.scope) for name in contracts)
        declaration = TypeDeclaration(node.identity, ".".join((*context.scope, node.header[0].text)), self.spelling,
                                      fields, (), traits, node.span, context.source.path, defaults,
                                      template_name=getattr(node, 'record_template', ''), arguments=getattr(node, 'record_arguments', ()))
        from dataclasses import replace
        from py_compiler.syntax.function.allocate import PROPERTIES
        defaults = {}
        for child in node.children:
            if not child.provider and child.tokens and child.tokens[0].text.startswith('_memory_'):
                key = child.tokens[0].text[len('_memory_'):]
                if key not in PROPERTIES or len(child.tokens) < 3 or child.tokens[1].text != '=' or key in defaults:
                    raise ValueError('invalid or duplicate class memory default')
                defaults[key] = ''.join(t.text for t in child.tokens[2:])
        declaration = replace(declaration, memory_defaults=tuple(defaults.items()))
        context.register(declaration)
        context.types[declaration.name] = ObjectType(declaration.name, "record", "!llvm.ptr", declaration=declaration,
            owns_fields=any(callable(getattr(context.type(t), "release", None)) for _, t in fields))
        context.tags[declaration.name] = len(context.tags) + 1
        for trait_name in traits:
            context.require(trait_name, lambda provider, target: provider.satisfy(target, declaration))
        for member in members:
            member.provider.declare_member(member, context, Receiver(declaration))

    def layout(self, declaration, context):
        types = [context.type(name) for _, name in declaration.fields]
        return "!llvm.struct<(" + ", ".join(self.storage_type(t, context) for t in types) + ")>"

    def storage_type(self, type_, context):
        if type_.family == 'record':
            return context.provider_by_name[type_.name].layout(type_.declaration, context)
        if type_.mlir.startswith('memref') and type_.family in ('array', 'pointer'):
            bits = context.type('usize').bits
            return f'!llvm.struct<(!llvm.ptr, !llvm.ptr, i{bits}, !llvm.array<1 x i{bits}>, !llvm.array<1 x i{bits}>)>'
        if not type_.mlir or type_.mlir.startswith('memref') or type_.mlir == 'index':
            raise ValueError('field type requires an aggregate storage/lifetime contract')
        return type_.mlir

    def field_place(self, value, field, cfg, span, write=False):
        from py_compiler.syntax.generic.storage import FieldPlace
        declaration = cfg.context.by_name[value.type.name]
        if field.startswith("__") or (write and field.startswith("_")):
            raise ValueError(f"field {field!r} is not externally {'writable' if write else 'readable'}")
        for index, (name, type_name) in enumerate(declaration.fields):
            if name == field:
                type_ = cfg.context.type(type_name)
                return FieldPlace(value, index, self.layout(declaration, cfg.context), type_, self.storage_type(type_, cfg.context))
        raise ValueError(f"{declaration.name} has no field {field!r}")

    def instantiate(self, declaration, arguments, cfg, env, span, allocation_plan=None):
        from py_compiler.syntax.complex.object import Allocate
        from py_compiler.syntax.generic.storage import FieldPlace
        from py_compiler.syntax.function.calls import invoke, literal_value, select
        if allocation_plan is None:
            from py_compiler.syntax.function.allocate import resolve_plan
            allocation_plan = resolve_plan(cfg.context.type(declaration.name), {}, cfg.context)
        if allocation_plan.location != 'stack':
            raise ValueError('class heap allocation requires an owning aggregate lowering contract')
        layout = self.layout(declaration, cfg.context)
        value = cfg.emit("allocate", cfg.context.type(declaration.name), (), Allocate(layout, allocation_plan), span)
        cfg.stack_values.add(value.identity)
        constructor_name = (declaration.template_name or declaration.name.split('[', 1)[0]).rsplit('.', 1)[-1]
        constructors = cfg.context.functions.get(declaration.name + "." + constructor_name, [])
        if constructors:
            entry, operands = select(constructors, arguments, cfg, env)
            invoke(entry, operands, cfg, span, value)
            return value
        if len(arguments) > len(declaration.fields):
            raise ValueError("too many constructor arguments")
        for index, ((_, type_name), default) in enumerate(zip(declaration.fields, declaration.defaults)):
            type_ = cfg.context.type(type_name)
            if index < len(arguments):
                place = FieldPlace(value, index, layout, type_, self.storage_type(type_, cfg.context))
                if callable(getattr(type_, "release", None)):
                    store_field(place, arguments[index], cfg, env, span, initialized=False)
                    continue
                field = cfg.expr(arguments[index], env, type_)
            elif default is not None:
                field = literal_value(default, type_, cfg, span)
            else:
                raise ValueError(f"constructor requires field {declaration.fields[index][0]!r}")
            FieldPlace(value, index, layout, type_, self.storage_type(type_, cfg.context)).write(cfg, field, span)
        return value

    def copy_value(self, value, cfg, span):
        from py_compiler.syntax.complex.object import Allocate
        from py_compiler.syntax.generic.storage import FieldPlace
        declaration = cfg.context.by_name[value.type.name]
        layout = self.layout(declaration, cfg.context)
        result = cfg.emit("allocate", value.type, (), Allocate(layout), span)
        cfg.stack_values.add(result.identity)
        for index, (_, name) in enumerate(declaration.fields):
            type_ = cfg.context.type(name)
            storage = self.storage_type(type_, cfg.context)
            field = FieldPlace(value, index, layout, type_, storage).read(cfg, span)
            if callable(getattr(type_, "release", None)):
                field = type_.copy_value(field, cfg, span)
            FieldPlace(result, index, layout, type_, storage).write(cfg, field, span)
        return result


    def release(self, value, cfg, span):
        from py_compiler.syntax.generic.storage import FieldPlace
        declaration = value.type.declaration
        layout = self.layout(declaration, cfg.context)
        for index in reversed(range(len(declaration.fields))):
            type_ = cfg.context.type(declaration.fields[index][1])
            release = getattr(type_, 'release', None)
            if release:
                field = FieldPlace(value, index, layout, type_, self.storage_type(type_, cfg.context)).read(cfg, span)
                # An inline record lives on the stack; its owned buffer does not.
                cfg.stack_values.discard(field.identity)
                release(field, cfg, span)


def store_field(place, expression, cfg, env, span, owner=None, initialized=True, value=None):
    from py_compiler.syntax.generic.storage import root_binding
    from py_compiler.syntax.grammar.expression import OWNERSHIP
    mode = expression.token.text if expression.kind == 'unary' and expression.token.text in OWNERSHIP else None
    term = expression.operands[0] if mode else expression
    source = root_binding(term, cfg, env)
    if value is None:
        value = cfg.expr(term, env, place.type)
    if owner:
        from py_compiler.syntax.generic.storage import BindingPlace
        with BindingPlace(owner, rebind=False).capture(cfg, 'borrow'):
            pass
    if callable(getattr(place.type, 'release', None)):
        if mode in ('copy', 'mirror'):
            value = value.type.copy_value(value, cfg, span)
        else:
            if mode in ('view', 'borrow') or (source and source.ownership in ('view', 'borrow')):
                raise ValueError('owned field requires ownership; use copy for a view or borrow')
            if term.kind == 'member' or (source and source.identity in cfg.storage):
                raise ValueError('moving an owned field requires replacing it; use copy')
            if value.type.family != 'record' and value.identity in cfg.stack_values:
                raise ValueError('owned field cannot retain stack storage')
            if value.identity in cfg.call_result_owners or cfg.value_sources.get(value.identity):
                raise ValueError('owned field cannot retain a borrowed result; use copy')
            if source:
                if owner and cfg.flow.resolve(source.identity) == cfg.flow.resolve(owner.identity):
                    raise ValueError('owned field cannot consume its containing object')
                cfg.flow.move(source)
        if initialized:
            previous = place.read(cfg, span)
            cfg.stack_values.discard(previous.identity)
            place.type.release(previous, cfg, span)
    place.write(cfg, value, span)


class Receiver:
    """Class syntax owns receiver layout and field access, not callable lowering."""
    def __init__(self, declaration):
        self.declaration = declaration
        self.name = declaration.name

    def configure(self, builder, node, parameters):
        from py_compiler.mir.ownership.flow import Binding
        from py_compiler.syntax.generic.storage import FieldPlace
        syntax = builder.syntax
        receiver_type = builder.context.types[self.declaration.name]
        self_value = builder.value(receiver_type)
        parameters.append(self_value)
        builder.self_binding = Binding(node.identity + ':self', 'self', receiver_type, True, getattr(builder, 'receiver_mode', 'borrow'), node.span)
        builder.body.bindings.append(builder.self_binding)
        builder.values[builder.self_binding.identity] = self_value
        builder.value_sources[self_value.identity] = {-1}
        fields = {}
        field_types = [builder.context.type(type_name) for _, type_name in self.declaration.fields]
        provider = builder.context.provider_by_name[self.declaration.name]
        record_type = provider.layout(self.declaration, builder.context)
        for index, ((field_name, _), type_) in enumerate(zip(self.declaration.fields, field_types)):
            mode = getattr(builder, 'receiver_mode', 'borrow')
            field = Binding(node.identity + ":self." + field_name, field_name, type_, False, mode, node.span)
            fields[field_name] = field
            builder.storage[field.identity] = FieldPlace(self_value, index, record_type, type_, provider.storage_type(type_, builder.context))
        builder.namespaces["self"] = fields
        builder.fallback_scopes.insert(0, fields)


import unittest


class ClassTests(unittest.TestCase):
    def compile(self, source):
        from py_compiler.frontend.src.lib import compile_source
        from py_compiler.syntax.recognition import Syntax
        return compile_source('class-contracts.sev', source, Syntax())

    def valid(self, source):
        result = self.compile(source)
        self.assertFalse(result.diagnostics, str(result.diagnostics))
        return result.program

    def test_generic_class_defaults_and_method_realizations(self):
        program = self.valid('class Cell[T = int]:\n    value: T\n    def read(self: view Self) -> T:\n        return self.value\ndef work() -> int:\n    a = Cell(3)\n    b = Cell[i32](4)\n    b.read()\n    return a.read()\n')
        self.assertEqual({d.name for d in program.declarations}, {'Cell[i64]', 'Cell[i32]'})
        methods = [body for body in program.bodies if body.declaration.endswith('.read')]
        self.assertEqual(len({body.name for body in methods}), 2)
        self.assertEqual({body.result_type.name for body in methods}, {'i32', 'i64'})

    def test_generic_constructor_and_explicit_receiver(self):
        self.valid('class Cell[T = int]:\n    value: T\n    def Cell(self: borrow Self, value: T):\n        self.value = value\n    def set(self: borrow Self, value: T):\n        self.value = value\ndef work() -> int:\n    a = Cell[int](2)\n    a.set(3)\n    return a.value\n')

    def test_generic_constructor_with_a_locally_declared_element(self):
        program = self.valid('''class Buffer[T]:
    items: array[T]
    def Buffer(self: borrow Self):
        unsafe:
            self.items = allocate[array[T]](0)
def work():
    class Entry:
        key: int
    values = Buffer[Entry]()
''')
        constructors = [body for body in program.bodies if body.declaration.endswith('.Buffer')]
        self.assertEqual(len(constructors), 1)
        self.assertIn('.Entry]', constructors[0].declaration)

    def test_unused_generic_member_does_not_require_element_ordering(self):
        source = '''class Cell[T]:
    value: T
    def less(self: view Self, other: T) -> bool:
        return self.value < other
class Point:
    x: int
def work():
    point = Point(1)
    cell = Cell[Point](point)
'''
        self.valid(source)
        result = self.compile(source + '    cell.less(point)\n')
        self.assertTrue(result.diagnostics)
        self.assertIn("F.operator grammar for '<'", str(result.diagnostics[0]))

    def test_nested_records_have_inline_layout_and_independent_copy(self):
        from py_compiler.mlir.src.lib import lower, render
        program = self.valid('class Point:\n    x: int\nclass Pair[T = Point]:\n    left: T\n    right: T\ndef work() -> int:\n    p = Point(1)\n    pair = Pair(p,p)\n    p.x = 2\n    return pair.left.x\n')
        text = render(lower(program))
        self.assertIn('!llvm.struct<(!llvm.struct<(i64)>, !llvm.struct<(i64)>)>', text)
        self.assertIn('llvm.load', text)
        self.assertIn('llvm.store', text)

    def test_view_receiver_cannot_mutate_fields(self):
        result = self.compile('class Counter:\n    value: int\n    def bad(self: view Self):\n        self.value = 1\n')
        self.assertTrue(result.diagnostics)

    def test_nested_record_view_preserves_address_and_owner(self):
        source = 'class Point:\n    x: int\nclass Outer:\n    point: Point\ndef work() -> int:\n    p = Point(1)\n    outer = Outer(p)\n    inner = outer.point\n    return inner.x\n'
        self.valid(source)
        result = self.compile(source.replace('    return inner.x', '    drop outer\n    return inner.x'))
        self.assertTrue(result.diagnostics)

    def test_nested_mutation_through_view_is_rejected(self):
        result = self.compile('class Point:\n    x: int\nclass Outer:\n    point: Point\ndef bad(outer: Outer):\n    outer.point.x = 2\n')
        self.assertTrue(result.diagnostics)

    def test_receiver_view_of_local_record_cannot_escape(self):
        result = self.compile('class Item:\n    value: int\n    def identity(self: view Self) -> Self:\n        return self\ndef bad() -> Item:\n    value = Item(1)\n    return value.identity()\n')
        self.assertTrue(result.diagnostics)
        self.assertIn('class by value requires ownership', str(result.diagnostics[0]))

    def test_infer_function_argument_from_class_specialization(self):
        self.valid('class Cell[T]:\n    value: T\ndef read[T](cell: Cell[T]) -> T:\n    return cell.value\ndef work() -> int:\n    value = Cell[int](2)\n    return read(value)\n')

    def test_generic_class_infers_concrete_scalar_constructor(self):
        program = self.valid('class Cell[T]:\n    value: T\ndef work() -> i32:\n    a = Cell(i32(1))\n    b = Cell(i64(2))\n    return a.value\n')
        self.assertEqual({d.name for d in program.declarations}, {'Cell[i32]', 'Cell[i64]'})

    def test_allocate_uses_class_defaults_without_adding_fields(self):
        self.valid('''class Item:
    _memory_location = memory.stack
    _memory_alignment = 16
    value: i32 = 0
def work() -> i32:
    unsafe:
        item = allocate[Item]()
        return item.value
''')
