from py_compiler.syntax.block.record import RecordProvider, TypeDeclaration, register_template, record_header
from py_compiler.syntax.function.contracts import signature
from py_compiler.syntax.complex.object import ObjectType


class Trait(RecordProvider):
    spelling = "trait"
    allow_empty_body = True

    def parse_header(self, items):
        header = tuple(items[1:-1] if items[-1].text == ':' else items[1:])
        record_header(header)
        return header

    def declare(self, node, context):
        if register_template(self, node, context):
            return
        from dataclasses import replace
        contracts = list(record_header(node.header)[2])
        children = []
        for child in node.children:
            if not child.provider and child.tokens and all(t.text not in (':', '=', ':=') for t in child.tokens):
                contracts.append(''.join(t.text for t in child.tokens))
            else:
                children.append(child)
        parents = []
        for contract in contracts:
            parent = context.type(contract)
            if parent.family != 'trait':
                raise ValueError('trait inheritance requires a trait contract')
            if parent.name not in [p.name for p in parents]:
                parents.append(context.by_name[parent.name])
        fields, defaults, members = self.fields(replace(node, children=children), context)
        grammars = tuple(member for member in members if getattr(member.provider, 'spelling', '') == 'grammar')
        members = tuple(member for member in members if member not in grammars)
        requirements = []
        for parent in parents:
            for requirement in parent.grammar_requirements:
                previous = next((item for item in requirements if grammar_shape(item[1]) == grammar_shape(requirement[1])), None)
                if previous is not None and previous[2] != requirement[2]:
                    raise ValueError('conflicting inherited trait grammar ' + requirement[0])
                if previous is None:
                    requirements.append(requirement)
        for grammar in grammars:
            if grammar.children:
                raise ValueError('trait grammar requirements must be bodyless')
            name, arrangement, result = grammar.header
            if not isinstance(arrangement, tuple):
                raise ValueError('trait grammar requires a typed sentence arrangement')
            canonical = lambda t: contract_annotation(t, context) if t != 'Self' else t
            arrangement = tuple(part if isinstance(part, str) else (part[0], canonical(part[1])) for part in arrangement)
            requirement = (name, arrangement, canonical(result) if result else None)
            previous = next((item for item in requirements if grammar_shape(item[1]) == grammar_shape(arrangement)), None)
            if previous is not None and previous[2] != requirement[2]:
                raise ValueError('conflicting inherited trait grammar ' + name)
            if previous is None:
                requirements.append(requirement)
        methods = tuple(signature(member, ".".join((*context.scope, node.header[0].text))) for member in members)
        for method in methods:
            method.scope = context.scope
            if not method.templates:
                method.parameters = tuple((name, contract_annotation(annotation, context) if annotation and annotation != 'Self' else annotation)
                                          for name, annotation in method.parameters)
                if method.result and method.result != 'Self':
                    method.result = contract_annotation(method.result, context)
            if any(annotation is None for _, annotation in method.parameters):
                raise ValueError("trait parameter contracts require type annotations")
        own_methods = methods
        combined_fields, combined_defaults, combined_methods = [], [], []
        for inherited_fields, inherited_defaults, inherited_methods in [
                *((p.fields, p.defaults, p.methods) for p in parents), (fields, defaults, methods)]:
            for field, default in zip(inherited_fields, inherited_defaults):
                previous = next((f for f in combined_fields if f[0] == field[0]), None)
                if previous is not None:
                    if previous != field:
                        raise ValueError('conflicting inherited trait field ' + field[0])
                else:
                    combined_fields.append(field); combined_defaults.append(default)
            for method in inherited_methods:
                leaf = method.name.rsplit('.', 1)[-1]
                previous = next((m for m in combined_methods if m.name.rsplit('.', 1)[-1] == leaf
                                 and tuple(t for _, t in m.parameters) == tuple(t for _, t in method.parameters)), None)
                if previous is not None:
                    if (previous.result != method.result or previous.result_ownership != method.result_ownership
                            or previous.receiver_ownership != method.receiver_ownership
                            or tuple(previous.parameter_ownership.get(n, 'view') for n, _ in previous.parameters)
                               != tuple(method.parameter_ownership.get(n, 'view') for n, _ in method.parameters)):
                        raise ValueError('conflicting inherited trait method ' + leaf)
                    if previous.node != method.node and previous.node.children and method.node.children:
                        raise ValueError('ambiguous inherited trait default ' + leaf)
                    if method.node.children and not previous.node.children:
                        combined_methods[combined_methods.index(previous)] = method
                else:
                    combined_methods.append(method)
        fields, defaults, methods = tuple(combined_fields), tuple(combined_defaults), tuple(combined_methods)
        context.register(TypeDeclaration(node.identity, ".".join((*context.scope, node.header[0].text)), self.spelling,
                                         fields, (), tuple(p.name for p in parents), node.span, context.source.path, defaults, methods,
                                         getattr(node, 'record_template', ''), getattr(node, 'record_arguments', ()),
                                         grammar_requirements=tuple(requirements)))
        name = ".".join((*context.scope, node.header[0].text))
        context.types[name] = ObjectType(name, "trait", "!llvm.struct<(!llvm.ptr, i32)>", declaration=context.by_name[name])
        for member in members:
            for annotation in member.decorators:
                if not hasattr(context, 'operator_traits'):
                    context.operator_traits={}
                context.operator_traits.setdefault(annotation.name,[]).append((name,member.header[0].text))
        if not getattr(context, 'importing_trait', False):
            from py_compiler.syntax.block.function import Function
            previous = context.scope
            context.scope = (*previous, node.header[0].text)
            try:
                for method in own_methods:
                    if method.node.children and method.receiver_ownership is None:
                        Function().register(method.node, context)
            finally:
                context.scope = previous


    def satisfy(self, target, implementation, context):
        def resolved(annotation):
            from py_compiler.syntax.complex.generic import substitute
            return contract_annotation(substitute(annotation, {'Self': context.types[implementation.name]}), context)
        for field in target.fields:
            if field not in implementation.fields:
                raise ValueError(f"{implementation.name} does not satisfy {target.name}.{field[0]}: {field[1]}")
        for method in target.methods:
            name = method.name.rsplit(".", 1)[-1]
            candidates = context.functions.get(implementation.name + "." + name, [])
            expected = tuple(resolved(t) for _, t in method.parameters)
            if not candidates:
                if method.node.children and method.receiver_ownership is None:
                    continue
                if method.node.children:
                    from py_compiler.syntax.block.function import Function
                    from importlib import import_module
                    previous = context.scope
                    context.scope = method.scope
                    try:
                        Function().register(method.node, context, import_module("py_compiler.syntax.block.class").Receiver(implementation))
                    finally:
                        context.scope = previous
                    candidates = context.functions[implementation.name + "." + name]
                    for entry in candidates:
                        entry.trait_fallback = True
                else:
                    raise ValueError(f"{implementation.name} is missing {target.name}.{name}")
            matches = []
            for entry in candidates:
                if entry.result_ownership != method.result_ownership:
                    continue
                actual = tuple(resolved(t) for _, t in entry.parameters)
                if actual == expected and (method.node.children and method.receiver_ownership is None or resolved(entry.result or "absent") == resolved(method.result or "absent")):
                    if any(entry.parameter_ownership.get(n, 'view') != method.parameter_ownership.get(m, 'view') for (n, _), (m, _) in zip(entry.parameters, method.parameters)):
                        continue
                    if method.receiver_ownership == 'view' and entry.receiver_ownership == 'borrow':
                        continue
                    matches.append(entry)
            if len(matches) != 1:
                raise ValueError(f"{implementation.name}.{name} requires exactly one implementation of the trait callable contract")
        actual = context.types[implementation.name]
        for requirement in target.grammar_requirements:
            matches = [grammar for grammar in context.sentence_grammars
                       if grammar_satisfies(requirement, grammar[1], grammar[3], actual, context)]
            if len(matches) != 1:
                raise ValueError(f'{implementation.name} requires exactly one implementation of {target.name} grammar {requirement[0]}')


def grammar_shape(arrangement):
    # Capture names are local bindings; their concrete types define the contract.
    return tuple(part if isinstance(part, str) else ('capture', part[1]) for part in arrangement)


def grammar_satisfies(requirement, arrangement, result, actual, context):
    from py_compiler.syntax.complex.generic import substitute
    if not isinstance(arrangement, tuple):
        return False
    resolve = lambda t: actual.name if t == 'Self' else contract_annotation(substitute(t, {'Self': actual}), context)
    expected = tuple(part if isinstance(part, str) else ('capture', resolve(part[1])) for part in requirement[1])
    found = tuple(part if isinstance(part, str) else ('capture', part[1].name) for part in arrangement)
    if found != expected:
        return False
    output = requirement[2]
    if output == 'iterator':
        # C permits both value iteration and mapping entry iteration, but every
        # implementation must provide a concrete iterator element contract.
        return result is not None and result.family == 'iterator' and hasattr(result, 'element')
    return (result.name if result is not None else 'absent') == resolve(output or 'absent')


import unittest


class GenericTraitTests(unittest.TestCase):
    def test_imported_default_keeps_its_library_scope_through_nested_storage(self):
        from py_compiler.frontend.src.lib import compile_source
        from py_compiler.syntax.recognition import Syntax
        source = ('class Result[T]:\n    value: T\n'
            'def wrap[T](value: T) -> Result[T]:\n    return Result[T](value)\n'
            'trait C[T]:\n    def get(self: view Self) -> T\n'
            '    def wrapped(self: view Self) -> Result[T]:\n        return wrap(self.get())\n'
            'class List[T]: C[T]\n    value: T\n'
            '    def get(self: view Self) -> T:\n        return value\n')
        producer = compile_source('shared.sev', source, Syntax())
        self.assertFalse(producer.diagnostics, str(producer.diagnostics))
        package = {'package': {'name': 'shared'}, 'templates': producer.templates,
                   'exports': [], 'types': [], 'objects': ['shared.o']}
        consumer = compile_source('consumer.sev', 'from shared import List\n'
            'class Stack[T]:\n    values: List[T]\n'
            'def work() -> int:\n    stack = Stack[int](List[int](42))\n'
            '    wrapped = stack.values.wrapped()\n    return wrapped.value\n',
            Syntax(), c_packages={'shared': package})
        self.assertFalse(consumer.diagnostics, str(consumer.diagnostics))

    def test_shared_syntax_keeps_parallel_trait_checks_in_their_module(self):
        from concurrent.futures import ThreadPoolExecutor
        from threading import Barrier
        from unittest.mock import patch
        from py_compiler.frontend.src.lib import compile_source
        from py_compiler.syntax.recognition import Syntax
        syntax = Syntax()
        barrier = Barrier(2)
        satisfy = Trait.satisfy

        def synchronized(provider, target, implementation, context):
            barrier.wait(timeout=5)
            return satisfy(provider, target, implementation, context)

        def compile_module(name):
            return compile_source(name + '.sev',
                'trait Readable:\n    def read(self: view Self) -> int\n'
                f'class {name}: Readable\n'
                '    def read(self: view Self) -> int:\n        return 1\n', syntax)

        with patch.object(Trait, 'satisfy', synchronized), ThreadPoolExecutor(max_workers=2) as pool:
            results = list(pool.map(compile_module, ('First', 'Second')))
        for result in results:
            self.assertFalse(result.diagnostics, str(result.diagnostics))

    source = ('trait Readable[T]:\n    def read(self: view Self) -> T\n'
              'class Cell[T = int]: Readable[T]\n    value: T\n'
              '    def read(self: view Self) -> T:\n        return self.value\n')

    def compile(self, source):
        from py_compiler.frontend.src.lib import compile_source
        from py_compiler.syntax.recognition import Syntax
        return compile_source('trait-contracts.sev', source, Syntax())

    def test_constraint_realizes_to_direct_method_call(self):
        result = self.compile(self.source + 'def read[T: Readable[int]](value: T) -> int:\n    return value.read()\ndef work() -> int:\n    value = Cell[int](3)\n    return read(value)\n')
        self.assertFalse(result.diagnostics, str(result.diagnostics))
        body = next(body for body in result.program.bodies if body.declaration == 'read')
        calls = [operation.payload.symbol for block in body.blocks for operation in block.operations if operation.kind == 'call']
        self.assertTrue(any('Cell[i64].read' in symbol for symbol in calls))
        self.assertEqual(len(body.blocks), 1)

    def test_wrong_trait_argument_is_rejected(self):
        result = self.compile(self.source + 'def read[T: Readable[i32]](value: T) -> int:\n    return 1\ndef work() -> int:\n    value = Cell[int](3)\n    return read(value)\n')
        self.assertTrue(result.diagnostics)
        self.assertIn('does not satisfy', str(result.diagnostics[0]))

    def test_generic_trait_default_method(self):
        source = ('trait Default[T]:\n    def value(self: view Self) -> T:\n        return T(1)\n'
                  'class Item: Default[int]\n    unused: int\n'
                  'def work() -> int:\n    item = Item(0)\n    return item.value()\n')
        result = self.compile(source)
        self.assertFalse(result.diagnostics, str(result.diagnostics))


class TraitInheritanceTests(GenericTraitTests):
    def test_diamond_retains_one_inherited_default(self):
        result = self.compile(
            'trait Root:\n    def value(self: view Self) -> int:\n        return 7\n'
            'trait Left: Root\n    left: int\ntrait Right: Root\n    right: int\n'
            'trait Leaf: Left + Right\n    tag: int\n'
            'class Item: Leaf\n    left: int\n    right: int\n    tag: int\n'
            'def use[T: Root](item: T) -> int:\n    return item.value()\n'
            'def work() -> int:\n    item = Item(1,2,3)\n    return use(item)\n')
        self.assertFalse(result.diagnostics, str(result.diagnostics))
        leaf = next(d for d in result.program.declarations if d.name == 'Leaf')
        self.assertEqual(len(leaf.methods), 1)
        self.assertEqual(len(leaf.fields), 3)

    def test_conflicting_contracts_are_rejected(self):
        for declarations, diagnostic in (
            ('trait A:\n    x: int\ntrait B: A\n    x: bool\n', 'conflicting inherited trait field'),
            ('trait A:\n    def x() -> int\ntrait B: A\n    def x() -> bool\n', 'conflicting inherited trait method'),
            ('class A:\n    x: int\ntrait B: A\n    y: int\n', 'requires a trait contract'),
            ('trait A:\n    def x() -> int:\n        return 1\ntrait B:\n    def x() -> int:\n        return 2\ntrait C: A + B\n    y: int\n', 'ambiguous inherited trait default'),
        ):
            with self.subTest(declarations=declarations):
                result = self.compile(declarations)
                self.assertTrue(result.diagnostics)
                self.assertIn(diagnostic, str(result.diagnostics[0]))


def contract_annotation(annotation, context):
    """Resolve names without instantiating recursive callable result types."""
    from py_compiler.syntax.complex.generic import split_arguments, substitute
    annotation = substitute(annotation, context.template_types)
    if '|' in annotation:
        members = split_arguments(annotation, '|')
        if len(members) > 1:
            return ' | '.join(contract_annotation(member, context) for member in members)
    if '[' in annotation and annotation.endswith(']'):
        base, arguments = annotation[:-1].split('[', 1)
        base = context.declaration_name(base, context.scope)
        if base in context.record_templates:
            base = context.record_templates[base].name
        return base + '[' + ','.join(contract_annotation(a,context) for a in split_arguments(arguments)) + ']'
    if annotation.isdigit() or annotation == 'Self':
        return annotation
    return context.type(annotation).name


class TraitGrammarTests(unittest.TestCase):
    def test_union_results_match_their_canonical_grammar_contract(self):
        result = self.compile('''trait Indexed[T]:
    grammar indexed[self, "[", key: int, "]"]() -> T
class Choice: Indexed[int | bool]
    value: int | bool
    grammar indexed[self, "[", index: int, "]"]() -> int | bool:
        return true
test:
    choice = Choice(true)
    assert(choice[0] == true)
''')
        self.assertFalse(result.diagnostics, str(result.diagnostics))

    def test_conflicting_parent_results_are_rejected_without_a_redeclaration(self):
        result = self.compile('''trait First:
    grammar indexed[self, "[", key: int, "]"]() -> int
trait Second:
    grammar indexed[self, "[", key: int, "]"]() -> bool
trait Both: First + Second
''')
        self.assertIn('conflicting inherited trait grammar', str(result.diagnostics))

    def compile(self, source):
        from py_compiler.frontend.src.lib import compile_source
        from py_compiler.syntax.recognition import Syntax
        return compile_source('trait-grammar.sev', source, Syntax())

    contract = '''trait Indexed[T]:
    grammar indexed[self, "[", key: int, "]"]() -> T
trait Traversable[T]: Indexed[T]
    grammar iterate["iterator", "(", self, ")"]() -> iterator
'''
    implementation = '''class Value: Traversable[int]
    value: int
    grammar indexed[self, "[", offset: int, "]"]() -> int:
        return self.value + offset
    grammar iterate["iterator", "(", self, ")"]() -> iterator[int]:
        yield self.value
'''

    def test_inherited_grammar_contract_checks_types_not_capture_names(self):
        result = self.compile(self.contract + self.implementation +
            'test:\n    item = Value(40)\n    assert(item[2] == 42)\n'
            '    for value in item:\n        assert(value == 40)\n')
        self.assertFalse(result.diagnostics, str(result.diagnostics))

    def test_missing_and_wrong_result_grammars_are_rejected(self):
        for implementation in (
                'class Value: Traversable[int]\n    value: int\n',
                self.implementation.replace('offset: int', 'offset: bool'),
                self.implementation.replace('() -> int:', '() -> bool:').replace('self.value + offset', 'true')):
            result = self.compile(self.contract + implementation)
            self.assertTrue(result.diagnostics)
            self.assertIn('grammar', str(result.diagnostics))

    def test_imported_generic_grammar_requirements_are_preserved(self):
        from py_compiler.frontend.src.lib import compile_source
        from py_compiler.syntax.recognition import Syntax
        producer = self.compile(self.contract)
        self.assertFalse(producer.diagnostics, str(producer.diagnostics))
        package = {'package': {'name': 'contracts'}, 'objects': ['contracts.o'],
                   'exports': [], 'types': [], 'templates': producer.templates}
        result = compile_source('consumer.sev', 'from contracts import Traversable\n' +
                                self.implementation, Syntax(), c_packages={'contracts': package})
        self.assertFalse(result.diagnostics, str(result.diagnostics))
        result = compile_source('consumer.sev', 'from contracts import Traversable\n'
            'class Missing: Traversable[int]\n    value: int\n', Syntax(), c_packages={'contracts': package})
        self.assertTrue(result.diagnostics)
        self.assertIn('grammar', str(result.diagnostics))
