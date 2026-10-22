"""C package imports consume checked interface contracts, never source bodies."""
from dataclasses import dataclass
from types import SimpleNamespace
from py_compiler.syntax.generic.block import DeclarationProvider
from py_compiler.syntax.function.contracts import Callable
from py_compiler.syntax.generic.atom import ExternalSymbol


@dataclass
class ImportedCallable(Callable):
    c_symbol: object = None

    @property
    def symbol(self):
        return self.c_symbol.symbol


@dataclass(frozen=True)
class CSymbol(ExternalSymbol):
    result_layout: str = ''

    def render_operation(self, operation):
        if self.result_layout:
            from py_compiler.syntax.complex.object import Invoke
            return Invoke(self.symbol, operation.result.type, result_layout=self.result_layout).render_operation(operation)
        return super().render_operation(operation)

    def __post_init__(self):
        if self.abi != 'severian' or not self.symbol or '\x00' in self.symbol:
            raise ValueError('invalid c Severian symbol')
        if not self.dependencies or any(not p.endswith('.o') for p in self.dependencies):
            raise ValueError('c import requires object dependencies')


class Import(DeclarationProvider):
    spelling = 'import'

    def has_body(self, node):
        return False

    def parse_header(self, items):
        if any(token.text == 'from' for token in items[1:]):
            position = next(i for i, token in enumerate(items) if token.text == 'from')
            if position != len(items) - 2:
                raise ValueError('expected import name[, name] from package_alias')
            return FromImport().parse_header((items[position], items[-1], items[0], *items[1:position]))
        if len(items) != 2 or items[1].kind != 'IDENTIFIER':
            raise ValueError('expected import package_alias')
        return tuple(items[1:])

    def declare(self, node, context):
        if len(node.header) > 1:
            return FromImport().declare(node, context)
        alias = node.header[0].text
        if context.scope:
            raise ValueError('package imports belong at module scope')
        package = context.c_packages.get(alias)
        if package is None:
            raise ValueError(f'package {alias!r} is not a resolved manifest dependency')
        if alias in context.package_imports:
            if alias in getattr(context, 'template_dependency_imports', set()):
                return
            raise ValueError(f'duplicate package import {alias!r}')
        context.package_imports.add(alias)
        # A template must retain the library imports used by its body.
        # Dependency interfaces come from the checked manifest closure.
        from dataclasses import replace
        required = {name for template in package.get('templates', ()) for name in template.get('imports', ())}
        # Exported ordinary signatures can also mention dependency-owned
        # types, even when this package has no generic templates of its own.
        required.update(package.get('template-dependencies', {}))
        for dependency in sorted(required):
            contract = package.get('template-dependencies', {}).get(dependency)
            if contract is None:
                contract = context.c_packages.get(dependency)
            if contract is None:
                raise ValueError('template dependency is unavailable: ' + dependency)
            existing = context.c_packages.get(dependency)
            if existing is not None and existing['package'] != contract['package']:
                raise ValueError('template dependency alias conflicts with consumer: ' + dependency)
            context.c_packages[dependency] = contract
            if dependency not in context.package_imports:
                self.declare(replace(node, header=(node.header[0].with_text(dependency),)), context)
                if not hasattr(context, 'template_dependency_imports'):
                    context.template_dependency_imports = set()
                context.template_dependency_imports.add(dependency)
        from py_compiler.frontend.source.source import SourceFile
        from py_compiler.frontend.lexer.lexer import lex
        from py_compiler.frontend.parser.blocks import parse_blocks
        from py_compiler.syntax.block.record import register_template
        type_contracts = {t['name']: t for t in package.get('types', ())}
        template_names = {item['name'] for item in package.get('templates', ())}
        def qualify(name, excluded=()):
            import re
            return re.sub(r'(?<![\w.])([A-Za-z_]\w*)', lambda match:
                alias + '.' + match[1] if match[1] not in excluded and
                (match[1] in type_contracts or match[1] in template_names) else match[1], name)
        trait_contracts = []
        for contract in package.get('templates', ()):
            if contract.get('kind') == 'class-grammars':
                continue
            source = SourceFile(package['package']['name'] + '/' + contract['path'], contract['source'])
            root, errors = parse_blocks(source, lex(source, context.syntax), context.syntax)
            if errors or len(root.children) != 1:
                raise ValueError('invalid imported record template interface')
            template = root.children[0]
            if not hasattr(context, 'imported_template_origins'):
                context.imported_template_origins = {}
            context.imported_template_origins[source.identity] = context.imported_template_origins.get(node.span.source, node.identity)
            if contract.get('kind') == 'enum':
                if getattr(template.provider, 'spelling', '') != 'enum' or template.header[0].text != contract['name']:
                    raise ValueError('imported enum identity does not match its declaration')
                previous = context.scope
                context.scope = (alias,)
                try:
                    template.provider.declare(template, context)
                finally:
                    context.scope = previous
                continue
            if contract.get('kind') == 'alias':
                from py_compiler.syntax.complex.generic import declare_alias
                previous = context.scope
                context.scope = (alias,)
                try:
                    declare_alias(template, context)
                    if alias + '.' + contract['name'] not in context.record_templates:
                        raise ValueError('imported generic alias identity does not match its declaration')
                finally:
                    context.scope = previous
                continue
            if getattr(template.provider, 'spelling', '') not in ('class', 'trait', 'def') or template.header[0].text != contract['name']:
                raise ValueError('imported template identity does not match its declaration')
            if contract.get('kind') == 'trait':
                if template.provider.spelling != 'trait':
                    raise ValueError('imported trait contract requires a trait declaration')
                trait_contracts.append(template)
                continue
            previous = context.scope
            context.scope = (alias,)
            try:
                if contract.get('kind') == 'function':
                    entry = template.provider.register(template, context)
                    excluded = {parameter.name for parameter in entry.templates}
                    entry.parameters = tuple((name, qualify(annotation, excluded) if annotation else None)
                                             for name, annotation in entry.parameters)
                    entry.result = qualify(entry.result, excluded) if entry.result else None
                elif not register_template(template.provider, template, context):
                    raise ValueError('imported class template requires generic parameters')
            finally:
                context.scope = previous

        from py_compiler.syntax.generic.storage import GlobalPlace
        for contract in package.get('globals', ()):
            symbol = contract['symbol']
            import re
            if not re.fullmatch(r'__sev_global_[0-9a-f]{64}', symbol):
                raise ValueError('invalid imported global symbol')
            type_ = context.type(contract['type'])
            type_.module_storage(symbol)
            context.imported_globals[alias + '.' + contract['name']] = GlobalPlace(
                symbol, type_, constant=contract['constant'], external_symbol=symbol)
        from py_compiler.syntax.block.record import TypeDeclaration
        from importlib import import_module
        Class = import_module('py_compiler.syntax.block.class').Class
        Receiver = import_module('py_compiler.syntax.block.class').Receiver
        from py_compiler.syntax.complex.object import ObjectType
        for contract in type_contracts.values():
            if contract.get('kind') == 'union':
                continue
            name = qualify(contract['name'])
            declaration = TypeDeclaration(node.identity + ':' + name, name, 'class',
                tuple((field, qualify(type_)) for field, type_ in contract['fields']), (), tuple(qualify(t) for t in contract.get('traits', ())), node.span,
                defaults=tuple(contract['defaults']), memory_defaults=tuple(map(tuple, contract['memory-defaults'])),
                destructor=contract.get('destructor', False))
            context.register(declaration)
            context.provider_by_name[name] = Class()
            context.types[name] = ObjectType(name, 'record', '!llvm.ptr', declaration=declaration)
        from py_compiler.syntax.block.union import UnionType, UnionDeclaration
        for contract in type_contracts.values():
            if contract.get('kind') != 'union':
                continue
            name = qualify(contract['name'])
            variants = tuple(qualify(member) for member in contract['variants'])
            declaration = TypeDeclaration(node.identity + ':' + name, name, 'union',
                                          (), variants, (), node.span)
            context.register(declaration)
            context.provider_by_name[name] = UnionDeclaration()
            context.types[name] = UnionType(tuple(context.type(member) for member in variants), name).with_layout(context)
        previous = context.scope, context.active_provider
        context.scope = (alias,)
        try:
            context.importing_trait = True
            for template in trait_contracts:
                context.active_provider = template.provider
                template.provider.declare(template, context)
        finally:
            context.importing_trait = False
            context.scope, context.active_provider = previous
        from dataclasses import replace
        from py_compiler.syntax.generic.storage import owns_storage
        def owned(name, active=()):
            if name in active:
                raise ValueError('recursive inline class layout')
            if type_contracts[name].get('kind') == 'union':
                return False
            return any((type_contracts[t].get('destructor', False) or owned(t, (*active, name))) if t in type_contracts else
                       owns_storage(context.type(qualify(t)))
                       for _, t in type_contracts[name]['fields'])
        for name in type_contracts:
            if type_contracts[name].get('kind') == 'union':
                continue
            context.types[qualify(name)] = replace(context.types[qualify(name)], owns_fields=owned(name))
        for contract in package.get('templates', ()):
            if contract.get('kind') != 'class-grammars':
                continue
            source = SourceFile(package['package']['name'] + '/' + contract['path'], contract['source'])
            root, errors = parse_blocks(source, lex(source, context.syntax), context.syntax)
            if errors or len(root.children) != 1 or root.children[0].header[0].text != contract['name']:
                raise ValueError('invalid imported class grammar interface')
            if not hasattr(context, 'imported_template_origins'):
                context.imported_template_origins = {}
            context.imported_template_origins[source.identity] = context.imported_template_origins.get(node.span.source, node.identity)
            owner = Receiver(context.by_name[qualify(contract['name'])])
            previous = context.scope
            context.scope = (alias,)
            try:
                for member in root.children[0].children:
                    if getattr(member.provider, 'spelling', '') == 'grammar':
                        member.provider.declare_member(member, context, owner)
                        node_, captures, _, result = context.sentence_grammars[-1]
                        context.sentence_grammars[-1] = (node_, captures, (), result)
            finally:
                context.scope = previous
        for contract in package['exports']:
            name = alias + '.' + contract['name']
            parameters = tuple((p['name'], qualify(p['type'])) for p in contract['parameters'])
            for _, type_ in parameters:
                context.type(type_)
            context.type(qualify(contract['result']))
            entry = ImportedCallable(node, name, parameters, qualify(contract['result']), (),
                                     receiver=Receiver(context.by_name[qualify(contract['receiver'])]) if contract.get('receiver') else None,
                                     receiver_ownership=contract.get('receiver-ownership'),
                                     parameter_ownership={p['name']: p['ownership'] for p in contract['parameters']},
                                     c_symbol=CSymbol(contract['symbol'], 'severian', tuple(package['objects'])))
            entry.body = SimpleNamespace(result_sources=tuple(contract['result-sources']),
                                         retained_result_sources=tuple(contract.get('retained-result-sources', ())),
                                         retained_parameter_sources=tuple((destination, tuple(sources)) for destination, sources in
                                                                          contract.get('retained-parameter-sources', ())))
            entry.result_ownership = contract.get('result-ownership', 'own')
            from py_compiler.syntax.grammar.expression import expression
            for parameter, spelling in contract.get('defaults', {}).items():
                if parameter not in dict(parameters):
                    raise ValueError('imported default has no matching parameter')
                source = SourceFile(name + ':default:' + parameter, spelling)
                entry.defaults[parameter] = expression(tuple(t for t in lex(source, context.syntax)
                    if t.kind not in ('NEWLINE', 'EOF', 'INDENT', 'DEDENT')))
            from py_compiler.syntax.function.contracts import pure_predicate
            guards = []
            for text in contract.get('acceptance', ()):
                source = SourceFile(name + ':acceptance', text)
                guard = expression(tuple(t for t in lex(source, context.syntax) if t.kind not in ('NEWLINE', 'EOF', 'INDENT', 'DEDENT')))
                pure_predicate(guard, {name for name, _ in parameters} | ({'self'} if entry.receiver else set()))
                guards.append(guard)
            entry.guards = tuple(guards)
            context.functions.setdefault(name, []).append(entry)
        from py_compiler.syntax.complex.generic import CallableAlias
        for binding in package.get('aliases', ()):
            name = alias + '.' + binding['name']
            if name in context.functions or name in context.function_aliases:
                raise ValueError('duplicate imported callable alias ' + name)
            context.function_aliases[name] = CallableAlias(alias + '.' + binding['target'], ())


import unittest


class ImportedPredicateTests(unittest.TestCase):
    def test_imported_global_memory_has_a_borrowable_owner(self):
        from py_compiler.frontend.src.lib import compile_source
        from py_compiler.syntax.recognition import Syntax
        package = {'package': {'name': 'values'}, 'objects': ['values.o'], 'exports': [], 'types': [],
                   'globals': [{'name': 'label', 'symbol': '__sev_global_' + '0' * 64,
                                'type': 'string', 'constant': True}]}
        result = compile_source('consumer.sev', 'import values\n'
            'def use(value: string) -> int:\n    return len(value)\n'
            'def work() -> int:\n    return use(values.label)\n', Syntax(), c_packages={'values': package})
        self.assertFalse(result.diagnostics, str(result.diagnostics))

    def test_imported_function_template_qualifies_nominal_parameters(self):
        from py_compiler.frontend.src.lib import compile_source
        from py_compiler.syntax.recognition import Syntax
        producer = compile_source('lib.sev', 'class Item:\n    value: int\n'
            'def choose[T](item: Item, value: T) -> T:\n    return value\n', Syntax())
        self.assertFalse(producer.diagnostics, str(producer.diagnostics))
        package = {'package': {'name': 'values'}, 'objects': ['values.o'], 'exports': [],
                   'templates': list(producer.templates),
                   'types': [{'name': 'Item', 'fields': [('value', 'int')], 'traits': [], 'defaults': [], 'memory-defaults': []}]}
        result = compile_source('consumer.sev', 'import values\n'
            'def work() -> int:\n    item = values.Item(1)\n    return values.choose[int](item, 42)\n',
            Syntax(), c_packages={'values': package})
        self.assertFalse(result.diagnostics, str(result.diagnostics))

    def test_json_roundtrip_preserves_all_lifetime_sources(self):
        import json
        from py_compiler.frontend.src.lib import compile_source
        from py_compiler.syntax.recognition import Syntax
        package = {'package': {'name': 'lifetimes'}, 'objects': ['lifetimes.o'], 'types': [],
                   'exports': [{'name': 'choose', 'symbol': 'choose', 'receiver': None,
                     'parameters': [{'name': name, 'type': 'array[int]', 'ownership': 'view'}
                                    for name in ('a', 'b')],
                     'result': 'array[int]', 'result-ownership': 'view', 'result-sources': (0, 1),
                     'retained-result-sources': (1,), 'retained-parameter-sources': ((0, (1,)),)}]}
        restored = json.loads(json.dumps(package))
        result = compile_source('roundtrip.sev', 'import lifetimes\n'
                                'def work(a: array[int], b: array[int]) -> int:\n'
                                '    return len(lifetimes.choose(a, b))\n', Syntax(),
                                c_packages={'lifetimes': restored})
        self.assertFalse(result.diagnostics, str(result.diagnostics))
        imported = next(op.atom.owner for body in result.program.bodies for block in body.blocks
                        for op in block.operations if isinstance(op.atom.owner, ImportedCallable))
        self.assertEqual(imported.body.result_sources, (0, 1))
        self.assertEqual(imported.body.retained_result_sources, (1,))
        self.assertEqual(imported.body.retained_parameter_sources, ((0, (1,)),))
        self.assertEqual(imported.result_ownership, 'view')
        self.assertEqual(imported.parameter_ownership, {'a': 'view', 'b': 'view'})

    def test_imported_union_is_a_named_constraint_and_value_type(self):
        from py_compiler.frontend.src.lib import compile_source
        from py_compiler.syntax.recognition import Syntax
        package = {'package': {'name': 'units'}, 'objects': ['units.o'], 'exports': [],
                   'types': [{'name': 'Unit', 'kind': 'union', 'variants': ['u8', 'u32']}]}
        source = ('import units\ndef identity[T: units.Unit](value: T) -> T:\n    return value\n'
                  'def work() -> units.Unit:\n    return identity(u32(955))\n')
        result = compile_source('consumer.sev', source, Syntax(), c_packages={'units': package})
        self.assertFalse(result.diagnostics, str(result.diagnostics))

    def compile_import(self, *, receiver=True, predicate='-self.count <= position < self.count'):
        from py_compiler.frontend.src.lib import compile_source
        from py_compiler.syntax.recognition import Syntax
        package = {
            'package': {'name': 'checked-vector'}, 'objects': ['checked-vector.o'],
            'types': [{'name': 'Vector', 'fields': [['count', 'int']], 'defaults': [None],
                       'memory-defaults': []}],
            'exports': [{'name': 'Vector.get' if receiver else 'get', 'symbol': 'checked_get',
                         'receiver': 'Vector' if receiver else None, 'receiver-ownership': 'view',
                         'parameters': [{'name': 'position', 'type': 'int', 'ownership': 'copy'}],
                         'result': 'int', 'result-sources': [], 'acceptance': [predicate]}],
        }
        return compile_source('consumer.sev', 'import vector\n', Syntax(include_tests=False),
                              c_packages={'vector': package})

    def test_imported_method_predicate_can_read_its_receiver(self):
        result = self.compile_import()
        self.assertFalse(result.diagnostics, str(result.diagnostics))

    def test_free_function_predicate_cannot_invent_a_receiver(self):
        result = self.compile_import(receiver=False)
        self.assertTrue(result.diagnostics)
        self.assertIn('stable-read provider', str(result.diagnostics[0]))

    def test_imported_method_predicate_still_rejects_arbitrary_calls(self):
        result = self.compile_import(predicate='self.mutate()')
        self.assertTrue(result.diagnostics)
        self.assertIn('stable-read provider', str(result.diagnostics[0]))


class FromImport(Import):
    spelling = 'from'

    def parse_header(self, items):
        if len(items) < 4 or items[1].kind not in ('IDENTIFIER', 'STRING') or items[2].text != 'import':
            raise ValueError('expected from package_alias import name[, name]')
        names = items[3::2]
        if items[1].kind == 'STRING' and len(names) == 1 and names[0].text == '*':
            return (items[1], *names)
        if any(t.kind != 'IDENTIFIER' for t in names) or any(t.text != ',' for t in items[4::2]) or len(items) % 2 != 0:
            raise ValueError('import names must be comma-separated identifiers')
        return (items[1], *names)

    def declare(self, node, context):
        from dataclasses import replace
        alias = node.header[0].text
        if node.header[0].kind == 'STRING':
            import ast
            from pathlib import Path
            from py_compiler.frontend.source.source import SourceFile
            from py_compiler.frontend.lexer.lexer import lex
            from py_compiler.frontend.parser.blocks import parse_blocks
            from py_compiler.syntax.generic.block import DeclarationProvider
            if len(node.header) != 2 or node.header[1].text != '*':
                raise ValueError('source imports currently require an explicit wildcard')
            origin = node.tokens[0].lexeme.source
            base = Path(origin.debug_path or origin.path).resolve().parent
            source_path = (base / ast.literal_eval(alias)).resolve()
            if source_path.suffix != '.sev' or not source_path.is_relative_to(base):
                raise ValueError('source import must stay inside its source directory')
            loaded = getattr(context, 'loaded_source_imports', set())
            if source_path in loaded:
                return
            context.loaded_source_imports = loaded | {source_path}
            source = SourceFile(str(source_path), source_path.read_text(), str(source_path))
            root, diagnostics = parse_blocks(source, lex(source, context.syntax), context.syntax)
            if diagnostics:
                raise ValueError('source import failed: ' + str(diagnostics[0]))
            if any(not isinstance(child.provider, DeclarationProvider) for child in root.children):
                raise ValueError('source import requires declarations; module initialization needs a package import')
            if not hasattr(context, 'imported_template_origins'):
                context.imported_template_origins = {}
            context.imported_template_origins[source.identity] = context.imported_template_origins.get(node.span.source, node.identity)
            context.declare_nodes(root.children, context.scope)
            return
        if alias == 'xxi':
            if any(token.text not in ('c', 'rust', 'python') for token in node.header[1:]):
                raise ValueError('unknown foreign provider')
            return
        if alias not in context.package_imports:
            previous = context.scope
            context.scope = ()
            try:
                super().declare(replace(node, header=node.header[:1]), context)
            finally:
                context.scope = previous
        for token in node.header[1:]:
            name = token.text
            qualified = alias + '.' + name
            name = '.'.join((*context.scope, name))
            functions = context.lookup_functions(qualified, ())
            if functions:
                context.check_prelude_conflict(name)
                if name in context.functions and context.functions[name] != functions:
                    raise ValueError('duplicate imported callable ' + name)
                context.functions[name] = list(functions)
            elif qualified in context.record_templates:
                if name in context.record_templates and context.record_templates[name] is not context.record_templates[qualified]:
                    raise ValueError('duplicate imported template ' + name)
                context.record_templates[name] = context.record_templates[qualified]
            elif qualified in context.types:
                if name in context.types and context.types[name] != context.types[qualified]:
                    raise ValueError('duplicate imported type ' + name)
                context.types[name] = context.types[qualified]
            else:
                raise ValueError('package does not export ' + qualified)


class SelectiveImportTests(unittest.TestCase):
    def test_selected_function_calls_the_exported_c_symbol(self):
        from py_compiler.frontend.src.lib import compile_source
        from py_compiler.syntax.recognition import Syntax
        from py_compiler.mlir.src.lib import lower, render
        package = {'package': {'name': 'numbers'}, 'objects': ['numbers.o'], 'types': [],
                   'exports': [{'name': 'answer', 'symbol': 'library_answer', 'parameters': [],
                                'result': 'int', 'result-sources': []}]}
        result = compile_source('selective.sev', 'from numbers import answer\ndef run() -> int:\n    return answer()\n',
                                Syntax(), c_packages={'numbers': package})
        self.assertFalse(result.diagnostics, str(result.diagnostics))
        self.assertIn('func.call @library_answer()', render(lower(result.program)))


class ImportedDestructorTests(unittest.TestCase):
    def test_imported_scalar_resource_and_containing_class_keep_cleanup(self):
        from py_compiler.frontend.src.lib import compile_source
        from py_compiler.syntax.recognition import Syntax
        from py_compiler.mlir.src.lib import lower, render
        package = {'package': {'name': 'resources'}, 'objects': ['resources.o'],
                   'types': [{'name': 'Handle', 'fields': [['id', 'int']], 'defaults': [None],
                              'memory-defaults': [], 'destructor': True}],
                   'exports': [{'name': 'Handle.drop', 'symbol': 'release_handle', 'receiver': 'Handle',
                                'receiver-ownership': 'view', 'parameters': [],
                                'result': 'absent', 'result-sources': []}]}
        result = compile_source('consumer.sev',
            'import resources\nclass Owner:\n    handle: resources.Handle\n'
            'def work():\n    handle = resources.Handle(7)\n    owner = Owner(handle)\n',
            Syntax(), c_packages={'resources': package})
        self.assertFalse(result.diagnostics, str(result.diagnostics))
        text = render(lower(result.program))
        self.assertIn('func.call @release_handle', text)
