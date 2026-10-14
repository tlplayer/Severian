"""Native package imports consume checked interface contracts, never source bodies."""
from dataclasses import dataclass
from types import SimpleNamespace
from py_compiler.syntax.generic.block import DeclarationProvider
from py_compiler.syntax.function.contracts import Callable
from py_compiler.syntax.generic.atom import ExternalSymbol


@dataclass
class ImportedCallable(Callable):
    native_symbol: object = None

    @property
    def symbol(self):
        return self.native_symbol.symbol


@dataclass(frozen=True)
class NativeSymbol(ExternalSymbol):
    result_layout: str = ''

    def render_operation(self, operation):
        if self.result_layout:
            from py_compiler.syntax.complex.object import Invoke
            return Invoke(self.symbol, operation.result.type, result_layout=self.result_layout).render_operation(operation)
        return super().render_operation(operation)

    def __post_init__(self):
        if self.abi != 'severian' or not self.symbol or '\x00' in self.symbol:
            raise ValueError('invalid native Severian symbol')
        if not self.dependencies or any(not p.endswith('.o') for p in self.dependencies):
            raise ValueError('native import requires object dependencies')


class Import(DeclarationProvider):
    spelling = 'import'

    def has_body(self, node):
        return False

    def parse_header(self, items):
        if len(items) != 2 or items[1].kind != 'IDENTIFIER':
            raise ValueError('expected import package_alias')
        return tuple(items[1:])

    def declare(self, node, context):
        alias = node.header[0].text
        if context.scope:
            raise ValueError('package imports belong at module scope')
        package = context.native_packages.get(alias)
        if package is None:
            raise ValueError(f'package {alias!r} is not a resolved manifest dependency')
        if alias in context.package_imports:
            raise ValueError(f'duplicate package import {alias!r}')
        context.package_imports.add(alias)
        from py_compiler.frontend.source.source import SourceFile
        from py_compiler.frontend.lexer.lexer import lex
        from py_compiler.frontend.parser.blocks import parse_blocks
        from py_compiler.syntax.block.record import register_template
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
            context.imported_template_origins[source.identity] = node.identity
            if getattr(template.provider, 'spelling', '') not in ('class', 'trait', 'def') or template.header[0].text != contract['name']:
                raise ValueError('imported template identity does not match its declaration')
            previous = context.scope
            context.scope = (alias,)
            try:
                if contract.get('kind') == 'function':
                    template.provider.declare(template, context)
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
        type_contracts = {t['name']: t for t in package.get('types', ())}
        template_names = {item['name'] for item in package.get('templates', ())}
        def qualify(name):
            return alias + '.' + name if name in type_contracts or name.split('[', 1)[0] in template_names else name
        for contract in type_contracts.values():
            if contract.get('kind') == 'union':
                continue
            name = qualify(contract['name'])
            declaration = TypeDeclaration(node.identity + ':' + name, name, 'class',
                tuple((field, qualify(type_)) for field, type_ in contract['fields']), (), tuple(qualify(t) for t in contract.get('traits', ())), node.span,
                defaults=tuple(contract['defaults']), memory_defaults=tuple(map(tuple, contract['memory-defaults'])))
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
        from dataclasses import replace
        def owned(name, active=()):
            if name in active:
                raise ValueError('recursive inline class layout')
            if type_contracts[name].get('kind') == 'union':
                return False
            return any(owned(t, (*active, name)) if t in type_contracts else
                       callable(getattr(context.type(t), 'release', None))
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
                                     native_symbol=NativeSymbol(contract['symbol'], 'severian', tuple(package['objects'])))
            entry.body = SimpleNamespace(result_sources=tuple(contract['result-sources']))
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


import unittest


class ImportedPredicateTests(unittest.TestCase):
    def test_imported_union_is_a_named_constraint_and_value_type(self):
        from py_compiler.frontend.src.lib import compile_source
        from py_compiler.syntax.recognition import Syntax
        package = {'package': {'name': 'units'}, 'objects': ['units.o'], 'exports': [],
                   'types': [{'name': 'Unit', 'kind': 'union', 'variants': ['u8', 'u32']}]}
        source = ('import units\ndef identity[T: units.Unit](value: T) -> T:\n    return value\n'
                  'def work() -> units.Unit:\n    return identity(u32(955))\n')
        result = compile_source('consumer.sev', source, Syntax(), native_packages={'units': package})
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
                              native_packages={'vector': package})

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
