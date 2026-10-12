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
            source = SourceFile(package['package']['name'] + '/' + contract['path'], contract['source'])
            root, errors = parse_blocks(source, lex(source, context.syntax), context.syntax)
            if errors or len(root.children) != 1:
                raise ValueError('invalid imported record template interface')
            template = root.children[0]
            if not hasattr(context, 'imported_template_origins'):
                context.imported_template_origins = {}
            context.imported_template_origins[source.identity] = node.identity
            if getattr(template.provider, 'spelling', '') != 'class' or template.header[0].text != contract['name']:
                raise ValueError('imported template identity does not match its declaration')
            previous = context.scope
            context.scope = (alias,)
            try:
                if not register_template(template.provider, template, context):
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
            name = qualify(contract['name'])
            declaration = TypeDeclaration(node.identity + ':' + name, name, 'class',
                tuple((field, qualify(type_)) for field, type_ in contract['fields']), (), (), node.span,
                defaults=tuple(contract['defaults']), memory_defaults=tuple(map(tuple, contract['memory-defaults'])))
            context.register(declaration)
            context.provider_by_name[name] = Class()
            context.types[name] = ObjectType(name, 'record', '!llvm.ptr', declaration=declaration)
        from dataclasses import replace
        def owned(name, active=()):
            if name in active:
                raise ValueError('recursive inline class layout')
            return any(owned(t, (*active, name)) if t in type_contracts else
                       callable(getattr(context.type(t), 'release', None))
                       for _, t in type_contracts[name]['fields'])
        for name in type_contracts:
            context.types[qualify(name)] = replace(context.types[qualify(name)], owns_fields=owned(name))
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
            context.functions.setdefault(name, []).append(entry)
