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
        for contract in package['exports']:
            name = alias + '.' + contract['name']
            parameters = tuple((p['name'], p['type']) for p in contract['parameters'])
            for _, type_ in parameters:
                context.type(type_)
            context.type(contract['result'])
            entry = ImportedCallable(node, name, parameters, contract['result'], (),
                                     parameter_ownership={p['name']: p['ownership'] for p in contract['parameters']},
                                     native_symbol=NativeSymbol(contract['symbol'], 'severian', tuple(package['objects'])))
            entry.body = SimpleNamespace(result_sources=tuple(contract['result-sources']))
            context.functions.setdefault(name, []).append(entry)
