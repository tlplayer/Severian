"""Shared field contract tools used by record-like syntax providers."""
from dataclasses import dataclass
from py_compiler.syntax.generic.block import DeclarationProvider
from py_compiler.syntax.grammar.expression import expression
from py_compiler.frontend.parser.contract import Literal
from py_compiler.hir.hir.src.program import resolve_literal


@dataclass(frozen=True)
class TypeDeclaration:
    identity: str
    name: str
    kind: str
    fields: tuple
    variants: tuple
    traits: tuple
    span: object
    source: str = ""
    defaults: tuple = ()
    methods: tuple = ()
    template_name: str = ''
    arguments: tuple = ()

    @property
    def binding_ownership(self):
        return "view"

    def grammars(self):
        # Record declarations explicitly expose their type name; they have no literal syntax.
        from py_compiler.syntax.generic.owned import WordGrammar
        return (WordGrammar(self, self.name, "IDENTIFIER"),)



class RecordProvider(DeclarationProvider):
    def declare_member(self, node, context, owner):
        context.declare_nodes([node], (*context.scope, owner.declaration.identity))

    def parse_header(self, items):
        header = super().parse_header(items)
        if len(header) != 1 or header[0].kind != "IDENTIFIER":
            raise ValueError(f"{self.spelling} requires a name")
        return header

    def field(self, child, context):
        items = list(child.tokens)
        if items[0].kind != "IDENTIFIER":
            raise ValueError("expected a named field")
        position, annotation = 1, None
        if len(items) > 2 and items[1].text == ":":
            position = next((i for i in range(2, len(items)) if items[i].text in ('=', ':=')), len(items))
            annotation = context.type(''.join(t.text for t in items[2:position])).name
        default = None
        if position < len(items):
            if items[position].text not in ("=", ":="):
                raise ValueError("expected a field initializer")
            term = expression(items[position + 1:])
            if term.kind != "literal":
                raise ValueError("field initializer requires a literal provider in this milestone")
            type_, default = resolve_literal(Literal(child.identity, term.token.span, term.token.text, term.token.kind, annotation), context.bound_syntax, context)
            annotation = type_.name
        if annotation is None:
            raise ValueError("field requires a type or initializer")
        return (items[0].text, context.type(annotation).name), default

    def fields(self, node, context):
        fields, defaults, members = [], [], []
        for child in node.children:
            if child.provider:
                members.append(child)
                continue
            field, default = self.field(child, context)
            if any(f[0] == field[0] for f in fields):
                raise ValueError("duplicate field declaration")
            fields.append(field)
            defaults.append(default)
        return tuple(fields), tuple(defaults), members


def record_header(header):
    from py_compiler.syntax.complex.generic import template_header
    parameters, items = template_header(list(header))
    if not items or items[0].kind != 'IDENTIFIER':
        raise ValueError('record declaration requires a name')
    if len(items) > 1 and items[1].text != ':':
        raise ValueError('expected record name followed by trait contracts')
    groups, group, depth = [], [], 0
    for token in items[2:]:
        if token.text == '+' and depth == 0:
            if not group:
                raise ValueError('missing trait contract')
            groups.append(''.join(t.text for t in group)); group = []
        else:
            group.append(token)
            depth += (token.text == '[') - (token.text == ']')
    if group:
        groups.append(''.join(t.text for t in group))
    if depth or (len(items) > 1 and not groups):
        raise ValueError('invalid trait contract list')
    return parameters, items, tuple(groups)


def register_template(provider, node, context):
    parameters, _, _ = record_header(node.header)
    if not parameters:
        return False
    name = '.'.join((*context.scope, node.header[0].text))
    if name in context.record_templates or name in context.types:
        raise ValueError(f'duplicate declaration {name!r}')
    context.record_templates[name] = RecordTemplate(name, node, provider, parameters, context.scope)
    return True


@dataclass
class RecordTemplate:
    name: str
    node: object
    provider: object
    parameters: tuple
    scope: tuple

    def realize(self, arguments, context):
        from dataclasses import replace
        from py_compiler.syntax.complex.generic import template_argument, substitute, check_constraints, record_clone
        if len(arguments) > len(self.parameters):
            raise ValueError('too many record template arguments')
        bindings = {p.name: template_argument(value, context) for p, value in zip(self.parameters, arguments)}
        previous = context.scope, context.template_types, context.active_provider
        context.scope = self.scope
        try:
            for parameter in self.parameters:
                if parameter.name not in bindings:
                    if parameter.default is None:
                        raise ValueError(f'missing record template argument {parameter.name}')
                    bindings[parameter.name] = template_argument(substitute(parameter.default, bindings), context)
            context.template_types = {**bindings}
            check_constraints(self.parameters, bindings, context)
            suffix = '[' + ','.join(bindings[p.name].name for p in self.parameters) + ']'
            name = self.name + suffix
            if name in context.types:
                return context.types[name]
            if name in context.realizing_records:
                raise ValueError('recursive record storage requires an indirection contract')
            context.realizing_records.add(name)
            try:
                _, header, _ = record_header(self.node.header)
                node = record_clone(self.node, bindings, suffix)
                node.header = tuple(replace(token, text=substitute(token.text, bindings)) for token in header)
                node.header = (replace(node.header[0], text=node.header[0].text + suffix), *node.header[1:])
                node.record_arguments = tuple(bindings[p.name] for p in self.parameters)
                node.record_template = self.name
                context.active_provider = self.provider
                self.provider.declare(node, context)
                context.scopes[node.identity] = node.scope
                context.resolve_contracts()
                return context.types[name]
            finally:
                context.realizing_records.remove(name)
        finally:
            context.scope, context.template_types, context.active_provider = previous
