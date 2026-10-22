"""Add methods through the existing class receiver and callable providers."""
from py_compiler.syntax.generic.block import DeclarationProvider


class Extend(DeclarationProvider):
    spelling = "extend"

    def declare(self, node, context):
        from importlib import import_module
        from py_compiler.syntax.function.contracts import signature
        if len(node.decorators) > 1:
            raise ValueError('extension declares exactly one operator namespace')
        namespace = node.decorators[0].name if node.decorators else None
        spelling=''.join(token.text for token in node.header)
        if '[' in spelling:
            from py_compiler.syntax.complex.generic import split_arguments
            base,args=spelling[:-1].split('[',1)
            template=context.record_templates.get(context.declaration_name(base,context.scope))
            if template is not None and tuple(split_arguments(args))==tuple(p.name for p in template.parameters):
                if template.provider.spelling != 'class':
                    raise ValueError('extend currently requires a class template')
                members = [(member, None) for member in template.node.children]
                for previous, extension, _ in getattr(context, 'generic_extensions', ()):
                    if previous is template:
                        scope = extension.decorators[0].name if extension.decorators else None
                        members.extend((member, scope) for member in extension.children)
                existing = {}
                for member, scope in members:
                    if getattr(member.provider, 'spelling', '') in ('def', 'operator'):
                        leaf = signature(member, template.name).name.rsplit('.', 1)[-1]
                        existing.setdefault(leaf, []).append(scope if member.provider.spelling == 'operator' else None)
                    elif not member.provider and member.tokens:
                        existing.setdefault(member.tokens[0].text, []).append(None)
                for member in node.children:
                    if getattr(member.provider, 'spelling', '') not in ('def', 'operator'):
                        raise ValueError('extend may add methods, not storage fields')
                    leaf = signature(member, template.name).name.rsplit('.', 1)[-1]
                    if leaf in ('drop', template.name.rsplit('.', 1)[-1]):
                        raise ValueError('extend cannot replace construction or destruction contracts')
                    scopes = existing.get(leaf, ())
                    scoped_operator = namespace is not None and member.provider.spelling == 'operator'
                    if scopes and (not scoped_operator or None in scopes or namespace in scopes):
                        raise ValueError('extension cannot overwrite existing member ' + leaf)
                    existing.setdefault(leaf, []).append(namespace if scoped_operator else None)
                context.generic_extensions=getattr(context,'generic_extensions',[])+[(template,node,context.scope)]
                # An annotation can realize storage before an extension appears.
                # Those existing specializations receive the same added members.
                from py_compiler.syntax.complex.generic import record_clone
                for target in {t.name: t for t in context.types.values()}.values():
                    if getattr(getattr(target, 'declaration', None), 'template_name', '') == template.name:
                        bindings = dict(zip((p.name for p in template.parameters), target.generic_arguments))
                        previous_types = context.template_types
                        context.template_types = bindings
                        try:
                            self.declare(record_clone(node, bindings, ':extend:' + target.name), context)
                        finally:
                            context.template_types = previous_types
                return
        target = context.type(spelling)
        if target.family != 'record':
            raise ValueError('extend currently requires a concrete class type')
        declaration = context.by_name[target.name]
        receiver = import_module('py_compiler.syntax.block.class').Receiver(declaration)
        additions = set()
        for member in node.children:
            if getattr(member.provider, 'spelling', None) not in ('def','operator'):
                raise ValueError('extend may add methods, not storage fields')
            method = signature(member, target.name)
            leaf = method.name.rsplit('.', 1)[-1]
            if leaf in ('drop', declaration.name.rsplit('.', 1)[-1].split('[', 1)[0]):
                raise ValueError('extend cannot replace construction or destruction contracts')
            entries = context.functions.get(method.name, ())
            conflict = bool(entries) and (namespace is None or member.provider.spelling != 'operator'
                or any(getattr(entry, 'operator_namespace', None) in (None, namespace) for entry in entries))
            if (conflict or leaf in additions
                    or any(name == leaf for name, _ in declaration.fields)):
                raise ValueError('extension cannot overwrite existing member ' + leaf)
            additions.add(leaf)
        for member in node.children:
            before = set(id(entry) for entry in context.functions.get(target.name + '.' + member.header[0].text, ()))
            member.provider.declare_member(member, context, receiver)
            if node.decorators:
                key=target.name+'.'+member.header[0].text
                for entry in context.functions[key]:
                    if id(entry) not in before and member.provider.spelling == 'operator':
                        entry.operator_namespace=namespace


import unittest


class ExtensionTests(unittest.TestCase):
    def test_generic_extension_reaches_an_already_realized_storage_type(self):
        result = self.compile('class Box[T]:\n    value: T\n'
            'class Holder:\n    storage: Box[int]\n'
            'extend Box[T]:\n    def read(self: view Self) -> T:\n        return value\n'
            'def work() -> int:\n    holder = Holder(Box[int](42))\n'
            '    return holder.storage.read()\n')
        self.assertFalse(result.diagnostics, str(result.diagnostics))

    def test_generic_overwrite_is_rejected_without_instantiating_the_class(self):
        result = self.compile('class Items[T]:\n    value: T\n'
            '    def get() -> T:\n        return value\n'
            'extend Items[T]:\n    def get() -> T:\n        return value\n')
        self.assertTrue(result.diagnostics)
        self.assertIn('cannot overwrite existing member get', str(result.diagnostics))

    def compile(self, source):
        from py_compiler.frontend.src.lib import compile_source
        from py_compiler.syntax.recognition import Syntax
        return compile_source('extensions.sev', source, Syntax())

    def test_extension_uses_the_original_class_and_receiver(self):
        result = self.compile('class Counter:\n    value: int\n'
            'extend Counter:\n    def increment():\n        value += 1\n'
            'def work() -> int:\n    counter = Counter(4)\n    counter.increment()\n    return counter.value\n')
        self.assertFalse(result.diagnostics, str(result.diagnostics))
        self.assertEqual([d.name for d in result.program.declarations], ['Counter'])
        method = next(b for b in result.program.bodies if b.declaration == 'Counter.increment')
        self.assertEqual(method.blocks[0].parameters[0].type.name, 'Counter')

    def test_extension_rejects_fields_existing_methods_and_lifetime_hooks(self):
        for member in ('    value: int\n', '    def read() -> int:\n        return 2\n',
                       '    def drop():\n        return\n', '    def Counter():\n        return\n'):
            result = self.compile('class Counter:\n    value: int\n    def read() -> int:\n        return value\n'
                                  'extend Counter:\n' + member)
            self.assertTrue(result.diagnostics)
