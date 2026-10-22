"""Explicit pipeline: syntax -> source -> lexer -> parser -> HIR -> THIR -> MIR."""
from dataclasses import dataclass, replace
from py_compiler.frontend.source.source import SourceFile, Diagnostic
from py_compiler.frontend.lexer.lexer import lex
from py_compiler.frontend.parser.parser import parse
from py_compiler.hir.hir.src.program import partition, Program


@dataclass(frozen=True)
class FrontendResult:
    source: SourceFile
    program: Program
    diagnostics: tuple[Diagnostic, ...]
    templates: tuple = ()
    aliases: tuple = ()


def compile_source(path, text, syntax, *, stop_after='mir', interface=None, c_packages=None, debug_path=None):
    if stop_after not in ('thir', 'mir'):
        raise ValueError('stop_after must be thir or mir')
    source = SourceFile(path, text, debug_path)
    try:
        tokens = lex(source, syntax)
    except Diagnostic as failure:
        return FrontendResult(source, Program((), ()), (failure,))
    from py_compiler.frontend.parser.blocks import parse_blocks
    from py_compiler.hir.thir.src.checking import Checker
    from py_compiler.hir.thir.src.lib import prepare, prepare_literals
    from py_compiler.mir.registration import register
    from py_compiler.hir.hir.src.modules import resolve
    root, block_errors = parse_blocks(source, tokens, syntax)
    from py_compiler.syntax.generic.grammar import SourceWindow
    if block_errors:
        # Recovery retains independently recognized literal declarations for diagnostics.
        graph, _ = parse(source, tokens, syntax)
        program, hir_errors = partition(source, graph, syntax)
        return FrontendResult(source, program, tuple(block_errors + hir_errors))
    if not text:
        return FrontendResult(source, Program((), (), stage=stop_after), ())
    literal_match = syntax.registry.recognize('X.literal-module', SourceWindow(source, 0, len(text), tuple(tokens)), optional=True)
    if literal_match is not None:
        graph, errors = literal_match.provider.construct(literal_match)
        program, hir_errors = partition(source, graph, syntax)
        if errors or hir_errors:
            return FrontendResult(source, program, tuple(errors + hir_errors))
        stored_globals = bool(interface and program.constants and all(c.name for c in program.constants))
        if stored_globals:
            try:
                for constant in program.constants:
                    storage = getattr(constant.type, 'module_storage', None)
                    if storage is None:
                        raise ValueError('literal type has no module storage contract')
                    storage('__check_module_storage')
            except ValueError:
                stored_globals = False
        if not stored_globals:
            typed = prepare_literals(source, program, interface)
            modules = (register(typed),) if stop_after == 'mir' else ()
            return FrontendResult(source, replace(program, typed_modules=(typed,), mir_modules=modules, stage=stop_after), ())

    phase = 'HIR'
    try:
        module = resolve(source, root, syntax, Checker, c_packages, interface)
        phase = 'THIR'
        typed = prepare(module, interface)
        constants = tuple(o.payload for b in typed.bodies for block in b.blocks for o in block.operations if o.kind == "constant")
        if stop_after == 'thir':
            return FrontendResult(source, Program(constants, module.submodules, (), typed.declarations,
                                                  (typed,), (), 'thir'), ())
        templates = tuple({'name': t.name, 'kind': 'alias' if type(t).__name__ == 'GenericAlias' else 'record', 'path': source.path,
                           'source': '\n' * (source.position(t.node.span.start)[0] - 1) + source.text[t.node.span.start:t.node.span.end] + '\n'}
                          for t in module.context.record_templates.values()
                          if not t.scope and not t.node.scope.development)
        templates += tuple({'name': entry.name, 'kind': 'function', 'path': source.path,
                            'source': '\n' * (source.position(entry.node.span.start)[0] - 1)
                                      + source.text[entry.node.span.start:entry.node.span.end] + '\n'}
                           for entry in module.context.callables
                           if (entry.templates or entry.variadic) and not entry.scope and not entry.node.scope.development)
        # Generic classes can depend on ordinary trait contracts too. Preserve
        # those signatures/defaults so consumers can check specializations.
        templates += tuple({'name': node.header[0].text, 'kind': 'trait', 'path': source.path,
                            'source': '\n' * (source.position(node.span.start)[0] - 1)
                                      + source.text[node.span.start:node.span.end] + '\n'}
                           for node in root.children
                           if getattr(node.provider, 'spelling', '') == 'trait'
                           and node.header[0].text not in module.context.record_templates
                           and not node.scope.development)
        templates += tuple({'name': node.header[0].text, 'kind': 'enum', 'path': source.path,
                            'source': source.text[node.span.start:node.span.end] + '\n'}
                           for node in root.children
                           if getattr(node.provider, 'spelling', '') == 'enum' and not node.scope.development)
        templates += tuple({'name': node.header[0].text, 'kind': 'class-grammars', 'path': source.path,
                            'source': '\n' * (source.position(node.span.start)[0] - 1)
                                      + source.text[node.span.start:node.span.end] + '\n'}
                           for node in root.children
                           if getattr(node.provider, 'spelling', '') == 'class'
                           and node.header[0].text not in module.context.record_templates
                           and any(getattr(child.provider, 'spelling', '') == 'grammar' for child in node.children))
        required_imports = tuple(node.header[0].text for node in root.children if getattr(node.provider, 'spelling', '') in ('import','from'))
        templates = tuple({**template, 'imports': required_imports} for template in templates)
        phase = 'MIR'
        mir = register(typed)
        constants = tuple({op.payload.identity: op.payload for body in mir.bodies
                           for block in body.blocks for op in block.operations if op.kind == 'constant'}.values())
        aliases = tuple({'name': name, 'target': alias.target}
                        for name, alias in module.context.function_aliases.items()
                        if not alias.prelude and not alias.scope and '.' not in name)
        return FrontendResult(source, Program(constants, module.submodules, mir.bodies, typed.declarations,
                                              (typed,), (mir,), 'mir'), (), templates, aliases)
    except Diagnostic as failure:
        return FrontendResult(source, Program((), ()), (failure,))
    except ValueError as failure:
        return FrontendResult(source, Program((), ()), (Diagnostic(phase, str(failure), source, root.span),))


import unittest
from py_compiler.syntax.recognition import Syntax


class FrontendTests(unittest.TestCase):
    def test_imported_generic_class_preserves_plain_trait_contract(self):
        producer = compile_source('lib.sev',
            'trait Sized:\n    def size(self: view Self) -> int\n'
            'class Box[T]: Sized\n    item: T\n'
            '    def size(self: view Self) -> int:\n        return 1\n'
            'def count[C: Sized](value: view C) -> int:\n    return value.size()\n', Syntax())
        self.assertFalse(producer.diagnostics, str(producer.diagnostics))
        self.assertTrue(any(t.get('kind') == 'trait' and t['name'] == 'Sized'
                            for t in producer.templates))
        package = {'package': {'name': 'collections'}, 'objects': ['collections.o'],
                   'exports': [], 'types': [], 'templates': producer.templates}
        consumer = compile_source('main.sev', 'import collections\n'
            'def work() -> int:\n    box = collections.Box[int](42)\n'
            '    return collections.count(box)\n', Syntax(),
            c_packages={'collections': package})
        self.assertFalse(consumer.diagnostics, str(consumer.diagnostics))

        # Import must preserve enforcement, not merely make the name resolve.
        invalid = compile_source('bad.sev', 'import collections\n'
            'class Wrong: collections.Sized\n'
            '    def size(self: view Self) -> bool:\n        return true\n', Syntax(),
            c_packages={'collections': package})
        self.assertTrue(invalid.diagnostics)
        self.assertIn('trait callable contract', str(invalid.diagnostics[0]))

    def test_imported_plain_trait_preserves_default_method(self):
        producer = compile_source('lib.sev',
            'trait Sized:\n    def size(self: view Self) -> int:\n        return 7\n'
            'class Box[T]: Sized\n    item: T\n', Syntax())
        self.assertFalse(producer.diagnostics, str(producer.diagnostics))
        package = {'package': {'name': 'collections'}, 'objects': ['collections.o'],
                   'exports': [], 'types': [], 'templates': producer.templates}
        consumer = compile_source('main.sev', 'import collections\n'
            'def work() -> int:\n    box = collections.Box[int](42)\n'
            '    return box.size()\n', Syntax(), c_packages={'collections': package})
        self.assertFalse(consumer.diagnostics, str(consumer.diagnostics))

    def test_primitives_to_hir_and_later_keyword_diagnostic(self):
        result = compile_source("x.sev", "small: i8 = -128\nletter = '😀'\nbytes = 4B\nif true:\n", Syntax())
        self.assertEqual([c.type.name for c in result.program.constants], ["i8", "char", "byte"])
        self.assertEqual(len(result.diagnostics), 1)
        self.assertEqual(result.diagnostics[0].stage, "parser")


class GenericLibraryInterfaceTests(unittest.TestCase):
    def test_named_generic_alias_survives_a_library_interface(self):
        producer = compile_source('alias.sev',
            'class Box[T]:\n    value: T\n'
            'Box[T:Element] as Stored[Element]\n', Syntax())
        self.assertFalse(producer.diagnostics, str(producer.diagnostics))
        package = {'package': {'name': 'alias'}, 'objects': ['alias.o'],
                   'exports': [], 'types': [], 'templates': producer.templates}
        consumer = compile_source('consumer.sev',
            'from alias import Stored\ndef work() -> int:\n'
            '    value = Stored[Element:int](42)\n    return value.value\n',
            Syntax(), c_packages={'alias': package})
        self.assertFalse(consumer.diagnostics, str(consumer.diagnostics))

    def test_tuple_annotation_keeps_string_key_and_value_types(self):
        source = ('def key(pair: (string,string)) -> string:\n'
                  '    result = copy pair[0]\n    return result\n'
                  'def work() -> string:\n'
                  '    pair: (string,string) = ("name","Alice")\n'
                  '    return key(pair)\n')
        result = compile_source('pairs.sev',source,Syntax())
        self.assertFalse(result.diagnostics,str(result.diagnostics))
        invalid = compile_source('wrong-key.sev',source +
            'def bad():\n    pair: (string,string) = (42,"Alice")\n',Syntax())
        self.assertTrue(invalid.diagnostics)
