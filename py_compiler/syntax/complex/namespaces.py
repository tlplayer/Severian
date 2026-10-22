"""Lexical operator selection never installs an extension as a global operator."""

def selected_operator(cfg,node,env):
    annotations=[a for a in getattr(cfg,'operator_namespaces',()) if node.token.text in a.arguments]
    if not annotations:
        return None
    if len(annotations)>1:
        raise ValueError('ambiguous operator namespace for '+node.token.text)
    annotation=annotations[0]
    opname='__operator_'+'_'.join(str(ord(c)) for c in node.token.text)
    traits=[name for name,method in getattr(cfg.context,'operator_traits',{}).get(annotation.name,()) if method==opname]
    if len(traits)>1:
        raise ValueError('ambiguous operator trait namespace')
    from py_compiler.syntax.function.calls import dispatch,select,invoke
    if traits:
        return dispatch(None,opname,node.operands,cfg,env,node.token.span,traits[0])
    if annotation.name in cfg.context.package_imports:
        entries = cfg.context.lookup_functions(annotation.name + '.' + opname, cfg.declaration_scope)
        if entries:
            entry, values = select(entries, node.operands, cfg, env)
            return invoke(entry, values, cfg, node.token.span)
    left=cfg.expr(node.operands[0],env)
    entries=[e for e in cfg.context.functions.get(left.type.name+'.'+opname,()) if getattr(e,'operator_namespace',None)==annotation.name]
    if not entries:
        raise ValueError('namespace has no applicable operator '+node.token.text)
    entry,values=select(entries,(node.operands[1],),cfg,env)
    return invoke(entry,values,cfg,node.token.span,left)


def namespace_syntax(syntax, annotations):
    """Selected word operators are lexical to the annotated declaration.

    Existing punctuation keeps its established binding strength. New word
    operators bind like multiplication; their callable belongs to the library.
    """
    if not annotations:
        return syntax
    from py_compiler.syntax.catalog import operator_bindings
    from py_compiler.syntax.grammar.expansion import ScopedSyntax
    from py_compiler.syntax.generic.owned import OwnedGrammar
    from functools import cached_property
    known = operator_bindings(syntax)
    symbols = {}
    for annotation in annotations:
        # Named decorator arguments (such as @c(symbol="name")) are not packs.
        if not all(word.isidentifier() or word in known for word in annotation.arguments):
            continue
        for word in annotation.arguments:
            if word.isidentifier() and word not in known:
                symbols.setdefault(word, annotation)
    if not symbols:
        return syntax

    class NamespaceOperator(OwnedGrammar):
        associativity = 'left'
        comparison = False

        def __init__(self, spelling, annotation):
            super().__init__(annotation, 'F.operator', (spelling,))
            self.spelling = spelling
            self.precedence = known['*'].precedence
            self.lexical_namespace = annotation.name

        def recognize(self, window):
            from py_compiler.syntax.generic.grammar import Match
            if len(window.tokens) == 1 and window.tokens[0].text == self.spelling:
                return Match(self, window, window.end, {})
            return None

    class NamespaceSyntax(ScopedSyntax):
        @cached_property
        def grammar_catalog(self):
            from py_compiler.syntax.catalog import grammars
            return (*grammars(self.parent), *(NamespaceOperator(word, annotation) for word, annotation in symbols.items()))

    return NamespaceSyntax(syntax, ())


import unittest


class NamespaceOperatorTests(unittest.TestCase):
    def compile(self, source):
        from py_compiler.frontend.src.lib import compile_source
        from py_compiler.syntax.recognition import Syntax
        return compile_source('word-operators.sev', source, Syntax(include_tests=True))

    def test_word_operator_uses_the_selected_extension_and_precedence(self):
        source = '''class Number:
    value: int
@arithmetic
extend Number:
    operator blend(other: Number) -> int:
        return value + other.value
@arithmetic(blend)
def combine(left: Number, right: Number) -> int:
    with Number(0) as temporary:
        if left blend right > 0:
            return 1 + left blend right * 2
    return 0
'''
        # Multiplication associates left-to-right with the selected word symbol.
        result = self.compile(source)
        self.assertFalse(result.diagnostics, str(result.diagnostics))

    def test_selection_does_not_reserve_generic_parameter_names(self):
        result = self.compile('def identity[X](value: X) -> X:\n    return value\n'
                              'test:\n    assert(identity(42) == 42)\n')
        self.assertFalse(result.diagnostics, str(result.diagnostics))

    def test_selected_word_survives_nested_grammar_imports(self):
        from py_compiler.syntax.recognition import Syntax
        from py_compiler.syntax.block.mlir import NamespaceAnnotation
        from py_compiler.syntax.grammar.expansion import expand
        from py_compiler.frontend.source.source import SourceFile
        from py_compiler.frontend.lexer.lexer import lex
        from py_compiler.syntax.grammar.expression import expression
        selected = namespace_syntax(Syntax(), (NamespaceAnnotation('arithmetic', ('blend',)),))
        nested, _ = expand(selected, ('test',))
        tokens = [t for t in lex(SourceFile('nested-word.sev', 'left blend right'), nested)
                  if t.kind not in ('EOF', 'NEWLINE')]
        self.assertEqual(expression(tokens).token.text, 'blend')

    def test_unknown_selected_operator_has_no_implicit_implementation(self):
        result = self.compile('@missing(combine)\ndef work() -> int:\n    return 1 combine 2\n')
        self.assertTrue(result.diagnostics)
        self.assertIn('no applicable operator', str(result.diagnostics))

    def test_word_operator_does_not_escape_its_declaration(self):
        result = self.compile('@missing(combine)\ndef unused():\n    return\n'
                              'def work() -> int:\n    return 1 combine 2\n')
        self.assertTrue(result.diagnostics)
        self.assertIn('unsupported expression suffix', str(result.diagnostics))
