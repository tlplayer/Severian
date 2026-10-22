"""Expected runtime failure boundary, retaining its lexical execution scope."""
from py_compiler.syntax.generic.block import BlockProvider


class Throws(BlockProvider):
    spelling = 'throws'
    required_scope = 'test'

    def parse_header(self, items):
        if len(items) > 1 and items[1].text == '(':
            return tuple(items[1:-1] if items[-1].text == ':' else items[1:])
        return super().parse_header(items)

    def has_body(self, node):
        return node.tokens[-1].text == ':'

    def expand(self, header, syntax):
        if header and header[0].text == '(':
            from py_compiler.syntax.grammar.expansion import ScopedSyntax, GrammarImport
            imported = GrammarImport('throws-pattern', (ErrorPattern(),))
            return ScopedSyntax(syntax, (imported,)), ()
        return syntax, ()

    def lower(self, node, cfg, env, nodes, index):
        if node.header and node.header[0].text == '(':
            return self.capture(node, cfg, env)

        if getattr(cfg, 'expected_failure_node', None) == node.identity:
            cfg.scope(node.children, dict(env))
            cfg.flow.end_scope(cfg.live_binding_ids(env))
            return 1
        import ast
        from py_compiler.hir.thir.src.model import Terminator
        from py_compiler.mir.ownership.flow import FlowState
        if len(node.header) != 1 or node.header[0].kind != 'STRING':
            raise ValueError('throws requires a quoted expected diagnostic')
        bindings = tuple(b for b in env.values() if b.identity not in cfg.storage and b.identity not in cfg.flow.moved)
        caught = cfg.block(tuple(b.type for b in bindings))
        handler = {'block':caught, 'bindings':bindings, 'message':ast.literal_eval(node.header[0].text),
                   'flows':[], 'scope-depth':len(cfg.active_scopes)}
        if not hasattr(cfg, 'error_handlers'):
            cfg.error_handlers=[]
        initial_values,initial_flow = dict(cfg.values),cfg.flow.fork()
        cfg.error_handlers.append(handler)
        try:
            cfg.scope(node.children, dict(env))
            if cfg.current.terminator is None:
                cfg.current.terminator=Terminator('panic',message='expected error was not raised')
        finally:
            cfg.error_handlers.pop()
        cfg.current,cfg.values=caught,initial_values
        cfg.values.update({b.identity:v for b,v in zip(bindings,caught.parameters)})
        cfg.flow=FlowState.join(handler['flows']) if handler['flows'] else initial_flow
        cfg.flow.end_scope(cfg.live_binding_ids(env))
        return 1

    def capture(self, node, cfg, env):
        from py_compiler.syntax.grammar.expression import expression
        from py_compiler.syntax.complex.error import is_error
        from py_compiler.syntax.block.union import OptionalField
        from py_compiler.syntax.generic.storage import release_storage
        from py_compiler.mir.ownership.flow import Binding
        tokens = list(node.header[1:-1])
        depth, split = 0, None
        for index, token in enumerate(tokens):
            if token.text in ('(', '['):
                depth += 1
            elif token.text in (')', ']'):
                depth -= 1
            elif token.text == '->' and depth == 0:
                split = index
                break
        patterns = node.children
        if split is not None:
            expected = cfg.context.type(''.join(t.text for t in tokens[split + 1:]))
            tokens = tokens[:split]
            if patterns:
                raise ValueError('throws arrow form does not take a pattern block')
        else:
            if len(patterns) != 1 or not isinstance(patterns[0].provider, ErrorPattern):
                raise ValueError('throws block requires one error pattern')
            expected = cfg.context.type(patterns[0].header[0].text)
        if not is_error(expected):
            raise ValueError('throws requires an Error type')
        term = expression(tokens)
        previous = getattr(cfg, 'captured_error_expression', None)
        cfg.captured_error_expression = term
        try:
            value = cfg.expr(term, env)
        finally:
            cfg.captured_error_expression = previous
        if value is None:
            raise ValueError('throws requires a typed error result')
        if value.type == expected:
            payload = value
        elif value.type.family == 'union' and expected in value.type.variants:
            payload = OptionalField(value, value.type.variants.index(expected), 'value', expected).read(cfg, node.span)
        else:
            raise ValueError('throws expected type is not in the callable result')
        if patterns:
            pattern = patterns[0]
            names = [token for token in pattern.header[2:-1] if token.text != ',']
            fields = expected.declaration.fields
            if len(names) != len(fields):
                raise ValueError('error pattern must bind every payload field')
            local = dict(env)
            for token, (field, _) in zip(names, fields):
                if token.kind != 'IDENTIFIER' or token.text in local:
                    raise ValueError('error pattern requires fresh binding names')
                field_value = cfg.context.provider_by_name[expected.name].field_place(payload, field, cfg, node.span).read(cfg, node.span)
                binding = Binding(pattern.identity + ':' + token.text, token.text, field_value.type, False, 'view', token.span)
                local[token.text] = binding
                cfg.body.bindings.append(binding)
                cfg.values[binding.identity] = field_value
            cfg.scope(pattern.children, local)
        if value.identity in cfg.owned_results:
            release_storage(value, cfg, node.span)
        return 1


class ErrorPattern(BlockProvider):
    spelling = ''
    required_scope = 'test'

    def recognize(self, window):
        from py_compiler.syntax.generic.grammar import Match
        tokens = window.tokens
        if (len(tokens) >= 4 and tokens[0].kind == 'IDENTIFIER'
                and tokens[1].text == '(' and tokens[-2].text == ')' and tokens[-1].text == ':'):
            return Match(self, window, window.end, {'header': tokens})
        return None

    def parse_header(self, items):
        return tuple(items[:-1])
