from py_compiler.syntax.operator.operation import PrefixOperator


class Drop(PrefixOperator):
    def __init__(self):
        super().__init__("drop")

    def matches(self, node):
        return node.tokens[0].text == "drop"

    def lower(self, node, cfg, env, local_names):
        items = list(node.tokens[1:])
        from py_compiler.syntax.grammar.expression import expression
        term = cfg.capture_expression(expression(items))
        if term.kind == 'index':
            from py_compiler.syntax.generic.place import analyze_target
            from py_compiler.syntax.generic.storage import AddressPlace
            place, owner, _, _ = analyze_target(term, cfg, env).lower(cfg, env, node.span)
            with AddressPlace(place, owner).capture(cfg, 'borrow'):
                destroy = getattr(place, 'destroy', None)
                if destroy:
                    destroy(cfg, node.span)
                else:
                    from py_compiler.syntax.function.calls import literal_value
                    zero = 0.0 if place.type.family == 'float' else 0
                    place.write(cfg, literal_value(zero, place.type, cfg, node.span), node.span)
            return
        if len(items) == 3 and items[0].text == '(' and items[-1].text == ')':
            items = items[1:-1]
        if len(items) != 1 or items[0].text not in env:
            raise ValueError("drop requires a visible binding")
        binding = env[items[0].text]
        if callable(getattr(binding.type, 'release', None)):
            binding.type.require_context(cfg)
            if binding.ownership in ('view', 'borrow'):
                raise ValueError('cannot drop memory through a view or borrow')
            cfg.flow.read(binding)
            cfg.cleanup_memory((binding,), node.span)
        else:
            cfg.flow.drop(binding)
