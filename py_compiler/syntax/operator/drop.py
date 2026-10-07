from py_compiler.syntax.operator.operation import PrefixOperator


class Drop(PrefixOperator):
    def __init__(self):
        super().__init__("drop")

    def matches(self, node):
        return node.tokens[0].text == "drop"

    def lower(self, node, cfg, env, local_names):
        items = list(node.tokens[1:])
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
