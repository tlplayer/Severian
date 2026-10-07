"""deallocate consumes an owning integer-memory binding."""


class Deallocate:
    def call(self, element, arguments, cfg, env, span):
        if len(arguments) != 1 or arguments[0].kind not in ('name', 'reference'):
            raise ValueError('deallocate requires one owning binding')
        binding = cfg.lookup(arguments[0], env)
        if binding is None or not callable(getattr(binding.type, 'release', None)):
            raise ValueError('deallocate requires an owning array, not a raw pointer')
        if binding.ownership not in ('own', 'move', 'copy', 'mirror'):
            raise ValueError('cannot deallocate a view or borrow')
        if element is not None and binding.type.element != element:
            raise ValueError('deallocate element type does not match allocation')
        value = cfg.expr(arguments[0], env)
        cfg.flow.drop(binding)
        binding.type.release(value, cfg, span)
        return None
