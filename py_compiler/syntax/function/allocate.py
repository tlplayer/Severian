"""allocate[T](N) owns allocation syntax and returns array[T]."""
from py_compiler.syntax.complex.array import ArrayType, MemoryOperation


class Allocate:
    def call(self, element, arguments, cfg, env, span):
        if element is None or len(arguments) != 1:
            raise ValueError('allocate[T] requires an integer element type and one element count')
        result = ArrayType(element)
        if not cfg.unsafe_depth:
            raise ValueError('raw allocation requires an unsafe block')
        from py_compiler.syntax.generic.owned import select
        count = select(cfg.context.type('index'), 'F.constructor', 'construct').expand(cfg, arguments, env)
        return cfg.emit('allocate-array', result, (count,), MemoryOperation('allocate'), span)
