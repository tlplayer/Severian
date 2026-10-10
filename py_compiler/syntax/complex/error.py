"""Error owns its nominal type declaration and existing pointer representation."""
from py_compiler.syntax.complex.object import ObjectType, NamedDefinition


class ErrorType(ObjectType):
    def __init__(self):
        super().__init__('Error', 'error', '!llvm.ptr', declaration=NamedDefinition('Error'))

    def raise_failure(self, arguments, cfg, span):
        import ast
        from py_compiler.mir.cfg.cfg import Terminator
        if len(arguments) != 1 or arguments[0].kind != 'literal' or arguments[0].token.kind != 'STRING':
            raise ValueError('contract Error requires one string literal message')
        message = ast.literal_eval(arguments[0].token.text)
        line, column = cfg.source.position(span.start)
        cfg.current.terminator = Terminator(
            'panic', message=f'{message} at {cfg.source.path}:{line}:{column}')
