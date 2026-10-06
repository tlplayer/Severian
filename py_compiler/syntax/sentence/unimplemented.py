from py_compiler.mir.cfg.cfg import Terminator


class Unimplemented:
    def matches(self, node):
        return node.tokens[0].text == "unimplemented"

    def lower(self, node, cfg, env, local_names):
        if len(node.tokens) != 1:
            raise ValueError("unimplemented takes no operands")
        line, column = cfg.source.position(node.span.start)
        message = f"unimplemented at {cfg.source.path}:{line}:{column}"
        cfg.current.terminator = Terminator("panic", message=message)
