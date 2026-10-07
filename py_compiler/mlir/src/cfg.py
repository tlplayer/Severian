"""Lower checked SSA execution bodies to func, cf and arith dialects."""
from hashlib import sha256


def name(value):
    return f"%v{value.identity}"


def render_body(body):
    from py_compiler.mlir.src.lib import symbol_ref
    body.verify()
    first = body.blocks[0]
    arguments = ", ".join(f"{name(v)}: {v.type.mlir}" for v in first.parameters)
    result = " -> " + body.result_type.mlir if body.result_type.mlir else ""
    lines = [f"  func.func {symbol_ref(body.name)}({arguments}){result} {{"]
    def edge(e):
        if not e.arguments:
            return f"^bb{e.target}"
        values = ", ".join(name(v) for v in e.arguments)
        types = ", ".join(v.type.mlir for v in e.arguments)
        return f"^bb{e.target}({values} : {types})"
    for block in body.blocks:
        if block.identity:
            args = ", ".join(f"{name(v)}: {v.type.mlir}" for v in block.parameters)
            lines.append(f"  ^bb{block.identity}" + (f"({args})" if args else "") + ":")
        for operation in block.operations:
            value = operation.result
            operands = operation.operands
            prefix = f"    {name(value)} = " if value is not None else "    "
            if operation.atom is not None:
                lines.extend("    " + line for line in operation.atom.render_operation(operation))
                continue
            if hasattr(operation.payload, "render_operation"):
                lines.extend("    " + line for line in operation.payload.render_operation(operation))
                continue
            if operation.kind == "constant":
                symbol = "__sev_constant_" + sha256(operation.payload.identity.encode()).hexdigest()
                lines.append(prefix + f"func.call @{symbol}() : () -> {value.type.mlir}")
            else:
                raise ValueError(f"missing MLIR operation provider {operation.kind}")
        terminator = block.terminator
        if terminator.kind == "jump":
            lines.append("    cf.br " + edge(terminator.edges[0]))
        elif terminator.kind == "conditional":
            lines.append(f"    cf.cond_br {name(terminator.value)}, {edge(terminator.edges[0])}, {edge(terminator.edges[1])}")
        elif terminator.kind == "panic":
            from py_compiler.mlir.src.lib import quoted
            lines.append(f"    %panic_{block.identity} = arith.constant false")
            lines.append(f"    cf.assert %panic_{block.identity}, {quoted(terminator.message)}")
            lines.append("    llvm.unreachable")
        else:
            value = terminator.value
            lines.append(f"    func.return {name(value)} : {value.type.mlir}" if value else "    func.return")
    return "\n".join(lines + ["  }"])
