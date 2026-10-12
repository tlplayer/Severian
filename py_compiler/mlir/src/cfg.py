"""Lower checked SSA execution bodies to func, cf and arith dialects."""
from hashlib import sha256
from dataclasses import replace


def name(value):
    return f"%v{value.identity}"


def temporary_suffix(operation):
    """Function-local names identify emitted operations, never source locations."""
    if operation.result is not None:
        return str(operation.result.identity)
    if operation.lowering_position is None:
        raise ValueError('MLIR effect operation requires a lowering position')
    block, position = operation.lowering_position
    return f'effect_{block}_{position}'


def source_location(body, span=None):
    from py_compiler.mlir.src.lib import quoted
    source = body.source_snapshot
    if source is None:
        return ''
    span = span or body.source_span
    line, column = source.position(span.start) if span else (1, 1)
    return f' loc({quoted(source.debug_path or source.path)}:{line}:{column})'


def located_operations(lines, location):
    # Region-bearing operations carry their location after the closing brace.
    return [line if line.rstrip().endswith('{') or line.lstrip().startswith('^') else line + location for line in lines]


def render_body(body):
    from py_compiler.mlir.src.lib import symbol_ref
    body.verify()
    first = body.blocks[0]
    arguments = ", ".join(f"{name(v)}: {v.type.mlir}" for v in first.parameters)
    result = " -> " + (body.result_layout or body.result_type.mlir) if body.result_type.mlir else ""
    lines = [f"  func.func {symbol_ref(body.name)}({arguments}){result} {{"]
    for region in getattr(body, 'memory_regions', ()):
        lines.extend([f'    %{region.name}_size = arith.constant {region.size} : index',
                      f'    %{region.name} = memref.alloc(%{region.name}_size) {{alignment = {region.alignment} : i64}} : memref<?xi8>'])
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
        for position, operation in enumerate(block.operations):
            # An expansion may repeat both the span and operands. The registered
            # block/operation coordinates distinguish each emission without
            # changing MIR identities, source locations, or Atom contracts.
            operation = replace(operation, lowering_position=(block.identity, position))
            value = operation.result
            operands = operation.operands
            prefix = f"    {name(value)} = " if value is not None else "    "
            if operation.atom is not None:
                lines.extend("    " + line for line in located_operations(
                    operation.atom.render_operation(operation), source_location(body, operation.span)))
                continue
            if hasattr(operation.payload, "render_operation"):
                lines.extend("    " + line for line in located_operations(
                    operation.payload.render_operation(operation), source_location(body, operation.span)))
                continue
            if operation.kind == "constant":
                symbol = "__sev_constant_" + sha256(operation.payload.identity.encode()).hexdigest()
                lines.append(prefix + f"func.call @{symbol}() : () -> {value.type.mlir}")
            else:
                raise ValueError(f"missing MLIR operation provider {operation.kind}")
        terminator = block.terminator
        start = len(lines)
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
            for region in getattr(body, 'memory_regions', ()):
                lines.append(f'    memref.dealloc %{region.name} : memref<?xi8>')
            if value and body.result_layout:
                aggregate = f'%return_record_{block.identity}'
                lines.append(f'    {aggregate} = llvm.load {name(value)} : !llvm.ptr -> {body.result_layout}')
                lines.append(f'    func.return {aggregate} : {body.result_layout}')
            else:
                lines.append(f"    func.return {name(value)} : {value.type.mlir}" if value else "    func.return")
        span = terminator.span or (block.operations[-1].span if block.operations else body.source_span)
        lines[start:] = located_operations(lines[start:], source_location(body, span))
    return "\n".join(lines + ["  }" + source_location(body)])


import unittest


class DebugLocationTests(unittest.TestCase):
    def test_repeated_effects_keep_locations_and_have_unique_temporaries(self):
        import re
        from py_compiler.frontend.source.source import SourceFile
        from py_compiler.mir.cfg.cfg import Body, ExecutionBlock, Operation, Value, Edge, Terminator
        from py_compiler.syntax.complex.array import ArrayType, MemoryOperation
        from py_compiler.syntax.recognition import Syntax
        from py_compiler.mlir.src.lib import verify_native
        source = SourceFile('repeated-store.sev', 'storage[index] = value\n')
        types = Syntax().types
        arguments = (Value(0, ArrayType(types['int'])), Value(1, types['index']), Value(2, types['int']))
        implementation = MemoryOperation('store')
        operation = Operation('memory-store', None, arguments, implementation, source.span(0, 22))
        operation = replace(operation, atom=implementation.atom(operation))
        body = Body('repeated-store', 'repeated_store', source.path, types['absent'])
        body.source_snapshot = source
        body.source_span = operation.span
        body.blocks = [ExecutionBlock(0, arguments, [operation, operation],
                                      Terminator('jump', (Edge(1),))),
                       ExecutionBlock(1, (), [operation], Terminator('finish'))]
        text = render_body(body)
        definitions = re.findall(r'^\s*(%[A-Za-z_0-9]+)\s*=', text, re.MULTILINE)
        self.assertEqual(len(definitions), len(set(definitions)))
        self.assertEqual(text.count('memref.store'), 3)
        self.assertEqual(text, render_body(body))
        self.assertTrue(all('loc("repeated-store.sev":1:1)' in line
                            for line in text.splitlines() if 'memref.store' in line))
        self.assertIsNone(operation.lowering_position)
        verify_native('module {\n' + text + '\n}')

    def test_region_headers_and_block_labels_are_not_operations(self):
        lines = ['scf.while (%arg = %initial) : (i64) -> i64 {',
                 '^bb0(%arg: i64):', 'scf.yield %arg : i64', '}']
        rendered = located_operations(lines, ' loc("example.sev":3:4)')
        self.assertEqual(rendered[:2], lines[:2])
        self.assertTrue(rendered[2].endswith('loc("example.sev":3:4)'))
        self.assertTrue(rendered[3].endswith('loc("example.sev":3:4)'))

    def test_function_calls_and_returns_keep_source_coordinates(self):
        from py_compiler.frontend.src.lib import compile_source
        from py_compiler.syntax.recognition import Syntax
        source = 'def first() -> int:\n    return 1\ndef second() -> int:\n    value = first()\n    return value\n'
        result = compile_source('logical.sev', source, Syntax(), debug_path='/tmp/source space.sev')
        self.assertFalse(result.diagnostics, str(result.diagnostics))
        body = next(body for body in result.program.bodies if body.declaration == 'second')
        rendered = render_body(body)
        call = next(line for line in rendered.splitlines() if 'func.call' in line)
        returned = next(line for line in rendered.splitlines() if 'func.return' in line)
        self.assertIn('loc("/tmp/source space.sev":4:', call)
        self.assertIn('loc("/tmp/source space.sev":5:5)', returned)
        self.assertTrue(rendered.endswith('loc("/tmp/source space.sev":3:1)'))
