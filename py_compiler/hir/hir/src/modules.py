"""Resolve syntax terms into declaration modules before executable MIR construction."""
from dataclasses import dataclass
from py_compiler.hir.hir.src.program import Submodule


@dataclass
class Module:
    identity: str
    source: object
    root: object
    context: object
    submodules: tuple


def resolve(source, root, syntax, builder):
    nodes = {}
    def collect(node):
        nodes[node.identity] = node
        for child in node.children:
            collect(child)
    collect(root)
    edges = {identity: set() for identity in nodes}
    # Index declaration captures without executing their declaration semantics.
    from py_compiler.syntax.generic.block import DeclarationProvider
    names = {}
    for node in nodes.values():
        if isinstance(node.provider, DeclarationProvider) and node.header and hasattr(node.header[0], 'text'):
            names.setdefault((node.parent, node.header[0].text), set()).add(node.identity)
    for identity, node in nodes.items():
        for token in node.tokens:
            if token.kind != 'IDENTIFIER':
                continue
            parent = node.identity
            while parent in nodes:
                candidates = names.get((parent, token.text))
                if candidates:
                    edges[identity].update(candidates)
                    break
                parent = nodes[parent].parent
        edges[identity].update(child.identity for child in node.children)
    # Tarjan partitions mutually dependent terms; independent components remain separate.
    active, indices, low, stack, components = set(), {}, {}, [], []
    def visit(identity):
        indices[identity] = low[identity] = len(indices)
        stack.append(identity)
        active.add(identity)
        for dependency in sorted(edges[identity]):
            if dependency not in nodes:
                continue
            if dependency not in indices:
                visit(dependency)
                low[identity] = min(low[identity], low[dependency])
            elif dependency in active:
                low[identity] = min(low[identity], indices[dependency])
        if low[identity] == indices[identity]:
            component = []
            while True:
                member = stack.pop()
                active.remove(member)
                component.append(member)
                if member == identity:
                    break
            components.append(tuple(sorted(component)))
    for identity in nodes:
        if identity not in indices:
            visit(identity)
    owners = {member: component[0] for component in components for member in component}
    submodules = tuple(Submodule(component[0], component, tuple(sorted({owners[d] for n in component for d in edges[n]
                       if d in owners and owners[d] != component[0]}))) for component in components)
    # Resolve the partitioned terms into their owning module before MIR construction.
    from py_compiler.hir.hir.src.declarations import DeclarationContext
    context = DeclarationContext(source, syntax, builder)
    context.declared_nodes = set()
    context.declare_nodes(root.children, ())
    context.resolve_contracts()
    return Module(source.identity, source, root, context, submodules)
