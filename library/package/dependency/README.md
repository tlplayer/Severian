# package.dependency

Indexed dependency graphs in Severian. This package has no filesystem or
compiler dependencies. Manifest discovery, version selection, and lock handling
remain in `package.resolve`; `package.dependency` handles the resolved graph.

Declare a dependency with package name `package.dependency` and an import alias
such as `dependency`:

```sev
import dependency

graph = dependency.graph(["app", "library"], [dependency.Edge(0, 1, "lib")])
assert(dependency.build_order(graph, [0]) == [1, 0])
assert(dependency.link_order(graph, [0]) == [0, 1])
assert(dependency.affected(graph, [1]) == [1, 0])
```

| API | Result |
| --- | --- |
| `graph(names, edges)` | Snapshot with forward and reverse edge indexes |
| `build_order(graph, roots=[])` | Dependencies before consumers; rejects cycles |
| `link_order(graph, roots=[])` | Consumers before providers for static linking |
| `closure(graph, roots, reverse=false)` | Reachable nodes, each once, roots first |
| `affected(graph, changed)` | Changed nodes and transitive consumers |

Edges retain aliases and development flags. Filter development dependencies
during resolution; the graph operates on every supplied edge. Empty roots mean
the whole graph for ordering and an empty result for closure/affected queries.
Node indices must be in range. Cycles include package names in the diagnostic.

Construction and each traversal take O(V + E) time and space, where V is the
number of nodes and E is the number of edges. Traversal uses explicit stacks
and queues, so deep dependency chains do not consume the call stack. Supplied
node and edge order determines reproducible results. Treat the returned graph
as immutable; reconstruct it after editing dependencies.

`affected` is a conservative candidate set. It does not assert that a private
implementation change alters a consumer's public interface.

Build, then test:

```sh
sev_rust build library/package/dependency
sev_rust test library/package/dependency
```
