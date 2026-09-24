# Compiler graph resolution

This dependency-free crate owns SCC decomposition, condensation, and a fixed-point
work queue. It has no syntax, HIR, package, MIR, or backend dependency.

`Graph<X>` preserves concrete node identities. Edges point from consumers to
providers and retain their `Requirement`. `condensation` returns dependency-first
units and the DAG between them. Members within an SCC have no dependency order.

`resolve_graph` processes each SCC to convergence. The caller owns its facts and
must monotonically join new information from a finite domain, reporting a change
only when knowledge grows. There is no path repetition or iteration limit.
`unresolved` preserves unsatisfied edges and requirement kinds for owner-specific
diagnostics after convergence; the engine does not classify speculative lookups
as required language facts.

Production users:

- `frontend/modules` constructs the package graph from discovered source imports
  and groups cyclic packages into resolution units. Scheduling preserves source
  discovery order inside a unit, including the requested root.
- `frontend/semantic/package/imports` discovers name requests after collecting
  declarations, constructs its own dependency graph, and joins symbol resolutions
  using the shared engine. HIR retains declaration identities, overload merging,
  visibility, and diagnostics. Package planning does not resolve names.

Runtime initialization cycles are still checked separately. This does not yet
replace source-compiler block discovery or source-package artifact planning.
