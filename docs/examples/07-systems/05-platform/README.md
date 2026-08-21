# Platform

`platform` is the neutral description of the selected execution environment.
It exposes facts and capabilities needed by libraries without exposing ABI
implementation internals or encouraging string comparisons against target
triples.

Target facts include architecture, operating-system family, pointer width,
endianness, and supported devices. Capabilities describe available services
such as processes, filesystem access, environment mutation, networking, and
dynamic loading.

ABI layout is derived from the selected target by the ABI boundary. Source code
may query stable layout operations when it genuinely needs them, but backends do
not resolve semantic types through `platform`.

Provider selection is explicit and testable. A package can substitute a
deterministic provider without pretending to run on a different target.
