Title: Module 

Status: Draft | Accepted | Implementing | Implemented | Rejected | Superseded
Type: Language | Compiler | Runtime | Tooling | Package | Interop | Process
Authors:
Created: YYYY-MM-DD
Target:
Supersedes:
Superseded by:


## Appendix

Package: Versioned collection of Libraries, dependencies, interfaces, and artifacts.
Library: Collection of modules available to the compiler.
Module: Named collection of submodules sharing a namespace and public interface.
Submodule: Independently analyzable unit containing one or more blocks.
Block: see block sip
Scope: Region governing symbol visibility and resolution.
Interface: Public declarations and contracts exposed by a module.
Implementation: Definitions providing the behavior of declared symbols.
Export: Declaration made accessible outside a module.
Import: Reference to declarations from another module or package.
Dependency: Relationship between compilation atom or symbols.
Compilation atom: Unit that can be compiled independently.
Artifact: Compiled output such as an object file, shared library, or executable.
Realization: Concrete implementation generated from a generic declaration.
SCC: Strongly connected component within a dependency graph.
ABI: Binary interface governing data layout, symbol linkage, and calling conventions.
Module identity: Stable identifier distinguishing a module from other modules.
Symbol identity: Stable identifier distinguishing a declaration from other declarations.
Interface hash: Fingerprint identifying a module's externally observable compilation contract.
Implementation hash: Fingerprint identifying a compilation unit's implementation.

## Goal(s)

Hierarchy: Represent source code as library → module → submodule → block → .
Independence: Modules must not depend on source-file boundaries for their identity.
Namespaces: Modules define namespaces for symbol declarations and resolution.
Visibility: Support public exports and private implementations.
Resolution: Resolve symbols through local, prelude, submodule, module, package, and external scopes.
Dependencies: Maintain a graph of dependencies between modules, submodules, symbols, and generic realizations.
Compilation: Allow submodules to compile independently when dependencies permit.
Incremental builds: Recompile only units invalidated by relevant changes.
Interfaces: Allow consumers to compile against module interfaces without implementation source.
Generics: Resolve and specialize generic declarations across module boundaries.
Cycles: Detect cyclic dependencies and distinguish resolvable declaration cycles from invalid dependencies.
Linking: Produce object files, static libraries, shared libraries, and executables.
Interoperability: Support linking Severian modules with compatible external languages and ABIs.
Determinism: Produce consistent symbol identities, dependency graphs, and build outputs independent of discovery order.

## Examples

```py
# package: math
# module: math.vector

class Vector[T]:
    data: pointer[T]
    size: int

    def length(self) -> int:
        return self.size

    def get(self, index: int) -> T:
        return self.data[index]


def add[T](a: Vector[T], b: Vector[T]) -> Vector[T]:
    ...


def multiply[T](a: Vector[T], b: Vector[T]) -> Vector[T]:
    ...
```

```sh
package random
└── module random.choice
    ├── generic definition choice[C]
    ├── submodule realization choice[list[string]]
    │   ├── substitution: C = list[string]
    │   └── block class choice
    │       ├── block length -> int
    │       └── block get(index: int) -> string
    └── submodule realization choice[list[int]]
        ├── substitution: C = list[int]
        └── block class choice
            ├── block length -> int
            └── block get(index: int) -> int
```

For this fixture, `T` selects the input type and `F` selects the callable. Omitting `F` resolves it to `simple_hash` before constructing the realization key.

```text
package geometry
└── module hash
    ├── definition hash[T, F=simple_hash](hashable: T) -> int
    ├── submodule hash[int, simple_hash]
    │   └── block function hash(hashable: int) -> int
    ├── submodule hash[float, simple_hash]
    │   └── block function hash(hashable: float) -> int
    └── submodule hash[char, simple_hash]
        └── block function hash(hashable: char) -> int
```

Each realization must resolve and type-check its corresponding call:

```text
hash[int, simple_hash]   -> simple_hash(int)   -> int
hash[float, simple_hash] -> simple_hash(float) -> int
hash[char, simple_hash]  -> simple_hash(char)  -> int

missing compatible simple_hash implementation -> diagnostic
```

These are three `hash` realizations. Their `simple_hash` dependencies may require additional realizations or reuse existing implementations.

For the **eventual static-library output**, assume the target explicitly requests all three realizations, uses one Atom per `hash` realization, and obtains `simple_hash` from a compatible dependency. Saved lowered IR is enabled. Object filenames below are readable examples; metadata carries canonical identities.

```text
hash/package.pkg/
├── package.pkgi/
│   ├── index.json
│   └── severian/<build-id>/
│       └── lib.sevi
├── metadata/
│   ├── realizations/
│   │   └── <build-id>.json
│   ├── symbols/
│   │   └── <build-id>.json
│   ├── dependencies/
│   │   └── <build-id>.json
│   └── layouts/
│       └── <abi-id>.json
├── artifacts/<target-triple>/<build-id>/
│   ├── object/
│   │   ├── hash_int_simple_hash.o
│   │   ├── hash_float_simple_hash.o
│   │   └── hash_char_simple_hash.o
│   ├── archive/
│   │   └── hash.a
│   └── ir/
│       ├── hash_int_simple_hash.mlirbc
│       ├── hash_float_simple_hash.mlirbc
│       └── hash_char_simple_hash.mlirbc
├── build/<build-id>/
│   └── <static-library-target>/
├── cache/
├── debug/
│   └── build/
│       ├── log.txt
│       ├── error.txt
│       └── warn.txt
└── source/
    ├── package.json
    ├── package.lock
    └── src/
        ├── lib.sev
        └── hash/
            └── mod.sev
```

This follows [SIP-0003](/home/tplayer/Documents/Severian/docs/sip/0003-Packages.md): `build/` holds release target outputs; `cache/` holds temporary artifacts. Optional outputs are absent. `debug/test/<build-id>/` appears when tests run; profiling and coverage outputs appear when requested.

For **this pass**, stop after registering the typed submodules in MIR. If that checkpoint writes package metadata, mark it non-linkable:

```text
metadata/realizations/<build-id>.json
  module: canonical geometry.hash identity
  realizations:
    - source submodule + {T: canonical int,   F: canonical simple_hash}
    - source submodule + {T: canonical float, F: canonical simple_hash}
    - source submodule + {T: canonical char,  F: canonical simple_hash}
  stage: typed-submodules-registered       # Proposed metadata field
  linkable: false                         # Proposed metadata field
  objects: []
```

Package-layout acceptance tests, expressed as pseudocode:

```text
test "typed submodules reach MIR":
    selected roots = the three hash calls
    assert hash realization count == 3
    assert all retain source identities and resolved callable contracts
    assert hash[int](42) and hash[int, simple_hash](42) share a realization
    assert no ReadyMir
    assert no c objects or archive recorded for this build

test "completed static target matches package layout":
    assert interface index references existing lib.sevi
    assert realization records reference existing Atoms and object paths
    assert dependencies identify each selected simple_hash implementation
    assert libgeometry.a contains the three hash objects
    assert source snapshot matches the manifest and lock
    assert every recorded artifact path resolves inside package.pkg
```

The three calls inside `test` select realizations for a test request. A normal library build needs its own selection/export roots to guarantee those same three outputs.
## Problem(s)

Examples of corner cases

## Tests

Blocks of severian code to test behavior


