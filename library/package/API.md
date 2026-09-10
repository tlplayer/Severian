`package` owns package operations in ordinary Severian source. The runnable
`sev_compiler` entry point imports this library; the Rust seed's existing CLI
remains a separate bootstrap implementation.

```sev
import package

project = package.open(".")
graph = package.resolve(project)
plan = package.plan(project, package.Build(profile="release"))
print(package.tree(graph))

compiler = package.Compiler("/path/to/sev_compiler", "/path/to/Severian")
result = package.build(project, compiler)
report = package.test(project, compiler)
```

The compiler boundary is a subprocess protocol for **one compilation unit**.
The library sends a versioned snapshot of package identities, manifests, roots,
and alias edges to `__compile-unit`; the compiler analyzes/emits that unit.
It does not invoke `sev build`, resolve dependencies a second time, or call the
Rust seed. This boundary works with the seed's current source-language support
and can later gain an in-process adapter without moving package semantics.

| Operation | API |
| --- | --- |
| Create/discover | `new(path, name="", library=false)`, `open(path=".")` |
| Inspect | `reference(text)`, `parse_manifest(text)`, `metadata(project)`, `tree(resolution)` |
| Resolve | `resolve(project, Resolve(mode="build", locked=false, registry=""))` |
| Plan/build | `plan(project, Build())`, `execute(plan, compiler)`, `build(project, compiler, Build())` |
| Check/test/run | `check(project, compiler, Check())`, `test(project, compiler, Test())`, `run(project, compiler, Run())` |
| Edit | `edit(project)`, `add(project, reference, alias="", development=false)`, `remove(project, alias)`, `update(project, alias)` |
| Distribute | `publish(project, compiler, Publish())`, `install(reference, compiler, Install())`, `run_reference(reference, compiler, Run())` |
| Archives | `read_archive(path)`, `write_archive(path, Archive(...))`, `extract(archive, new_directory)` |
| Select | `select(distribution_root, Requirements())` |
| Clean | `clean(project, Clean())` |

Request fields have defaults; empty platform/profile values inherit manifest
settings and then `host`/`dev`. `Build.target` selects a declared target;
`Build.platform` selects the compilation destination. Results expose concrete
artifacts and the plan that produced them. Fallible APIs return `T | Error`;
ordinary `=` propagates errors and `try`/`catch` can handle them.

```sev
transaction = package.edit(project)
transaction.add("geometry@2", alias="shapes")
transaction.remove("old-math")
project = transaction.commit()
```

Edits resolve before changing the manifest. A project lock serializes readers
and writers; a rollback journal lets the next open recover an interrupted pair
of replacements. Concurrent manifest/lock edits are rejected. This is a
cooperating-reader transaction, not a claim that two filesystem renames are one
atomic OS operation. TOML rendering preserves values but normalizes formatting
and does not preserve comments. Array tables retain their table kind.

Resolution keeps aliases on edges, canonicalizes path dependencies, includes
only root dev dependencies for tests/edits, and reports dependency chains.
Normal resolution does not rewrite `sev.lock`. Lock enforcement is opt-in.
With `locked=true`, registry selections stay pinned to recorded versions and resolved content must match the lock. Numeric latest, prefix and exact selectors use the filesystem registry layout
`packages/<name>/<version>` and `SEVERIAN_REGISTRY`. Archives retain the seed's
SEVPKG v1/v2 byte layout; publication uses v2. Installation defaults to
`~/.local/bin` and refuses to overwrite an existing command.

Test reports currently describe test executables, and `Test.filter` selects target/source names.

`package.pipeline` runs an ordered dependency graph of commands and preserves a
report even when steps fail:

```sev
steps = [
    package.Step("build", ["compiler", "build", "app.sev", "-o", "/tmp/app"], ".", stage="build"),
    package.Step("run", ["/tmp/app"], ".", ["build"], stage="run"),
]
report = package.pipeline(steps, "/tmp/app-reports")
assert(report.failed == 0 and report.blocked == 0)
```

Each invocation owns its report directory. `results.json` and `REPORT.md` are
updated after every step; separate files retain stdout and stderr. Failed steps
block their dependents while independent steps continue. Disabled steps count
as absent, and timeouts count as failures. Optional stdout/stderr fixture paths
check exact output. Stage coverage measures passing commands against eligible
commands, including blocked and pending commands in the denominator; it does
not claim instrumented line coverage. Dependencies must precede their consumers.
Commands run through the hosted process boundary with quoted arguments and a
per-step timeout (60 seconds by default).

Current limits are explicit: the source compiler emits host executables;
backend-artifact loading, general `.pkgi` generation/consumption, remote registry
transport/authentication, and container/network-policy execution are not
implemented. The selector reports unsupported alternatives. Source-free
libraries are rejected. Publication includes Severian source files and currently
requires separately published versions for path dependencies. Hosted filesystem,
hashing, Git and process primitives use the existing runtime and standard Unix
tools. The source compiler's existing language subset still limits which
application/library bodies it can compile; adding this package manager does not
make that compiler self-hosting.

Validation from the repository root:

```sh
package.pkg/debug/sev test test/validation/packages/api
package.pkg/debug/sev build sev_compiler
python3 test/validation/packages/source_package.py
```
