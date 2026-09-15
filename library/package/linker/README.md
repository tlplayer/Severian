# package.linker

Native linking in ordinary Severian source, independently of package resolution
and compiler-owned artifact layouts. Declare package name `package.linker` with
an import alias such as `linker`.

```sev
import linker

object = linker.input("build/math.o", target="host")
plan = linker.plan_inputs("host", "static-archive", "artifacts/math.a",
                          "ar", [object], incremental=true)
result = linker.execute_link(plan)
```

`LinkInput` records a canonical path, input kind, SHA-256 checksum, and optional
target. Nonempty input targets must match the plan target. This validates the
declared target contract; it does not decode object headers to infer an ABI.

`plan_inputs(target, kind, output, tool, inputs, options=[], incremental=false)`
constructs commands for Unix `ar` and compiler drivers. Kinds are
`static-archive`, `shared-library`, and `executable`. Input order is preserved;
place consuming archives before their providers. `plan_link` accepts existing
clang, llvm-ar, lld-link, and ld64 command vectors for compatibility.

Execution validates inputs before and after running the tool, locks the output,
links to a fresh file in the destination filesystem, and renames the completed
file into place. Failed commands and commands that omit their output preserve
an existing artifact. Fresh archives cannot retain obsolete members. Successful
links write `<output>.link.json` with arguments, target, input hashes, linker
identity, working directory, and output checksum. Package records use JSON.

Opt-in incremental reuse requires an exact record and verified output digest.
The caller must provide every external input and a hermetic toolchain/environment;
implicit headers, `-l` libraries, linker scripts, and response-file contents are
not recursively discovered. Raw command plans always execute. System hashing
and tool discovery currently require `sha256sum` and `which` on the build host.
Cross-target compatibility commands run those target tools on that host.

The primary artifact is published atomically. The JSON receipt is a subsequent
rename, so interruption between them forces a rebuild. Auxiliary outputs such
as Windows import libraries and PDBs remain tool-managed and are not covered by
the primary-output transaction.

See the [diamond dependency example](../../../docs/examples/10-building/02-linker-dependency/README.md)
for actual C objects, static archives, an executable, and JSON link records.

Build, then test (tests need `cc` and `ar`):

```sh
sev_rust build library/package/linker
sev_rust test library/package/linker
```
