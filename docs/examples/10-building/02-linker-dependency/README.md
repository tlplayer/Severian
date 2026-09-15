# Dependency graphs and native linking

This example uses the `package.dependency` and `package.linker` libraries to
build a diamond: `app` uses `left` and `right`, both of which use `shared`.
Each dependency is built once. The final link places both consumers before the
shared archive, then runs the executable and checks its exit status.

From this directory, with `cc`, `ar`, `sha256sum`, and `which` installed:

```sh
sev_rust build --bin linker-dependency-example -o /tmp/linker-dependency-example
/tmp/linker-dependency-example
```

The runner leaves concrete artifacts here:

```text
package.pkg/
├── artifacts/host/dev/
│   ├── app.c, app.o
│   ├── left.c, left.o, left.a, left.a.link.json
│   ├── right.c, right.o, right.a, right.a.link.json
│   └── shared.c, shared.o, shared.a, shared.a.link.json
└── bin/host/dev/
    ├── diamond
    └── diamond.link.json
```

Output lock files are also retained for safe concurrent linking. Repeated runs
reuse unchanged archives after verifying their inputs, tool, and output hashes.
