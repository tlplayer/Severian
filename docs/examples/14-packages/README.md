# Packages

Severian packaging follows a Cargo-like official tool model: one manifest, one
lockfile, one standard package manager, and one build/test/doc command family.

```text
package/
├── package.toml
├── sev.lock
├── src/
│   ├── lib.sev
│   └── main.sev
├── tests/
└── examples/
```

Workspaces use a manifest containing `[workspace]` and `members`. Packages use
`[package]`, optional `[lib]` and `[[bin]]` targets, `[dependencies]`, and
`[dev-dependencies]`. Path dependencies use the same explicit shape as Cargo:

```toml
[dependencies]
geometry = { path = "../geometry", version = "0.1.0" }
```

The manifest resolves package names to source roots. Source files still use the
readable import syntax `from geometry import Point`; imports do not download or
select dependency versions.

```sh
sev init
sev add geometry --path ../geometry --version 0.1
sev build
sev test
sev doc
sev publish
```

Version-only dependencies resolve from `SEVERIAN_REGISTRY`, are verified against
the registry SHA-256 record, cached below `SEVERIAN_HOME/packages`, and pinned in
`sev.lock`. Source imports only the dependency-table name; registry and cache
paths never appear in `.sev` files. `sev update` is the explicit operation that
moves a lock entry to a newer compatible version.
