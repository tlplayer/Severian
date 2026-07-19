# Packages

Severian packaging follows a Cargo-like official tool model: one manifest, one
lockfile, one standard package manager, and one build/test/doc command family.

```text
package/
├── Severian.toml
├── Severian.lock
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
sev build
sev test
sev doc
sev publish
```
