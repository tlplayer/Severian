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

## External requirements

`package.toml` is also the only package-controlled input to native and toolchain
installation. It describes intent and cannot contain installer code:

```toml
[package]
name = "tensor_rocm"
version = "0.1.0"

[dependencies]
tensor = "1.2"

[system]
cmake = ">=3.30"
rocm = ">=7.0"

[install.rocm]
publisher = "amd"
package = "rocm"
source = "vendor"
```

The source must be `vendor`; a package cannot provide a URL, command, shell
fragment, setup script, or privilege-escalation instruction. `install.sh`,
`setup.py`, PowerShell installers, and executable manifest hooks are rejected
before the package is cached or built.

`sev install --dry-run` resolves the plan without installing or changing the
external lock state. `sev install` verifies the publisher trust window,
namespace, HTTPS domain, Ed25519 signature, and SHA-256 artifact digest before
placing the artifact below `SEVERIAN_HOME/external`. HTTPS redirects are not
followed, so an allowlisted source cannot redirect installation to another
domain. Approval defaults to no. `sev install --locked` rejects any difference
from `sev.lock`, and `sev verify` rechecks the exact lock and installed bytes.

Trust configuration belongs to Severian, outside every package, under
`SEVERIAN_HOME/trust`. Packages cannot add or modify publishers. Inspect it with
`sev trust list` and `sev trust show <publisher>`. Ordinary builds preserve
external lock entries and never update or download them.

A Severian distribution or administrator provisions publishers at
`SEVERIAN_HOME/trust/publishers.toml`:

```toml
[[publisher]]
name = "amd"
allowed_domains = ["repo.radeon.com"]
signing_keys = ["<32-byte Ed25519 public key encoded as hex>"]
package_namespaces = ["rocm"]
trusted_from = "2026-01-01"
trusted_until = "2027-01-01"
allow_system_install = true
```

The compiler-owned `trust/vendor-catalog.toml` binds an exact vendor version to
its HTTPS source, SHA-256 digest, and Ed25519 signature. The signature covers
the name, version, publisher, source, and digest. An optional administrator
artifact path supports offline mirrors; it still has to match the signed digest.
Packages cannot select or override that path.

Successful installation records the authorization used at resolution time:

```toml
[[external]]
name = "rocm"
version = "7.0.1"
publisher = "amd"
source = "https://repo.radeon.com/..."
sha256 = "..."
signature = "..."
trusted_from = "2026-01-01"
trusted_until = "2027-01-01"
```

Build-time code is a separate authority boundary. Publisher trust does not
grant build scripts network, process-spawning, unrestricted filesystem, sudo,
or root access; Severian currently rejects executable build and installer hooks
entirely.
