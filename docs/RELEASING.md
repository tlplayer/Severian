# Releasing Severian

GitHub Releases are the canonical binary distribution. Normal users install a
tested archive; they do not clone the repository or install Rust, LLVM, MLIR,
CMake, ROCm, or CUDA.

## Supported targets

Publish only architectures continuously built and tested by the release
workflow:

- `x86_64-unknown-linux-gnu`
- `aarch64-unknown-linux-gnu`

macOS and Windows targets belong here only after their full release and
installer paths run in CI.

## Tag flow

The workspace version and tag must match exactly:

```bash
git tag v0.1.0
git push origin v0.1.0
```

[release.yml](../.github/workflows/release.yml) then performs, natively on each
architecture:

```text
checkout
  -> validate tag against Cargo.toml
  -> install the build-only LLVM/MLIR toolchain
  -> cargo test --locked --workspace
  -> build the standalone Severian payload
  -> compress severian-VERSION-TARGET.tar.zst
  -> create signed Sigstore provenance
  -> aggregate SHA-256 checksums
  -> create a draft GitHub Release
  -> upload every asset
  -> publish the complete release
```

The draft-first sequence is compatible with GitHub immutable releases: all
assets are attached before publication. Repository administrators should enable
immutable releases in the GitHub repository settings.

Each release contains:

```text
severian-VERSION-x86_64-unknown-linux-gnu.tar.zst
severian-VERSION-x86_64-unknown-linux-gnu.tar.zst.sigstore.json
severian-VERSION-aarch64-unknown-linux-gnu.tar.zst
severian-VERSION-aarch64-unknown-linux-gnu.tar.zst.sigstore.json
checksums.txt
checksums.txt.sigstore.json
```

Verify provenance manually with:

```bash
gh attestation verify severian-VERSION-TARGET.tar.zst \
  -R tlplayer/Severian
```

## Payload contract

The portable builder emits:

```text
severian-VERSION-TARGET/
├── bin/sev
├── lib/severian/
│   ├── bin/                 # Clang, LLD, MLIR, and the real sev executable
│   └── lib/                 # LLVM/MLIR runtime libraries
├── share/severian/library/  # compiler-visible Severian libraries
├── LICENSE.md
├── RELEASE.toml
└── VERSION
```

Optional GPU/XLA components are deliberately absent. `sev doctor` discovers
them independently, so a missing accelerator never invalidates the base
compiler installation.

The payload can be built locally with:

```bash
scripts/release/build_portable_release.sh
```

## Installer contract

[`install.sh`](../install.sh) resolves the latest stable release or
`SEV_VERSION`, detects the host target, downloads the corresponding archive and
`checksums.txt`, verifies SHA-256 before extraction, and atomically activates
the selected installation through `~/.local/bin/sev`.
The activated payload records `method = "standalone"` in `INSTALLATION.toml`,
which gives a future self-updater an explicit ownership boundary.

Attestation policy is configurable:

- `SEV_ATTESTATION=auto` verifies through `gh` when available.
- `SEV_ATTESTATION=required` refuses installation unless provenance verifies.
- `SEV_ATTESTATION=skip` exists for controlled/offline mirrors and tests.

Package managers should install the same release payload but record their own
management method. A future `sev self update` must update standalone installs
and refuse to replace installations managed by apt, Homebrew, or another
package manager.
