# Published preludes

`sev` selects `sev-prelude:latest` from the configured package registry. The
release pins its grammar, runtime, testing, and math dependencies. Its frontend
archive contains syntax descriptors, generic bodies, resolved signatures, and
validated prelude bodies. Each compilation decodes an independent state before
adding the application's declarations and generic specializations.

The original provider sources remain the authoring locations. `packages.json`
lists their package ownership and public import order. To stage and publish the
packages with a freshly built source compiler:

```sh
python3 tools/compiler/prelude_packages.py
```

`SEVERIAN_REGISTRY` chooses the registry; otherwise the ordinary package registry
under `~/.severian/packages/registry` is used. `--stage-only` produces standalone
packages under `library/prelude/package.pkg/staged`. Imports between packages use
`import * from "package:dependency/path.sev"`; the dependency alias must be
declared, and the path must stay inside that dependency. Published manifests
replace staging paths with exact registry versions.

Native providers referenced by the source declarations and their local headers
are included in the release. When a consumer also imports the same provider
from a local library, identical C/header inputs are linked once.

Publication validates the combined prelude before committing its frontend
archive. The archive schema is generated from concrete compiler IR contracts by
`sev_compiler/build/frontend_codec.py`. The compiler package declares this recipe
in `build.generators`; builds regenerate it automatically before checking cached
compilation results. The generated codec lives under
`sev_compiler/package.pkg/build/<digest>/staging/`, with a generated import entry
in `package.pkg/build/frontend_archive.sev`. Its digest includes the recipe and
concrete IR sources. Cleaning build output is safe; the next package build
recreates it. After changing these contracts, rebuild the compiler and republish.
Changed source needs a new version in
`packages.json`; published versions remain immutable.

An installation without a registered prelude can bootstrap from the sysroot.
Direct prelude-provider tests retain that source path to exercise local edits. Prelude
exclusions, open extensions, compiler tests, and full IR/editor queries use the
published syntax with joint semantic validation, since those operations need the
complete declaration environment. Ordinary run and test consumers reuse the
validated state. New registry publications invalidate existing unit selections.
