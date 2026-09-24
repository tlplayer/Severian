# Directory consolidation

This change groups files by concept without changing compiler phase ownership.
No builds or tests were run.

## Layout

- Primitive declarations, catalog, and behavior: `syntax/primitive/`.
- Large primitive concepts: `int/int.sev`, `float/float.sev`, `string/string.sev`, `char/char.sev`, and corresponding concept folders.
- Package sources: `<package>/<package>/src/`; manifests stay at each package root. MIR uses `mir/package.json` with `mir/mir/src/`.
- Sentence composition and declarations: `syntax/sentence/`.
- Expression conversion: `syntax/function/expression_conversion.sev`.
- Annotation and lowered contracts remain at `syntax/annotation.sev` and `syntax/lowered.sev`, outside `type/`.
- Ownership implementation and pipeline: `mir/ownership/`.

## Collisions and resolutions

- Purged placeholder: `sev_compiler/mir/mir/src/model/operation/mod.sev` (only `import universal`).
- Purged placeholder: `sev_compiler/mir/mir/src/model/value/mod.sev` (only `import universal`).
- Ownership manifests overlap: retained the implementation package `sev-compiler-mir-ownership`; retired the compatibility manifest.
- Ownership README overlap: retained the implementation documentation.
- Ownership `mlir.pipeline` collided with a dangling symlink: installed the real pipeline.
- Primitive README overlap: retained the catalog README and moved implementation guidance to `IMPLEMENTATIONS.md`.
- Type forwarding wrappers overlap the real contracts: imports now target the retained files; no implementation bodies were discarded.
- `primitive.sev` names describe two roles: `contract.sev` retains the syntax contract; `primitive.sev` retains the compatibility exports including the error declaration.

## Existing unresolved references

- `syntax/function/operator/{syntax,kind,scalar,lowered}.sev` forward to missing `syntax/operator/` files. They are preserved because their implementations are absent.
- Deleted packages such as `universal` and backend boundary packages remain missing. This consolidation does not reconstruct them.
- Some pre-existing source imports refer to absent files; path relocation is not a substitute for implementing those files.
