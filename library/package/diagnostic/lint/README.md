# package.diagnostic.lint

Owns token-based package quality analysis, lint policy, suppressions, severity,
report formatting and enforcement. `lint(project, identity, force, testing,
immutable)` returns diagnostics; `enforce` applies their failure policy.

The package coordinator supplies an identity covering source, configuration,
compiler and revision. Reports and disposable analysis state live under the
analyzed package's `package.pkg/debug/quality/lint/`. Damaged cached analysis
falls back to source analysis. Coverage and contribution reports remain in
`package.diagnostic`; shared file/document operations are in
`package.diagnostic.support`. Neither subpackage depends on its parent.

Existing `package.lint` and `diagnostic.lint` call sites forward to this package.
