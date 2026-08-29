# Severian code-health compiler

`severian-health` is the repository-owned analysis driver. It combines facts
from source, workspace architecture, Git history, compiler pass contracts,
coverage, and mutation results into one versioned finding model. It is written
in Rust and does not delegate repository validation to Python.

## Commands

```console
cargo xtask health
cargo xtask health --changed origin/main
cargo xtask health --all-targets --all-features
cargo xtask health --coverage target/coverage.json
cargo xtask health --changed origin/main --coverage target/coverage.json
cargo xtask health --mutation-report mutants.out/outcomes.json
cargo xtask health --format human
cargo xtask health --format json
cargo xtask health --format sarif
```

`--coverage` consumes `cargo llvm-cov --branch --json` output. When combined
with `--changed`, executable changed lines and branches with zero execution are
hard findings. Workspace line and branch totals below 95% are also hard
findings. `--mutation-report` consumes JSON and rejects records whose outcome,
status, or result is `Survived`.

The scanner never follows symlinks. It excludes every `.git`, `target`,
`third_party`, `.codex`, and `.agents` directory at any depth. CI and local
runs should still use one Cargo job and an explicit timeout on memory-limited
machines.

## Finding contract

Every finding carries:

- a rule ID, severity, and confidence;
- one primary and zero or more related source spans;
- concrete evidence and named metrics;
- ranked remediation choices;
- a stable fingerprint that does not depend on line numbers;
- whether the finding belongs to the checked-in debt baseline.

Confidence and severity are deliberately separate. Proven invariant failures
are denied. High-confidence source analysis is review evidence unless its rule
is an architecture gate. Heuristics warn. Trend findings only rank cleanup.
The hotspot score never fails a build.

Human, JSON, and SARIF reporters are deterministic. Structured formats escape
source text and include the fingerprint so review systems can track findings.

## Implemented rules

Hard rules:

- `source_file_limit`
- `forbidden_dependency`
- `frontend_target_leak`
- `unsafe_without_contract`
- `bootstrap_semantic_drift`
- `coverage_floor`
- `changed_code_uncovered`
- `mutation_survived`

Review rules:

- `user_input_panic`
- `stringly_semantic_dispatch`
- `unverified_transform`
- `nondeterministic_artifact`
- `parallel_semantic_catalog`
- `exact_clone` and `renamed_clone`
- `unreachable_private_candidate`
- `low_module_cohesion`
- `vertical_test_missing`

Ranking:

- `hotspot_risk`, combining size, nesting, branch signals, panic signals,
  fan-out, and 90-day churn.

The reachability rule is intentionally not labeled proven: its current
repository graph is conservative source analysis. A future rustc HIR/MIR lint
driver can promote a candidate only after resolving dynamic roots, supported
features, trait dispatch, registries, and address-taken functions.

## Debt ratchet

[`baseline.toml`](baseline.toml) records pre-existing hard findings. Each entry
must have an owner, reason, issue, and ISO date. Expired or malformed entries
are configuration errors. Baselined findings remain visible but do not fail the
gate; a new deny finding does.

To inspect a prospective baseline without modifying the repository:

```console
cargo xtask health --write-baseline /tmp/severian-health-baseline.toml
```

Do not refresh the baseline to make a regression green. Fix the regression or
add a reviewed, owned, expiring entry.

## Analysis boundary

The source scanner automates facts it can support today. It does not pretend
text parsing is rustc HIR, semantic clone proof, or whole-program proof. The
next precision layer is a project-owned rustc/Dylint driver feeding the same
finding model. Compiler IR checks live at their actual stage rather than being
reconstructed by the repository scanner; MIR pass contracts are the first
enforced integration.

Nightly jobs are the appropriate home for full feature/target reachability,
full mutation testing, structure-aware fuzzing, Miri, sanitizers, differential
backend corpora, and historical co-change mining. Their results should enter
this command as findings rather than creating independent policy formats.
