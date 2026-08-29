# Harness library

Stable ID: `library.harness`.

The harness package orchestrates executable examples, benchmarks, and runtime
checks. It consumes language-level `test` declarations and library testing
helpers but does not define parser test syntax. Test syntax belongs to the
language; harness behavior belongs to this package; CI policy belongs to the
repository.

Timing, environment, and process effects must be declared by the selected
harness operation. Reproducibility guarantees are not yet specified per helper.
