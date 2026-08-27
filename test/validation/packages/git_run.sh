#!/usr/bin/env bash
set -euo pipefail

REPOSITORY_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
SEVERIAN_BIN=${SEVERIAN_BIN:-"$REPOSITORY_ROOT/target/debug/sev"}
TEST_ROOT=$(mktemp -d "${TMPDIR:-/tmp}/severian-git-run.XXXXXX")
trap 'rm -rf "$TEST_ROOT"' EXIT

export SEVERIAN_REGISTRY="$TEST_ROOT/registry"
PACKAGE="$TEST_ROOT/git-tool"
"$SEVERIAN_BIN" new "$PACKAGE"
git -C "$PACKAGE" init -q
git -C "$PACKAGE" add package.toml sev.lock src/main.sev
git -C "$PACKAGE" -c user.name=Severian -c user.email=packages@severian.test \
    commit -qm initial

OUTPUT=$("$SEVERIAN_BIN" run "git+file://$PACKAGE")
test "$OUTPUT" = hello
test -f "$SEVERIAN_REGISTRY/cache/git/"*/package.toml

echo "Git ephemeral run golden path passed"
