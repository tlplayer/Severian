#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIRECTORY=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
REPOSITORY_ROOT=$(CDPATH= cd -- "$SCRIPT_DIRECTORY/../../.." && pwd)
SEVERIAN_BIN=${SEVERIAN_BIN:-"$REPOSITORY_ROOT/target/debug/sev"}

if [[ ! -x "$SEVERIAN_BIN" ]]; then
    echo "missing Severian executable: $SEVERIAN_BIN" >&2
    exit 1
fi

TEST_ROOT=$(mktemp -d "${TMPDIR:-/tmp}/severian-dependency-commands.XXXXXX")
trap 'rm -rf -- "$TEST_ROOT"' EXIT
export SEVERIAN_REGISTRY="$TEST_ROOT/registry"

publish_calculator() {
    local version=$1
    local answer=$2
    local root="$TEST_ROOT/calculator-$version"
    "$SEVERIAN_BIN" new "$root"
    cat > "$root/package.toml" <<TOML
[package]
name = "calculator"
version = "$version"
edition = "2026"

[lib]
path = "src/lib.sev"
TOML
    cat > "$root/src/lib.sev" <<SEV
def answer() -> int:
    return $answer
SEV
    "$SEVERIAN_BIN" publish "$root"
    mv "$root" "$TEST_ROOT/calculator-$version-checkout-removed"
}

publish_calculator "1.0.0" 10
publish_calculator "2.1.0" 21

APPLICATION_ROOT="$TEST_ROOT/application"
"$SEVERIAN_BIN" new "$APPLICATION_ROOT"
cat > "$APPLICATION_ROOT/src/main.sev" <<'SEV'
import calculator

print(calculator.answer())
SEV

echo "adding latest calculator"
(
    cd "$APPLICATION_ROOT"
    "$SEVERIAN_BIN" add calculator
)
grep -Fq 'calculator = "2.1.0"' "$APPLICATION_ROOT/package.toml"
grep -Fq 'name = "calculator"' "$APPLICATION_ROOT/sev.lock"
grep -Fq 'version = "2.1.0"' "$APPLICATION_ROOT/sev.lock"
OUTPUT=$("$SEVERIAN_BIN" run "$APPLICATION_ROOT")
[[ "$OUTPUT" == "21" ]]

publish_calculator "2.2.0" 22

echo "updating calculator"
(
    cd "$APPLICATION_ROOT"
    "$SEVERIAN_BIN" update calculator
)
grep -Fq 'calculator = "2.2.0"' "$APPLICATION_ROOT/package.toml"
OUTPUT=$("$SEVERIAN_BIN" run "$APPLICATION_ROOT")
[[ "$OUTPUT" == "22" ]]

echo "removing calculator"
(
    cd "$APPLICATION_ROOT"
    "$SEVERIAN_BIN" remove calculator
)
if grep -Eq '^calculator[[:space:]]*=' "$APPLICATION_ROOT/package.toml"; then
    echo "remove left calculator in the manifest" >&2
    exit 1
fi
if grep -Fq 'name = "calculator"' "$APPLICATION_ROOT/sev.lock"; then
    echo "remove left calculator in the lockfile" >&2
    exit 1
fi

echo "adding calculator major version 2"
(
    cd "$APPLICATION_ROOT"
    "$SEVERIAN_BIN" add calculator@2
)
grep -Fq 'calculator = "2.2.0"' "$APPLICATION_ROOT/package.toml"

echo "pinning calculator 2.1.0"
(
    cd "$APPLICATION_ROOT"
    "$SEVERIAN_BIN" add calculator@2.1.0
)
grep -Fq 'calculator = "2.1.0"' "$APPLICATION_ROOT/package.toml"
OUTPUT=$("$SEVERIAN_BIN" run "$APPLICATION_ROOT")
[[ "$OUTPUT" == "21" ]]

echo "registry dependency command golden path passed"
