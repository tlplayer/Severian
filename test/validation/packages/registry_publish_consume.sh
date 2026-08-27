#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIRECTORY=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
REPOSITORY_ROOT=$(CDPATH= cd -- "$SCRIPT_DIRECTORY/../../.." && pwd)
SEVERIAN_BIN=${SEVERIAN_BIN:-"$REPOSITORY_ROOT/target/debug/sev"}

if [[ ! -x "$SEVERIAN_BIN" ]]; then
    echo "missing Severian executable: $SEVERIAN_BIN" >&2
    echo "build it first or set SEVERIAN_BIN" >&2
    exit 1
fi

TEST_ROOT=$(mktemp -d "${TMPDIR:-/tmp}/severian-registry-golden.XXXXXX")
trap 'rm -rf -- "$TEST_ROOT"' EXIT

REGISTRY_ROOT="$TEST_ROOT/registry"
PRODUCER_ROOT="$TEST_ROOT/greeting"
CONSUMER_ROOT="$TEST_ROOT/consumer"
export SEVERIAN_REGISTRY="$REGISTRY_ROOT"

echo "creating producer"
"$SEVERIAN_BIN" new "$PRODUCER_ROOT"

# The package identity intentionally differs from the library target name.
# Publication and dependency resolution must use `package.name`; `[lib].name`
# controls only the exported library target.
cat > "$PRODUCER_ROOT/package.toml" <<'TOML'
[package]
name = "greeting"
version = "0.1.0"
edition = "2026"

[lib]
name = "greeting_api"
path = "src/lib.sev"

[publish]
registry = "default"
include-source = true
include-interface = true
TOML

cat > "$PRODUCER_ROOT/src/lib.sev" <<'SEV'
def message() -> string:
    return "hello from registry"
SEV

echo "publishing producer"
"$SEVERIAN_BIN" publish "$PRODUCER_ROOT"

# Prove the consumer is not accidentally compiling through the producer's
# development checkout after publication.
mv "$PRODUCER_ROOT" "$TEST_ROOT/producer-checkout-removed"

echo "creating consumer"
"$SEVERIAN_BIN" new "$CONSUMER_ROOT"

cat > "$CONSUMER_ROOT/package.toml" <<'TOML'
[package]
name = "consumer"
version = "0.1.0"
edition = "2026"
default-run = "consumer"

[[bin]]
name = "consumer"
path = "src/main.sev"

[dependencies]
greeting = "0.1.0"
salutation = { package = "greeting", version = "0.1.0" }
TOML

cat > "$CONSUMER_ROOT/src/main.sev" <<'SEV'
import greeting
import salutation

print(greeting.message() + " / " + salutation.message())
SEV

if grep -Eq '^(greeting|salutation)[[:space:]]*=.*path[[:space:]]*=' "$CONSUMER_ROOT/package.toml"; then
    echo "consumer unexpectedly contains a path dependency" >&2
    exit 1
fi

echo "building consumer"
"$SEVERIAN_BIN" build "$CONSUMER_ROOT"

echo "running consumer"
OUTPUT=$("$SEVERIAN_BIN" run "$CONSUMER_ROOT")
if [[ "$OUTPUT" != "hello from registry / hello from registry" ]]; then
    echo "unexpected consumer output: $OUTPUT" >&2
    exit 1
fi

RELEASE_ROOT="$REGISTRY_ROOT/packages/greeting/0.1.0"
test -f "$RELEASE_ROOT/source/package.toml"
test -f "$RELEASE_ROOT/greeting-0.1.0.pkg"
grep -Fq 'name = "greeting"' "$RELEASE_ROOT/source/package.toml"
grep -Fq 'name = "greeting_api"' "$RELEASE_ROOT/source/package.toml"

echo "registry publish/consume golden path passed"
