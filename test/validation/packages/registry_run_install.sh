#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIRECTORY=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
REPOSITORY_ROOT=$(CDPATH= cd -- "$SCRIPT_DIRECTORY/../../.." && pwd)
SEVERIAN_BIN=${SEVERIAN_BIN:-"$REPOSITORY_ROOT/target/debug/sev"}

if [[ ! -x "$SEVERIAN_BIN" ]]; then
    echo "missing Severian executable: $SEVERIAN_BIN" >&2
    exit 1
fi

TEST_ROOT=$(mktemp -d "${TMPDIR:-/tmp}/severian-run-install.XXXXXX")
trap 'rm -rf -- "$TEST_ROOT"' EXIT
export SEVERIAN_REGISTRY="$TEST_ROOT/registry"
export SEVERIAN_BIN_HOME="$TEST_ROOT/bin"

TOOL_ROOT="$TEST_ROOT/hello-tool"
"$SEVERIAN_BIN" new "$TOOL_ROOT"
cat > "$TOOL_ROOT/package.toml" <<'TOML'
[package]
name = "hello-tool"
version = "1.0.0"
edition = "2026"
default-run = "hello-tool"

[[bin]]
name = "hello-tool"
path = "src/main.sev"

[lib]
name = "hello_tool_api"
path = "src/lib.sev"
TOML
cat > "$TOOL_ROOT/src/lib.sev" <<'SEV'
def message() -> string:
    return "hello from packaged executable"
SEV
cat > "$TOOL_ROOT/src/main.sev" <<'SEV'
import "lib.sev" as hello_tool_api

print(hello_tool_api.message())
SEV

"$SEVERIAN_BIN" publish "$TOOL_ROOT"
mv "$TOOL_ROOT" "$TEST_ROOT/tool-checkout-removed"

RELEASE="$SEVERIAN_REGISTRY/packages/hello-tool/1.0.0"
ARTIFACT="$RELEASE/artifacts/host/dev/bin/hello-tool"
SOURCE="$RELEASE/source"
PACKAGE="$RELEASE/hello-tool-1.0.0.pkg"
test -x "$ARTIFACT"
test -f "$RELEASE/metadata/package.toml"
test -f "$RELEASE/metadata/sev.lock"
test -f "$RELEASE/metadata/build.toml"
test -f "$PACKAGE"
cmp -n 8 <(printf 'SEVPKG\0\002') "$PACKAGE"

echo "running exact package from its precompiled artifact"
mv "$SOURCE" "$TEST_ROOT/source-unavailable"
OUTPUT=$(
    cd "$TEST_ROOT"
    "$SEVERIAN_BIN" run hello-tool@1.0.0
)
[[ "$OUTPUT" == "hello from packaged executable" ]]
mv "$TEST_ROOT/source-unavailable" "$SOURCE"

echo "running from the v2 package with its exploded realization removed"
EXPLODED="$TEST_ROOT/exploded-unavailable"
mkdir "$EXPLODED"
mv "$RELEASE/metadata" "$RELEASE/source" "$RELEASE/artifacts" "$EXPLODED/"
OUTPUT=$(
    cd "$TEST_ROOT"
    "$SEVERIAN_BIN" run hello-tool@1.0.0
)
[[ "$OUTPUT" == "hello from packaged executable" ]]
test -x "$SEVERIAN_REGISTRY/cache/distributions/hello-tool/1.0.0/artifacts/host/dev/bin/hello-tool"
mv "$EXPLODED/metadata" "$EXPLODED/source" "$EXPLODED/artifacts" "$RELEASE/"

echo "running latest package through source fallback"
mv "$ARTIFACT" "$TEST_ROOT/artifact-unavailable"
OUTPUT=$(
    cd "$TEST_ROOT"
    "$SEVERIAN_BIN" run hello-tool
)
[[ "$OUTPUT" == "hello from packaged executable" ]]
mv "$TEST_ROOT/artifact-unavailable" "$ARTIFACT"

echo "installing package executable"
(
    cd "$TEST_ROOT"
    "$SEVERIAN_BIN" install hello-tool@1.0.0
)
test -x "$SEVERIAN_BIN_HOME/hello-tool"
[[ $("$SEVERIAN_BIN_HOME/hello-tool") == "hello from packaged executable" ]]

echo "publishing an executable-only package"
BIN_ONLY="$TEST_ROOT/bin-only"
"$SEVERIAN_BIN" new "$BIN_ONLY"
sed -i 's/version = "0.1.0"/version = "3.0.0"/' "$BIN_ONLY/package.toml"
"$SEVERIAN_BIN" publish "$BIN_ONLY"
OUTPUT=$("$SEVERIAN_BIN" run bin-only@3)
[[ "$OUTPUT" == "hello" ]]

echo "registry run/install golden path passed"
