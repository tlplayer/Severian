#!/usr/bin/env bash
set -euo pipefail

REPOSITORY_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
TEST_ROOT=$(mktemp -d "${TMPDIR:-/tmp}/severian-portable-release.XXXXXX")
trap 'rm -rf "$TEST_ROOT"' EXIT

VERSION=$("$REPOSITORY_ROOT/target/debug/sev" --version | awk '{print $2}')
TARGET=$(rustc -vV | awk '/^host:/ {print $2}')
RELEASE="$TEST_ROOT/severian-$VERSION-$TARGET"
"$REPOSITORY_ROOT/scripts/release/build_portable_release.sh" \
    --profile dev --version "$VERSION" --target "$TARGET" --output "$RELEASE"

test -x "$RELEASE/bin/sev"
test -x "$RELEASE/lib/severian/bin/clang"
test -x "$RELEASE/lib/severian/bin/ld.lld"
test -f "$RELEASE/lib/severian/lib/libLLVM.so.21.1"
test -f "$RELEASE/share/severian/library/compute/tensor/package.toml"
test "$(cat "$RELEASE/VERSION")" = "$VERSION"
test -f "$RELEASE.tar.zst"

PROJECT="$TEST_ROOT/application"
"$RELEASE/bin/sev" new "$PROJECT"
"$RELEASE/bin/sev" run "$PROJECT" > "$TEST_ROOT/output"
grep -Fxq 'hello' "$TEST_ROOT/output"

ASSETS="$TEST_ROOT/assets"
mkdir "$ASSETS"
cp "$RELEASE.tar.zst" "$ASSETS/"
(
    cd "$ASSETS"
    sha256sum "$(basename "$RELEASE").tar.zst" > checksums.txt
)
SEV_VERSION="$VERSION" \
SEV_TARGET="$TARGET" \
SEV_RELEASE_BASE_URL="file://$ASSETS" \
SEV_INSTALL_ROOT="$TEST_ROOT/installed" \
SEV_BIN_DIR="$TEST_ROOT/bin" \
SEV_ATTESTATION=skip \
sh "$REPOSITORY_ROOT/install.sh"
test "$("$TEST_ROOT/bin/sev" --version)" = "sev $VERSION"

INSTALLED_PROJECT="$TEST_ROOT/installed-application"
"$TEST_ROOT/bin/sev" new "$INSTALLED_PROJECT"
"$TEST_ROOT/bin/sev" run "$INSTALLED_PROJECT" > "$TEST_ROOT/installed-output"
grep -Fxq 'hello' "$TEST_ROOT/installed-output"

echo "portable release golden path passed"
