#!/usr/bin/env bash
set -euo pipefail

REPOSITORY_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
TEST_ROOT=$(mktemp -d "${TMPDIR:-/tmp}/severian-installer.XXXXXX")
trap 'rm -rf "$TEST_ROOT"' EXIT

VERSION=9.8.7
TARGET=x86_64-unknown-linux-gnu
ASSET="severian-$VERSION-$TARGET.tar.zst"
RELEASE="$TEST_ROOT/releases/v$VERSION"
PAYLOAD="$TEST_ROOT/payload/severian-$VERSION-$TARGET"
mkdir -p "$RELEASE" "$PAYLOAD/bin" "$PAYLOAD/share/severian/library"

cat > "$PAYLOAD/bin/sev" <<'SEV'
#!/usr/bin/env sh
printf '%s\n' 'sev 9.8.7'
SEV
chmod +x "$PAYLOAD/bin/sev"
printf '%s\n' "$VERSION" > "$PAYLOAD/VERSION"
tar --zstd -C "$TEST_ROOT/payload" -cf "$RELEASE/$ASSET" "$(basename "$PAYLOAD")"
(
    cd "$RELEASE"
    sha256sum "$ASSET" > checksums.txt
)

INSTALL_ROOT="$TEST_ROOT/installations"
BIN_DIR="$TEST_ROOT/bin"
SEV_VERSION="$VERSION" \
SEV_TARGET="$TARGET" \
SEV_RELEASE_BASE_URL="file://$RELEASE" \
SEV_INSTALL_ROOT="$INSTALL_ROOT" \
SEV_BIN_DIR="$BIN_DIR" \
SEV_ATTESTATION=skip \
sh "$REPOSITORY_ROOT/install.sh"

test -L "$BIN_DIR/sev"
test -x "$INSTALL_ROOT/$VERSION-$TARGET/bin/sev"
grep -Fq 'method = "standalone"' "$INSTALL_ROOT/$VERSION-$TARGET/INSTALLATION.toml"
test "$("$BIN_DIR/sev" --version)" = "sev $VERSION"

cp "$RELEASE/$ASSET" "$RELEASE/$ASSET.tampered"
printf 'tampered\n' >> "$RELEASE/$ASSET.tampered"
mv "$RELEASE/$ASSET.tampered" "$RELEASE/$ASSET"
if SEV_VERSION="$VERSION" \
    SEV_TARGET="$TARGET" \
    SEV_RELEASE_BASE_URL="file://$RELEASE" \
    SEV_INSTALL_ROOT="$TEST_ROOT/tampered-installations" \
    SEV_BIN_DIR="$TEST_ROOT/tampered-bin" \
    SEV_ATTESTATION=skip \
    sh "$REPOSITORY_ROOT/install.sh" > "$TEST_ROOT/tampered.stdout" 2> "$TEST_ROOT/tampered.stderr"; then
    echo "tampered release unexpectedly installed" >&2
    exit 1
fi
grep -qi checksum "$TEST_ROOT/tampered.stderr"

echo "standalone installer golden path passed"
