#!/bin/sh
set -eu

REPOSITORY=${SEV_GITHUB_REPOSITORY:-tlplayer/Severian}
ATTESTATION=${SEV_ATTESTATION:-auto}
TEMPORARY=

say() {
    printf '%s\n' "$*"
}

fail() {
    printf 'error: %s\n' "$*" >&2
    exit 1
}

usage() {
    cat <<'USAGE'
usage: ./install.sh [--source]

With no arguments, download and install a verified Severian release.
With --source, build this checkout and install its `sev` through Cargo.

source-install environment:
  SEV_CARGO_INSTALL_ROOT  Cargo installation root (default: $CARGO_HOME or ~/.cargo)
USAGE
}

install_from_source() {
    command -v cargo >/dev/null 2>&1 || fail \
        "cargo is required for a source installation"
    SOURCE_ROOT=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd) || fail \
        "could not resolve the Severian checkout"
    [ -f "$SOURCE_ROOT/Cargo.toml" ] || fail \
        "--source must be run from a Severian source checkout"
    [ -f "$SOURCE_ROOT/rust_compiler/boundaries/driver/Cargo.toml" ] || fail \
        "the Severian driver crate is missing from $SOURCE_ROOT"

    CARGO_INSTALL_ROOT=${SEV_CARGO_INSTALL_ROOT:-${CARGO_HOME:-$HOME/.cargo}}
    COMMAND="$CARGO_INSTALL_ROOT/bin/sev"
    say "Building and installing Severian from $SOURCE_ROOT"
    cargo install \
        --locked \
        --force \
        --root "$CARGO_INSTALL_ROOT" \
        --path "$SOURCE_ROOT/rust_compiler/boundaries/driver" \
        --bin sev

    [ -x "$COMMAND" ] || fail "cargo did not install $COMMAND"
    HELP=$($COMMAND --help 2>&1) || fail "installed sev did not start"
    printf '%s\n' "$HELP" | grep -Fq 'agent-ir' || fail \
        "installed sev does not contain this checkout's Agent IR support"

    say "Installed this Severian checkout to $COMMAND"
    case ":${PATH:-}:" in
        *":$CARGO_INSTALL_ROOT/bin:"*) ;;
        *) say "Add $CARGO_INSTALL_ROOT/bin to PATH to invoke sev directly." ;;
    esac
    say "Run 'hash -r' if this shell cached an older sev command."
}

case ${1:-} in
    --source)
        [ "$#" -eq 1 ] || fail "--source accepts no additional arguments"
        install_from_source
        exit 0
        ;;
    -h|--help)
        usage
        exit 0
        ;;
    '') ;;
    *)
        usage >&2
        fail "unknown installer option '$1'"
        ;;
esac

cleanup() {
    if [ -n "$TEMPORARY" ] && [ -d "$TEMPORARY" ]; then
        rm -rf "$TEMPORARY"
    fi
}
trap cleanup EXIT HUP INT TERM

command -v curl >/dev/null 2>&1 || fail "curl is required to install Severian"
command -v tar >/dev/null 2>&1 || fail "tar is required to install Severian"

VERSION=${SEV_VERSION:-}
if [ -z "$VERSION" ]; then
    [ -z "${SEV_RELEASE_BASE_URL:-}" ] || fail \
        "SEV_VERSION is required with SEV_RELEASE_BASE_URL"
    LATEST_URL="https://github.com/$REPOSITORY/releases/latest"
    RESOLVED=$(curl --proto '=https' --tlsv1.2 -fsSLI \
        -o /dev/null -w '%{url_effective}' "$LATEST_URL") || fail \
        "could not resolve the latest Severian release"
    VERSION=${RESOLVED##*/}
fi
VERSION=${VERSION#v}
printf '%s\n' "$VERSION" | grep -Eq '^[0-9]+\.[0-9]+\.[0-9]+$' || fail \
    "invalid Severian version '$VERSION'; expected MAJOR.MINOR.PATCH"

TARGET=${SEV_TARGET:-}
if [ -z "$TARGET" ]; then
    case $(uname -s) in
        Linux) OPERATING_SYSTEM=unknown-linux-gnu ;;
        Darwin) OPERATING_SYSTEM=apple-darwin ;;
        *) fail "Severian does not publish an installer artifact for $(uname -s)" ;;
    esac
    case $(uname -m) in
        x86_64|amd64) ARCHITECTURE=x86_64 ;;
        aarch64|arm64) ARCHITECTURE=aarch64 ;;
        *) fail "Severian does not publish an installer artifact for $(uname -m)" ;;
    esac
    TARGET="$ARCHITECTURE-$OPERATING_SYSTEM"
fi
case "$TARGET" in
    x86_64-unknown-linux-gnu|aarch64-unknown-linux-gnu) ;;
    *) fail "Severian does not publish a tested release for $TARGET" ;;
esac

ASSET="severian-$VERSION-$TARGET.tar.zst"
TAG="v$VERSION"
BASE_URL=${SEV_RELEASE_BASE_URL:-"https://github.com/$REPOSITORY/releases/download/$TAG"}
TEMPORARY=$(mktemp -d "${TMPDIR:-/tmp}/severian-install.XXXXXX") || fail \
    "could not create an installer workspace"
ARCHIVE="$TEMPORARY/$ASSET"
CHECKSUMS="$TEMPORARY/checksums.txt"

download() {
    SOURCE=$1
    DESTINATION=$2
    case "$SOURCE" in
        https://*) curl --proto '=https' --tlsv1.2 -fsSL "$SOURCE" -o "$DESTINATION" ;;
        file://*)
            [ -n "${SEV_RELEASE_BASE_URL:-}" ] || fail "file downloads require an explicit release base"
            curl -fsSL "$SOURCE" -o "$DESTINATION"
            ;;
        *) fail "release downloads require HTTPS" ;;
    esac
}

say "Downloading Severian $VERSION for $TARGET"
download "$BASE_URL/$ASSET" "$ARCHIVE" || fail "could not download $ASSET"
download "$BASE_URL/checksums.txt" "$CHECKSUMS" || fail "could not download checksums.txt"

EXPECTED=$(awk -v asset="$ASSET" '
    {
        name = $2
        sub(/^\*/, "", name)
        if (name == asset) print $1
    }
' "$CHECKSUMS" | tail -n 1)
[ -n "$EXPECTED" ] || fail "checksums.txt does not contain $ASSET"
if command -v sha256sum >/dev/null 2>&1; then
    ACTUAL=$(sha256sum "$ARCHIVE" | awk '{print $1}')
elif command -v shasum >/dev/null 2>&1; then
    ACTUAL=$(shasum -a 256 "$ARCHIVE" | awk '{print $1}')
else
    fail "sha256sum or shasum is required to verify Severian"
fi
[ "$ACTUAL" = "$EXPECTED" ] || fail "checksum verification failed for $ASSET"
say "Verified SHA-256 checksum"

case "$ATTESTATION" in
    skip) ;;
    auto)
        if command -v gh >/dev/null 2>&1; then
            if gh attestation verify "$ARCHIVE" -R "$REPOSITORY" >/dev/null 2>&1; then
                say "Verified GitHub Sigstore provenance"
            else
                say "GitHub provenance was unavailable; checksum verification succeeded" >&2
            fi
        else
            say "GitHub CLI not found; checksum verified (use SEV_ATTESTATION=required to enforce provenance)" >&2
        fi
        ;;
    required)
        command -v gh >/dev/null 2>&1 || fail \
            "GitHub CLI is required when SEV_ATTESTATION=required"
        gh attestation verify "$ARCHIVE" -R "$REPOSITORY" >/dev/null || fail \
            "GitHub Sigstore provenance verification failed"
        say "Verified GitHub Sigstore provenance"
        ;;
    *) fail "SEV_ATTESTATION must be auto, required, or skip" ;;
esac

EXTRACTED="$TEMPORARY/extracted"
mkdir -p "$EXTRACTED"
if tar --help 2>/dev/null | grep -q -- '--zstd'; then
    tar --zstd -xf "$ARCHIVE" -C "$EXTRACTED" || fail "could not extract $ASSET"
elif command -v zstd >/dev/null 2>&1; then
    DECOMPRESSED="$TEMPORARY/release.tar"
    zstd -q -d "$ARCHIVE" -o "$DECOMPRESSED" || fail "could not decompress $ASSET"
    tar -xf "$DECOMPRESSED" -C "$EXTRACTED" || fail "could not extract $ASSET"
else
    fail "tar with Zstandard support or the zstd command is required"
fi

PAYLOAD="$EXTRACTED/severian-$VERSION-$TARGET"
[ -x "$PAYLOAD/bin/sev" ] || fail "release does not contain bin/sev"
[ -f "$PAYLOAD/VERSION" ] || fail "release does not contain VERSION"
[ "$(sed -n '1p' "$PAYLOAD/VERSION")" = "$VERSION" ] || fail \
    "release VERSION does not match $VERSION"

INSTALL_ROOT=${SEV_INSTALL_ROOT:-"${XDG_DATA_HOME:-$HOME/.local/share}/severian"}
BIN_DIR=${SEV_BIN_DIR:-"${XDG_BIN_HOME:-$HOME/.local/bin}"}
DESTINATION="$INSTALL_ROOT/$VERSION-$TARGET"
mkdir -p "$INSTALL_ROOT" "$BIN_DIR"
printf '%s\n' "$EXPECTED" > "$PAYLOAD/.archive.sha256"
cat > "$PAYLOAD/INSTALLATION.toml" <<INSTALLATION
format = 1
method = "standalone"
version = "$VERSION"
target = "$TARGET"
archive_sha256 = "$EXPECTED"
INSTALLATION
if [ -e "$DESTINATION" ]; then
    [ -x "$DESTINATION/bin/sev" ] || fail \
        "existing installation is incomplete: $DESTINATION"
    [ -f "$DESTINATION/.archive.sha256" ] || fail \
        "existing installation predates verified standalone installs: $DESTINATION"
    [ "$(sed -n '1p' "$DESTINATION/.archive.sha256")" = "$EXPECTED" ] || fail \
        "existing installation does not match the verified release archive"
else
    mv "$PAYLOAD" "$DESTINATION"
fi

COMMAND="$BIN_DIR/sev"
if [ -e "$COMMAND" ] && [ ! -L "$COMMAND" ] && [ "${SEV_FORCE:-0}" != 1 ]; then
    fail "$COMMAND is not managed by the Severian installer; set SEV_FORCE=1 to replace it"
fi
LINK="$BIN_DIR/.sev.$$.tmp"
rm -f "$LINK"
ln -s "$DESTINATION/bin/sev" "$LINK"
mv -f "$LINK" "$COMMAND"

INSTALLED=$($COMMAND --version 2>/dev/null) || fail "installed sev did not start"
case "$INSTALLED" in
    *"$VERSION"*) ;;
    *) fail "installed sev reported an unexpected version: $INSTALLED" ;;
esac

say "Installed Severian $VERSION to $DESTINATION"
say "Command: $COMMAND"
case ":${PATH:-}:" in
    *":$BIN_DIR:"*) ;;
    *) say "Add $BIN_DIR to PATH to invoke sev directly." ;;
esac
