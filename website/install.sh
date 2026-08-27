#!/bin/sh
set -eu

TEMPORARY=$(mktemp "${TMPDIR:-/tmp}/severian-bootstrap.XXXXXX")
cleanup() {
    rm -f "$TEMPORARY"
}
trap cleanup EXIT HUP INT TERM
curl --proto '=https' --tlsv1.2 -fsSL \
    https://raw.githubusercontent.com/tlplayer/Severian/main/install.sh \
    -o "$TEMPORARY"
sh "$TEMPORARY"
