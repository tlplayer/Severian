#!/usr/bin/env bash
set -euo pipefail

if (($# != 1)); then
    echo "usage: $0 vMAJOR.MINOR.PATCH" >&2
    exit 2
fi
TAG=$1
if [[ ! "$TAG" =~ ^v([0-9]+)\.([0-9]+)\.([0-9]+)$ ]]; then
    echo "release tags must use vMAJOR.MINOR.PATCH; got $TAG" >&2
    exit 1
fi
VERSION=${TAG#v}
REPOSITORY_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
WORKSPACE_VERSION=$(awk '
    /^\[workspace.package\]$/ { workspace = 1; next }
    /^\[/ { workspace = 0 }
    workspace && /^version = / {
        value = $3
        gsub(/"/, "", value)
        print value
        exit
    }
' "$REPOSITORY_ROOT/Cargo.toml")
if [[ "$VERSION" != "$WORKSPACE_VERSION" ]]; then
    echo "tag $TAG does not match workspace version $WORKSPACE_VERSION" >&2
    exit 1
fi
echo "release tag $TAG matches workspace version"
