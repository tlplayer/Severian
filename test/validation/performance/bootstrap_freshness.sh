#!/usr/bin/env bash
# Use the real bootstrap CLI; verify content and output invalidation, not mtimes.
set -euo pipefail
root=$(cd "$(dirname "$0")/../../.." && pwd)
seed=${SEVERIAN_BOOTSTRAP:-"$root/package.pkg/release/sev"}
workspace=$(mktemp -d /tmp/sev-bootstrap-regression-XXXXXX)
mkdir -p "$workspace/src"
cat > "$workspace/package.toml" <<'MANIFEST'
[package]
name = "freshness"
version = "0.1.0"
edition = "2026"
[[bin]]
name = "freshness"
path = "src/main.sev"
MANIFEST
cat > "$workspace/src/main.sev" <<'SOURCE'
def main():
    pass
SOURCE
output="$workspace/package.pkg/bin/freshness"
logs=$(mktemp -d /tmp/sev-bootstrap-regression-logs-XXXXXX)
check_build() {
    /usr/bin/time -f '%e' -o "$logs/$1.seconds" "$seed" build "$workspace" -o "$output" > "$logs/$1.log" 2>&1
    rg "^$2 " "$logs/$1.log"
    "$output"
}
check_build cold built
check_build warm fresh
# A source edit must invalidate even when its timestamp is preserved.
cp -p "$workspace/src/main.sev" "$logs/original"
cat >> "$workspace/src/main.sev" <<'SOURCE'
# source revision two
SOURCE
touch -r "$logs/original" "$workspace/src/main.sev"
check_build changed built
check_build changed_warm fresh
# Corrupt a completed result without damaging its executable header.
printf '\0' >> "$output"
check_build corrupt built
# Failed compilation must preserve the previous working output.
cp "$output" "$logs/working"
printf 'def broken(\n' > "$workspace/src/main.sev"
if "$seed" build "$workspace" -o "$output" > "$logs/failure.log" 2>&1; then
    exit 1
fi
cmp "$output" "$logs/working"
awk -v limit="${SEVERIAN_BOOTSTRAP_WARM_SECONDS:-1}" '($1 > limit) { exit 1 }' "$logs/warm.seconds"
printf 'Bootstrap freshness regression passed. Measurements: %s\n' "$logs"
