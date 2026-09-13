#!/usr/bin/env bash
# Baselines use actual CLI executions; evidence belongs to local debug output.
set -euo pipefail
root=$(cd "$(dirname "$0")/../../.." && pwd)
compiler=${SEVERIAN_SOURCE_COMPILER:-"$root/bin/sev"}
workspace=$(mktemp -d /tmp/sev-native-performance-XXXXXX)
profile_root="$root/package.pkg/debug/profile"
mkdir -p "$profile_root"
evidence=$(mktemp -d "$profile_root/run-XXXXXX")
trap 'cp "$workspace"/*.log "$workspace"/*.seconds "$evidence/" 2>/dev/null || true' EXIT
export SEVERIAN_SYSROOT="$root"
export SEVERIAN_HOME="$workspace/home"
cp "$root/test/validation/performance/fixtures/string_growth.sev" "$workspace/string_growth.sev"
"$compiler" test "$workspace/string_growth.sev" 2>&1 | tee "$evidence/string-growth.log"
"$compiler" test "$root/library/system/file/tests/buffer.sev" 2>&1 | tee "$evidence/file-io.log"
# A false resource budget must fail; counting zero or ignoring contracts fails
# this regression even when the positive case happens to finish quickly.
sed 's/defer allocations < 50000/defer allocations < 0/' "$workspace/string_growth.sev" > "$workspace/impossible.sev"
if "$compiler" test "$workspace/impossible.sev" > "$workspace/impossible.log" 2>&1; then
    printf 'Resource budget was ignored\n' >&2
    exit 1
fi
rg 'assertion failed' "$workspace/impossible.log"
"$compiler" init "$workspace/baseline"
cd "$workspace/baseline"
cat > src/main.sev <<'SOURCE'
def main():
    value = string()
    for index in range(4096):
        value.append('x')
    assert(value.length() == 4096)
SOURCE
SEVERIAN_PROFILE_ACTIVE=1 /usr/bin/time -f '%e' -o "$workspace/cold.seconds" "$compiler" build --build-profile release > "$workspace/cold.log" 2>&1
for iteration in 1 2 3; do
    /usr/bin/time -f '%e' -o "$workspace/warm-$iteration.seconds" "$compiler" build --build-profile release --locked > "$workspace/warm-$iteration.log" 2>&1
    if rg '^compiling ' "$workspace/warm-$iteration.log"; then exit 1; fi
    awk -v limit="${SEVERIAN_WARM_BUILD_SECONDS:-2}" '($1 > limit) { exit 1 }' "$workspace/warm-$iteration.seconds"
done
awk -v limit="${SEVERIAN_SEMANTIC_SECONDS:-12}" '/Stage semantic:/ { found=1; if (($3 + 0) > limit) exit 1 } END { if (!found) exit 1 }' "$workspace/cold.log"
awk -v cold="$(cat "$workspace/cold.seconds")" -v w1="$(cat "$workspace/warm-1.seconds")" -v w2="$(cat "$workspace/warm-2.seconds")" -v w3="$(cat "$workspace/warm-3.seconds")" 'BEGIN { printf "{\"schema_version\":1,\"cold_build_seconds\":%s,\"warm_build_seconds\":[%s,%s,%s]}\n", cold,w1,w2,w3 }' > "$evidence/baseline.json"
cp "$evidence/baseline.json" "$evidence/baseline.next"
mv "$evidence/baseline.next" "$profile_root/baseline.json"
cat "$profile_root/baseline.json"
printf 'Performance regression passed. Evidence: %s\n' "$evidence"
