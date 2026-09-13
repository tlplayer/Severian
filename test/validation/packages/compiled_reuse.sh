#!/usr/bin/env bash
# Acceptance gate for compiled package reuse, not whole-program cache reuse.
# Defaults: build/publish <= 90s, a fresh consumer <= 5s, tests <= 15s,
# complete gate <= 300s. Every timeout covers the invocation's process group.
set -euo pipefail

repo=$(CDPATH= cd -- "$(dirname -- "$0")/../../.." && pwd)
compiler=$(readlink -f -- "${1:-$repo/bin/sev_compiler}")
build_seconds=${SEV_REUSE_BUILD_SECONDS:-90}
consumer_seconds=${SEV_REUSE_CONSUMER_SECONDS:-5}
test_seconds=${SEV_REUSE_TEST_SECONDS:-15}
total_seconds=${SEV_REUSE_TOTAL_SECONDS:-300}
for limit in "$build_seconds" "$consumer_seconds" "$test_seconds" "$total_seconds"; do
    [[ "$limit" =~ ^[1-9][0-9]*$ ]] || { echo 'timeouts must be positive seconds' >&2; exit 2; }
done
if [[ ${SEV_REUSE_GUARDED:-} != 1 ]]; then
    exec timeout --kill-after=2s "${total_seconds}s" env SEV_REUSE_GUARDED=1 bash "$0" "$compiler"
fi
[[ -x "$compiler" ]] || { echo "missing compiler: $compiler" >&2; exit 2; }
for tool in strace timeout sha256sum find sort cmp; do
    command -v "$tool" >/dev/null || { echo "missing tool: $tool" >&2; exit 2; }
done
work=$(mktemp -d /tmp/sev-compiled-reuse.XXXXXX)
# Preserve evidence for failures as well as success.
echo "package reuse evidence: $work"
export SEVERIAN_HOME="$work/home"
export SEVERIAN_SYSROOT=${SEVERIAN_SYSROOT:-$repo}
unset SEVERIAN_REGISTRY
failures=0

reject() {
    echo "REJECT: $*" >&2
    failures=$((failures + 1))
}

invoke() {
    local label=$1 seconds=$2
    shift 2
    local start=$SECONDS status=0
    timeout --kill-after=2s "${seconds}s" "$compiler" "$@" >"$work/$label.stdout" 2>"$work/$label.stderr" || status=$?
    echo "$label: $((SECONDS - start))s, status=$status, bound=${seconds}s"
    if ((status != 0)); then
        cat "$work/$label.stderr" >&2
        reject "$label failed or exceeded its time bound"
        return 1
    fi
}

mkdir -p "$work/producer/src" "$work/first" "$work/unrelated/second"
cat >"$work/producer/package.toml" <<'TOML'
[package]
name = "reuse_probe"
version = "1.0.0"
[lib]
path = "src/lib.sev"
TOML
cat >"$work/producer/src/lib.sev" <<'SEV'
def answer() -> int:
    return 42
SEV

invoke build "$build_seconds" build "$work/producer" || exit 1
if [[ ! -e "$work/producer/package.pkg/package.pkgi" ]]; then
    reject 'sev build produced no package.pkgi semantic interface'
fi
if [[ ! -d "$work/producer/package.pkg/build" ]]; then
    reject 'sev build produced no reusable build state'
fi
invoke publish "$build_seconds" publish "$work/producer" --local --build-profile dev || exit 1
release="$SEVERIAN_HOME/packages/registry/reuse_probe/1.0.0"
if [[ ! -d "$release" ]]; then
    reject "publication is missing from $release"
    exit 1
fi
if [[ ! -e "$release/package.pkg/package.pkgi" ]]; then
    reject 'published package has no package.pkgi semantic interface'
fi
# Move our disposable fixture to prove there is no checkout fallback.
mv "$work/producer" "$work/producer-unavailable"
find "$release" -type f -printf '%P %T@\n' -exec sha256sum -- {} \; | sort >"$work/release.before"
cat >"$work/first/main.sev" <<'SEV'
import reuse_probe
print(reuse_probe.answer())
SEV
cat >"$work/unrelated/second/main.sev" <<'SEV'
import reuse_probe
print(reuse_probe.answer() + 1)
SEV

index=0
for directory in "$work/first" "$work/unrelated/second"; do
    index=$((index + 1))
    status=0
    start=$SECONDS
    (cd "$directory" && timeout --kill-after=2s "${consumer_seconds}s" \
        strace -f -qq -e trace=openat -o "$work/consumer-$index.trace" \
        "$compiler" main.sev) >"$work/consumer-$index.stdout" 2>"$work/consumer-$index.stderr" || status=$?
    echo "fresh consumer $index: $((SECONDS - start))s, status=$status, bound=${consumer_seconds}s"
    if ((status != 0)); then
        reject "fresh consumer $index failed or exceeded its time bound"
    elif [[ $(cat "$work/consumer-$index.stdout") != "$((41 + index))" ]]; then
        reject "fresh consumer $index produced the wrong result"
    fi
    # Require source-free consumption as well as fast execution. Source reads
    # alone do not distinguish hashing from parsing; missing interfaces and
    # timing failures are reported separately above.
    if grep -E '"[^" ]*(/source/[^" ]*\.sev|/sev_compiler/universal/prelude\.sev)".*O_RDONLY' "$work/consumer-$index.trace" >"$work/consumer-$index.source-reads"; then
        reject "fresh consumer $index reopened dependency/prelude implementation source"
    fi
done

cat >"$work/first/test.sev" <<'SEV'
import reuse_probe
test "published compiled dependency":
    assert(reuse_probe.answer() == 42)
SEV
invoke test "$test_seconds" test "$work/first/test.sev" || true
find "$release" -type f -printf '%P %T@\n' -exec sha256sum -- {} \; | sort >"$work/release.after"
cmp -s "$work/release.before" "$work/release.after" || reject 'consuming a publication modified its files'
if ((failures != 0)); then
    echo "package reuse: REJECTED ($failures failures); evidence: $work" >&2
    exit 1
fi
echo "package reuse: PASSED; evidence: $work"
