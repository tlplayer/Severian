#!/usr/bin/env bash
# Exercise library APIs and their providers using concrete compiler binaries.
set -euo pipefail
root=$(cd "$(dirname "$0")/../../.." && pwd)
cd "$root"
profile_root="$root/package.pkg/debug/profile"
mkdir -p "$profile_root"
evidence=$(mktemp -d "$profile_root/libraries-XXXXXX")
workspace=$(mktemp -d /tmp/sev-libraries-XXXXXX)
export SEVERIAN_HOME="$workspace/home"
export SEVERIAN_SYSROOT="$root"
bootstrap_home="$workspace/bootstrap"
mkdir -p "$bootstrap_home"
ln -s "$root/library" "$bootstrap_home/library"
ln -s "$root/sev_compiler" "$bootstrap_home/sev_compiler"
compiler=${SEVERIAN_SOURCE_COMPILER:-"$root/bin/sev"}
bootstrap=${SEVERIAN_BOOTSTRAP:-"$root/package.pkg/release/sev"}
cc=${CC:-clang-21}
"$cc" -O2 -Wall -Wextra -Werror library/system/file/tests/native_io.c \
    library/system/file/native/text.c library/core/memory/native/memory.c \
    -o "$workspace/file"
"$workspace/file"
"$cc" -O2 -Wall -Wextra -Werror -DSEV_TEST_OWNERSHIP \
    library/system/file/tests/native_io.c library/system/file/native/owned.c \
    library/system/file/native/text.c library/core/memory/native/memory.c \
    library/core/storage/native/storage.c -o "$workspace/file-owned"
"$workspace/file-owned"
for ownership in physical combined; do
    extra=()
    if [[ "$ownership" == combined ]]; then
        extra=(-DSEV_TEST_OWNERSHIP library/core/storage/native/storage.c)
    fi
    "$cc" -O2 -Wall -Wextra -Werror library/core/storage/tests/statistics.c \
        library/core/storage/native/statistics.c library/core/memory/native/memory.c \
        "${extra[@]}" -o "$workspace/storage-$ownership"
    "$workspace/storage-$ownership"
done
"$cc" -O2 -Wall -Wextra -Werror library/data/json/tests/native_json.c \
    library/data/json/native/json.c rust_compiler/runtime/native/list.c \
    rust_compiler/runtime/native/string.c rust_compiler/runtime/native/any.c \
    library/core/storage/native/storage.c -lm -o "$workspace/json"
"$workspace/json" | tee "$evidence/json.json"
"$compiler" test library/core/text/tests/format.sev 2>&1 | tee "$evidence/text.log"
"$compiler" test library/core/text/tests/profile.sev 2>&1 | tee "$evidence/text-profile.log"
# These entry points use namespace dispatch and Data, currently implemented by
# the bootstrap frontend. Keep its validation separate and explicitly named.
SEVERIAN_HOME="$bootstrap_home" "$bootstrap" test library/system/file/tests/dispatch.sev 2>&1 | tee "$evidence/bootstrap-file.log"
SEVERIAN_HOME="$bootstrap_home" "$bootstrap" test library/data/json 2>&1 | tee "$evidence/bootstrap-json.log"
awk -v json="$(cat "$evidence/json.json")" '
    /^profile:/ { ++count; elapsed[count]=$2; calls[count]=$4; bytes[count]=$6 }
    END {
        if (count != 2) exit 1
        printf "{\"schema_version\":1,\"integers\":{\"nanoseconds\":%s,\"allocations\":%s,\"allocated_bytes\":%s},\"floats\":{\"nanoseconds\":%s,\"allocations\":%s,\"allocated_bytes\":%s},\"json\":%s}\n", elapsed[1],calls[1],bytes[1],elapsed[2],calls[2],bytes[2],json
    }
' "$evidence/text-profile.log" > "$evidence/baseline.json"
printf 'Library regression passed. Evidence: %s\n' "$evidence"
