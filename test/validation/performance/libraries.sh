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
bootstrap=${SEVERIAN_BOOTSTRAP:-"$root/package.pkg/debug/sev"}
cc=${CC:-clang-21}
"$cc" -O2 -Wall -Wextra -Werror library/system/io/tests/descriptor.c \
    library/system/io/extern/posix/descriptor.c -o "$workspace/io"
"$workspace/io" | tee "$evidence/io.log"
"$cc" -O2 -Wall -Wextra -Werror library/interop/xxi/tests/loans.c \
    rust_compiler/runtime/native/list.c rust_compiler/runtime/native/string.c \
    rust_compiler/runtime/native/any.c library/core/storage/native/storage.c \
    -lm -o "$workspace/xxi-loans"
"$workspace/xxi-loans" | tee "$evidence/xxi-loans.json"
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
"$compiler" test library/system/file/tests/buffer.sev 2>&1 | tee "$evidence/file-buffer.log"
"$compiler" test library/core/text/tests/format.sev 2>&1 | tee "$evidence/text.log"
"$compiler" test library/core/text/tests/profile.sev 2>&1 | tee "$evidence/text-profile.log"
# These entry points use namespace dispatch and Data, currently implemented by
# the bootstrap frontend. Keep its validation separate and explicitly named.
SEVERIAN_HOME="$bootstrap_home" "$bootstrap" test library/system/file/tests/dispatch.sev 2>&1 | tee "$evidence/bootstrap-file.log"
SEVERIAN_HOME="$bootstrap_home" "$bootstrap" test library/data/json 2>&1 | tee "$evidence/bootstrap-json.log"
for case in library/system/file/tests/reading.sev library/system/file/tests/text.sev library/interop/xxi/tests/contracts.sev library/interop/xxi/tests/worker.sev library/interop/xxi/tests/bridge; do
    name=$(basename "$case" .sev)
    SEVERIAN_HOME="$bootstrap_home" "$bootstrap" test "$case" 2>&1 | tee "$evidence/$name.log"
done
python3 - "$evidence" <<'REPORT'
import json
import pathlib
import re
import sys
root = pathlib.Path(sys.argv[1])
pattern = r"profile: (\d+) ns; (\d+) allocations; (\d+) allocated bytes"
def measurement(values):
    return dict(zip(("nanoseconds", "allocations", "allocated_bytes"), map(int, values)))
text = re.findall(pattern, (root / "text-profile.log").read_text())
file = re.findall(pattern, (root / "file-buffer.log").read_text())
assert len(text) == 2 and len(file) == 1
report = {"schema_version": 2, "integers": measurement(text[0]), "floats": measurement(text[1]), "file_buffers": measurement(file[0]), "json": json.loads((root / "json.json").read_text()), "xxi": json.loads((root / "xxi-loans.json").read_text())}
(root / "baseline.json").write_text(json.dumps(report, indent=2) + "\n")
REPORT
printf 'Library regression passed. Evidence: %s\n' "$evidence"
