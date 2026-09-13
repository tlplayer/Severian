#!/usr/bin/env bash
# Exercise the installed CLI, its native compiler and real registry protocol.
set -euo pipefail
workspace=$(mktemp -d /tmp/sev-native-packages-XXXXXX)
export SEVERIAN_HOME="$workspace/home"
cd "$workspace"
sev init hello_world --lib
cat > hello_world/package.toml <<'EOF'
[package]
name = "hello_world"
version = "0.1.0"
edition = "2026"
export = ["hello", "answer"]
[lib]
path = "src/lib.sev"
EOF
cat > hello_world/src/lib.sev <<'EOF'
def hello():
    print("hello, severian")

def answer() -> i64:
    return 42

test "library returns its answer":
    assert(answer() == 42)
EOF
cd hello_world
sev build --build-profile release
sev test
sev publish --local
archive=$(find package.pkg/artifacts -name '*.a' -print -quit)
test -n "$archive"
file "$archive"
nm --defined-only "$archive" | rg '__sev_pkg_.*_(hello|answer)$'
if nm --defined-only "$archive" | rg ' T main$'; then
    exit 1
fi
if nm -g --defined-only "$archive" | rg '__sev_scalar_'; then
    exit 1
fi
payload=$(find "$SEVERIAN_HOME/packages/registry" -type d -name package.pkg -print -quit)
test -f "$payload/source/src/lib.sev"
test -f "$payload/metadata/source-index.toml"
if find "$payload" -type d \( -name build -o -name cache -o -name debug \) | rg .; then
    exit 1
fi
cd "$workspace"
sev init consumer
cd consumer
sev add hello_world
cat > src/main.sev <<'EOF'
import hello_world

def main():
    hello_world.hello()

test "published functions link and run":
    hello_world.hello()
    assert(hello_world.answer() == 42)
EOF
/usr/bin/time -p sev build --build-profile release 2>"$workspace/first.log"
cat "$workspace/first.log"
# A matching publication must avoid compiling dependency source even cold.
if rg 'compiling .*hello_world.*src/lib.sev' "$workspace/first.log"; then
    exit 1
fi
sev test --build-profile release
/usr/bin/time -p sev build --build-profile release --locked 2>"$workspace/warm.log"
cat "$workspace/warm.log"
if rg '^compiling ' "$workspace/warm.log"; then
    exit 1
fi
cd "$workspace"
mv hello_world moved_producer
mv consumer moved_consumer
cd moved_consumer
/usr/bin/time -p sev build --build-profile release --locked 2>"$workspace/moved.log"
cat "$workspace/moved.log"
if rg '^compiling ' "$workspace/moved.log"; then
    exit 1
fi
sev test --build-profile release
# Source edits in the consumer must still link the same dependency archive.
printf '\n# consumer-only change\n' >> src/main.sev
sev build --build-profile release 2>"$workspace/changed.log"
cat "$workspace/changed.log"
if rg 'compiling .*hello_world.*src/lib.sev' "$workspace/changed.log"; then
    exit 1
fi
printf 'Native package acceptance workspace: %s\n' "$workspace"
