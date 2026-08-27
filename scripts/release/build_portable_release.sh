#!/usr/bin/env bash
set -euo pipefail

REPOSITORY_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
PROFILE=release
OUTPUT=
REQUESTED_VERSION=
REQUESTED_TARGET=

while (($#)); do
    case "$1" in
        --profile)
            PROFILE=${2:?--profile requires a value}
            shift 2
            ;;
        --output)
            OUTPUT=${2:?--output requires a value}
            shift 2
            ;;
        --version)
            REQUESTED_VERSION=${2:?--version requires a value}
            shift 2
            ;;
        --target)
            REQUESTED_TARGET=${2:?--target requires a value}
            shift 2
            ;;
        *)
            echo "usage: $0 [--profile dev|release] [--version VERSION] [--target TRIPLE] [--output DIRECTORY]" >&2
            exit 2
            ;;
    esac
done

case "$PROFILE" in
    dev)
        CARGO_PROFILE=dev
        TARGET_PROFILE=debug
        ;;
    release)
        CARGO_PROFILE=release
        TARGET_PROFILE=release
        ;;
    *)
        echo "unsupported release profile: $PROFILE" >&2
        exit 2
        ;;
esac

cargo build --locked --manifest-path "$REPOSITORY_ROOT/Cargo.toml" \
    -p severian-driver --bin sev --profile "$CARGO_PROFILE"

SEV_BINARY="$REPOSITORY_ROOT/target/$TARGET_PROFILE/sev"
BINARY_VERSION=$($SEV_BINARY --version | awk '{print $2}')
VERSION=${REQUESTED_VERSION#v}
VERSION=${VERSION:-$BINARY_VERSION}
if [[ "$VERSION" != "$BINARY_VERSION" ]]; then
    echo "requested release $VERSION does not match sev $BINARY_VERSION" >&2
    exit 1
fi

HOST=$(rustc -vV | awk '/^host:/ {print $2}')
TARGET=${REQUESTED_TARGET:-$HOST}
if [[ "$TARGET" != "$HOST" ]]; then
    echo "release target $TARGET must be built natively on $HOST" >&2
    exit 1
fi

NAME="severian-$VERSION-$TARGET"
if [[ -z "$OUTPUT" ]]; then
    OUTPUT="$REPOSITORY_ROOT/target/release-distributions/$NAME"
fi
if [[ $(basename "$OUTPUT") != "$NAME" ]]; then
    echo "release output must end in $NAME" >&2
    exit 1
fi
if [[ -e "$OUTPUT" || -e "$OUTPUT.tar.zst" ]]; then
    echo "release output already exists: $OUTPUT" >&2
    exit 1
fi

RUNTIME="$OUTPUT/lib/severian"
SHARE="$OUTPUT/share/severian"
mkdir -p "$OUTPUT/bin" "$RUNTIME/bin" "$RUNTIME/lib" "$SHARE/library"
cp "$SEV_BINARY" "$RUNTIME/bin/sev-real"
(
    cd "$REPOSITORY_ROOT/library"
    tar --exclude='*/target' --exclude='*/.git' -cf - .
) | tar -C "$SHARE/library" -xf -

LLVM_CONFIG=${SEVERIAN_LLVM_CONFIG:-$(command -v llvm-config-21)}
LLVM_BINDIR=$($LLVM_CONFIG --bindir)
LLVM_LIBDIR=$($LLVM_CONFIG --libdir)
for tool in clang ld.lld mlir-opt mlir-translate llvm-config; do
    cp "$LLVM_BINDIR/$tool" "$RUNTIME/bin/$tool"
done
for library in libLLVM.so.21.1 libMLIR.so.21.1 libclang-cpp.so.21.1; do
    candidate="$LLVM_LIBDIR/$library"
    if [[ ! -e "$candidate" ]]; then
        candidate=$(ldconfig -p | awk -v name="$library" '$1 == name { print $NF; exit }')
    fi
    if [[ -z "$candidate" || ! -e "$candidate" ]]; then
        echo "could not locate required LLVM runtime library $library" >&2
        exit 1
    fi
    cp -L "$candidate" "$RUNTIME/lib/$library"
done

RESOURCE_DIR=$($LLVM_BINDIR/clang --print-resource-dir)
mkdir -p "$RUNTIME/lib/clang"
cp -a "$RESOURCE_DIR" "$RUNTIME/lib/clang/"

cat > "$OUTPUT/bin/sev" <<'WRAPPER'
#!/usr/bin/env bash
set -euo pipefail
SOURCE=${BASH_SOURCE[0]}
while [[ -L "$SOURCE" ]]; do
    DIRECTORY=$(cd "$(dirname "$SOURCE")" && pwd)
    SOURCE=$(readlink "$SOURCE")
    [[ "$SOURCE" = /* ]] || SOURCE="$DIRECTORY/$SOURCE"
done
SEVERIAN_HOME=$(cd "$(dirname "$SOURCE")/.." && pwd)
RUNTIME="$SEVERIAN_HOME/lib/severian"
export SEVERIAN_HOME
export SEVERIAN_LIBRARY_ROOT="$SEVERIAN_HOME/share/severian/library"
export SEVERIAN_COMPONENT_ROOT="$RUNTIME/components"
export SEVERIAN_CLANG="$RUNTIME/bin/clang"
export SEVERIAN_MLIR_OPT="$RUNTIME/bin/mlir-opt"
export SEVERIAN_MLIR_TRANSLATE="$RUNTIME/bin/mlir-translate"
export SEVERIAN_LLVM_CONFIG="$RUNTIME/bin/llvm-config"
export SEVERIAN_LINKER="$RUNTIME/bin/ld.lld"
export LD_LIBRARY_PATH="$RUNTIME/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
exec "$RUNTIME/bin/sev-real" "$@"
WRAPPER
chmod +x "$OUTPUT/bin/sev" "$RUNTIME/bin/sev-real"

printf '%s\n' "$VERSION" > "$OUTPUT/VERSION"
cat > "$OUTPUT/RELEASE.toml" <<METADATA
format = 1
name = "severian"
version = "$VERSION"
target = "$TARGET"
includes_native_compiler = true
includes_optional_accelerators = false
METADATA
cp "$REPOSITORY_ROOT/LICENSE.md" "$OUTPUT/LICENSE.md"

SOURCE_DATE_EPOCH=${SOURCE_DATE_EPOCH:-$(git -C "$REPOSITORY_ROOT" log -1 --format=%ct)}
tar --sort=name --mtime="@$SOURCE_DATE_EPOCH" --owner=0 --group=0 --numeric-owner \
    --zstd -C "$(dirname "$OUTPUT")" -cf "$OUTPUT.tar.zst" "$NAME"
echo "portable Severian release: $OUTPUT.tar.zst"
