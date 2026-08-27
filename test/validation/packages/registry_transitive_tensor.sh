#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIRECTORY=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
REPOSITORY_ROOT=$(CDPATH= cd -- "$SCRIPT_DIRECTORY/../../.." && pwd)
SEVERIAN_BIN=${SEVERIAN_BIN:-"$REPOSITORY_ROOT/target/debug/sev"}

if [[ ! -x "$SEVERIAN_BIN" ]]; then
    echo "missing Severian executable: $SEVERIAN_BIN" >&2
    echo "build it first or set SEVERIAN_BIN" >&2
    exit 1
fi

TEST_ROOT=$(mktemp -d "${TMPDIR:-/tmp}/severian-registry-transitive.XXXXXX")
trap 'rm -rf -- "$TEST_ROOT"' EXIT

REGISTRY_ROOT="$TEST_ROOT/registry"
MATRIX_ROOT="$TEST_ROOT/matrix-kernel"
SERVICE_ROOT="$TEST_ROOT/matrix-service"
APPLICATION_ROOT="$TEST_ROOT/matrix-application"
export SEVERIAN_REGISTRY="$REGISTRY_ROOT"

echo "publishing tensor"
"$SEVERIAN_BIN" publish "$REPOSITORY_ROOT/library/tensor"

echo "creating matrix package"
"$SEVERIAN_BIN" new "$MATRIX_ROOT"
cat > "$MATRIX_ROOT/package.toml" <<'TOML'
[package]
name = "matrix-kernel"
version = "0.1.0"
edition = "2026"

[lib]
name = "matrix_kernel_api"
path = "src/lib.sev"

[dependencies]
tensor = "0.1.0"
TOML

cat > "$MATRIX_ROOT/src/lib.sev" <<'SEV'
import tensor

def product() -> list[float]:
    left = tensor.tensor([1.0, 2.0, 3.0, 4.0], [2, 2])
    right = tensor.tensor([5.0, 6.0, 7.0, 8.0], [2, 2])
    return tensor.values(tensor.matmul(left, right))
SEV

echo "publishing matrix package"
"$SEVERIAN_BIN" publish "$MATRIX_ROOT"

echo "creating service package"
"$SEVERIAN_BIN" new "$SERVICE_ROOT"
cat > "$SERVICE_ROOT/package.toml" <<'TOML'
[package]
name = "matrix-service"
version = "0.1.0"
edition = "2026"

[lib]
name = "matrix_service_api"
path = "src/lib.sev"

[dependencies]
matrix_kernel = { package = "matrix-kernel", version = "0.1.0" }
TOML

cat > "$SERVICE_ROOT/src/lib.sev" <<'SEV'
import matrix_kernel

def compute() -> list[float]:
    return matrix_kernel.product()
SEV

echo "publishing service package"
"$SEVERIAN_BIN" publish "$SERVICE_ROOT"

# Only registry realizations may remain available to the final application.
mv "$MATRIX_ROOT" "$TEST_ROOT/matrix-checkout-removed"
mv "$SERVICE_ROOT" "$TEST_ROOT/service-checkout-removed"

echo "creating application package"
"$SEVERIAN_BIN" new "$APPLICATION_ROOT"
cat > "$APPLICATION_ROOT/package.toml" <<'TOML'
[package]
name = "matrix-application"
version = "0.1.0"
edition = "2026"
default-run = "matrix-application"

[[bin]]
name = "matrix-application"
path = "src/main.sev"

[dependencies]
matrix_service = { package = "matrix-service", version = "0.1.0" }
TOML

cat > "$APPLICATION_ROOT/src/main.sev" <<'SEV'
import matrix_service

def main():
    result = matrix_service.compute()
    assert(result == [19.0, 22.0, 43.0, 50.0])
    print("transitive matrix result ok")
SEV

# Package 3 knows only package 2. Package 1 and tensor must be discovered from
# published transitive metadata.
if grep -Eq '^(matrix_kernel|tensor)[[:space:]]*=' "$APPLICATION_ROOT/package.toml"; then
    echo "application leaks an implementation dependency" >&2
    exit 1
fi

# A missing transitive release must be diagnosed through the dependency chain,
# without requiring package 3 to duplicate package 2's implementation details.
MATRIX_RELEASE="$REGISTRY_ROOT/packages/matrix-kernel/0.1.0"
MISSING_MATRIX_RELEASE="$TEST_ROOT/matrix-kernel-release-missing"
mv "$MATRIX_RELEASE" "$MISSING_MATRIX_RELEASE"
if MISSING_OUTPUT=$("$SEVERIAN_BIN" build "$APPLICATION_ROOT" 2>&1); then
    echo "application unexpectedly built without matrix-kernel" >&2
    exit 1
fi
if [[ "$MISSING_OUTPUT" != *'package `matrix-application` dependency `matrix_service`'* \
    || "$MISSING_OUTPUT" != *'package `matrix-service` dependency `matrix_kernel`'* \
    || "$MISSING_OUTPUT" != *'package `matrix-kernel` version `0.1.0` is not present'* ]]; then
    echo "missing dependency diagnostic lost its transitive chain:" >&2
    echo "$MISSING_OUTPUT" >&2
    exit 1
fi
mv "$MISSING_MATRIX_RELEASE" "$MATRIX_RELEASE"

# Transitive availability is not namespace re-export. Package 3 can execute
# package 1 through package 2, but cannot import package 1 without declaring it.
cp "$APPLICATION_ROOT/src/main.sev" "$APPLICATION_ROOT/src/main.valid.sev"
sed -i '1i import matrix_kernel' "$APPLICATION_ROOT/src/main.sev"
if PRIVATE_OUTPUT=$("$SEVERIAN_BIN" build "$APPLICATION_ROOT" 2>&1); then
    echo "application unexpectedly imported an undeclared transitive package" >&2
    exit 1
fi
if [[ "$PRIVATE_OUTPUT" != *'package import `matrix_kernel` has not been resolved'* ]]; then
    echo "undeclared transitive import produced the wrong diagnostic:" >&2
    echo "$PRIVATE_OUTPUT" >&2
    exit 1
fi
mv "$APPLICATION_ROOT/src/main.valid.sev" "$APPLICATION_ROOT/src/main.sev"

echo "building application"
"$SEVERIAN_BIN" build "$APPLICATION_ROOT"

echo "running application"
OUTPUT=$("$SEVERIAN_BIN" run "$APPLICATION_ROOT")
if [[ "$OUTPUT" != "transitive matrix result ok" ]]; then
    echo "unexpected application output: $OUTPUT" >&2
    exit 1
fi

grep -Fq 'matrix_kernel' "$REGISTRY_ROOT/packages/matrix-service/0.1.0/source/package.toml"
grep -Fq 'tensor' "$REGISTRY_ROOT/packages/matrix-kernel/0.1.0/source/package.toml"

echo "transitive tensor package golden path passed"
