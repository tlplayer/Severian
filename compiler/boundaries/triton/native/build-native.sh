#!/usr/bin/env bash
set -euo pipefail

bridge_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
severian_root=$(cd -- "${bridge_dir}/../../../.." && pwd)
triton_source=${SEVERIAN_TRITON_SOURCE_DIR:-"${severian_root}/../triton"}
build_dir=${SEVERIAN_TRITON_BUILD_DIR:-"${severian_root}/target/severian-triton-native-v5"}
cache_dir=${SEVERIAN_TRITON_CACHE_PATH:-"${severian_root}/target/severian-triton-cache"}
bridge_library="${build_dir}/libseverian_triton_bridge.so"

# This is a toolchain component, not an application object. Reuse a bridge
# whose ABI source is unchanged so repeated setup/build commands are instant.
# Set SEVERIAN_TRITON_FORCE_REBUILD=1 when deliberately rebuilding the donor.
if [[ -f "${bridge_library}" && "${SEVERIAN_TRITON_FORCE_REBUILD:-0}" != "1" ]]; then
  changed=$(find "${bridge_dir}" -type f \
    \( -name '*.cpp' -o -name '*.h' -o -name 'CMakeLists.txt' \) \
    -newer "${bridge_library}" -print -quit)
  if [[ -z "${changed}" ]]; then
    echo "${bridge_library}"
    exit 0
  fi
fi

expected=8957b9aac23e526fb1252c7c3b592e6f43c175c8
actual=$(git -C "${triton_source}" rev-parse HEAD)
if [[ "${actual}" != "${expected}" ]]; then
  echo "pinned Triton mismatch: expected ${expected}, found ${actual}" >&2
  exit 2
fi

generator="Unix Makefiles"
if command -v ninja >/dev/null 2>&1; then
  generator=Ninja
fi

cmake -S "${bridge_dir}" -B "${build_dir}" -G "${generator}" \
  -DSEVERIAN_TRITON_SOURCE_DIR="${triton_source}" \
  -DTRITON_CACHE_PATH="${cache_dir}" \
  -DCMAKE_BUILD_TYPE=Release
cmake --build "${build_dir}" --target severian_triton_bridge \
  --parallel "${SEVERIAN_TRITON_BUILD_JOBS:-2}"
if command -v strip >/dev/null 2>&1; then
  strip --strip-unneeded "${bridge_library}"
fi
echo "${bridge_library}"
